# kernel.py — Graph-Native V4 ABR Kernel Operators
#
# Canonical ordering: A -> B -> R -> E
# V4: no kernel C. C is a declared projection.
#
# ALL operators act on declared adjacency lists.
# No np.roll. No ring assumption. No uniform degree.
#
# A: NodeField -> EdgeField
#   Produces downstream edges, upstream edges, and component edges
#   over declared adjacencies only.
#
# B: EdgeField -> EdgeField
#   Accumulates edges along declared paths. At each adjacency,
#   B sums the edge with edges at continuation nodes along the
#   declared downstream/upstream direction.
#
# R: EdgeField -> EdgeField
#   Cross-couples downstream, upstream, and component edges
#   antisymmetrically.

import numpy as np
from .types import NodeField, EdgeField
from ..topology.declaration import TopologyDeclaration


def operator_a(f: NodeField) -> EdgeField:
    """A — Relational Gradient Extraction.
    NodeField -> EdgeField. The unique representation transition.

    For each declared adjacency (source -> target):
      downstream[adj, c] = node[c, source] - node[c, target]
      upstream[adj, c]   = node[c, target] - node[c, source]

    For each declared component pair (a, b) at each node:
      comp[pair, node] = node[a, node] - node[b, node]

    A(x) depends on pairs of indices. Not index-local."""
    topo = f.topo
    n_adj = topo.n_spatial_adjacencies
    k = topo.k_components
    n = topo.n_nodes

    downstream = np.empty((n_adj, k))
    upstream = np.empty((n_adj, k))

    for adj_idx, adj in enumerate(topo.spatial_adjacencies):
        for c in range(k):
            downstream[adj_idx, c] = f.data[c, adj.source] - f.data[c, adj.target]
            upstream[adj_idx, c] = f.data[c, adj.target] - f.data[c, adj.source]

    n_pairs = len(topo.component_pairs)
    comp = np.empty((n_pairs, n))
    for p_idx, (a, b) in enumerate(topo.component_pairs):
        comp[p_idx] = f.data[a] - f.data[b]

    return EdgeField(downstream=downstream, upstream=upstream,
                     comp=comp, topo=topo)


def operator_b(g: EdgeField) -> EdgeField:
    """B — Local Relational Accumulation.
    EdgeField -> EdgeField. Accumulates along declared paths.

    For each adjacency, B looks for continuation edges:
    - Downstream continuation: edges whose source is this edge's target
      (continuing downstream along the flow)
    - Upstream continuation: edges whose target is this edge's source
      (continuing upstream against the flow)

    At fan-out nodes (multiple downstream continuations), each
    continuation edge is accumulated independently.
    At fan-in nodes (multiple upstream continuations), each
    continuation edge is accumulated independently.

    B does NOT cross-couple downstream and upstream.
    B does NOT cross-couple spatial and component edges.

    Component edges accumulate by averaging continuations at
    downstream neighbors (declared choice — accumulation direction
    for component edges is a declared open condition)."""
    topo = g.topo
    n_adj = topo.n_spatial_adjacencies
    k = topo.k_components
    n = topo.n_nodes

    # Build adjacency index lookup for continuation
    # target_to_adjs[node] = list of adj indices where source == node
    #   (edges leaving this node — downstream continuations)
    # source_to_adjs[node] = list of adj indices where target == node
    #   (edges arriving at this node — upstream continuations)
    target_to_adjs = {i: [] for i in range(n)}
    source_to_adjs = {i: [] for i in range(n)}
    for idx, adj in enumerate(topo.spatial_adjacencies):
        target_to_adjs[adj.source].append(idx)
        source_to_adjs[adj.target].append(idx)

    # --- Downstream accumulation ---
    # For edge adj_idx (source -> target):
    #   continuation edges are those leaving target (downstream from target)
    #   B_down[adj, c] = down[adj, c] + mean(down[cont, c] for cont leaving target)
    downstream = g.downstream.copy()
    for adj_idx, adj in enumerate(topo.spatial_adjacencies):
        cont_indices = target_to_adjs[adj.target]
        if cont_indices:
            cont_mean = np.mean(
                [g.downstream[ci] for ci in cont_indices], axis=0)
            downstream[adj_idx] += cont_mean

    # --- Upstream accumulation ---
    # For edge adj_idx (target -> source in upstream direction):
    #   continuation edges are those arriving at source (upstream from source)
    #   B_up[adj, c] = up[adj, c] + mean(up[cont, c] for cont arriving at source)
    upstream = g.upstream.copy()
    for adj_idx, adj in enumerate(topo.spatial_adjacencies):
        cont_indices = source_to_adjs[adj.source]
        if cont_indices:
            cont_mean = np.mean(
                [g.upstream[ci] for ci in cont_indices], axis=0)
            upstream[adj_idx] += cont_mean

    # --- Component edge accumulation ---
    # Accumulate at each node by averaging component edges at
    # downstream neighbors (declared choice)
    comp = g.comp.copy()
    for node_idx in range(n):
        down_neighbors = topo.downstream_neighbors(node_idx)
        if down_neighbors:
            neighbor_mean = np.mean(
                [g.comp[:, nb] for nb in down_neighbors], axis=0)
            comp[:, node_idx] += neighbor_mean

    return EdgeField(downstream=downstream, upstream=upstream,
                     comp=comp, topo=topo)


def compute_rho(a_out: EdgeField, rho_base: float) -> np.ndarray:
    """Per-node circulation strength derived from A(x).
    rho[node] = rho_base * max_grad[node] / (1 + max_grad[node])

    max_grad at each node is the maximum absolute edge magnitude
    across all edge types touching that node.

    Per-node. No aggregation beyond the local neighborhood."""
    assert 0 < rho_base <= 0.5, f"rho_base must be in (0, 0.5]: got {rho_base}"

    topo = a_out.topo
    n = topo.n_nodes
    max_grad = np.zeros(n)

    # Spatial edges touching each node
    for adj_idx, adj in enumerate(topo.spatial_adjacencies):
        for c in range(topo.k_components):
            mag = max(abs(a_out.downstream[adj_idx, c]),
                      abs(a_out.upstream[adj_idx, c]))
            max_grad[adj.source] = max(max_grad[adj.source], mag)
            max_grad[adj.target] = max(max_grad[adj.target], mag)

    # Component edges at each node
    for p_idx in range(len(topo.component_pairs)):
        for node_idx in range(n):
            max_grad[node_idx] = max(max_grad[node_idx],
                                     abs(a_out.comp[p_idx, node_idx]))

    return rho_base * max_grad / (1.0 + max_grad)


def operator_r(bg: EdgeField, rho: np.ndarray) -> EdgeField:
    """R — Antisymmetric Circulation.
    EdgeField -> EdgeField. Cross-topology coupling.

    Three coupling modes (all antisymmetric, local, additive):

    1. Downstream <-> Upstream coupling:
       At each adjacency, downstream edges receive the asymmetry
       between upstream edges at continuation nodes, and vice versa.

    2. Spatial <-> Component coupling:
       Downstream/upstream edges receive component-edge asymmetry
       at their source and target nodes.
       Component edges receive the asymmetry between downstream
       and upstream spatial edges at their node.

    R(e) depends on multiple local edge relationships across
    declared topologies. Not index-local."""
    assert rho.shape == (bg.topo.n_nodes,), \
        f"rho must be per-node ({bg.topo.n_nodes},): got {rho.shape}"

    topo = bg.topo
    n_adj = topo.n_spatial_adjacencies
    k = topo.k_components
    n = topo.n_nodes
    pairs = topo.component_pairs

    downstream = bg.downstream.copy()
    upstream = bg.upstream.copy()
    comp = bg.comp.copy()

    # --- 1. Downstream <-> Upstream coupling at each adjacency ---
    # Asymmetry between downstream and upstream at same adjacency
    for adj_idx in range(n_adj):
        adj = topo.spatial_adjacencies[adj_idx]
        rho_src = rho[adj.source]
        rho_tgt = rho[adj.target]
        rho_edge = 0.5 * (rho_src + rho_tgt)

        # downstream receives upstream asymmetry, upstream receives downstream
        down_up_asym = bg.downstream[adj_idx] - bg.upstream[adj_idx]
        downstream[adj_idx] += rho_edge * (-down_up_asym)  # push toward upstream
        upstream[adj_idx] += rho_edge * down_up_asym        # push toward downstream

    # --- 2. Spatial edges receive component-edge asymmetry ---
    for adj_idx, adj in enumerate(topo.spatial_adjacencies):
        rho_src = rho[adj.source]
        rho_tgt = rho[adj.target]

        for p_idx, (ca, cb) in enumerate(pairs):
            # Component edge asymmetry at source and target
            comp_asym_src = bg.comp[p_idx, adj.source]
            comp_asym_tgt = bg.comp[p_idx, adj.target]
            comp_asym = 0.5 * (comp_asym_src + comp_asym_tgt)

            # Antisymmetric: first component in pair gets +, second gets -
            downstream[adj_idx, ca] += rho_src * comp_asym
            downstream[adj_idx, cb] -= rho_src * comp_asym
            upstream[adj_idx, ca] += rho_tgt * comp_asym
            upstream[adj_idx, cb] -= rho_tgt * comp_asym

    # --- 3. Component edges receive spatial-edge asymmetry ---
    # At each node, compute the asymmetry between downstream and
    # upstream spatial edges, then couple into component edges
    for node_idx in range(n):
        rho_n = rho[node_idx]

        # Gather spatial edge magnitudes at this node
        # Downstream edges leaving this node
        down_out = [(adj_idx, adj) for adj_idx, adj
                    in enumerate(topo.spatial_adjacencies)
                    if adj.source == node_idx]
        # Upstream edges arriving at this node
        up_in = [(adj_idx, adj) for adj_idx, adj
                 in enumerate(topo.spatial_adjacencies)
                 if adj.target == node_idx]

        if not down_out and not up_in:
            continue

        # Average downstream and upstream spatial edges at this node
        if down_out:
            avg_down = np.mean(
                [bg.downstream[ai] for ai, _ in down_out], axis=0)
        else:
            avg_down = np.zeros(k)

        if up_in:
            avg_up = np.mean(
                [bg.upstream[ai] for ai, _ in up_in], axis=0)
        else:
            avg_up = np.zeros(k)

        # Spatial asymmetry: downstream vs upstream at this node
        spatial_asym = avg_down - avg_up

        # Couple into component edges
        for p_idx, (ca, cb) in enumerate(pairs):
            # Component edge receives asymmetry between the two
            # components' spatial behavior
            comp[p_idx, node_idx] += rho_n * (spatial_asym[ca] - spatial_asym[cb])

    return EdgeField(downstream=downstream, upstream=upstream,
                     comp=comp, topo=topo)


def operator_e(f: NodeField, rho_base: float = 0.3) -> EdgeField:
    """E — V4 Kernel Composition.
    E(x, rho) = R(B(A(x)), rho(A(x)))

    Full operator sequence on declared topology. No kernel C.
    Returns EdgeField (not NodeField — no implicit projection).

    Topology is carried by the NodeField and propagates through
    every operator. No topology is inferred or defaulted."""
    a = operator_a(f)
    rho = compute_rho(a, rho_base)
    b = operator_b(a)
    return operator_r(b, rho)
