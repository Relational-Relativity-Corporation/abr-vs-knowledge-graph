# declaration.py — Graph-Native Topology Declaration
#
# V4 DISCIPLINE: There is no topological default.
# Ring topology is the PROOF topology (V3 canonical).
# Every V4 application declares its own spatial and component
# topology through Origin as part of M.
#
# Spatial topology: directed graph with explicit adjacency lists.
# Each adjacency produces TWO directed edge types (downstream, upstream).
# Nodes may have irregular degree.
#
# Component topology: declared subset of component pairs.
# NOT all-pairs by default. Origin declares which components
# are operationally adjacent.

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


@dataclass
class DirectedAdjacency:
    """A single declared directed adjacency between two nodes.
    source -> target defines the downstream direction.
    target -> source is the upstream direction.
    Both are first-class edges with distinct identity."""
    source: int
    target: int

    def __post_init__(self):
        assert self.source != self.target, \
            f"Self-adjacency ({self.source} -> {self.source}) is inadmissible"


@dataclass
class TopologyDeclaration:
    """Complete topology declaration for a V4 application.
    Declared by Origin as part of M before any operator acts.

    n_nodes: number of nodes in the spatial graph
    k_components: number of measurement components per node
    node_names: human-readable names for each node
    component_names: human-readable names for each component
    spatial_adjacencies: list of directed adjacencies (source -> target)
    component_pairs: list of declared component coupling pairs (a, b)

    Every node must have at least one adjacency (no isolated nodes).
    Every component pair must reference valid component indices.
    No self-coupling in component pairs."""

    n_nodes: int
    k_components: int
    node_names: List[str]
    component_names: List[str]
    spatial_adjacencies: List[DirectedAdjacency]
    component_pairs: List[Tuple[int, int]]

    def __post_init__(self):
        # Node count matches names
        assert len(self.node_names) == self.n_nodes, \
            f"node_names length {len(self.node_names)} != n_nodes {self.n_nodes}"

        # Component count matches names
        assert len(self.component_names) == self.k_components, \
            f"component_names length {len(self.component_names)} != k_components {self.k_components}"

        # All adjacency indices in range
        for adj in self.spatial_adjacencies:
            assert 0 <= adj.source < self.n_nodes, \
                f"source {adj.source} out of range for n_nodes={self.n_nodes}"
            assert 0 <= adj.target < self.n_nodes, \
                f"target {adj.target} out of range for n_nodes={self.n_nodes}"

        # No isolated nodes — every node appears in at least one adjacency
        connected = set()
        for adj in self.spatial_adjacencies:
            connected.add(adj.source)
            connected.add(adj.target)
        isolated = set(range(self.n_nodes)) - connected
        assert len(isolated) == 0, \
            f"Isolated nodes (no declared adjacency): {isolated}"

        # Component pairs valid
        for a, b in self.component_pairs:
            assert 0 <= a < self.k_components, \
                f"component pair index {a} out of range"
            assert 0 <= b < self.k_components, \
                f"component pair index {b} out of range"
            assert a != b, \
                f"self-coupling ({a}, {a}) is inadmissible"

    @property
    def n_spatial_adjacencies(self) -> int:
        """Number of declared spatial adjacencies."""
        return len(self.spatial_adjacencies)

    @property
    def n_downstream_edges(self) -> int:
        """Total downstream edges: adjacencies * components."""
        return self.n_spatial_adjacencies * self.k_components

    @property
    def n_upstream_edges(self) -> int:
        """Total upstream edges: same count as downstream."""
        return self.n_spatial_adjacencies * self.k_components

    @property
    def n_component_edges(self) -> int:
        """Total component edges: pairs * nodes."""
        return len(self.component_pairs) * self.n_nodes

    @property
    def total_edge_count(self) -> int:
        """Total edges in the multi-topology edge field."""
        return self.n_downstream_edges + self.n_upstream_edges + self.n_component_edges

    def downstream_neighbors(self, node: int) -> List[int]:
        """Nodes that receive from this node (downstream targets)."""
        return [adj.target for adj in self.spatial_adjacencies
                if adj.source == node]

    def upstream_neighbors(self, node: int) -> List[int]:
        """Nodes that supply to this node (upstream sources)."""
        return [adj.source for adj in self.spatial_adjacencies
                if adj.target == node]

    def node_degree(self, node: int) -> Dict[str, int]:
        """Degree breakdown for a node."""
        down = len(self.downstream_neighbors(node))
        up = len(self.upstream_neighbors(node))
        return {'downstream_out': down, 'upstream_in': up,
                'total': down + up}

    def summary(self) -> str:
        """Human-readable topology summary for Origin verification."""
        lines = [
            f"Topology Declaration",
            f"  Nodes: {self.n_nodes} ({', '.join(self.node_names)})",
            f"  Components: {self.k_components} ({', '.join(self.component_names)})",
            f"  Spatial adjacencies: {self.n_spatial_adjacencies}",
            f"  Component pairs: {len(self.component_pairs)}",
            f"  Edge field dimensionality:",
            f"    Downstream edges: {self.n_downstream_edges}",
            f"    Upstream edges:   {self.n_upstream_edges}",
            f"    Component edges:  {self.n_component_edges}",
            f"    Total:            {self.total_edge_count}",
            f"",
            f"  Spatial structure:",
        ]
        for adj in self.spatial_adjacencies:
            s = self.node_names[adj.source]
            t = self.node_names[adj.target]
            lines.append(f"    {s} -> {t}")

        lines.append(f"")
        lines.append(f"  Component coupling:")
        for a, b in self.component_pairs:
            lines.append(
                f"    {self.component_names[a]} <-> {self.component_names[b]}")

        lines.append(f"")
        lines.append(f"  Node degrees:")
        for i in range(self.n_nodes):
            d = self.node_degree(i)
            lines.append(
                f"    {self.node_names[i]}: "
                f"out={d['downstream_out']}, in={d['upstream_in']}")

        return "\n".join(lines)
