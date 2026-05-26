# types.py — Graph-Native V4 Type Definitions
#
# No ring assumption. No np.roll. No uniform edge count.
#
# NodeField: k components, n nodes, irregular graph.
# EdgeField: directed edges on declared adjacency with
#   downstream/upstream distinction, plus component edges
#   on declared component pairs.
#
# Representation type discipline:
#   NodeField -> EdgeField transition occurs ONLY at operator A
#   EdgeField -> NodeField occurs ONLY through declared Projection
#   No implicit interchange

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from ..topology.declaration import TopologyDeclaration


@dataclass
class NodeField:
    """k-component field on n nodes of an irregular graph.
    data[c, i] = value of component c at node i.

    This is a NodeField. Operators B and R do not accept this type.
    Only operator A transitions NodeField -> EdgeField."""
    data: np.ndarray          # shape (k, n)
    topo: TopologyDeclaration

    def __post_init__(self):
        assert self.data.ndim == 2, "data must be 2D: (k, n)"
        assert self.data.shape == (self.topo.k_components, self.topo.n_nodes), \
            f"data shape {self.data.shape} != ({self.topo.k_components}, {self.topo.n_nodes})"
        assert np.all(np.isfinite(self.data)), \
            "all values must be finite (D membership)"


@dataclass
class EdgeField:
    """Multi-topology edge field on a declared irregular graph.

    downstream[adj_idx, c]: downstream edge for adjacency adj_idx,
        component c. Direction: source -> target as declared.
        downstream[adj_idx, c] = node[c, source] - node[c, target]

    upstream[adj_idx, c]: upstream edge for same adjacency.
        Direction: target -> source (reverse of declared flow).
        upstream[adj_idx, c] = node[c, target] - node[c, source]
        = -downstream[adj_idx, c]  at construction by A.
        After B and R, upstream != -downstream.

    comp[pair_idx, node_idx]: component edge for declared pair
        pair_idx at node node_idx.
        comp[pair_idx, i] = node[a, i] - node[b, i]
        for pair (a, b).

    This is an EdgeField. It is NOT a NodeField.
    Do not index into it as if values represent per-node quantities.
    EdgeField -> NodeField occurs ONLY through declared Projection."""

    downstream: np.ndarray    # shape (n_adjacencies, k_components)
    upstream: np.ndarray      # shape (n_adjacencies, k_components)
    comp: np.ndarray          # shape (n_component_pairs, n_nodes)
    topo: TopologyDeclaration

    def __post_init__(self):
        n_adj = self.topo.n_spatial_adjacencies
        k = self.topo.k_components
        n = self.topo.n_nodes
        n_pairs = len(self.topo.component_pairs)

        assert self.downstream.shape == (n_adj, k), \
            f"downstream shape {self.downstream.shape} != ({n_adj}, {k})"
        assert self.upstream.shape == (n_adj, k), \
            f"upstream shape {self.upstream.shape} != ({n_adj}, {k})"
        assert self.comp.shape == (n_pairs, n), \
            f"comp shape {self.comp.shape} != ({n_pairs}, {n})"

        # D-membership: all edge values must be finite
        assert np.all(np.isfinite(self.downstream)), \
            "downstream edges must be finite (D membership)"
        assert np.all(np.isfinite(self.upstream)), \
            "upstream edges must be finite (D membership)"
        assert np.all(np.isfinite(self.comp)), \
            "comp edges must be finite (D membership)"
