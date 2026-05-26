# test_topology_declaration.py — Topology Admissibility
# Verifies the supply chain topology declaration satisfies
# the admissibility conditions from the topology paper:
#   - Deterministic continuation at every node
#   - No undeclared boundaries
#   - Finite indexing
#   - Structural properties of the declared graph

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.topology.declaration import TopologyDeclaration, DirectedAdjacency
from src.topology.supply_chain import (
    declare_supply_chain_topology,
    SUPPLIER_SHANGHAI, SUPPLIER_VIETNAM, PORT_LONGBEACH,
    WAREHOUSE_WEST, WAREHOUSE_CENTRAL, WAREHOUSE_EAST,
    RETAILER_BIGBOX, RETAILER_ONLINE,
)


def test_topology_creates():
    """Topology declaration constructs without error."""
    topo = declare_supply_chain_topology()
    assert topo.n_nodes == 8
    assert topo.k_components == 5
    print("PASS: Topology creates (8 nodes, 5 components)")


def test_no_isolated_nodes():
    """Every node appears in at least one adjacency."""
    topo = declare_supply_chain_topology()
    connected = set()
    for adj in topo.spatial_adjacencies:
        connected.add(adj.source)
        connected.add(adj.target)
    assert connected == set(range(8)), \
        f"Not all nodes connected: missing {set(range(8)) - connected}"
    print("PASS: No isolated nodes")


def test_no_self_adjacency():
    """No node is adjacent to itself."""
    topo = declare_supply_chain_topology()
    for adj in topo.spatial_adjacencies:
        assert adj.source != adj.target
    print("PASS: No self-adjacency")


def test_no_self_coupling():
    """No component is coupled to itself."""
    topo = declare_supply_chain_topology()
    for a, b in topo.component_pairs:
        assert a != b
    print("PASS: No self-coupling in component pairs")


def test_component_pairs_in_range():
    """All component pair indices within [0, k)."""
    topo = declare_supply_chain_topology()
    for a, b in topo.component_pairs:
        assert 0 <= a < topo.k_components
        assert 0 <= b < topo.k_components
    print("PASS: Component pairs in range")


def test_declared_subset():
    """Component topology is a declared subset, not all-pairs."""
    topo = declare_supply_chain_topology()
    max_pairs = topo.k_components * (topo.k_components - 1) // 2
    assert len(topo.component_pairs) < max_pairs, \
        f"Expected declared subset, got all {max_pairs} pairs"
    print(f"PASS: Declared {len(topo.component_pairs)} of {max_pairs} possible pairs")


def test_irregular_degree():
    """Graph has irregular node degree (not a ring)."""
    topo = declare_supply_chain_topology()
    degrees = [topo.node_degree(i)['total'] for i in range(topo.n_nodes)]
    assert len(set(degrees)) > 1, "All nodes have same degree — ring-like"

    # Port_LongBeach should have higher degree (fan-in from 2 suppliers + fan-out)
    port_degree = topo.node_degree(PORT_LONGBEACH)['total']
    assert port_degree >= 3, f"Port should have degree >= 3, got {port_degree}"
    print(f"PASS: Irregular degree (degrees: {degrees})")


def test_directional_structure():
    """Downstream and upstream neighbors are distinct."""
    topo = declare_supply_chain_topology()

    # Suppliers have downstream only (they supply, receive nothing)
    sh_down = topo.downstream_neighbors(SUPPLIER_SHANGHAI)
    sh_up = topo.upstream_neighbors(SUPPLIER_SHANGHAI)
    assert len(sh_down) > 0, "Shanghai should have downstream neighbors"
    assert len(sh_up) == 0, "Shanghai should have no upstream neighbors"

    # Retailers have upstream only (they receive, don't supply further)
    bb_down = topo.downstream_neighbors(RETAILER_BIGBOX)
    bb_up = topo.upstream_neighbors(RETAILER_BIGBOX)
    assert len(bb_down) == 0, "BigBox should have no downstream neighbors"
    assert len(bb_up) > 0, "BigBox should have upstream neighbors"

    print("PASS: Directional structure (suppliers out-only, retailers in-only)")


def test_edge_field_dimensionality():
    """Edge field has correct total dimensionality."""
    topo = declare_supply_chain_topology()

    # 7 adjacencies * 5 components * 2 directions + 6 pairs * 8 nodes
    expected_down = 7 * 5
    expected_up = 7 * 5
    expected_comp = 6 * 8
    expected_total = expected_down + expected_up + expected_comp

    assert topo.n_downstream_edges == expected_down
    assert topo.n_upstream_edges == expected_up
    assert topo.n_component_edges == expected_comp
    assert topo.total_edge_count == expected_total

    print(f"PASS: Edge field dimensionality = {expected_total} "
          f"(down={expected_down}, up={expected_up}, comp={expected_comp})")


def test_summary_prints():
    """Summary produces readable output for Origin verification."""
    topo = declare_supply_chain_topology()
    s = topo.summary()
    assert "Supplier_Shanghai" in s
    assert "Retailer_Online" in s
    assert "inventory_level" in s
    assert "downstream" in s.lower() or "Downstream" in s
    print("PASS: Summary produces readable output")
    print()
    print(s)


if __name__ == '__main__':
    test_topology_creates()
    test_no_isolated_nodes()
    test_no_self_adjacency()
    test_no_self_coupling()
    test_component_pairs_in_range()
    test_declared_subset()
    test_irregular_degree()
    test_directional_structure()
    test_edge_field_dimensionality()
    test_summary_prints()
    print("\n=== ALL TOPOLOGY TESTS PASSED ===")
