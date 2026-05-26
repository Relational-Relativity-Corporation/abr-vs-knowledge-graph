# test_operator_invariants.py — Graph-Native Kernel Verification
# Must pass before simulation or detection code is written.
#
# All tests use the declared supply chain topology.
# No ring assumptions. No np.roll.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.topology.supply_chain import declare_supply_chain_topology
from src.operators.types import NodeField, EdgeField
from src.operators.kernel import operator_a, operator_b, operator_r, compute_rho, operator_e
from src.operators.diagnostics import (
    sigma_sq_total, sigma_sq_downstream, sigma_sq_upstream,
    sigma_sq_comp, gamma_decomposed, gamma_total, format_gamma_report,
)


def make_field(data: np.ndarray) -> NodeField:
    """Helper: wrap data array as NodeField on supply chain topology."""
    topo = declare_supply_chain_topology()
    return NodeField(data=data, topo=topo)


def make_steady_state() -> NodeField:
    """Synthetic steady-state supply chain field.
    Different components at different scales, smooth spatial variation."""
    topo = declare_supply_chain_topology()
    n, k = topo.n_nodes, topo.k_components
    np.random.seed(42)
    data = np.zeros((k, n))
    # inventory_level: decreasing along chain (suppliers have more)
    data[0] = np.array([1000, 800, 600, 400, 350, 300, 200, 150], dtype=float)
    # lead_time: increasing along chain
    data[1] = np.array([2, 3, 5, 4, 5, 6, 3, 4], dtype=float)
    # unit_cost: varies
    data[2] = np.array([10, 12, 15, 18, 19, 20, 25, 22], dtype=float)
    # quality_score: high, slight degradation along chain
    data[3] = np.array([0.95, 0.93, 0.90, 0.88, 0.87, 0.85, 0.82, 0.84], dtype=float)
    # throughput: varies by node type
    data[4] = np.array([500, 400, 800, 600, 500, 400, 700, 600], dtype=float)
    return NodeField(data=data, topo=topo)


# === TESTS ===

def test_a_produces_edge_field():
    """A transitions NodeField -> EdgeField on irregular graph."""
    f = make_steady_state()
    a = operator_a(f)

    assert isinstance(a, EdgeField), "A must produce EdgeField"
    assert a.downstream.shape == (7, 5), f"downstream shape: {a.downstream.shape}"
    assert a.upstream.shape == (7, 5), f"upstream shape: {a.upstream.shape}"
    assert a.comp.shape == (6, 8), f"comp shape: {a.comp.shape}"
    print("PASS: A produces EdgeField (downstream=7x5, upstream=7x5, comp=6x8)")


def test_a_correct_differences():
    """A computes correct directed differences on declared adjacencies."""
    f = make_steady_state()
    topo = f.topo
    a = operator_a(f)

    # Check first adjacency: Shanghai(0) -> Port(2)
    adj0 = topo.spatial_adjacencies[0]
    assert adj0.source == 0 and adj0.target == 2

    for c in range(topo.k_components):
        expected_down = f.data[c, 0] - f.data[c, 2]
        expected_up = f.data[c, 2] - f.data[c, 0]
        assert abs(a.downstream[0, c] - expected_down) < 1e-12, \
            f"downstream[0, {c}]: expected {expected_down}, got {a.downstream[0, c]}"
        assert abs(a.upstream[0, c] - expected_up) < 1e-12, \
            f"upstream[0, {c}]: expected {expected_up}, got {a.upstream[0, c]}"

    print("PASS: A computes correct directed differences")


def test_a_antisymmetry():
    """At construction, upstream = -downstream for each adjacency."""
    f = make_steady_state()
    a = operator_a(f)
    np.testing.assert_allclose(a.upstream, -a.downstream, atol=1e-12)
    print("PASS: A produces antisymmetric downstream/upstream")


def test_a_translation_invariance():
    """A(x + c) produces same edge field as A(x). (Axiom 1)"""
    f1 = make_steady_state()
    f2 = NodeField(data=f1.data + 999.0, topo=f1.topo)
    a1 = operator_a(f1)
    a2 = operator_a(f2)
    np.testing.assert_allclose(a1.downstream, a2.downstream, atol=1e-12)
    np.testing.assert_allclose(a1.upstream, a2.upstream, atol=1e-12)
    np.testing.assert_allclose(a1.comp, a2.comp, atol=1e-12)
    print("PASS: A is translation invariant (Axiom 1)")


def test_b_preserves_edge_type():
    """B produces EdgeField, same shape as input."""
    f = make_steady_state()
    a = operator_a(f)
    b = operator_b(a)
    assert isinstance(b, EdgeField)
    assert b.downstream.shape == a.downstream.shape
    assert b.upstream.shape == a.upstream.shape
    assert b.comp.shape == a.comp.shape
    print("PASS: B preserves EdgeField shape")


def test_b_modifies_values():
    """B actually accumulates (output differs from input)."""
    f = make_steady_state()
    a = operator_a(f)
    b = operator_b(a)
    assert not np.allclose(b.downstream, a.downstream), \
        "B should modify downstream edges"
    print("PASS: B modifies edge values (accumulation occurring)")


def test_b_no_downstream_upstream_cross():
    """B does not cross-couple downstream and upstream."""
    f = make_steady_state()
    a = operator_a(f)

    # Zero out upstream, run B, check downstream unchanged
    a_mod = EdgeField(
        downstream=a.downstream.copy(),
        upstream=np.zeros_like(a.upstream),
        comp=a.comp.copy(),
        topo=a.topo
    )
    b_mod = operator_b(a_mod)
    b_orig = operator_b(a)

    np.testing.assert_allclose(b_mod.downstream, b_orig.downstream, atol=1e-12)
    print("PASS: B does not cross-couple downstream and upstream")


def test_r_output_finite():
    """R produces finite output on supply chain field."""
    f = make_steady_state()
    a = operator_a(f)
    rho = compute_rho(a, 0.3)
    b = operator_b(a)
    r = operator_r(b, rho)

    assert np.all(np.isfinite(r.downstream)), "R downstream must be finite"
    assert np.all(np.isfinite(r.upstream)), "R upstream must be finite"
    assert np.all(np.isfinite(r.comp)), "R comp must be finite"
    print("PASS: R produces finite output")


def test_r_breaks_antisymmetry():
    """After B and R, upstream != -downstream (coupling has broken initial antisymmetry)."""
    f = make_steady_state()
    e = operator_e(f, rho_base=0.3)

    # After full ABR, antisymmetry should be broken
    diff = np.max(np.abs(e.upstream + e.downstream))
    assert diff > 1e-6, \
        f"After ABR, upstream should != -downstream (max diff = {diff})"
    print(f"PASS: R breaks downstream/upstream antisymmetry (max diff = {diff:.4f})")


def test_r_nonzero_variance_all_types():
    """R output has nonzero variance in all declared edge types."""
    f = make_steady_state()
    e = operator_e(f, rho_base=0.3)

    k = f.topo.k_components
    for c in range(k):
        assert sigma_sq_downstream(e, c) > 0, \
            f"downstream variance for {f.topo.component_names[c]} must be > 0"
        assert sigma_sq_upstream(e, c) > 0, \
            f"upstream variance for {f.topo.component_names[c]} must be > 0"

    for p in range(len(f.topo.component_pairs)):
        assert sigma_sq_comp(e, p) > 0, \
            f"comp variance for pair {p} must be > 0"

    print("PASS: R output has nonzero variance in all edge types")


def test_gamma_computable():
    """Gamma is computable and finite per edge type."""
    f = make_steady_state()
    gd = gamma_decomposed(f, rho_base=0.3)

    assert np.isfinite(gd['gamma_total']), "Total Gamma must be finite"
    for v in gd['gamma_downstream']:
        assert np.isfinite(v), "Downstream Gamma must be finite"
    for v in gd['gamma_upstream']:
        assert np.isfinite(v), "Upstream Gamma must be finite"
    for v in gd['gamma_comp']:
        assert np.isfinite(v), "Component Gamma must be finite"

    print(f"PASS: Gamma computable and finite (total = {gd['gamma_total']:.6f})")
    print()
    print(format_gamma_report(gd))


def test_operator_ordering_matters():
    """Reordering operators produces different output."""
    f = make_steady_state()
    a = operator_a(f)
    rho = compute_rho(a, 0.3)
    b = operator_b(a)

    # Correct: A -> B -> R
    r_correct = operator_r(b, rho)

    # Wrong: A -> R -> B
    r_wrong_input = operator_r(a, rho)
    b_wrong = operator_b(r_wrong_input)

    diff = np.max(np.abs(r_correct.downstream - b_wrong.downstream))
    assert diff > 1e-6, "Reordering must produce different output"
    print(f"PASS: Operator ordering matters (max diff = {diff:.4f})")


def test_identical_components_zero_comp_edges():
    """If all components are identical, component edges are zero."""
    topo = declare_supply_chain_topology()
    base = np.array([100, 80, 60, 40, 35, 30, 20, 15], dtype=float)
    data = np.tile(base, (topo.k_components, 1))
    f = NodeField(data=data, topo=topo)
    e = operator_e(f, rho_base=0.3)

    comp_max = np.max(np.abs(e.comp))
    assert comp_max < 1e-10, \
        f"Identical components: comp edges must be ~0, got {comp_max}"
    print("PASS: Identical components produce zero component edges")


if __name__ == '__main__':
    test_a_produces_edge_field()
    test_a_correct_differences()
    test_a_antisymmetry()
    test_a_translation_invariance()
    test_b_preserves_edge_type()
    test_b_modifies_values()
    test_b_no_downstream_upstream_cross()
    test_r_output_finite()
    test_r_breaks_antisymmetry()
    test_r_nonzero_variance_all_types()
    test_gamma_computable()
    test_operator_ordering_matters()
    test_identical_components_zero_comp_edges()
    print("\n=== ALL OPERATOR TESTS PASSED ===")
