# test_simulation_dynamics.py — Simulation Verification
# MUST PASS before detection is layered on top.
#
# Protocol: verify dynamics produce correct baseline behavior,
# correct perturbation propagation, and realistic cross-component
# coupling BEFORE any ABR or graph detection touches the data.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.simulation.supply_chain import (
    SupplyChainSimulation, steady_state_baseline,
)
from src.simulation.measurement_mapping import (
    scale_to_domain, SUPPLY_CHAIN_SCALES,
)
from src.topology.supply_chain import (
    SUPPLIER_SHANGHAI, SUPPLIER_VIETNAM, PORT_LONGBEACH,
    WAREHOUSE_WEST, WAREHOUSE_CENTRAL, WAREHOUSE_EAST,
    RETAILER_BIGBOX, RETAILER_ONLINE,
    INVENTORY_LEVEL, LEAD_TIME, UNIT_COST, QUALITY_SCORE, THROUGHPUT,
)


def test_baseline_physical_bounds():
    """Baseline values are physically reasonable."""
    base = steady_state_baseline()
    assert np.all(base[INVENTORY_LEVEL] >= 0), "inventory must be >= 0"
    assert np.all(base[LEAD_TIME] > 0), "lead time must be > 0"
    assert np.all(base[UNIT_COST] > 0), "cost must be > 0"
    assert np.all(base[QUALITY_SCORE] >= 0) and np.all(base[QUALITY_SCORE] <= 1), \
        "quality must be in [0, 1]"
    assert np.all(base[THROUGHPUT] >= 0), "throughput must be >= 0"
    print("PASS: Baseline physical bounds")


def test_baseline_structure():
    """Baseline has correct shape and no NaN/inf."""
    base = steady_state_baseline()
    assert base.shape == (5, 8), f"expected (5, 8), got {base.shape}"
    assert np.all(np.isfinite(base)), "baseline must be finite"
    print("PASS: Baseline structure (5, 8), all finite")


def test_scaling_admissible():
    """M scaling preserves pairwise differences (uniform per component)."""
    base = steady_state_baseline()
    scaled = scale_to_domain(base, SUPPLY_CHAIN_SCALES)

    # For each component, check that ratio of pairwise differences
    # is preserved (uniform scaling)
    for c in range(5):
        raw_diff = base[c, 0] - base[c, 1]
        scaled_diff = scaled[c, 0] - scaled[c, 1]
        if abs(raw_diff) > 1e-10:
            expected_ratio = 1.0 / SUPPLY_CHAIN_SCALES[c]
            actual_ratio = scaled_diff / raw_diff
            assert abs(actual_ratio - expected_ratio) < 1e-10, \
                f"component {c}: scaling not uniform"
    print("PASS: M scaling is admissible (uniform per component)")


def test_steady_state_stable():
    """Before perturbation, field is approximately constant (+ noise)."""
    sim = SupplyChainSimulation(t_event=30, n_steps=80, noise_level=0.02)
    data = sim.generate()

    # Check first 25 steps (well before t_event=30)
    pre_event = data[:25]
    for c in range(5):
        for n in range(8):
            series = pre_event[:, c, n]
            cv = np.std(series) / max(np.mean(series), 1e-6)
            assert cv < 0.10, \
                f"Pre-event instability: comp {c}, node {n}, CV = {cv:.4f}"
    print("PASS: Steady state stable before perturbation (CV < 10%)")


def test_perturbation_hits_shanghai_first():
    """Shanghai lead_time increases before any other node's lead_time."""
    sim = SupplyChainSimulation(t_event=30, n_steps=80)
    data = sim.generate()
    base = steady_state_baseline()

    # Shanghai lead_time should deviate from baseline at t_event
    lt_shanghai = data[:, LEAD_TIME, SUPPLIER_SHANGHAI]
    lt_baseline = base[LEAD_TIME, SUPPLIER_SHANGHAI]

    # At t=29 (before event), should be near baseline
    assert abs(lt_shanghai[29] - lt_baseline) / lt_baseline < 0.10, \
        "Shanghai lead_time should be near baseline before t_event"

    # At t=35 (during ramp), should be elevated
    assert lt_shanghai[35] > lt_baseline * 1.5, \
        f"Shanghai lead_time should be elevated at t=35: {lt_shanghai[35]:.1f} vs {lt_baseline:.1f}"

    # Other suppliers should NOT be directly affected
    lt_vietnam = data[:, LEAD_TIME, SUPPLIER_VIETNAM]
    lt_vn_baseline = base[LEAD_TIME, SUPPLIER_VIETNAM]
    assert abs(lt_vietnam[35] - lt_vn_baseline) / lt_vn_baseline < 0.15, \
        "Vietnam lead_time should not be directly perturbed"

    print("PASS: Perturbation hits Shanghai first")


def test_downstream_propagation_delay():
    """Perturbation arrives at downstream nodes with declared delay."""
    sim = SupplyChainSimulation(t_event=30, n_steps=80)
    timeline = sim.perturbation_timeline()

    # Port should arrive before warehouses, warehouses before retailers
    port_arrival = timeline['arrival_times']['Port_LongBeach']['arrival_step']
    west_arrival = timeline['arrival_times']['Warehouse_West']['arrival_step']
    bigbox_arrival = timeline['arrival_times']['Retailer_BigBox']['arrival_step']

    assert port_arrival < west_arrival, \
        f"Port ({port_arrival}) should arrive before West ({west_arrival})"
    assert west_arrival <= bigbox_arrival, \
        f"West ({west_arrival}) should arrive before/with BigBox ({bigbox_arrival})"

    print(f"PASS: Downstream propagation delay "
          f"(Port={port_arrival}, West={west_arrival}, BigBox={bigbox_arrival})")


def test_propagation_actually_occurs():
    """Downstream nodes show lead_time increase after propagation delay."""
    sim = SupplyChainSimulation(t_event=30, n_steps=80)
    data = sim.generate()
    base = steady_state_baseline()
    timeline = sim.perturbation_timeline()

    # Check Port_LongBeach after its arrival time
    port_full = timeline['arrival_times']['Port_LongBeach']['full_arrival']
    lt_port_after = data[min(port_full + 5, 79), LEAD_TIME, PORT_LONGBEACH]
    lt_port_base = base[LEAD_TIME, PORT_LONGBEACH]

    assert lt_port_after > lt_port_base * 1.1, \
        f"Port lead_time should increase after arrival: " \
        f"{lt_port_after:.2f} vs baseline {lt_port_base:.2f}"

    print(f"PASS: Downstream propagation occurs "
          f"(Port lead_time: {lt_port_base:.1f} -> {lt_port_after:.1f})")


def test_cross_component_coupling():
    """Lead_time increase drives inventory decrease (declared coupling)."""
    sim = SupplyChainSimulation(t_event=30, n_steps=80, noise_level=0.0)
    data = sim.generate()
    base = steady_state_baseline()

    # At Shanghai, after perturbation:
    # lead_time up -> inventory should decrease (coupling strength -0.15)
    t_check = 50  # well after perturbation
    inv_shanghai_after = data[t_check, INVENTORY_LEVEL, SUPPLIER_SHANGHAI]
    inv_shanghai_base = base[INVENTORY_LEVEL, SUPPLIER_SHANGHAI]

    assert inv_shanghai_after < inv_shanghai_base, \
        f"Shanghai inventory should decrease when lead_time increases: " \
        f"{inv_shanghai_after:.1f} vs baseline {inv_shanghai_base:.1f}"

    print(f"PASS: Cross-component coupling "
          f"(Shanghai inventory: {inv_shanghai_base:.0f} -> {inv_shanghai_after:.0f})")


def test_no_early_scalar_threshold():
    """No downstream node crosses a 20% deviation threshold before
    propagation actually arrives. This is the condition under which
    ABR detection can provide lead time over scalar thresholds."""
    sim = SupplyChainSimulation(t_event=30, n_steps=80, noise_level=0.02)
    data = sim.generate()
    base = steady_state_baseline()
    timeline = sim.perturbation_timeline()

    threshold = 0.20  # 20% deviation from baseline

    for node_name, info in timeline['arrival_times'].items():
        node_idx = sim.topo.node_names.index(node_name)
        arrival = info['arrival_step']

        # Check all components at this node before arrival
        for c in range(5):
            pre_arrival = data[:arrival, c, node_idx]
            baseline_val = base[c, node_idx]
            if baseline_val < 1e-6:
                continue
            max_deviation = np.max(np.abs(pre_arrival - baseline_val)) / baseline_val

            assert max_deviation < threshold, \
                f"{node_name} comp {c} crosses {threshold*100}% threshold " \
                f"before arrival at t={arrival}: max dev = {max_deviation:.3f}"

    print("PASS: No early scalar threshold crossing before propagation arrival")


def test_simulation_output_finite():
    """All simulation output is finite (D membership)."""
    sim = SupplyChainSimulation(t_event=30, n_steps=80)
    data = sim.generate()
    assert np.all(np.isfinite(data)), "simulation output must be finite"

    scaled = sim.generate_scaled()
    assert np.all(np.isfinite(scaled)), "scaled output must be finite"

    print(f"PASS: All output finite (shape {data.shape})")


def test_timeline_readable():
    """Perturbation timeline produces readable output for Origin."""
    sim = SupplyChainSimulation(t_event=30, n_steps=80)
    timeline = sim.perturbation_timeline()

    assert 'perturbation_start' in timeline
    assert 'arrival_times' in timeline
    assert len(timeline['arrival_times']) > 0

    print("PASS: Timeline readable")
    print(f"  Perturbation start: t={timeline['perturbation_start']}")
    print(f"  Full perturbation: t={timeline['perturbation_full']}")
    for name, info in timeline['arrival_times'].items():
        print(f"  {name}: arrives t={info['arrival_step']}, "
              f"full t={info['full_arrival']}")


if __name__ == '__main__':
    test_baseline_physical_bounds()
    test_baseline_structure()
    test_scaling_admissible()
    test_steady_state_stable()
    test_perturbation_hits_shanghai_first()
    test_downstream_propagation_delay()
    test_propagation_actually_occurs()
    test_cross_component_coupling()
    test_no_early_scalar_threshold()
    test_simulation_output_finite()
    test_timeline_readable()
    print("\n=== ALL SIMULATION DYNAMICS TESTS PASSED ===")
