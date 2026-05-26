# test_detection_comparison.py — Detection Comparison
# Tests:
#   1. ABR detection time is well-defined (topology-specific Gamma triggers)
#   2. Graph detection time is well-defined (rule fires downstream)
#   3. Under declared simulation conditions, ABR detects perturbation
#   4. Lead time is reported, not asserted as universal

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.simulation.supply_chain import SupplyChainSimulation, steady_state_baseline
from src.topology.supply_chain import declare_supply_chain_topology
from src.detection.abr_detector import ABRDetector
from src.detection.graph_detector import GraphDetector


def test_abr_detects():
    """ABR detection triggers on the supply chain perturbation."""
    topo = declare_supply_chain_topology()
    sim = SupplyChainSimulation(t_event=30, n_steps=80)
    scaled = sim.generate_scaled()

    detector = ABRDetector(topo, rho_base=0.3,
                           baseline_window=20, threshold_sigma=3.0,
                           confirmation_window=3)
    result = detector.run(scaled)

    assert result['detection_time'] is not None, \
        "ABR should detect the perturbation"
    assert result['detection_time'] >= 20, \
        "Detection should not occur during baseline window"

    print(f"PASS: ABR detects at t={result['detection_time']} "
          f"via {result['detection_component']}")


def test_graph_detects():
    """Graph detection triggers on the supply chain perturbation."""
    topo = declare_supply_chain_topology()
    sim = SupplyChainSimulation(t_event=30, n_steps=80)
    raw = sim.generate()
    baseline = steady_state_baseline()

    detector = GraphDetector(topo, baseline)
    result = detector.run(raw)

    assert result['detection_time'] is not None, \
        "Graph system should eventually detect the perturbation"

    print(f"PASS: Graph detects at t={result['detection_time']} "
          f"via {result['detection_rule']}")


def test_detection_comparison():
    """Compare detection times under declared simulation conditions.

    This test REPORTS the comparison. It does not assert ABR is
    universally superior. The result is conditional on the declared
    simulation dynamics, topology, and detection parameters."""
    topo = declare_supply_chain_topology()
    sim = SupplyChainSimulation(t_event=30, n_steps=80)
    raw = sim.generate()
    scaled = sim.generate_scaled()
    baseline = steady_state_baseline()

    # ABR detection
    abr = ABRDetector(topo, rho_base=0.3,
                      baseline_window=20, threshold_sigma=3.0,
                      confirmation_window=3)
    abr_result = abr.run(scaled)

    # Graph detection
    graph = GraphDetector(topo, baseline)
    graph_result = graph.run(raw)

    abr_t = abr_result['detection_time']
    graph_t = graph_result['detection_time']

    print(f"\nDetection Comparison")
    print(f"{'='*50}")
    print(f"  Perturbation start:  t={sim.t_event}")
    print(f"  ABR detection:       t={abr_t} ({abr_result['detection_component']})")
    print(f"  Graph detection:     t={graph_t} ({graph_result['detection_rule']})")

    if abr_t is not None and graph_t is not None:
        lead_time = graph_t - abr_t
        print(f"  Lead time:           {lead_time} steps")
        if lead_time > 0:
            print(f"  ABR detected {lead_time} steps before graph system")
        elif lead_time < 0:
            print(f"  Graph system detected {-lead_time} steps before ABR")
        else:
            print(f"  Both detected at the same time")
    else:
        print(f"  Lead time: not computable (one or both failed to detect)")

    # This is a report, not an assertion
    print(f"\n  NOTE: This result is conditional on declared simulation")
    print(f"  conditions and detection parameters. It does not assert")
    print(f"  universal superiority of either approach.")

    print(f"\nPASS: Detection comparison completed")


def test_abr_reports_readable():
    """ABR report is human-readable for Origin verification."""
    topo = declare_supply_chain_topology()
    sim = SupplyChainSimulation(t_event=30, n_steps=80)
    scaled = sim.generate_scaled()

    detector = ABRDetector(topo, rho_base=0.3,
                           baseline_window=20, threshold_sigma=3.0,
                           confirmation_window=3)
    result = detector.run(scaled)
    report = detector.format_report(result)

    assert "ABR Detection Report" in report
    assert "Confirmation window" in report
    print("PASS: ABR report readable")


def test_graph_reports_readable():
    """Graph report is human-readable for Origin verification."""
    topo = declare_supply_chain_topology()
    sim = SupplyChainSimulation(t_event=30, n_steps=80)
    raw = sim.generate()
    baseline = steady_state_baseline()

    detector = GraphDetector(topo, baseline)
    result = detector.run(raw)
    report = detector.format_report(result)

    assert "Graph Detection Report" in report
    assert len(report) > 100
    print("PASS: Graph report readable")


if __name__ == '__main__':
    test_abr_detects()
    test_graph_detects()
    test_detection_comparison()
    test_abr_reports_readable()
    test_graph_reports_readable()
    print("\n=== ALL DETECTION COMPARISON TESTS PASSED ===")
