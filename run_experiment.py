# run_experiment.py
#
# Operator-Based Relational Field Analysis vs Explicitly Declared Graph Structure
# Supply Chain Demonstration
#
# Metatron Dynamics, Inc. — May 2026
#
# This demonstration isolates a specific representational distinction
# between declared-relationship processing and continuous-field-operator
# processing under declared simulation conditions.
#
# It is not intended to show that ABR is universally superior to graph
# systems. It is intended to isolate a specific representational
# distinction under declared simulation conditions.
#
# Sequence:
#   1. Declare topology (Origin)
#   2. Generate and verify simulation dynamics
#   3. Run ABR detection (continuous field operators)
#   4. Run Graph detection (declared relationship rules)
#   5. Compare and report

import numpy as np
from src.topology.supply_chain import declare_supply_chain_topology
from src.simulation.supply_chain import SupplyChainSimulation, steady_state_baseline
from src.detection.abr_detector import ABRDetector
from src.detection.graph_detector import GraphDetector
from src.operators.diagnostics import gamma_decomposed, format_gamma_report
from src.operators.types import NodeField


def main():
    print("=" * 60)
    print("Operator-Based Relational Field Analysis")
    print("vs Explicitly Declared Graph Structure")
    print("Supply Chain Demonstration")
    print("Metatron Dynamics, Inc.")
    print("=" * 60)

    # === 1. TOPOLOGY DECLARATION ===
    print("\n--- 1. TOPOLOGY DECLARATION (Origin) ---\n")
    topo = declare_supply_chain_topology()
    print(topo.summary())

    # === 2. SIMULATION ===
    print("\n--- 2. SIMULATION DYNAMICS ---\n")
    sim = SupplyChainSimulation(t_event=30, n_steps=80)
    raw = sim.generate()
    scaled = sim.generate_scaled()
    baseline = steady_state_baseline()

    timeline = sim.perturbation_timeline()
    print(f"Perturbation: Shanghai lead_time x{sim.perturbation_magnitude}")
    print(f"  Start: t={timeline['perturbation_start']}")
    print(f"  Full:  t={timeline['perturbation_full']}")
    print(f"\nPropagation timeline:")
    for name, info in timeline['arrival_times'].items():
        print(f"  {name}: arrives t={info['arrival_step']}, "
              f"full t={info['full_arrival']}")

    # === 3. BASELINE GAMMA ===
    print("\n--- 3. BASELINE RELATIONAL FIELD (t=0) ---\n")
    f0 = NodeField(data=scaled[0], topo=topo)
    gd0 = gamma_decomposed(f0, rho_base=0.3)
    print(format_gamma_report(gd0))

    # === 4. ABR DETECTION ===
    print("\n--- 4. ABR DETECTION (Continuous Field Operators) ---\n")
    abr = ABRDetector(topo, rho_base=0.3,
                      baseline_window=20, threshold_sigma=3.0,
                      confirmation_window=3)
    abr_result = abr.run(scaled)
    print(abr.format_report(abr_result))

    # === 5. GRAPH DETECTION ===
    print("\n--- 5. GRAPH DETECTION (Declared Relationship Rules) ---\n")
    graph = GraphDetector(topo, baseline)
    graph_result = graph.run(raw)
    print(graph.format_report(graph_result))

    # === 6. COMPARISON ===
    print("\n--- 6. COMPARISON ---\n")
    abr_t = abr_result['detection_time']
    graph_t = graph_result['detection_time']

    print(f"  Perturbation onset:     t={sim.t_event}")
    print(f"  ABR detection:          t={abr_t}")
    print(f"  Graph detection:        t={graph_t}")

    if abr_t is not None and graph_t is not None:
        lead_time = graph_t - abr_t
        print(f"  Lead time difference:   {lead_time} steps")
        print()
        if lead_time > 0:
            print(f"  Result: ABR field operators detected the perturbation")
            print(f"  {lead_time} steps before the declared graph rules.")
            print()
            print(f"  The ABR detection triggered via topology-specific Gamma")
            print(f"  component: {abr_result['detection_component']}")
            print(f"  This represents relational field reorganization — a")
            print(f"  change in cross-topology coupling structure — that")
            print(f"  preceded any scalar threshold crossing at individual nodes.")
        elif lead_time < 0:
            print(f"  Result: Graph rules detected {-lead_time} steps before ABR.")
            print(f"  Under these simulation conditions, the scalar threshold")
            print(f"  approach detected the perturbation earlier.")
        else:
            print(f"  Result: Both approaches detected at the same time step.")
    else:
        print(f"  Result: One or both systems failed to detect.")

    print()
    print("  NOTE: This result is conditional on the declared simulation")
    print("  dynamics, topology, and detection parameters. It demonstrates")
    print("  a specific representational distinction, not universal")
    print("  superiority of either approach.")

    # === 7. POST-PERTURBATION GAMMA ===
    print("\n--- 7. POST-PERTURBATION RELATIONAL FIELD (t=50) ---\n")
    f50 = NodeField(data=scaled[min(50, len(scaled)-1)], topo=topo)
    gd50 = gamma_decomposed(f50, rho_base=0.3)
    print(format_gamma_report(gd50))

    print("\n" + "=" * 60)
    print("Experiment complete.")
    print("=" * 60)


if __name__ == '__main__':
    main()
