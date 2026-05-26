# Operator-Based Relational Field Analysis vs Explicitly Declared Graph Structure

**Metatron Dynamics, Inc. — May 2026**

Supply chain demonstration isolating a specific representational distinction between continuous topology-conditioned field operators and explicitly declared relationship processing under bounded simulation conditions.

This demonstration is not intended to show that ABR is universally superior to graph systems. It is intended to isolate a specific representational distinction under declared simulation conditions.

---

## The Distinction

**Declared-relationship processing** (knowledge graphs, rule-based systems): relational structure is explicitly declared by a domain expert or learned through modeling layers external to the graph. The system monitors and propagates along declared edges using threshold rules.

**Continuous-field-operator processing** (ABR kernel): relational operators compute over the measured field on declared topology at every time step. Cross-topology coupling structure is computed, not declared. Changes in coupling structure (relational field reorganization) are observable before scalar thresholds fire at individual nodes.

The structural gap is not that graph systems are weak — modern graph systems support weighted edges, temporal reasoning, multi-hop propagation, and learned ontologies. The gap is between systems that require relational structure to be articulated before processing and systems that compute relational structure from the measured field continuously. Capturing field-level reorganization requires introducing additional continuous relational dynamics beyond the explicitly declared graph structure itself. Graph dynamical systems, message-passing networks, graph PDE approaches, and temporal graph neural fields all move in this direction — and to the extent they do, they converge toward the same structural commitments the ABR kernel makes explicit.

---

## Result

Under the declared simulation conditions (8-node supply chain, Shanghai supplier delay, 5 measurement components):

- **ABR detection:** t=31 via sustained upstream lead_time Gamma deviation (relational field reorganization detected 1 step after perturbation onset, before any scalar threshold crossed)
- **Graph detection:** t=32 via lead_time exceeding 20% at perturbation source node
- **Downstream propagation:** Graph system traces perturbation node-by-node from t=32 to t=73. ABR detected field restructuring at t=31 before the perturbation had physically arrived at any downstream node.

This result is conditional on declared simulation dynamics, topology, and detection parameters. It demonstrates a specific representational distinction, not universal superiority of either approach.

---

## Framework

**ABR Kernel (V4):** Operator composition E(x, ρ) = R(B(A(x)), ρ(A(x)))

- **A** — Relational Gradient Extraction: NodeField → EdgeField (the unique representation transition)
- **B** — Local Relational Accumulation: EdgeField → EdgeField (along declared adjacencies)
- **R** — Antisymmetric Circulation: EdgeField → EdgeField (cross-topology coupling)
- **No kernel C** — boundedness is a declared projection at the application layer (V4 discipline)

**V4 Topology Discipline:** Ring topology is the proof topology (V3 canonical, used in Object Error theorems). V4 applications must NOT default to ring. Every application declares its own spatial and component topology through Origin as part of M. There is no topological default in V4. Undeclared topology is inadmissible.

**Declared topology for this application:**

- Spatial: directed acyclic graph with 8 nodes, 7 adjacencies, downstream/upstream edge types
- Components: inventory_level, lead_time, unit_cost, quality_score, throughput
- Component coupling: 6 declared operational pairs out of 10 possible
- Edge field dimensionality: 118 (35 downstream + 35 upstream + 48 component)

---

## Structure

```
src/
  topology/
    declaration.py       — Graph-native topology declaration
    supply_chain.py      — This application's Origin-declared topology
  operators/
    types.py             — NodeField / EdgeField on irregular graphs
    kernel.py            — Graph-native V4 ABR operators (A, B, R, E)
    diagnostics.py       — Topology-specific Gamma decomposition
    projections.py       — Declared projections with preserved/discarded invariants
    canonical_ring.py    — V3 ring operators (proof reference only)
  simulation/
    supply_chain.py      — Steady state + perturbation dynamics
    measurement_mapping.py — M : O → D with declared scaling
  detection/
    abr_detector.py      — Sustained Gamma deviation detection
    graph_detector.py    — Competent graph rules with trend confirmation
docs/
  build_plan.md          — Full build specification
  topology_discipline.md — V4 topology discipline documentation
tests/
  test_topology_declaration.py   — 10 topology admissibility tests
  test_operator_invariants.py    — 13 operator invariant tests
  test_simulation_dynamics.py    — 11 simulation verification tests
  test_detection_comparison.py   — 5 detection comparison tests
run_experiment.py        — Full experiment runner
```

## Dependencies

- Python 3.10+
- numpy

## Usage

```bash
# Run all tests (must pass in order)
python -m tests.test_topology_declaration
python -m tests.test_operator_invariants
python -m tests.test_simulation_dynamics
python -m tests.test_detection_comparison

# Run full experiment
python run_experiment.py
```

---

## References

- Macomber, R. (2026). Invariant Relational Evolution over Bounded Domains. arXiv:2601.22389.
- [The Object Error: A Formal Argument](https://github.com/Relational-Relativity-Corporation)
- [Metatron Dynamics](https://relationalrelativity.dev)

---

*All statements bounded over D. No claim beyond D. This demonstration describes a representational distinction under declared conditions. Interpretation and engagement remain projection-layer decisions.*
