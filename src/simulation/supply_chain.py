# supply_chain.py — Supply Chain Dynamics
#
# 8 nodes on declared directed graph, 5 components.
# Downstream propagation (material flow) and upstream
# propagation (demand/information flow) are structurally
# distinct with different dynamics and speeds.
#
# PROTOCOL: verify dynamics produce correct baseline behavior
# BEFORE layering any detection.

import numpy as np
from ..topology.supply_chain import (
    declare_supply_chain_topology,
    SUPPLIER_SHANGHAI, SUPPLIER_VIETNAM, PORT_LONGBEACH,
    WAREHOUSE_WEST, WAREHOUSE_CENTRAL, WAREHOUSE_EAST,
    RETAILER_BIGBOX, RETAILER_ONLINE,
    INVENTORY_LEVEL, LEAD_TIME, UNIT_COST, QUALITY_SCORE, THROUGHPUT,
)
from ..topology.declaration import TopologyDeclaration
from .measurement_mapping import scale_to_domain, SUPPLY_CHAIN_SCALES


# === STEADY STATE BASELINE ===

def steady_state_baseline() -> np.ndarray:
    """Raw steady-state values for 8 nodes, 5 components.
    Returns shape (k=5, n=8) in physical units before M scaling.

    These represent a normally operating supply chain:
    - Inventory decreases along chain (suppliers hold more)
    - Lead time increases with supply chain depth
    - Unit cost increases along chain (value added)
    - Quality score degrades slightly along chain
    - Throughput varies by node role"""

    data = np.zeros((5, 8))

    # inventory_level (units)
    data[INVENTORY_LEVEL] = [
        1000.0,  # Shanghai: large raw materials stock
        800.0,   # Vietnam: secondary supplier stock
        600.0,   # LongBeach: port buffer
        400.0,   # West: primary distribution
        350.0,   # Central: secondary distribution
        300.0,   # East: tertiary distribution
        200.0,   # BigBox: retail floor stock
        150.0,   # Online: fulfillment center
    ]

    # lead_time (days)
    data[LEAD_TIME] = [
        2.0,   # Shanghai: internal processing
        3.0,   # Vietnam: internal processing
        5.0,   # LongBeach: port clearance
        3.0,   # West: distribution processing
        4.0,   # Central: distribution processing
        4.0,   # East: distribution processing
        2.0,   # BigBox: shelf restocking
        3.0,   # Online: pick-pack-ship
    ]

    # unit_cost (USD)
    data[UNIT_COST] = [
        10.0,  # Shanghai: raw cost
        12.0,  # Vietnam: slightly higher raw cost
        15.0,  # LongBeach: + shipping/clearance
        18.0,  # West: + handling
        19.0,  # Central: + transfer
        20.0,  # East: + transfer
        25.0,  # BigBox: + retail margin
        22.0,  # Online: + fulfillment (lower than retail)
    ]

    # quality_score (0-1)
    data[QUALITY_SCORE] = [
        0.95,  # Shanghai: factory QC
        0.93,  # Vietnam: factory QC
        0.91,  # LongBeach: slight handling degradation
        0.89,  # West: slight storage degradation
        0.88,  # Central: transfer handling
        0.86,  # East: transfer handling
        0.83,  # BigBox: shelf life, handling
        0.85,  # Online: better controlled storage
    ]

    # throughput (units/day)
    data[THROUGHPUT] = [
        500.0,  # Shanghai: factory capacity
        400.0,  # Vietnam: smaller factory
        800.0,  # LongBeach: high port throughput
        600.0,  # West: distribution capacity
        500.0,  # Central: distribution capacity
        400.0,  # East: smaller distribution
        700.0,  # BigBox: high retail volume
        600.0,  # Online: fulfillment capacity
    ]

    return data


# === PERTURBATION MODEL ===

class SupplyChainSimulation:
    """Time-stepped supply chain simulation with perturbation.

    Dynamics:
    - Downstream propagation: delays and inventory depletion
      travel from suppliers toward retailers
    - Upstream propagation: demand signals and cost pressure
      travel from retailers toward suppliers
    - Cross-component coupling: lead_time affects inventory,
      inventory affects throughput, cost affects quality, etc.

    Perturbation: Shanghai supplier lead_time ramp starting
    at t_event, propagating downstream through declared adjacencies."""

    def __init__(self,
                 t_event: int = 30,
                 n_steps: int = 80,
                 perturbation_magnitude: float = 3.0,
                 perturbation_ramp: int = 5,
                 noise_level: float = 0.02,
                 seed: int = 42):
        """
        Args:
            t_event: time step when Shanghai delay begins
            n_steps: total simulation steps
            perturbation_magnitude: multiplier on Shanghai lead_time
            perturbation_ramp: steps over which perturbation ramps up
            noise_level: relative stochastic variation (fraction of baseline)
            seed: random seed for reproducibility
        """
        self.topo = declare_supply_chain_topology()
        self.t_event = t_event
        self.n_steps = n_steps
        self.perturbation_magnitude = perturbation_magnitude
        self.perturbation_ramp = perturbation_ramp
        self.noise_level = noise_level
        self.rng = np.random.RandomState(seed)

        self.baseline = steady_state_baseline()
        self.scales = SUPPLY_CHAIN_SCALES

        # Propagation delay: how many time steps for a perturbation
        # to cross each adjacency (downstream direction)
        # Declared per-adjacency — not uniform
        self.propagation_delays = {
            0: 3,  # Shanghai -> LongBeach (shipping delay)
            1: 3,  # Vietnam -> LongBeach (shipping delay)
            2: 2,  # LongBeach -> West (trucking)
            3: 2,  # West -> Central (transfer)
            4: 2,  # Central -> East (transfer)
            5: 1,  # West -> BigBox (local delivery)
            6: 1,  # East -> Online (local delivery)
        }

        # Cross-component coupling strengths (declared)
        # How strongly a change in one component drives change in another
        self.coupling = {
            # lead_time increase -> inventory decrease
            (LEAD_TIME, INVENTORY_LEVEL): -0.15,
            # lead_time increase -> cost increase (expediting)
            (LEAD_TIME, UNIT_COST): 0.08,
            # inventory decrease -> throughput decrease
            (INVENTORY_LEVEL, THROUGHPUT): 0.10,
            # quality decrease -> inventory decrease (rejection)
            (QUALITY_SCORE, INVENTORY_LEVEL): 0.05,
            # quality decrease -> throughput decrease
            (QUALITY_SCORE, THROUGHPUT): 0.06,
            # cost increase -> throughput decrease (budget constraints)
            (UNIT_COST, THROUGHPUT): -0.04,
        }

    def _perturbation_at(self, t: int) -> float:
        """Shanghai lead_time perturbation factor at time t.
        Ramps from 1.0 to perturbation_magnitude over perturbation_ramp steps."""
        if t < self.t_event:
            return 1.0
        elapsed = t - self.t_event
        if elapsed >= self.perturbation_ramp:
            return self.perturbation_magnitude
        # Linear ramp
        frac = elapsed / self.perturbation_ramp
        return 1.0 + (self.perturbation_magnitude - 1.0) * frac

    def _downstream_perturbation_arrival(self, adj_idx: int, t: int) -> float:
        """How much of the perturbation has arrived at this adjacency's
        target node by time t. Returns 0-1 fraction.

        Perturbation propagates from Shanghai through declared adjacencies
        with cumulative delay."""
        # Compute cumulative delay from Shanghai to this adjacency's target
        # by tracing the shortest declared path
        paths = self._paths_from_shanghai()
        target = self.topo.spatial_adjacencies[adj_idx].target

        if target not in paths:
            return 0.0

        cumulative_delay = paths[target]
        arrival_t = self.t_event + self.perturbation_ramp + cumulative_delay

        if t < arrival_t:
            return 0.0
        elif t >= arrival_t + 3:  # 3-step arrival ramp
            return 1.0
        else:
            return (t - arrival_t) / 3.0

    def _paths_from_shanghai(self) -> dict:
        """Compute cumulative propagation delay from Shanghai to each node.
        BFS along declared downstream adjacencies.

        DECLARED SIMPLIFICATION: uses shortest single path via BFS.
        Real supply chains may branch, reroute, dynamically compensate,
        or propagate along multiple pathways simultaneously. This is
        a declared modeling choice within M, not an assertion about
        intrinsic supply chain propagation behavior. Multi-path
        propagation models are a declared open condition."""
        delays = {SUPPLIER_SHANGHAI: 0}
        queue = [SUPPLIER_SHANGHAI]

        while queue:
            node = queue.pop(0)
            for adj_idx, adj in enumerate(self.topo.spatial_adjacencies):
                if adj.source == node and adj.target not in delays:
                    delays[adj.target] = delays[node] + self.propagation_delays[adj_idx]
                    queue.append(adj.target)

        return delays

    def generate(self) -> np.ndarray:
        """Generate full simulation: n_steps time steps of (k, n) fields.

        Returns shape (n_steps, k, n) — raw physical units.
        Apply M (scale_to_domain) before operator processing."""
        k = self.topo.k_components
        n = self.topo.n_nodes
        result = np.zeros((self.n_steps, k, n))

        paths = self._paths_from_shanghai()

        for t in range(self.n_steps):
            # Start from baseline
            state = self.baseline.copy()

            # --- Apply Shanghai perturbation ---
            shanghai_factor = self._perturbation_at(t)
            state[LEAD_TIME, SUPPLIER_SHANGHAI] *= shanghai_factor

            # --- Propagate downstream through declared adjacencies ---
            for node in range(n):
                if node == SUPPLIER_SHANGHAI:
                    continue
                if node not in paths:
                    continue

                cumulative_delay = paths[node]
                arrival_t = self.t_event + self.perturbation_ramp + cumulative_delay

                if t < arrival_t:
                    arrival_frac = 0.0
                elif t >= arrival_t + 3:
                    arrival_frac = 1.0
                else:
                    arrival_frac = (t - arrival_t) / 3.0

                if arrival_frac > 0:
                    # Lead time increases at downstream nodes
                    # Attenuated by distance (each hop reduces effect by 30%)
                    hops = len([1 for d in paths.values() if d <= paths[node]]) - 1
                    attenuation = 0.7 ** max(0, hops - 1)
                    lt_increase = (shanghai_factor - 1.0) * attenuation * arrival_frac
                    state[LEAD_TIME, node] *= (1.0 + lt_increase * 0.5)

            # --- Cross-component coupling ---
            # Changes in one component drive changes in coupled components
            # DECLARED SIMPLIFICATION: coupling is instantaneous at each
            # node within a time step. Real systems exhibit lagged coupling,
            # hysteresis, and delayed response (inventory effects, quality
            # degradation, throughput collapse often lag causally).
            # Component-specific temporal lag is a declared open condition.
            deltas = state - self.baseline
            for (src_comp, tgt_comp), strength in self.coupling.items():
                for node in range(n):
                    relative_change = deltas[src_comp, node] / max(
                        abs(self.baseline[src_comp, node]), 1e-6)
                    state[tgt_comp, node] += (
                        self.baseline[tgt_comp, node] * relative_change * strength
                    )

            # --- Stochastic variation ---
            noise = self.rng.randn(k, n) * self.noise_level
            state *= (1.0 + noise)

            # --- Clamp to physical bounds ---
            state[INVENTORY_LEVEL] = np.maximum(state[INVENTORY_LEVEL], 0.0)
            state[LEAD_TIME] = np.maximum(state[LEAD_TIME], 0.5)
            state[UNIT_COST] = np.maximum(state[UNIT_COST], 1.0)
            state[QUALITY_SCORE] = np.clip(state[QUALITY_SCORE], 0.0, 1.0)
            state[THROUGHPUT] = np.maximum(state[THROUGHPUT], 0.0)

            result[t] = state

        return result

    def generate_scaled(self) -> np.ndarray:
        """Generate simulation and apply M (scale_to_domain).
        Returns shape (n_steps, k, n) in domain D."""
        raw = self.generate()
        scaled = np.zeros_like(raw)
        for t in range(self.n_steps):
            scaled[t] = scale_to_domain(raw[t], self.scales)
        return scaled

    def perturbation_timeline(self) -> dict:
        """Return key time points for Origin verification."""
        paths = self._paths_from_shanghai()
        timeline = {
            'perturbation_start': self.t_event,
            'perturbation_full': self.t_event + self.perturbation_ramp,
            'arrival_times': {},
        }
        for node_idx, delay in sorted(paths.items()):
            if node_idx == SUPPLIER_SHANGHAI:
                continue
            arrival = self.t_event + self.perturbation_ramp + delay
            name = self.topo.node_names[node_idx]
            timeline['arrival_times'][name] = {
                'cumulative_delay': delay,
                'arrival_step': arrival,
                'full_arrival': arrival + 3,
            }
        return timeline
