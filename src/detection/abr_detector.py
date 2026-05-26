# abr_detector.py — ABR Field Detection
#
# Applies graph-native V4 kernel at each time step.
# Monitors topology-specific Gamma decomposition:
#   - Per-component downstream spatial Gamma
#   - Per-component upstream spatial Gamma
#   - Per-pair component Gamma
#
# Detection: significant change from baseline in any
# single Gamma component — not in the total.
#
# This represents continuous-field-operator processing.

import numpy as np
from ..operators.types import NodeField
from ..operators.diagnostics import gamma_decomposed
from ..topology.declaration import TopologyDeclaration


class ABRDetector:
    """Monitors relational field evolution through topology-specific
    Gamma decomposition. Detects perturbation as significant deviation
    from baseline Gamma trajectory.

    Detection criterion: any single Gamma component (downstream,
    upstream, or component-pair) exceeds baseline_mean + threshold_sigma
    standard deviations of baseline variation.

    This is continuous-field-operator processing. The detector operates
    on the full 118-edge relational field at every time step."""

    def __init__(self,
                 topo: TopologyDeclaration,
                 rho_base: float = 0.3,
                 baseline_window: int = 20,
                 threshold_sigma: float = 3.0,
                 confirmation_window: int = 3):
        """
        Args:
            topo: declared topology
            rho_base: circulation strength for ABR kernel
            baseline_window: number of initial steps to establish baseline
            threshold_sigma: number of standard deviations for detection
            confirmation_window: number of consecutive steps the deviation
                must persist to trigger detection. Distinguishes transient
                relational reorganization from sustained restructuring.
                Structurally comparable to graph detector's trend_window.
        """
        self.topo = topo
        self.rho_base = rho_base
        self.baseline_window = baseline_window
        self.threshold_sigma = threshold_sigma
        self.confirmation_window = confirmation_window

        # Storage for Gamma trajectory
        self.gamma_history = []
        self.detection_time = None
        self.detection_component = None

    def process_step(self, field_data: np.ndarray) -> dict:
        """Process one time step. field_data shape (k, n), already scaled.

        Returns the Gamma decomposition for this step."""
        f = NodeField(data=field_data, topo=self.topo)
        gd = gamma_decomposed(f, rho_base=self.rho_base)
        self.gamma_history.append(gd)
        return gd

    def run(self, scaled_data: np.ndarray) -> dict:
        """Run detection over full simulation.

        Args:
            scaled_data: shape (n_steps, k, n) — M-scaled field values

        Returns dict with:
            detection_time: first step where any Gamma component triggers
            detection_component: which component triggered
            gamma_trajectory: full history
            baseline_stats: mean and std of baseline Gamma per component
        """
        n_steps = scaled_data.shape[0]
        self.gamma_history = []
        self.detection_time = None
        self.detection_component = None

        # Process all steps
        for t in range(n_steps):
            self.process_step(scaled_data[t])

        # Compute baseline statistics from initial window
        baseline_stats = self._compute_baseline_stats()

        # Scan for detection after baseline window
        # Requires confirmation_window consecutive steps exceeding threshold
        for t in range(self.baseline_window, n_steps):
            triggered, component = self._check_sustained(t, baseline_stats)
            if triggered and self.detection_time is None:
                # Detection time is the START of the sustained deviation
                self.detection_time = t - self.confirmation_window + 1
                self.detection_component = component

        return {
            'detection_time': self.detection_time,
            'detection_component': self.detection_component,
            'gamma_trajectory': self.gamma_history,
            'baseline_stats': baseline_stats,
        }

    def _compute_baseline_stats(self) -> dict:
        """Compute mean and std of each Gamma component over baseline window."""
        k = self.topo.k_components
        n_pairs = len(self.topo.component_pairs)
        window = self.gamma_history[:self.baseline_window]

        stats = {}

        # Downstream Gamma per component
        for c in range(k):
            values = [gd['gamma_downstream'][c] for gd in window]
            name = f"downstream_{self.topo.component_names[c]}"
            stats[name] = {
                'mean': np.mean(values),
                'std': max(np.std(values), 1e-10),  # floor to prevent div/0
                'type': 'downstream',
                'index': c,
            }

        # Upstream Gamma per component
        for c in range(k):
            values = [gd['gamma_upstream'][c] for gd in window]
            name = f"upstream_{self.topo.component_names[c]}"
            stats[name] = {
                'mean': np.mean(values),
                'std': max(np.std(values), 1e-10),
                'index': c,
                'type': 'upstream',
            }

        # Component Gamma per pair
        for p in range(n_pairs):
            a, b = self.topo.component_pairs[p]
            values = [gd['gamma_comp'][p] for gd in window]
            name = f"comp_{self.topo.component_names[a]}_{self.topo.component_names[b]}"
            stats[name] = {
                'mean': np.mean(values),
                'std': max(np.std(values), 1e-10),
                'type': 'component',
                'index': p,
            }

        return stats

    def _check_sustained(self, t: int, baseline_stats: dict) -> tuple:
        """Check if any Gamma component exceeds threshold for
        confirmation_window consecutive steps ending at t.

        This distinguishes transient relational reorganization from
        sustained restructuring. A single-step deviation is a real
        field event but is not treated as detection of persistent
        structural change.

        Returns (triggered: bool, component_name: str or None)."""
        if t < self.baseline_window + self.confirmation_window - 1:
            return False, None

        for name, stat in baseline_stats.items():
            # Check all steps in the confirmation window
            sustained = True
            for dt in range(self.confirmation_window):
                step = t - dt
                gd = self.gamma_history[step]

                if stat['type'] == 'downstream':
                    value = gd['gamma_downstream'][stat['index']]
                elif stat['type'] == 'upstream':
                    value = gd['gamma_upstream'][stat['index']]
                elif stat['type'] == 'component':
                    value = gd['gamma_comp'][stat['index']]
                else:
                    sustained = False
                    break

                deviation = abs(value - stat['mean']) / stat['std']
                if deviation <= self.threshold_sigma:
                    sustained = False
                    break

            if sustained:
                return True, name

        return False, None

    def format_report(self, result: dict) -> str:
        """Human-readable detection report for Origin."""
        lines = [
            "ABR Detection Report",
            "=" * 40,
            f"  Baseline window: {self.baseline_window} steps",
            f"  Threshold: {self.threshold_sigma} sigma",
            f"  Confirmation window: {self.confirmation_window} consecutive steps",
            f"  rho_base: {self.rho_base}",
            "",
        ]

        if result['detection_time'] is not None:
            lines.append(
                f"  DETECTION at t={result['detection_time']} "
                f"via {result['detection_component']}")
        else:
            lines.append("  No detection triggered")

        lines.append("")
        lines.append("  Baseline Gamma statistics:")
        for name, stat in result['baseline_stats'].items():
            lines.append(f"    {name}: mean={stat['mean']:.2f}, std={stat['std']:.2f}")

        return "\n".join(lines)
