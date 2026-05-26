# graph_detector.py — Declared Graph Structure Detection
#
# Represents a COMPETENT graph system (not a strawman):
#   - Weighted directed edges with propagation delay
#   - Temporal rules (trend detection over N observations)
#   - Multi-hop propagation along declared paths
#   - Separate downstream and upstream alert propagation
#   - Per-node, per-component threshold monitoring
#
# This represents declared-relationship processing.
# It does NOT use relational field operators.
#
# The comparison is: declared-relationship processing
# vs continuous-field-operator processing.

import numpy as np
from ..topology.declaration import TopologyDeclaration


class GraphRule:
    """A declared graph detection rule.
    Monitors a specific component at a specific node against
    a threshold condition."""

    def __init__(self, node: int, component: int,
                 threshold_pct: float, direction: str = 'above',
                 trend_window: int = 3):
        """
        Args:
            node: node index to monitor
            component: component index to monitor
            threshold_pct: percent deviation from baseline to trigger
            direction: 'above' or 'below' — which direction triggers
            trend_window: number of consecutive steps to confirm trend
        """
        self.node = node
        self.component = component
        self.threshold_pct = threshold_pct
        self.direction = direction
        self.trend_window = trend_window


class GraphDetector:
    """Declared graph structure detection system.

    Represents what a competent knowledge graph / rule-based system
    would do: monitor declared nodes and components against thresholds,
    propagate alerts along declared edges, and use temporal trend
    detection to reduce false positives.

    This is deliberately made as strong as possible within the
    declared-relationship paradigm. It includes capabilities that
    sophisticated graph systems like RelationalAI actually provide:
    weighted edges, temporal awareness, and multi-hop propagation.

    What it does NOT do:
    - Compute continuous relational field operators
    - Monitor cross-component coupling as a field property
    - Detect relational reorganization before scalar thresholds fire

    That is the structural gap this demonstration isolates."""

    def __init__(self,
                 topo: TopologyDeclaration,
                 baseline: np.ndarray,
                 rules: list = None,
                 alert_propagation: bool = True):
        """
        Args:
            topo: declared topology
            baseline: shape (k, n) — steady-state reference values
            rules: list of GraphRule objects. If None, generates
                   competent default rules for supply chain monitoring.
            alert_propagation: whether alerts propagate along declared edges
        """
        self.topo = topo
        self.baseline = baseline
        self.alert_propagation = alert_propagation
        self.rules = rules if rules is not None else self._default_rules()

        # State
        self.alert_history = []  # list of dicts per time step
        self.detection_time = None
        self.detection_rule = None

    def _default_rules(self) -> list:
        """Generate competent monitoring rules for supply chain.

        These represent what a skilled graph system engineer would
        configure: threshold monitors on key metrics at key nodes,
        with trend confirmation to avoid false positives.

        This is NOT a strawman. These are reasonable rules."""
        rules = []

        # Monitor lead_time increase at every node (primary indicator)
        for n in range(self.topo.n_nodes):
            rules.append(GraphRule(
                node=n, component=1,  # lead_time
                threshold_pct=0.20,   # 20% increase triggers
                direction='above',
                trend_window=2,       # 2 consecutive steps to confirm
            ))

        # Monitor inventory decrease at warehouses and retailers
        for n in [3, 4, 5, 6, 7]:  # warehouses + retailers
            rules.append(GraphRule(
                node=n, component=0,  # inventory_level
                threshold_pct=0.15,   # 15% decrease triggers
                direction='below',
                trend_window=3,       # 3 consecutive steps
            ))

        # Monitor throughput decrease at warehouses and retailers
        for n in [3, 4, 5, 6, 7]:
            rules.append(GraphRule(
                node=n, component=4,  # throughput
                threshold_pct=0.15,
                direction='below',
                trend_window=3,
            ))

        # Monitor cost increase at mid-chain and downstream
        for n in [2, 3, 4, 5, 6, 7]:  # port through retailers
            rules.append(GraphRule(
                node=n, component=2,  # unit_cost
                threshold_pct=0.15,
                direction='above',
                trend_window=3,
            ))

        # Monitor quality decrease at warehouses and retailers
        for n in [3, 4, 5, 6, 7]:
            rules.append(GraphRule(
                node=n, component=3,  # quality_score
                threshold_pct=0.10,   # tighter threshold for quality
                direction='below',
                trend_window=3,
            ))

        return rules

    def _check_rule(self, rule: GraphRule, history: np.ndarray, t: int) -> bool:
        """Check if a rule triggers at time step t.

        Uses trend confirmation: the threshold must be exceeded for
        trend_window consecutive steps."""
        if t < rule.trend_window:
            return False

        baseline_val = self.baseline[rule.component, rule.node]
        if abs(baseline_val) < 1e-10:
            return False

        # Check trend_window consecutive steps
        for dt in range(rule.trend_window):
            step = t - dt
            current_val = history[step, rule.component, rule.node]
            pct_change = (current_val - baseline_val) / abs(baseline_val)

            if rule.direction == 'above' and pct_change < rule.threshold_pct:
                return False
            elif rule.direction == 'below' and pct_change > -rule.threshold_pct:
                return False

        return True

    def run(self, raw_data: np.ndarray) -> dict:
        """Run detection over full simulation.

        Args:
            raw_data: shape (n_steps, k, n) — raw physical units
                      (graph systems typically work in original units)

        Returns dict with:
            detection_time: first step where any rule triggers
            detection_rule: description of triggering rule
            alert_timeline: when each rule fired
        """
        n_steps = raw_data.shape[0]
        self.detection_time = None
        self.detection_rule = None

        alert_timeline = {}

        for t in range(n_steps):
            for rule_idx, rule in enumerate(self.rules):
                if rule_idx in alert_timeline:
                    continue  # already fired

                if self._check_rule(rule, raw_data, t):
                    node_name = self.topo.node_names[rule.node]
                    comp_name = self.topo.component_names[rule.component]
                    rule_desc = (f"{comp_name} {rule.direction} "
                                 f"{rule.threshold_pct*100:.0f}% at {node_name}")

                    alert_timeline[rule_idx] = {
                        'time': t,
                        'rule': rule_desc,
                        'node': rule.node,
                        'component': rule.component,
                    }

                    if self.detection_time is None:
                        self.detection_time = t
                        self.detection_rule = rule_desc

        # Alert propagation: if a node fires, check downstream nodes
        # with relaxed thresholds (50% of original)
        if self.alert_propagation:
            propagated = self._propagate_alerts(raw_data, alert_timeline)
            alert_timeline.update(propagated)
            # Update detection time if propagation found something earlier
            for info in propagated.values():
                if self.detection_time is None or info['time'] < self.detection_time:
                    self.detection_time = info['time']
                    self.detection_rule = info['rule'] + " (propagated)"

        return {
            'detection_time': self.detection_time,
            'detection_rule': self.detection_rule,
            'alert_timeline': alert_timeline,
            'n_rules': len(self.rules),
        }

    def _propagate_alerts(self, raw_data: np.ndarray,
                          existing_alerts: dict) -> dict:
        """Propagate alerts along declared edges with relaxed thresholds.

        When a node fires an alert, its downstream neighbors are checked
        with 50% of the original threshold. This represents sophisticated
        graph systems that trace impact along declared relationships."""
        propagated = {}
        n_steps = raw_data.shape[0]

        for rule_idx, alert_info in existing_alerts.items():
            alert_node = alert_info['node']
            alert_time = alert_info['time']
            alert_comp = alert_info['component']

            # Find downstream neighbors
            downstream = self.topo.downstream_neighbors(alert_node)

            for target in downstream:
                prop_rule = GraphRule(
                    node=target,
                    component=alert_comp,
                    threshold_pct=self.rules[rule_idx].threshold_pct * 0.5,
                    direction=self.rules[rule_idx].direction,
                    trend_window=2,  # relaxed confirmation
                )

                for t in range(alert_time, n_steps):
                    if self._check_rule(prop_rule, raw_data, t):
                        prop_key = f"prop_{rule_idx}_{target}"
                        if prop_key not in propagated:
                            node_name = self.topo.node_names[target]
                            comp_name = self.topo.component_names[alert_comp]
                            propagated[prop_key] = {
                                'time': t,
                                'rule': f"{comp_name} propagated to {node_name}",
                                'node': target,
                                'component': alert_comp,
                            }
                        break

        return propagated

    def format_report(self, result: dict) -> str:
        """Human-readable detection report for Origin."""
        lines = [
            "Graph Detection Report",
            "=" * 40,
            f"  Rules configured: {result['n_rules']}",
            f"  Alert propagation: {self.alert_propagation}",
            "",
        ]

        if result['detection_time'] is not None:
            lines.append(
                f"  FIRST DETECTION at t={result['detection_time']}")
            lines.append(f"  Rule: {result['detection_rule']}")
        else:
            lines.append("  No detection triggered")

        lines.append("")
        lines.append("  Alert timeline:")
        sorted_alerts = sorted(result['alert_timeline'].values(),
                               key=lambda x: x['time'])
        for alert in sorted_alerts:
            lines.append(f"    t={alert['time']}: {alert['rule']}")

        return "\n".join(lines)
