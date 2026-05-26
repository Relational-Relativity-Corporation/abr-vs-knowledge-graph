# diagnostics.py — Graph-Native Diagnostics
#
# sigma_sq: variance of edge field subsets (declared projection)
# Gamma: R-sustained circulation, TOPOLOGY-SPECIFIC
#
# V4 discipline: Gamma is computed per edge type, not as a
# monolithic scalar. Total Gamma is itself a declared projection
# that discards topology-specific information.
#
# Per-component spatial Gamma (downstream and upstream separately)
# Per-pair component Gamma
# These are the primary observables. Total is secondary.

import numpy as np
from .types import NodeField, EdgeField
from .kernel import operator_a, operator_b, operator_r, compute_rho


def _variance(arr: np.ndarray) -> float:
    """Variance of array. Declared projection: scalar summary."""
    if arr.size == 0:
        return 0.0
    return float(np.var(arr))


# --- Per-edge-type sigma_sq ---

def sigma_sq_downstream(g: EdgeField, comp: int) -> float:
    """Variance of downstream edges for a single component."""
    return _variance(g.downstream[:, comp])


def sigma_sq_upstream(g: EdgeField, comp: int) -> float:
    """Variance of upstream edges for a single component."""
    return _variance(g.upstream[:, comp])


def sigma_sq_comp(g: EdgeField, pair_idx: int) -> float:
    """Variance of component edges for a single declared pair."""
    return _variance(g.comp[pair_idx])


def sigma_sq_total(g: EdgeField) -> float:
    """Total relational variance across all edge types.

    Declared projection:
      Preserves: total variance magnitude.
      Discards: per-edge-type distribution, directional distinction
        (downstream vs upstream), sign structure, spatial arrangement."""
    k = g.topo.k_components
    sv = sum(sigma_sq_downstream(g, c) + sigma_sq_upstream(g, c)
             for c in range(k))
    cv = sum(sigma_sq_comp(g, p)
             for p in range(len(g.topo.component_pairs)))
    return sv + cv


# --- Topology-specific Gamma ---

def gamma_decomposed(f: NodeField, rho_base: float = 0.3) -> dict:
    """Topology-specific Gamma decomposition.

    Computes Gamma per edge type:
      gamma_downstream[c]: Gamma for downstream edges of component c
      gamma_upstream[c]: Gamma for upstream edges of component c
      gamma_comp[p]: Gamma for component pair p

    These are the PRIMARY observables in V4.
    Total Gamma is a secondary summary.

    Gamma can be positive, negative, or zero per edge type.
    The sign is a diagnostic observable, not a pass/fail condition.
    Negative Gamma in one edge type while positive in another
    indicates R is redistributing variance across topologies —
    which is itself the coupling signal."""
    a = operator_a(f)
    rho = compute_rho(a, rho_base)
    b = operator_b(a)
    r = operator_r(b, rho)

    k = f.topo.k_components
    n_pairs = len(f.topo.component_pairs)

    g_down = []
    g_up = []
    for c in range(k):
        g_down.append(sigma_sq_downstream(r, c) - sigma_sq_downstream(b, c))
        g_up.append(sigma_sq_upstream(r, c) - sigma_sq_upstream(b, c))

    g_comp = []
    for p in range(n_pairs):
        g_comp.append(sigma_sq_comp(r, p) - sigma_sq_comp(b, p))

    total_e = sigma_sq_total(r)
    total_ba = sigma_sq_total(b)

    return {
        'gamma_downstream': g_down,
        'gamma_upstream': g_up,
        'gamma_comp': g_comp,
        'gamma_total': total_e - total_ba,
        'sigma_sq_e': total_e,
        'sigma_sq_ba': total_ba,
        'component_names': f.topo.component_names,
        'component_pairs': f.topo.component_pairs,
    }


def gamma_total(f: NodeField, rho_base: float = 0.3) -> float:
    """Total Gamma (secondary summary).

    Declared projection:
      Preserves: net R effect across all edge types.
      Discards: per-type attribution, directional distinction,
        redistribution between edge types."""
    return gamma_decomposed(f, rho_base)['gamma_total']


def format_gamma_report(gd: dict) -> str:
    """Human-readable Gamma decomposition for Origin review."""
    lines = [
        f"Gamma Decomposition",
        f"  Total Gamma: {gd['gamma_total']:.6f}",
        f"  sigma_sq(E): {gd['sigma_sq_e']:.6f}",
        f"  sigma_sq(BA): {gd['sigma_sq_ba']:.6f}",
        f"",
        f"  Downstream Gamma by component:",
    ]
    for c, name in enumerate(gd['component_names']):
        lines.append(f"    {name}: {gd['gamma_downstream'][c]:.6f}")

    lines.append(f"")
    lines.append(f"  Upstream Gamma by component:")
    for c, name in enumerate(gd['component_names']):
        lines.append(f"    {name}: {gd['gamma_upstream'][c]:.6f}")

    lines.append(f"")
    lines.append(f"  Component Gamma by pair:")
    for p, (a, b) in enumerate(gd['component_pairs']):
        na = gd['component_names'][a]
        nb = gd['component_names'][b]
        lines.append(f"    {na} <-> {nb}: {gd['gamma_comp'][p]:.6f}")

    return "\n".join(lines)
