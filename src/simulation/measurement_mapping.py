# measurement_mapping.py — M : O -> D
#
# Maps synthetic observables to declared domain D.
# Unit scaling only (admissible pre-A transformation).
#
# M declares the embedding of supply chain observables into D,
# with stated preserved and discarded invariants.
#
# Preserved by M:
#   - Pairwise difference ratios within each component
#     (uniform scaling does not alter relational content)
#   - Sign structure of inter-node differences
#   - Declared topology (carried through from topology declaration)
#
# Discarded by M:
#   - Original physical units (replaced by domain-scaled values)
#   - Sub-node structure (each node is a single point in D)
#   - Continuous temporal dynamics (discretized to time steps)

import numpy as np
from ..topology.declaration import TopologyDeclaration


def scale_to_domain(raw: np.ndarray,
                    component_scales: np.ndarray) -> np.ndarray:
    """Uniform per-component scaling. Admissible pre-A transformation.

    T(x) = x / s  where s is a declared unit choice per component.
    This does not alter pairwise differences within a component
    (scales them uniformly). It brings components to comparable
    magnitude ranges so that rho computation is not dominated
    by a single component's physical units.

    Args:
        raw: shape (k, n) — raw observable values
        component_scales: shape (k,) — declared scale per component

    Returns:
        shape (k, n) — domain-scaled values"""
    assert raw.ndim == 2
    assert component_scales.shape == (raw.shape[0],)
    assert np.all(component_scales > 0), "scales must be positive"
    return raw / component_scales[:, np.newaxis]


# --- Declared component scales for supply chain ---
# These are part of M. They define the unit choice.
# Changing these changes D, not the relational content within D.

SUPPLY_CHAIN_SCALES = np.array([
    100.0,    # inventory_level: raw units ~100-1000, scale to ~1-10
    1.0,      # lead_time: raw units ~1-30 days, keep as-is
    10.0,     # unit_cost: raw units ~10-50 USD, scale to ~1-5
    0.1,      # quality_score: raw units ~0.5-1.0, scale to ~5-10
    100.0,    # throughput: raw units ~100-800 units/day, scale to ~1-8
])
