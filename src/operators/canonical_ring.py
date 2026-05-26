# canonical_ring.py — V3 Ring Operators (Proof Reference)
#
# This module preserves the ring-topology operators used in
# Object Error theorems and V3 canonical proofs.
#
# NOT for V4 applications. Retained as:
#   - validation reference against operators_V4.rs
#   - proof-topology test suite
#   - baseline for comparing graph-native operator behavior
#
# All operators use np.roll on periodic ring.