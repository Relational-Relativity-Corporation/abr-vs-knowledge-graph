# V4 Topology Discipline

Ring topology is the PROOF topology (V3 canonical, Object Error theorems).

V4 applications must NOT default to ring. Every application declares
its own spatial and component topology through Origin as part of M.

There is no topological default in V4. Undeclared topology is inadmissible.

## Why not ring for applications

1. **Periodic boundary violation:** Ring wraps last node to first,
   encoding coupling that doesn't exist in the operational system.

2. **Directional violation:** Ring produces bidirectional symmetric
   edges. Real systems have directionally distinct propagation modes
   (e.g., downstream material flow vs upstream demand flow) with
   different dynamics, speeds, and coupling behavior. The ring
   collapses this distinction.

3. **Uniform degree violation:** Ring imposes degree-2 at every node.
   Real systems have irregular degree (fan-in, fan-out, hub nodes).

## Consequence

Application topology is declared by Origin as part of M.
Operators act on declared adjacency only.
Undeclared adjacency is not computed.