# supply_chain.py — Supply Chain Topology Declaration
#
# Origin-declared topology for this specific application.
# This is M's topological commitment. Operators act on this structure.
#
# Spatial: directed graph with 8 nodes, 7 adjacencies.
#   Each adjacency produces downstream and upstream edge types.
#   Downstream: material/product flow (supplier -> retailer)
#   Upstream: demand/information flow (retailer -> supplier)
#   These are structurally distinct with different dynamics.
#
# Components (k=5): inventory_level, lead_time, unit_cost,
#   quality_score, throughput
#
# Component topology: 6 declared operational pairs out of 10 possible.
#   Undeclared pairs may exhibit indirect coupling through declared
#   pairs — if R surfaces that coupling, it is a result, not an input.

from .declaration import TopologyDeclaration, DirectedAdjacency


# --- Node indices ---
SUPPLIER_SHANGHAI = 0
SUPPLIER_VIETNAM = 1
PORT_LONGBEACH = 2
WAREHOUSE_WEST = 3
WAREHOUSE_CENTRAL = 4
WAREHOUSE_EAST = 5
RETAILER_BIGBOX = 6
RETAILER_ONLINE = 7

# --- Component indices ---
INVENTORY_LEVEL = 0
LEAD_TIME = 1
UNIT_COST = 2
QUALITY_SCORE = 3
THROUGHPUT = 4


def declare_supply_chain_topology() -> TopologyDeclaration:
    """Origin declares the topology for the supply chain application.

    This function IS the M declaration. It commits to:
    - Which nodes exist
    - Which adjacencies are declared (and which are NOT)
    - Which components are measured
    - Which component pairs are operationally coupled

    Preserved by this declaration:
    - Directional flow structure (downstream vs upstream)
    - Operational coupling between declared adjacent nodes
    - Declared component coupling at each node
    - Irregular degree structure (fan-in at port, fan-out at warehouse)

    Discarded by this declaration:
    - Non-adjacent node coupling (e.g., Supplier -> Retailer direct)
    - Undeclared component pairs (4 of 10 possible)
    - Continuous geographic positioning (nodes are discrete)
    - Sub-node structure (e.g., multiple docks at a port)
    """

    return TopologyDeclaration(
        n_nodes=8,
        k_components=5,

        node_names=[
            "Supplier_Shanghai",    # 0
            "Supplier_Vietnam",     # 1
            "Port_LongBeach",       # 2
            "Warehouse_West",       # 3
            "Warehouse_Central",    # 4
            "Warehouse_East",       # 5
            "Retailer_BigBox",      # 6
            "Retailer_Online",      # 7
        ],

        component_names=[
            "inventory_level",  # 0
            "lead_time",        # 1
            "unit_cost",        # 2
            "quality_score",    # 3
            "throughput",       # 4
        ],

        # Spatial adjacencies: directed, downstream = source -> target
        # Material flows from suppliers through port through warehouses
        # to retailers. 7 declared adjacencies.
        spatial_adjacencies=[
            DirectedAdjacency(SUPPLIER_SHANGHAI, PORT_LONGBEACH),
            DirectedAdjacency(SUPPLIER_VIETNAM, PORT_LONGBEACH),
            DirectedAdjacency(PORT_LONGBEACH, WAREHOUSE_WEST),
            DirectedAdjacency(WAREHOUSE_WEST, WAREHOUSE_CENTRAL),
            DirectedAdjacency(WAREHOUSE_CENTRAL, WAREHOUSE_EAST),
            DirectedAdjacency(WAREHOUSE_WEST, RETAILER_BIGBOX),
            DirectedAdjacency(WAREHOUSE_EAST, RETAILER_ONLINE),
        ],

        # Component topology: 6 declared operational pairs
        # Origin declares these based on supply chain physics.
        # The 4 undeclared pairs are NOT assumed absent —
        # they are undeclared. If R surfaces indirect coupling
        # through declared pairs, that is a result.
        component_pairs=[
            (INVENTORY_LEVEL, THROUGHPUT),        # direct operational
            (LEAD_TIME, INVENTORY_LEVEL),          # supply delay -> stock
            (LEAD_TIME, UNIT_COST),                # expediting costs money
            (UNIT_COST, THROUGHPUT),                # cost constrains capacity
            (QUALITY_SCORE, INVENTORY_LEVEL),      # rejection reduces stock
            (QUALITY_SCORE, THROUGHPUT),            # quality slows processing
        ],
    )
