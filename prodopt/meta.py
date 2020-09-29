"""
Created 29 September 2020
Bjarne Grimstad, bjarne.grimstad@gmail.no
"""

# Names of wells and flowlines
WELL_NAMES = [str(i) for i in range(1, 9)]
FLOWLINE_NAMES = ['1', '1']

# GORs (scf/STB)
GOR = [1053.0, 805.0, 789.0, 790.0, 633.0, 745.0, 742.0, 759.0]

# Change GOR unit to mmscf/mSTB
GOR = [1e-3*g for g in GOR]

# WCs (%)
WCT = [12.7, 23.7, 41.7, 55.4, 54.81, 27.0, 0, 0]

# Separator pressure (bara) * 0.01
P_SEP = [0.20, 0.30]

# Gas capacity (mmscf / day) * 0.01
C_GAS = 3.40

# Gas-lift capacity (mmscf / day) * 0.01
C_GL = 0.20

# Variable bounds
WELL_PWH_BOUNDS = [
    (0.3, 1.65329),
    (0.3, 1.55873),
    (0.3, 1.76193),
    (0.3, 1.51174),
    (0.3, 1.44545),
    (0.3, 1.32290),
    (0.3, 1.62300),
    (0.3, 1.24020),
]

WELL_QOIL_BOUNDS = [
    (0.0, 0.240734),
    (0.0, 0.284504),
    (0.0, 0.190091),
    (0.0, 0.162084),
    (0.0, 0.090637),
    (0.0, 0.232649),
    (0.0, 0.269781),
    (0.0, 0.323648),
]

# Bounds on flowline variables: QOIL, QGAS, QWAT, PUS, PDS
FLOWLINE_BOUNDS = [
    (0.040000, 1.60000),
    (0.024000, 1.92000),
    (0.000000, 0.96000),
    (0.299867, 2.09987),
    (0.005022, 1.33179),
]


class Node:
    def __init__(self, id, pres_var):
        self.id = id
        self.pres_var = pres_var


# Implementation of directed acyclic graph for production optimization problem

class SourceNode(Node):
    def __init__(self, id, pres_var, qoil_var, qgas_var, qwat_var):
        super(SourceNode, self).__init__(id, pres_var)
        self.qoil_var = qoil_var
        self.qgas_var = qgas_var
        self.qwat_var = qwat_var


class Edge:
    def __init__(self, from_node, to_node, qoil_var, qgas_var, qwat_var):
        self.from_node = from_node
        self.to_node = to_node
        self.qoil_var = qoil_var
        self.qgas_var = qgas_var
        self.qwat_var = qwat_var


class DiscreteEdge(Edge):
    def __init__(self, from_node, to_node, qoil_var, qgas_var, qwat_var, dp_var, on_off_var):
        super(DiscreteEdge, self).__init__(from_node, to_node, qoil_var, qgas_var, qwat_var)
        self.dp_var = dp_var
        self.on_off_var = on_off_var


def get_ingoing_edges(node, edges):
    ingoing_edges = []
    for e in edges:
        if e.to_node == node:
            ingoing_edges.append(e)
    return ingoing_edges


def get_outgoing_edges(node, edges):
    outgoing_edges = []
    for e in edges:
        if e.from_node == node:
            outgoing_edges.append(e)
    return outgoing_edges

# Graph topology
NUM_NODES = 12
NODES = list(range(NUM_NODES))

# Edges given as (from, to, discrete)
EDGES = []

# Well edges (discrete edges)
for i in range(len(WELL_NAMES)):
    EDGES.append((i, 8, True))
    EDGES.append((i, 9, True))

# Flowline edges
EDGES += [(8, 10, False), (9, 11, False)]




