"""
Created 08 November 2018
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

A simple production optimization problem
- 8 wells with routing and on/off
- two flowlines/risers with gas lift
- two topside separators

12 nodes
- 8 source nodes (wells)
- 2 nodes for the manifolds
 - 2 sink nodes (separators)
(- 1 artificial node to include gas-lift at injection point)

18 edges
- 16 discrete edges for the wells (choke and tie-ins)
- 2 flowline/riser edges

Each node has a pressure variable.
The nodes are connected by edges. Each edge has 3 flow variables (one per phase).
In addition there are gas-lift variables, binary variables for well switching and routing,
and other auxiliary variables.

Variable summary:
15 oil rates (mSTB/day) * 0.01
15 gas rates (mmscf/day) * 0.01
15 water rates (mSTB/day) * 0.01
1 gas-lift rate (mmscf/day) * 0.01
16 node pressures (bara) * 0.01, separator pressure is fixed
10 binary well inflow/source switches (on/off)

Total number of variables: 72

"""

import numpy as np
import gurobipy as gpy
from prodopt.meta import *
from prodopt.nn_models import *
from optimization.nn_milp_builder import build_milp_and_run_bt
from optimization.bound_tightening.bt_lp import bt_lrr, bt_rr


def solve_simple_prodopt(deep, bt_procedures, time_limit=None):
    """
    Solve production optimization problem
    :param deep: If True, use deep networks, otherwise, use shallow networks
    :param bt_procedures: List of callable bound tightening procedures
    :param time_limit: Time limit in seconds
    :return: Optimization results
    """

    # Create model
    model = gpy.Model('Simple Production Optimization Problem')

    # Enable eager model updates (lazy updates are default) - still need to use model.update() to update bounds, etc.
    model.Params.UpdateMode = 1
    if time_limit and time_limit > 0:
        model.setParam('TimeLimit', time_limit)  # Set time limit to 'time_limit' sec
        model.update()

    # NN surrogate constraint counter
    c_id = 0

    # Create nodes
    nodes = []

    # Well nodes (SourceNodes)
    for i in range(len(WELL_NAMES)):
        pvar = model.addVar(lb=0, ub=10, vtype=gpy.GRB.CONTINUOUS, name='pres_{}'.format(i))
        qoil = model.addVar(lb=0, ub=10, vtype=gpy.GRB.CONTINUOUS, name='qoil_{}'.format(i))
        qgas = model.addVar(lb=0, ub=10, vtype=gpy.GRB.CONTINUOUS, name='qgas_{}'.format(i))
        qwat = model.addVar(lb=0, ub=10, vtype=gpy.GRB.CONTINUOUS, name='qwat_{}'.format(i))
        nodes.append(SourceNode(i, pvar, qoil, qgas, qwat))

    # Manifold and riser nodes
    for i in range(4):
        j = i + len(WELL_NAMES)
        pvar = model.addVar(lb=0, ub=10, vtype=gpy.GRB.CONTINUOUS, name='pres_{}'.format(j))
        nodes.append(Node(j, pvar))

    edges = []
    for i, j, discrete in EDGES:
        qoil = model.addVar(lb=0, ub=10, vtype=gpy.GRB.CONTINUOUS, name='qoil_{}_{}'.format(i, j))
        qgas = model.addVar(lb=0, ub=10, vtype=gpy.GRB.CONTINUOUS, name='qgas_{}_{}'.format(i, j))
        qwat = model.addVar(lb=0, ub=10, vtype=gpy.GRB.CONTINUOUS, name='qwat_{}_{}'.format(i, j))

        if not discrete:
            edges.append(Edge(i, j, qoil, qgas, qwat))
        else:
            dp = model.addVar(lb=0, vtype=gpy.GRB.CONTINUOUS, name='dp_{}_{}'.format(i, j))
            on_off = model.addVar(lb=0, vtype=gpy.GRB.BINARY, name='on_off_{}_{}'.format(i, j))
            edges.append(DiscreteEdge(i, j, qoil, qgas, qwat, dp, on_off))

    # WPC, GOR and WCT constraints for wells
    for i in range(len(WELL_NAMES)):
        n = nodes[i]
        assert isinstance(n, SourceNode)
        model.addConstr(GOR[i]*n.qoil_var - n.qgas_var == 0, name='GOR_{}'.format(i))
        model.addConstr(WCT[i] / (100 - WCT[i])*n.qoil_var - n.qwat_var == 0, name='WCT_{}'.format(i))

        # Set variable bounds
        pwh_min, pwh_max = WELL_PWH_BOUNDS[i]
        qoil_min, qoil_max = WELL_QOIL_BOUNDS[i]

        model.update()
        n.pres_var.setAttr(gpy.GRB.Attr.LB, pwh_min)
        n.pres_var.setAttr(gpy.GRB.Attr.UB, pwh_max)
        model.update()

        n.qoil_var.setAttr(gpy.GRB.Attr.LB, qoil_min)
        n.qoil_var.setAttr(gpy.GRB.Attr.UB, qoil_max)
        model.update()

        n.qgas_var.setAttr(gpy.GRB.Attr.LB, GOR[i] * n.qoil_var.getAttr(gpy.GRB.Attr.LB))
        n.qgas_var.setAttr(gpy.GRB.Attr.UB, GOR[i] * n.qoil_var.getAttr(gpy.GRB.Attr.UB))
        model.update()

        n.qwat_var.setAttr(gpy.GRB.Attr.LB, WCT[i] / (100 - WCT[i]) * n.qoil_var.getAttr(gpy.GRB.Attr.LB))
        n.qwat_var.setAttr(gpy.GRB.Attr.UB, WCT[i] / (100 - WCT[i]) * n.qoil_var.getAttr(gpy.GRB.Attr.UB))
        model.update()

        # Load NN model
        if deep:
            nn_model = build_deep_well_model()
            nn_model.load_weights('./trained_networks/deep/wells/' + WELL_NAMES[i] + '/')
        else:
            nn_model = build_shallow_well_model()
            nn_model.load_weights('./trained_networks/shallow/wells/' + WELL_NAMES[i] + '/')
        nn_model._name = WELL_NAMES[i]

        # Call prediction to build model (since input shape is not specified by Sequential model)
        x = np.array([0]).reshape((1, -1))
        y_pred = nn_model.predict(x)

        # Build NN model constraints
        x_vars = [n.pres_var]  # PWH
        y_vars = [n.qoil_var]
        build_milp_and_run_bt(model, x_vars, y_vars, nn_model, c_id=c_id, bt_procedures=bt_procedures)

        # Increment counter
        c_id += 1

    # Mass balances
    for node in nodes:
        in_edges = get_ingoing_edges(node.id, edges)
        out_edges = get_outgoing_edges(node.id, edges)

        if len(out_edges) == 0:
            # Sink node
            continue
        elif len(in_edges) == 0 or isinstance(node, SourceNode):
            # Source node
            oil_balance = node.qoil_var - sum(e.qoil_var for e in out_edges)
            model.addConstr(oil_balance == 0)

            gas_balance = node.qgas_var - sum(e.qgas_var for e in out_edges)
            model.addConstr(gas_balance == 0)

            wat_balance = node.qwat_var - sum(e.qwat_var for e in out_edges)
            model.addConstr(wat_balance == 0)
        else:
            # Interior node
            oil_balance = sum(e.qoil_var for e in in_edges) - sum(e.qoil_var for e in out_edges)
            model.addConstr(oil_balance == 0)

            gas_balance = sum(e.qgas_var for e in in_edges) - sum(e.qgas_var for e in out_edges)
            model.addConstr(gas_balance == 0)

            wat_balance = sum(e.qwat_var for e in in_edges) - sum(e.qwat_var for e in out_edges)
            model.addConstr(wat_balance == 0)

    # Flowline momentum balances
    for i in range(2):
        e = edges[-i-1]  # Riser edges are stored last

        pus_var = nodes[e.from_node].pres_var
        pds_var = nodes[e.to_node].pres_var

        # Separator pressure constraint (fixed pressure)
        pds_var.setAttr(gpy.GRB.Attr.LB, P_SEP[i])
        pds_var.setAttr(gpy.GRB.Attr.UB, P_SEP[i])
        model.update()

        # Set variable bounds
        flowline_vars = [e.qoil_var, e.qgas_var, e.qwat_var, pus_var]

        for var, bounds in zip(flowline_vars, FLOWLINE_BOUNDS):
            vmin, vmax = bounds
            var.setAttr(gpy.GRB.Attr.LB, vmin)
            var.setAttr(gpy.GRB.Attr.UB, vmax)
        model.update()

        # Load NN model and build constraints
        if deep:
            nn_model = build_deep_flowline_model()
            nn_model.load_weights('./trained_networks/deep/flowlines/' + FLOWLINE_NAMES[i] + '/')
        else:
            nn_model = build_shallow_flowline_model()
            nn_model.load_weights('./trained_networks/shallow/flowlines/' + FLOWLINE_NAMES[i] + '/')
        nn_model._name = FLOWLINE_NAMES[i] + '_' + str(i+1)

        # Call prediction to build model (since input shape is not specified by Sequential model)
        x = np.array([0, 0, 0, 0]).reshape((1, -1))
        y_pred = nn_model.predict(x)

        # Build NN model constraints
        x_vars = [e.qoil_var, e.qgas_var, e.qwat_var, pus_var]
        y_vars = [pds_var]
        build_milp_and_run_bt(model, x_vars, y_vars, nn_model, c_id=c_id, bt_procedures=bt_procedures)

        # Increment counter
        c_id += 1

    # Momentum balances for discrete edges
    for e in edges:
        if isinstance(e, DiscreteEdge):
            pus = nodes[e.from_node].pres_var
            pds = nodes[e.to_node].pres_var

            # Big-M constraints:
            # -M(1 - ye) <= pi - pj - Dpe <= M(1 - ye),
            # where M = (pi_U - pi_L) + (pj_U - pj_L)
            M = (pus.getAttr(gpy.GRB.Attr.UB) - pus.getAttr(gpy.GRB.Attr.LB)) + \
                (pds.getAttr(gpy.GRB.Attr.UB) - pds.getAttr(gpy.GRB.Attr.LB))

            model.addConstr(pus - pds - e.dp_var <= M*(1 - e.on_off_var))
            model.addConstr(-M*(1 - e.on_off_var) <= pus - pds - e.dp_var)

            # Alternative to Big-M constraints: Indicator constraint
            # model.addGenConstrIndicator(e.on_off_var, True, pus - pds == e.dp_var)

    # Routing constraints
    for i in range(len(WELL_NAMES)):
        n = nodes[i]
        out_edges = get_outgoing_edges(n.id, edges)
        routing_expr = 0
        for e in out_edges:
            assert isinstance(e, DiscreteEdge)
            routing_expr += e.on_off_var
        model.addConstr(routing_expr <= 1)

    for e in edges:
        if isinstance(e, DiscreteEdge):
            model.addConstr(e.qoil_var - e.qoil_var.getAttr(gpy.GRB.Attr.UB) * e.on_off_var <= 0)
            model.addConstr(e.qgas_var - e.qgas_var.getAttr(gpy.GRB.Attr.UB) * e.on_off_var <= 0)
            model.addConstr(e.qwat_var - e.qwat_var.getAttr(gpy.GRB.Attr.UB) * e.on_off_var <= 0)

    # Gas capacity constraint
    qgas_cap_expr = 0
    for i in range(2):
        e = edges[-i-1]  # Riser edges are stored last
        qgas_cap_expr += e.qgas_var
    model.addConstr(qgas_cap_expr <= C_GAS)

    # Add objective function
    obj_expr = 0
    for i in range(2):
        e = edges[-i-1]  # Riser edges are stored last
        obj_expr += e.qoil_var
    model.setObjective(obj_expr, gpy.GRB.MAXIMIZE)

    lbs = [v.getAttr(gpy.GRB.Attr.LB) for v in model.getVars()]
    ubs = [v.getAttr(gpy.GRB.Attr.UB) for v in model.getVars()]
    print("Bounds range:", min(lbs), '-', max(ubs))

    #########################################
    # Optimize
    #########################################
    model.optimize()

    # Print solution
    # for v in model.getVars():
    #     print('%s %g' % (v.VarName, v.X))
    # print('Obj: %g' % model.ObjVal)
    # print('')

    results = dict()
    results['status'] = model.Status
    results['objective_value'] = model.ObjVal
    results['objective_bound'] = model.ObjBound
    results['optimality_gap'] = model.MIPGap
    results['runtime'] = model.RunTime
    results['num_binary_variables'] = model.NumBinVars
    results['num_variables'] = model.NumVars
    results['num_constraints'] = model.NumConstrs
    try:
        results['solution'] = {v.VarName: v.X for v in model.getVars()}
    except AttributeError as e:
        print('AttributeError: No solution available')

    return results


if __name__ == '__main__':

    # If False, use shallow networks, otherwise, use deep networks
    deep = False

    # Bound tightening procedures
    # Set to [bt_lrr, bt_rr] to use LRR followed by RR (note: RR should not be used alone)
    # Set to [bt_lrr] to use LRR
    bt_procedures = [bt_lrr, bt_rr]

    # Solve problem
    print(f'Solving production optimization problem')
    test_data = solve_simple_prodopt(deep, bt_procedures, time_limit=3600)
    print(test_data)
