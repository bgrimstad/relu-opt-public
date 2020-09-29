"""
Created 17 January 2019
Bjarne Grimstad, bjarne.grimstad@gmail.com

Some bound tightening utilities
"""

from copy import deepcopy
import gurobipy as gpy
from optimization.bound_tightening.bt_stats import bounds_mad


def init_empty_bounds(weights):
    """
    Initialize list of empty bounds for a neural network
    :param weights: weights in network (from which we can decide number of inputs and neurons)
    :return: empty bounds list
    """
    # Add input bounds
    w, b = weights[0]
    bounds = [[None] * w.shape[0]]

    # Add neuron bounds (including output neuron)
    for w, b in weights:
        bounds += [[None] * w.shape[1]]

    return bounds


def extract_bounds(bounds):
    """
    Extract input, hidden, and output bounds from bounds list
    :param bounds: list of lists of bounds (in order inputs bounds, hidden bounds, output bounds)
    :return: input bounds, hidden bounds, output bounds
    """
    # We expect at least one input layer and one output layer (zero hidden layers)
    assert len(bounds) >= 2
    input_bounds = deepcopy(bounds[0])
    hidden_bounds = deepcopy(bounds[1:-1])
    output_bounds = deepcopy(bounds[-1])
    return input_bounds, hidden_bounds, output_bounds


def build_bt_milp(weights, bounds, relax=False):
    """
    Build bound tightening optimization model
    :param weights: Model weights
    :param bounds: Model bounds
    :param relax: Build LP relaxation if True, otherwise build MILP model
    :return:
    """
    assert len(weights) == len(bounds) - 1

    K = len(bounds) - 1
    c_id = 0

    # Create optimization model
    model = gpy.Model()
    model.setParam('OutputFlag', False)  # Disable terminal output to reduce spam
    model.setParam(gpy.GRB.Param.MIPFocus, 3)  # Focus on best bound (seem to give more consistent results)
    model.update()

    # Create input variables
    x = []
    s = []
    z = []
    for i in range(K + 1):
        xi = []
        si = []
        zi = []
        for j, (lb, ub) in enumerate(bounds[i]):
            if i == 0 or i == K:
                # Input or output layer
                xi.append(model.addVar(lb=lb, ub=ub, vtype=gpy.GRB.CONTINUOUS, name=f'x_{i}_{j}_{c_id}'))
            else:
                # Hidden layer
                xi.append(model.addVar(lb=0, vtype=gpy.GRB.CONTINUOUS, name=f'x_{i}_{j}_{c_id}'))
                si.append(model.addVar(lb=0, vtype=gpy.GRB.CONTINUOUS, name=f's_{i}_{j}_{c_id}'))
                if relax:
                    zi.append(model.addVar(lb=0, ub=1, vtype=gpy.GRB.CONTINUOUS, name=f'z_{i}_{j}_{c_id}'))
                else:
                    zi.append(model.addVar(vtype=gpy.GRB.BINARY, name=f'z_{i}_{j}_{c_id}'))
        x.append(xi)
        s.append(si)
        z.append(zi)

    model.update()

    # Hidden layers (ReLU)
    # Hidden layers are numbered from 1 to K-1
    lb_con = [None]
    ub_con = [None]

    for i in range(1, K):
        w, b = weights[i - 1]
        nx, nu = w.shape  # Number of inputs to layer and number of units in layer
        assert nx == len(x[i - 1])
        assert w.shape[1] == b.shape[0]

        lb_con_i = []
        ub_con_i = []

        # Layer variables
        for j in range(nu):

            if not bounds[i][j]:
                raise Exception(f'No bounds on node ({i}, {j})')
            lb, ub = bounds[i][j]

            # Affine combination constraints
            model.addConstr(gpy.quicksum(w[k, j] * x[i - 1][k] for k in range(nx)) + b[j] == x[i][j] - s[i][j])

            # ReLU logic constraints
            ub_con_ij = model.addConstr(x[i][j] <= ub * z[i][j])
            lb_con_ij = model.addConstr(s[i][j] <= -lb * (1 - z[i][j]))
            ub_con_i.append(ub_con_ij)
            lb_con_i.append(lb_con_ij)

            if lb > 0:
                z[i][j].setAttr(gpy.GRB.Attr.LB, 1)  # Neuron activated
            if ub < 0:
                z[i][j].setAttr(gpy.GRB.Attr.UB, 0)  # Neuron deactivated

        lb_con.append(lb_con_i)
        ub_con.append(ub_con_i)

    # Output layer (affine)
    w, b = weights[-1]
    nx, nu = w.shape  # Number of inputs to layer and number of units in layer
    assert (nu == len(x[K]))
    for i in range(nu):
        model.addConstr(gpy.quicksum(w[j, i] * x[K - 1][j] for j in range(nx)) + b[i] == x[K][i])
    model.update()

    return model, x, s, z, lb_con, ub_con


def run_bound_tightening(weights, input_bounds, output_bounds, bt_procedures):
    """
    Run bound tightening procedures until no improvement in bounds
    :param weights: Network weights
    :param input_bounds: bounds on network inputs
    :param output_bounds: bounds on network outputs
    :param bt_procedures: list of bound tightening procedures to run (in listed order)
    :return: bounds on nodes in neural network
    """
    bounds = init_empty_bounds(weights)
    bounds[0] = input_bounds
    bounds[-1] = output_bounds

    # Perform bound tightening
    bounds_prev = None
    mad_prev = 0
    if bt_procedures:
        for bt in bt_procedures:
            bounds = bt(weights, bounds)
            mad = bounds_mad(bounds)
            if bounds_prev:
                print("Bounds improvement:", mad / mad_prev)
            bounds_prev = deepcopy(bounds)
            mad_prev = mad

    return bounds
