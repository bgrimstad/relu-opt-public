"""
Created 17 January 2019
Bjarne Grimstad, bjarne.grimstad@gmail.com
"""

import gurobipy as gpy
from copy import deepcopy
from optimization.bound_tightening.bt_utils import extract_bounds, build_bt_milp


def bt_rr(weights, bounds):
    """
    LP-based OBBT using ReLU relaxation. The procedure is named RR in the paper.

    Assuming that NN model has K-1 hidden layers with ReLU activation and 1 linear output layer

    Variables are named as x_i_j, where i is the layer number, and j is the unit number (of layer i)

    NOTE: Important to tighten bounds before calling this function to avoid the use of indicator variables
          conditioned on continuous variables

    :param weights: Neural network weights as list of tuples
    :param bounds: Bounds on variables in the neural network
    :return: tightened bounds
    """
    new_bounds = deepcopy(bounds)

    model, x, s, z, lb_con, ub_con = build_bt_milp(weights, new_bounds, relax=True)

    # Perform tightening for each neuron
    for i in range(len(bounds)):
        for j in range(len(bounds[i])):

            # Here, we build a new optimization problem for each iteration, using the updated bounds
            # To shorten run times, it is possible to update the coefficients in the model directly using the method
            # model.chgCoeff(constr, var, newvalue), but this requires us to keep track of the constraint objects.
            # model, x, s, z, _, _ = build_lp_relaxation(weights, new_bounds)

            # Input or output layer
            if i == 0 or i == len(bounds) - 1:
                # Find maximum value of x
                model.setObjective(x[i][j], gpy.GRB.MAXIMIZE)
                model.optimize()
                assert model.Status == gpy.GRB.OPTIMAL
                n_max = model.ObjVal

                # Find minimum value x
                model.setObjective(x[i][j], gpy.GRB.MINIMIZE)
                model.optimize()
                assert model.Status == gpy.GRB.OPTIMAL
                n_min = model.ObjVal

                # Save bounds
                new_bounds[i][j] = (n_min, n_max)

            else:
                # Find maximum value of x-s
                model.setObjective(x[i][j] - s[i][j], gpy.GRB.MAXIMIZE)
                model.optimize()
                assert model.Status == gpy.GRB.OPTIMAL
                n_max = model.ObjVal

                # Find minimum value of x-s
                model.setObjective(x[i][j] - s[i][j], gpy.GRB.MINIMIZE)
                model.optimize()
                assert model.Status == gpy.GRB.OPTIMAL
                n_min = model.ObjVal

                # Save bounds
                new_bounds[i][j] = (n_min, n_max)

                # Update constraints
                lb_con[i][j].rhs = -n_min
                model.chgCoeff(lb_con[i][j], z[i][j], -n_min)
                model.chgCoeff(ub_con[i][j], z[i][j], -n_max)
                model.update()

    return new_bounds


def fbbt_linear(x_lb, x_ub, w, b):
    """
    Feasibility-based bound tightening for linear constraints w*x + b
    :param x_lb: Lower bounds on x
    :param x_ub: Upper bounds on x
    :param w: Weight vector
    :param b: Bias
    :return: lower bound, upper bound
    """
    nx = len(x_lb)
    assert(nx == len(x_ub) and nx == len(w))

    m = gpy.Model()
    m.setParam('OutputFlag', False)  # Disable terminal output to reduce spam
    m.update()

    x = [m.addVar(lb=x_lb[i], ub=x_ub[i], vtype=gpy.GRB.CONTINUOUS) for i in range(nx)]
    y = m.addVar(lb=-gpy.GRB.INFINITY, vtype=gpy.GRB.CONTINUOUS)
    m.addConstr(gpy.quicksum(w[i] * x[i] for i in range(len(x))) + b, gpy.GRB.EQUAL, y)

    # Find maximum value
    m.setObjective(y, gpy.GRB.MAXIMIZE)
    m.optimize()
    assert m.Status == gpy.GRB.OPTIMAL
    y_max = m.ObjVal

    # Find minimum value
    m.setObjective(y, gpy.GRB.MINIMIZE)
    m.optimize()
    assert m.Status == gpy.GRB.OPTIMAL
    y_min = m.ObjVal

    return y_min, y_max


def bt_lrr(weights, bounds):
    """
    Feasibility-based bound tightening (FBBT) for ReLU networks. The procedure is called LRR in the paper.
    Propagate input bounds in a forward pass.
    :param weights: Neural network weights
    :param bounds: Bounds on variables in the neural network
    :return: List of tuples (lb, ub) for each neuron
    """
    assert len(weights) == len(bounds[1:])

    input_bounds, hidden_bounds, output_bounds = extract_bounds(bounds)

    # Perform FBBT for each neuron
    new_bounds = deepcopy(bounds)
    x_prev_lb = [lb for lb, ub in input_bounds]
    x_prev_ub = [ub for lb, ub in input_bounds]

    for i, weights_i in enumerate(weights):
        w, b = weights_i
        _, nu = w.shape  # Number of inputs to layer and number of units in layer
        for j in range(nu):
            x_lb, x_ub = fbbt_linear(x_prev_lb, x_prev_ub, w[:, j], b[j])

            # Intersect with given bounds
            if new_bounds[i + 1][j]:
                h_lb, h_ub = new_bounds[i + 1][j]
                x_lb = max(x_lb, h_lb)
                x_ub = min(x_ub, h_ub)
                assert x_lb <= x_ub

            new_bounds[i + 1][j] = (x_lb, x_ub)

        # Set bounds of neurons in previous layer
        # NOTE: Lower bound is 0 due to ReLU activation
        x_prev_lb = [max(0, x_lb) for x_lb, x_ub in new_bounds[i + 1]]
        x_prev_ub = [max(0, x_ub) for x_lb, x_ub in new_bounds[i + 1]]

    # Print some stats
    # bounds_stats(new_bounds)

    return new_bounds
