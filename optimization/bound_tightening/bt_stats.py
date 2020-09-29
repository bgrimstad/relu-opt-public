"""
Created 17 January 2019
Bjarne Grimstad, bjarne.grimstad@gmail.com

Various statistics for assessing various bound properties
"""


import numpy as np


def compare_bounds(a, b):
    """
    Bounds given as list of lists of tuples
    :param a: bounds a
    :param b: bounds b
    :return: True if all bounds are close, False otherwise
    """
    return np.allclose(np.array(a), np.array(b))


def bounds_stats(bounds):
    # Stats about bounds
    lower_bounds = []
    upper_bounds = []
    for b in bounds:
        for lb, ub in b:
            lower_bounds.append(lb)
            upper_bounds.append(ub)

    # Size of bounds
    mean_lb = [np.array([lb for lb, ub in hb]).mean() for hb in bounds]
    mean_ub = [np.array([ub for lb, ub in hb]).mean() for hb in bounds]

    print("\n*********Bounds stats*********")
    print("Min lower bound:", max(lower_bounds))
    print("Max upper bound:", max(upper_bounds))
    print("Mean lower bound:", mean_lb)
    print("Mean upper bound:", mean_ub)
    print("Dead neurons:   ", count_dead_neurons(bounds))
    print("Mean abs distance:", bounds_mad(bounds), "\n")


def count_dead_neurons(bounds):
    """
    Count number of hidden neurons with upper bound <= 0
    :param bounds: bounds on hidden neurons
    :return: number of dead neurons
    """
    ub_threshold = 1e-5  # Feasibility threshold (default feasibility tolerance of Gurobi is 1e-6)
    ub = np.array([bij for bi in bounds for bij in bi])[:, 1]
    return ub[ub <= ub_threshold].size


def bounds_mad(bounds, include_dead=True):
    """
    Compute mean absolute distance of bounds
    1/N sum_i |ub_i - lb_i|, for N bounds [lb_i, ub_i]
    :param bounds: list of bound tuples (lb_i, ub_i)
    :param include_dead: Include dead neurons in stats if True
    :return: mean absolute distance
    """
    bounds_np = np.array([bij for bi in bounds for bij in bi])
    assert bounds_np.shape[1] == 2
    lb = bounds_np[:, 0]
    ub = bounds_np[:, 1]

    if not include_dead:
        alive = ub >= 0
        lb = lb[alive]
        ub = ub[alive]

    return np.mean(np.abs(ub - lb))


def bounds_mrd(bounds, best_bounds, worst_bounds, include_dead=True):
    """
    Compute mean relative distance of bound intervals
    MRD(B, B*, B0) = |MAD(B) - MAD(B*)| / |MAD(B0) - MAD(B*)|
    :param bounds: bounds to measure
    :param best_bounds: optimal bounds
    :param worst_bounds: bounds computed with FBBT
    :param include_dead: Include dead neurons in stats if True otherwise exclude all neurons which are dead in b
    :return:
    """

    mad = bounds_mad(bounds, include_dead=include_dead)
    mad_best = bounds_mad(best_bounds, include_dead=include_dead)
    mad_worst = bounds_mad(worst_bounds, include_dead=include_dead)

    numerator = np.abs(mad - mad_best)
    denumerator = np.abs(mad_worst - mad_best)

    if np.isclose(denumerator, 0):
        return 0
    else:
        return numerator/denumerator

