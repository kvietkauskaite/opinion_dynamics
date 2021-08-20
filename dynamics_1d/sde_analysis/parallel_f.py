def f(sigma, n, x0, h, stop, bound_cond, max_steps, r):

    """
    Function that is used by parallel_HK_2.py for dividing
    the computations through multiple processing units.
    """
    import numpy as np
    from functions.HK_sde import HK_sde
    from functions.calc_order_par import calc_order_par

    # the time points for taking the order parameter
    times = [1000, 2000, 5000, 10000, 50000, 100000]

    # simulation (get only the profile of the last step)
    x = HK_sde(r, n, x0, h, sigma, stop, result = 'FULL', include_self = True, max_steps = max_steps, bound_cond = bound_cond)

    # calculate the order parameters
    res = []

    for t in times:
        Q = calc_order_par(x[min(t, len(x)-1)], r, n, bound_cond)
        res.append(Q)

    # append the final result of Q
    Q = calc_order_par(x[-1], r, n, bound_cond)
    res.append(Q)

    # return the list of results
    return res
