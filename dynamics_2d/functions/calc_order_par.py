def calc_order_par(x, R, n, bound_cond = 'adsorb', dist_norm = 2):
    """
    This function calculates the order parameter of a given opinion profile in 2D.

    @param x              - a distribution of opinions;
    @param R              - a bound (confidence level);
    @param n              - a number of agents;
    @param bound_cond     - boundary conditions: 'reflect', 'adsorb', or 'period';
    @param dist           - a norm for measuring the distance (L1, L2): 1, 2.
    """

    import numpy as np

    count = 0

    # loop through all pairwise combinations
    for i in range(n):

        for j in range(n):

            if bound_cond != 'period':

                # check if agent j influences agent i using L1 or L2 norm
                if np.linalg.norm(x[i] - x[j], ord = dist_norm) <= R:

                    count = count + 1

            # periodic case needs to be added!

    # calculate the order parameters
    order_par = count / n**2

    return order_par
