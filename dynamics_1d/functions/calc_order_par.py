def calc_order_par(x, R, n, bound_cond = 'reflect'):
    """
    This function calculates the order parameter of a given opinion profile in 1D.

    @param x              - a distribution of opinions;
    @param R              - a bound (confidence level);
    @param n              - a number of agents;
    @param bound_cond     - boundary conditions: 'reflect', 'adsorb', or 'period'.
    """

    import numpy as np

    count = 0

    # loop through all pairwise combinations
    for i in range(n):

        for j in range(n):

            if bound_cond != 'period':

                if abs(x[i] - x[j]) <= R:

                    count = count + 1

            elif bound_cond == 'period':

                # check if agent j influences agent i
                # for periodic boundary condition, the agents can communicate in two directions: <-- and --> on the interval
                if min(abs(x[i] - x[j]), 1 - abs(x[i] - x[j])) <= R:

                    count = count + 1

    # calculate the order parameters
    order_par = count / n**2

    return order_par
