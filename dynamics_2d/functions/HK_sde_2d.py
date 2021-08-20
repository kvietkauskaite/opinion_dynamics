def HK_sde_2d (R, n, x0, h, sigma, stop = 10**(-5), result = 'FULL', max_steps = 100, dist_norm = 2, bound_cond = 'adsorb'):

    """
    This function generates the opinion dynamics n 2D using an SDE HK model,
    where opinions are distributed within a region [0,1]x[0,1].
    This function does not take radicals into account.

    @param R            - a bound (confidence level);
    @param n            - a number of agents;
    @param x0           - an initial distribution of opinions (2D matrix: n x 2);
    @param h            - a step size;
    @param sigma        - level of noise;
    @param stop         - a stopping criterion;
    @param result       - a flag indicating if the whole array is returned ('FULL')
                          or only the last step values (!= 'FULL');
    @param max_steps    - a maximum number of steps to take;
    @param dist_norm    - a norm for measuring the distance (L1, L2): 1, 2;
    @param bound_cond   - boundary conditions: 'reflect', 'adsorb', or 'period'.
    """

    import numpy as np
    import math

    # the array of results (2D)
    res = []
    res.append(x0)

    x = np.copy(x0)
    y = np.copy(x)

    # the flag indicating if the calculation should proceed
    calculate = True

    # number of time steps taken
    steps = 0

    while calculate:

        steps += 1

        # -------------------------------------------------
        # update one step
        for i in range(n):

            # reset the set I and sum of opinions for agent i for both dimensions
            I = []
            sum_1 = 0
            sum_2 = 0

            # loop through other agents
            for j in range(n):

                # check if agent j influences agent i using L1 or L2 norm
                if np.linalg.norm(x[i] - x[j], ord = dist_norm) <= R:

                    # add it to the set I
                    I.append(j)

                    # update the sum of opinions
                    sum_1 = sum_1 + (x[i][0] - x[j][0])
                    sum_2 = sum_2 + (x[i][1] - x[j][1])

            # simulate the Wienner increment for both dimensions
            W_inc = np.sqrt(h) * np.random.randn(2, 1)
            W_inc = np.array([item for sublist in W_inc for item in sublist])

            # update both coordinates with Euler's scheme
            y[i][0] = x[i][0] + h * (-1/n) * sum_1 + sigma * W_inc[0]
            y[i][1] = x[i][1] + h * (-1/n) * sum_2 + sigma * W_inc[1]

            # ------------------------------------------------------------------
            # boundary conditions
            if bound_cond == 'adsorb':

                for i in range(len(y)):

                    # if the opinion is not in [0, 1], it is adsorbed on the boundary
                    # x axis
                    if y[i][0] < 0:
                        y[i][0] = 0

                    elif y[i][0] > 1:
                        y[i][0] = 1

                    # y axis
                    if y[i][1] < 0:
                        y[i][1] = 0

                    elif y[i][1] > 1:
                        y[i][1] = 1
            # ------------------------------------------------------------------
            # periodic and reflecting conditions need to be added!


        # -------------------------------------------------
        # check if the calculation should terminate
        if np.linalg.norm(x - y, ord = dist_norm) <= stop or steps > max_steps:

            # terminate the calculation
            calculate = False

        else:
            # update the opinions x only when all agents are considered
            x = np.copy(y)

            # append the opinions to the major array
            res.append(x)

        # -------------------------------------------------

    # prepare the results
    if result == 'FULL':
        # return the full 2D array
        return np.array(res)

    else:
        # return the last step only
        return x
