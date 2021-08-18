def HK_ode_r (R, n, x0, n_rad, x0_rad, h, stop = 10**(-5), result = 'FULL', include_self = True, max_steps = 1000):

    """
    This function generates the opinion dynamics using an ODE version of HK model,
    where opinions are distributed within a range [0,1].
    For that purpose, Euler's method is used.
    The function takes radicals into account.

    @param R            - a bound (confidence level);
    @param n            - a number of agents;
    @param x0           - an initial distribution of opinions;
    @param n_rad        - a number of radicals;
    @param x0_rad       - a distribution of radicals;
    @param h            - a step size;
    @param stop         - a stopping criterion;
    @param result       - a flag indicating if the whole 2D array is returned ('FULL')
                          or only the last step values (!= 'FULL');
    @param include_self - a flag indicating if self opinion plays a part in the stepping process;
    @param max_steps    - a maximum number of steps to take (used if self-opinion is not included).
    """

    import numpy as np
    import math
    from functions.dxdt import dxdt

    # combine the radicals and normal agents
    n_all = n + n_rad
    x0_all = np.concatenate((x0, x0_rad))
    x0_all.sort()

    # find radical ids
    radicals_idx = []

    for i in range(len(x0_rad)):

        for j in range(len(x0_all)):

            if j in radicals_idx:
                continue

            if x0_rad[i] == x0_all[j]:
                radicals_idx.append(j)
                break

    # the array of results (2D)
    res = []
    res.append(x0_all)

    x = np.copy(x0_all)
    y = np.copy(x)

    # the flag indicating if the calculation should proceed
    calculate = True

    # number of time steps taken
    steps = 0

    # if self-opinion is included, no oscillations should appear and the process terminates -
    # steps do not have to be cut
    if include_self:
        max_steps =  math.inf

    while calculate:

        # -------------------------------------------------
        # increase the step
        steps += 1

        # update one step using Euler's method
        y = x + h * dxdt(None, x, R, n_all, include_self, radicals_idx)

        # check the stopping conditions
        if max(abs(x - y)) <= stop or steps > max_steps:
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
