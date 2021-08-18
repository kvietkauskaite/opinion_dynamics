def HK_sde_r (R, n, x0, n_rad, x0_rad, h, sigma, stop = 10**(-5), result = 'FULL', include_self = True, max_steps = 1000, bound_cond = 'reflect'):

    """
    This function generates the opinion dynamics using an SDE version of HK model,
    where opinions are distributed within a range [0,1] and noise sigma is added.
    For that purpose, Euler-Maruyama method is used.
    The function takes radicals into account.
    The SDE case required the boundary conditions to be defined.

    @param R            - a bound (confidence level);
    @param n            - a number of agents;
    @param x0           - an initial distribution of opinions;
    @param n_rad        - a number of radicals;
    @param x0_rad       - a distribution of radicals;
    @param h            - a step size;
    @param sigma        - level of noise;
    @param stop         - a stopping criterion;
    @param result       - a flag indicating if the whole 2D array is returned ('FULL')
                          or only the last step values (!= 'FULL');
    @param include_self - a flag indicating if self opinion plays a part in the stepping process;
    @param max_steps    - a maximum number of steps to take (used if self-opinion is not included);
    @param bound_cond   - boundary conditions: 'reflect', 'adsorb', or 'period'.
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

    while calculate:

        # -------------------------------------------------
        # increase the step
        steps += 1

        # simulate the Wienner increment
        W_inc = np.sqrt(h) * np.random.randn(n_all, 1)
        W_inc = np.array([item for sublist in W_inc for item in sublist])

        W_inc[radicals_idx] = 0

        # update one step using Euler - Maruyama method
        y = x + h * dxdt(None, x, R, n_all, include_self, radicals_idx) + sigma * W_inc

        # boundary conditions
        if bound_cond == 'reflect':

            for i in range(len(y)):

                # if the opinion is not in [0, 1], it is reflected on the boundary
                while y[i] < 0 or y[i] > 1:

                    if y[i] < 0:
                        y[i] = - y[i]

                    elif y[i] > 1:
                        y[i] = 1 - (y[i] - 1)

        elif bound_cond == 'adsorb':

            for i in range(len(y)):

                if y[i] < 0:
                    y[i] = 0

                elif y[i] > 1:
                    y[i] = 1

        elif bound_cond == 'period':

            for i in range(len(y)):

                # if the opinion is not in [0, 1], it moves in the period
                if y[i] < 0 or y[i] > 1:

                    # the i'th opinion is its mod
                    y[i] = y[i] % 1

        # check the stopping conditions
        if max(abs(x - y)) <= stop or steps > max_steps:

            # terminate the calculation
            calculate = False

        else:

            # update the opinions x
            x = np.copy(y)

            # append the opinions to the major array
            if result == 'FULL':
                res.append(x)

        # -------------------------------------------------

    # prepare the results
    if result == 'FULL':
        # return the full 2D array
        return np.array(res)

    else:
        # return the last step only
        return x
