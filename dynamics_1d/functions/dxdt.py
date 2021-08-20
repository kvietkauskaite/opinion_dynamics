def dxdt(t, x, R, n, include_self = True, radicals_idx = [], bound_cond = 'reflect'):
    """
    This function generates the RHS of the ODE version of HK opinion dynamics in 1D.

    @param t              - a time variable (empty);
    @param x              - a current distribution of opinions;
    @param R              - a bound (confidence level);
    @param n              - a number of agents;
    @param include_self   - a flag indicating if self opinion plays a part in the stepping process;
    @param radicals_idx   - a set of radical ids in the distribution array x (may be empty);
    @param bound_cond     - boundary conditions: 'reflect', 'adsorb', or 'period'.
    """

    import numpy as np

    y = np.copy(x)

    # update one step
    for i in range(n):

        # radical opinions stay constant
        if i in radicals_idx:
            y[i] = 0
            continue

        # reset the set I and sum of opinions
        I = []
        sum = 0

        for j in range(n):

            if not include_self:
                # do not include the agent i himself
                if i == j:
                    continue

            if bound_cond != 'period':

                # check if agent j influences agent i
                if abs(x[i] - x[j]) <= R:

                    # add it to the set I
                    I.append(j)

                    # update the sum of opinion differences
                    sum = sum + (x[i] - x[j])

            elif bound_cond == 'period':

                # check if agent j influences agent i
                # for periodic boundary condition, the agents can communicate in two directions: <-- and --> on the interval
                if min(abs(x[i] - x[j]), 1 - abs(x[i] - x[j])) <= R:

                    # add it to the set I
                    I.append(j)

                    # option 1: standard case
                    if abs(x[i] - x[j]) <= 1 - abs(x[i] - x[j]):

                        # update the sum of opinion differences
                        sum = sum + (x[i] - x[j])

                    # option 2: the shorter distance is going through boundary
                    elif 1 - abs(x[i] - x[j]) < abs(x[i] - x[j]):

                        if x[i] < x[j]:

                            # update the sum of opinion differences
                            sum = sum + (1 - abs(x[i] - x[j]))

                        elif x[i] > x[j]:

                            # update the sum of opinion differences
                            sum = sum - (1 - abs(x[i] - x[j]))

        # apply the RHS of ODE
        if include_self:
            y[i] = (-1/n) * sum
        else:
            y[i] = (-1/(n-1)) * sum

    return y
