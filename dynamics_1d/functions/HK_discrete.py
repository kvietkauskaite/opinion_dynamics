def HK_discrete (R, n, x0, stop = 10**(-5), result = 'FULL', include_self = True, max_steps = 100):

    """
    This function generates the opinion dynamics using a discrete HK model,
    where opinions are distributed within a range [0,1].
    This function does not take radicals into account.

    @param R            - a bound (confidence level);
    @param n            - a number of agents;
    @param x0           - an initial distribution of opinions;
    @param stop         - a stopping criterion;
    @param result       - a flag indicating if the whole 2D array is returned ('FULL')
                          or only the last step values (!= 'FULL');
    @param include_self - a flag indicating if self opinion plays a part in the stepping process;
    @param max_steps    - a maximum number of steps to take (used if self-opinion is not included).
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

    # if self-opinion is included, no oscillations should appear and the process terminates -
    # steps do not have to be cut
    if include_self:
        max_steps =  math.inf

    while calculate:

        steps += 1

        # -------------------------------------------------
        # update one step
        for i in range(n):
            # reset the set I and sum of opinions
            I = []
            sum = 0

            for j in range(n):

                if not include_self:
                    # do not include the agent i himself
                    if i == j:
                        continue

                # check if agent j influences agent i
                if abs(x[i] - x[j]) <= R:
                    # add it to the set I
                    I.append(j)
                    # update the sum of opinions
                    sum = sum + x[j]

            # apply the stepping rule
            if len(I) != 0:
                y[i] = sum/len(I)
            else:
                y[i] = x[i]

        # -------------------------------------------------

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
