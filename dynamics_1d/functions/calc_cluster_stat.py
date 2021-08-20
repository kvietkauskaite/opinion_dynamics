def calc_cluster_stat (data, R, n):

    """
    This function calculates the number of clusters existing in the data and
    also provides the mean and variance of the clusters.

    @param data   - data (the last step of opinion dynamics);
    @param R      - a bound (confidence level);
    @param n      - a number of agents (radicals included).
    """

    import numpy as np

    # the cluster list
    cl = [data[0]]

    # loop through remaining agents
    for i in range(n):

        # loop through the clusters found
        for j in range(len(cl)):

            exists = False

            # if the distance is < R between the agenti and cluster j, the cluster already exists
            if abs(data[i] - cl[j]) < R:
                exists = True

        # add a new cluster
        if not exists:
            cl.append(data[i])

    # calculate statistics
    mean = np.mean(cl)
    var = np.var(cl)

    # return the number of clusters and statistics
    return (len(cl), mean, var)
