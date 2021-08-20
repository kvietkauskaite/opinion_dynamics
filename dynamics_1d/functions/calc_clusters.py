def calc_clusters (data, R, n):

    """
    This function calculates the number of clusters existing in the data.

    @param data   - data (the last step of opinion dynamics);
    @param R      - a bound (confidence level);
    @param n      - a number of agents (radicals included).
    """

    import numpy as np

    # sort the data
    data_sort = np.sort(data)

    # the cluster list
    cl = [data_sort[0]]

    # loop through remaining agents
    for i in range(n):

        # loop through the clusters found
        for j in range(len(cl)):

            exists = False

            # if the distance is < R between the agent i and cluster j, the cluster already exists
            if abs(data_sort[i] - cl[j]) < R:
                exists = True

        # add a new cluster
        if not exists:
            cl.append(data_sort[i])

    # return the number of clusters found
    return len(cl)
