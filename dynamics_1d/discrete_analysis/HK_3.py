"""
Analysis of the distribution of number of clusters for initial conditions
sampled from the uniform distribution. The calculated data is first saved in a csv file
and then results are plotted in a bar plot.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from functions.HK_discrete import HK_discrete
from functions.calc_clusters import calc_clusters
from functions.calc_cluster_stat import calc_cluster_stat
import  csv
import ast


# PARAMETERS
#----------------------------------------------
# the number of agents
N = [20, 40, 60]

# the bound
R = [0.1, 0.175, 0.25]

# the array of end clusters
clusters = []
variances = []

# the stopping criterion
stop = 10**(-5)

# the number of total runs for each pair of parameters
total_runs = 1000

# SIMULATION
#----------------------------------------------
for r in R:

    for n in N:

        # array of clusters for one pair of parameters R and N
        c = []

        # array of cluster variances for one pair of parameters R and N
        # first values represent the label (number of clusters from 1 to 8)
        v = [[i] for i in range(1, 9)]

        # SAMPLED VALUES
        #----------------------------------------------
        for t in range(total_runs):

            # INITIAL CONDITIONS: simulate the initial opinions
            x0 = np.random.uniform(0, 1, n)
            x0.sort()

            # SIMULATION
            x = HK_discrete(r, n, x0, stop, 'LAST_STEP', include_self = True)

            # check the number of clusters in the last step
            clust_no = calc_cluster_stat(x, r, n)[0]
            c.append(clust_no)

            # get the statistics (0th row of v is for simulations with one end cluster; 1st - for 2 clusters; etc.)
            v[clust_no - 1].append(calc_cluster_stat(x, r, n)[2])

        clusters.append(c)
        variances.append(v)

# save the cluster results
np.savetxt('discrete analysis/HK 3 results.txt', clusters, fmt='%d')

# write variance CSV file
with open("discrete analysis/HK 3 results (var).csv","w") as f:
    wr = csv.writer(f, lineterminator = '\n')
    wr.writerows(variances)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# read cluster CSV file
with open("discrete_analysis/HK 3 results (var).csv", "r") as f:
    reader = csv.reader(f, delimiter=",", quotechar='"')
    var = [row for row in reader]

# convert string lists to lists
for i in range(len(var)):
    for j in range(len(var[0])):
        var[i][j] = ast.literal_eval(var[i][j])

# load the cluster results
clusters = np.loadtxt('discrete_analysis/HK 3 results.txt', dtype=int)

# PLOT
#----------------------------------------------
fig = plt.figure(figsize = (35, 25))
i = 1

for r in R:

    for n in N:

        # the cluster results of one calculation (one pair of n and r)
        c = clusters[i-1]

        # the cluster variances of one calculation
        v = var[i-1]

        # REFERENCE VALUE
        #----------------------------------------------
        # INITIAL CONDITION: use expected value distribution
        rx0 = [i/(n+1) for i in range(1, n+1)] #np.linspace(0, 1, n, endpoint = True)

        # SIMULATION
        rx = HK_discrete(r, n, rx0, stop, 'LAST_STEP', include_self = True)

        # check the number of clusters in the last step
        rc = calc_clusters(rx, r, n)


        # PLOT
        #----------------------------------------------
        plt.subplot(3, 3, i)
        bins = np.arange(10)-0.5
        values = plt.hist(c, bins = bins, rwidth = 0.8, color = 'darkgrey')

        # text over the bars with variance final numbers
        for t in range(1, 9):

            # cluster no 1 provides variance = 0, so it is skipped
            if t == 1:
                continue

            # if there is at least one instance in the simulation with t clusters
            if values[0][t] != 0:

                # calculate the final variance for one bar:
                # mean of variances / number of clusters) ?
                final_var = np.mean(v[t-1][1:]) # / v[t-1][0])

                # print the final variance value for the cluster
                plt.annotate(round(final_var, 3), xy = (t, values[0][t]), ha = 'center', fontsize = 15)

        # add reference value
        plt.axvline(x = rc, c = 'orange', linestyle='--', lw = 4)

        # calculate the ratio of the peak bar
        peak = int(max(values[0])) / total_runs

        # add title and axis names
        plt.title('Parameters: n = {}, R = {}. Peak bar ratio: {}.'.format(n, r, peak), fontsize = 25)

        if i in [10, 11, 12]:
            plt.xlabel('Number of clusters', fontsize = 20)

        if i in [1, 4, 7, 10]:
            plt.ylabel('Frequency', fontsize = 20)

        i += 1
