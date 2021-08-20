"""
Analysis of a number of clusters and steps needed for varying parameters R and N in 2D.
This script simulates the opinion dynamics generated by using an ODE HK model in 2D
for varying parameters R and N. The number of clusters/steps for each pair of parameters
is given on a 2D raster. The analysis is performed using the Monte Carlo approach:
100 samples are taken and the cluster/convergence time results are averaged.
The results are saved in a csv file.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import colors
from analysis_2D.functions.HK_ode_2d import HK_ode_2d
from analysis_2D.functions.calc_clusters import calc_clusters
import warnings
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statistics as stat
import csv

# set the seed
np.random.seed(27)

#----------------------------------------------
# PARAMETERS

# the number of agents
N = np.linspace(1, 25, 25)
N = N.astype(np.int64)

# the bound
R = np.linspace(0, 0.375, 26)
R = R[1:]

# the stopping criterion
stop = 10**(-5)

# the step size
h = 0.1

# the distance norm for checking neighbours
dist_norm = 2

# the type of region: square/circle
region = 'circle'

# the array for clusters of all simulations
clusters_MC = []

# the array for steps of all simulations
steps_MC = []

#----------------------------------------------
# SIMULATION

for i in range(100):

    # the array of end clusters/steps
    clusters = []
    steps = []

    for n in N:

        # array of clusters/steps for one n
        c = []
        s = []

        # INITIAL CONDITIONS:
        #----------------------------------------
        if region == 'square':

            # OPTION 1a SQUARE: sample the initial opinions from a uniform distribution
            x0_x = np.random.uniform(0, 1, n)
            x0_y = np.random.uniform(0, 1, n)

            x0 = [[x0_x[i], x0_y[i]] for i in range(n)]

        elif region == 'circle':

            # OPTION 1b CIRCLE: sample the initial opinions from a uniform distribution
            length = 0.5 * np.sqrt(np.random.uniform(0, 1, n))
            angle = 2 * np.pi * np.random.uniform(0, 1, n)

            x0 = []

            x0_x = 0.5 + length * np.cos(angle)
            x0_y = 0.5 + length * np.sin(angle)

            x0 = [[x0_x[i], x0_y[i]] for i in range(n)]


        for r in R:

            # SIMULATION
            x = HK_ode_2d (r, n, x0, h, stop, result = 'FULL', max_steps = 2000, dist_norm = dist_norm)

            # check the number of clusters in the last step
            c.append(calc_clusters(x[-1], r, n, dist_norm = dist_norm))

            # check the number of steps
            s.append(len(x) * h)

        # append an array of clusters/steps for a specific n
        clusters.append(c)
        steps.append(s)

    clusters_MC.append(clusters)
    steps_MC.append(steps)

clusters_MC = np.array(clusters_MC)
steps_MC = np.array(steps_MC)

#----------------------------------------------
# AVERAGING

# reset the arrays - they will be used to hold the averages
clusters = []
steps = []

for i in range(len(N)):

    # temporary arrays for values of one n
    cl = []
    st = []

    for j in range(len(R)):

        # average the clusters
        cl_MC = [float(cl[i][j]) for cl in clusters_MC]
        cl.append(stat.mean(cl_MC))

        # average the steps
        st_MC = [float(st[i][j]) for st in steps_MC]
        st.append(stat.mean(st_MC))

    clusters.append(cl)
    steps.append(st)

clusters = np.array(clusters)
steps = np.array(steps)

#----------------------------------------------
# SAVE

# save results to a CSV file
with open('analysis_2D/ode_analysis/clusters_MC_100_L' + str(dist_norm) + '_' + region + '.csv',"w") as f:
    wr = csv.writer(f, lineterminator = '\n')
    wr.writerows(clusters)

with open('analysis_2D/ode_analysis/steps_MC_100_L' + str(dist_norm) + '_' + region + '.csv',"w") as f:
    wr = csv.writer(f, lineterminator = '\n')
    wr.writerows(steps)

#----------------------------------------------
# PLOT

nrows = len(N)
ncols = len(R)

# prepare data
data1 = np.ma.array(clusters.reshape((nrows, ncols)), mask=clusters==0)
data2 = np.ma.array(steps.reshape((nrows, ncols)), mask=steps==0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
im1 = ax1.imshow(data1, cmap = "Oranges", origin = "lower", vmin = 0, vmax = 25)
im2 = ax2.imshow(data2, cmap = "Oranges", origin = "lower", vmin = 0, vmax = 70)

# prepare the corresponding ticks for the axis
ytick_list = [None, 1, 6, 11, 16, 21]

xtick_list = [None]
for i in range(len(R)):
    if i%5==0:
        xtick_list.append(round(R[i], 3))

# workaround for the warning
warnings.filterwarnings("ignore")

for ax in [ax1, ax2]:

    ax.set_yticks(np.arange(nrows+1)-0.5, minor=True)
    ax.set_yticklabels(ytick_list)

    ax.set_xticks(np.arange(ncols+1)-0.499, minor=True)
    ax.set_xticklabels(xtick_list)

    ax.grid(which="minor")
    ax.tick_params(which="minor", size=0)

    ax.set_xlabel('R', fontsize=18)
    h = ax.set_ylabel('N', fontsize=18)
    h.set_rotation(0)

ax1.set_title('Cluster Analysis', fontsize = 20)
ax2.set_title('Step Analysis', fontsize = 20)

# colorbar 1
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')

# colorbar 2
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')

fig.suptitle('Analysis of End Clusters and Number of Steps', fontsize = 25)

# set the warnings back on
warnings.filterwarnings("default")
plt.show()