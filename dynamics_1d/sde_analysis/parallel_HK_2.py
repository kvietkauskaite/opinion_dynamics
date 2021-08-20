"""
Copy of HK_2.py for SDE model, but divided through 7 processing units
using multiprocessing package.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import colors
from functions.HK_sde import HK_sde
from functions.calc_order_par import calc_order_par
import warnings
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable
import csv
import ast
import multiprocessing as mp
from functools import partial
from parallel_f import f

#----------------------------------------------
# PARAMETERS

# the level of noise
sigma = np.linspace(0, 0.16, 21)

# the bound
R = np.linspace(0, 0.35, 21)

# the number of agents
n = 100

# the step limit allowing to reach a stable position
max_steps = 10000

# the array of order parameters
order_p = []

# the stopping criterion
stop = 10**(-5)

# the step size
h = 0.05

# the boundary condition
bound_cond = 'adsorb'

#----------------------------------------------
# INITIAL CONDITIONS: use expected value distribution
x0 = [i/(n+1) for i in range(1, n+1)]

#----------------------------------------------
# SIMULATION

for s in sigma:

    if __name__ == '__main__':
        pool = mp.Pool(mp.cpu_count())
        func = partial(f, s, n, x0, h, stop, bound_cond, max_steps)
        result1 = pool.map(func, R[0:7])
        result2 = pool.map(func, R[7:14])
        result3 = pool.map(func, R[14:21])

    result1.extend(result2)
    result1.extend(result3)

    # append an array of order parameter for a specific sigma
    order_p.append(result1)

order_p = np.array(order_p)

#----------------------------------------------
# SAVE

# save results to a CSV file
with open("sde_analysis/HK 2 results_local.csv","w") as f:
    wr = csv.writer(f, lineterminator = '\n')
    wr.writerows(order_p)

#----------------------------------------------
# READ

# read the file
with open("sde_analysis/HK 2 results_server_period_7_times.csv", "r") as f:
    reader = csv.reader(f, delimiter=",", quotechar='"')
    order_p = [row for row in reader]

# convert string lists to lists
for i in range(len(order_p)):
    for j in range(len(order_p[0])):
        order_p[i][j] = re.sub("\s+", ", ", order_p[i][j].strip())
        order_p[i][j] = ast.literal_eval(order_p[i][j])

order_p = np.array(order_p)

#----------------------------------------------
# PLOT
fig, ax = plt.subplots(3, 3, figsize = (20, 20))
times = [2000, 5000, 10000, 50000, 100000, 200000]
times = [int(t * h) for t in times]

for pic in range(9):

    # prepare the first elements of sublists
    temp_order_p = np.array([item[pic] for list in order_p for item in list])
    # 0: 1000, t = 50
    # 1: 2000, t = 100
    # 2: 5000, t = 250
    # 3: 10000, t = 500
    # 4: 50000, t = 2500
    # 5: 100000, t = 5000
    # 6: 200000, t = 10000

    nrows = len(sigma)
    ncols = len(R)

    # prepare data
    data1 = np.ma.array(temp_order_p.reshape((nrows, ncols)), mask=temp_order_p==0)
    im1 = ax[pic//3][pic%3].imshow(data1, cmap = "Oranges", origin = "lower", vmin = 0)

    # prepare the corresponding ticks for the axis
    ytick_list = [None]
    for i in range(len(sigma)):
        if i%5==0:
            ytick_list.append(round(sigma[i], 3))
            ytick_list.append(None)

    xtick_list = [None]
    for i in range(len(R)):
        if i%5==0:
            xtick_list.append(round(R[i], 3))
            xtick_list.append(None)

    # workaround for the warning
    warnings.filterwarnings("ignore")

    ax[pic//3][pic%3].set_yticks(np.arange(nrows+1)-0.5, minor=True)
    ax[pic//3][pic%3].set_yticklabels(ytick_list)

    ax[pic//3][pic%3].set_xticks(np.arange(ncols+1)-0.499, minor=True)
    ax[pic//3][pic%3].set_xticklabels(xtick_list)

    # ax.grid(which="minor")
    ax[pic//3][pic%3].tick_params(which="minor", size=0)

    ax[pic//3][pic%3].set_xlabel('R', fontsize=18)
    h = ax[pic//3][0].set_ylabel(r'$\sigma$', fontsize=18)
    h.set_rotation(0)

    ax[pic//3][pic%3].set_title('t = {}'.format(times[pic]), fontsize = 20)

    # colorbar 1
    divider = make_axes_locatable(ax[pic//3][pic%3])
    cax = divider.append_axes('right', size='5%', pad=0.2)
    fig.colorbar(im1, cax=cax, orientation='vertical')

    # set the warnings back on
    warnings.filterwarnings("default")

plt.show()
