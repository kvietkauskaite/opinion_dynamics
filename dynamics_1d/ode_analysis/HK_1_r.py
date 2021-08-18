"""
This script simulates the opinion dynamics generated by using an ODE version of HK model
where radicals are included. A function HK_ode_r is called after defining the required
parameters. The results are plotted on a 'Timesteps vs Opinions' figure.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os
from functions.HK_ode_r import HK_ode_r


# PARAMETERS
#----------------------------------------
# the number of agents
n = 23

# the bound
R = 0.24 #0.315

# the stopping criterion
stop = 10**(-5)

# the step size
h = 0.1


# RADICALS
#----------------------------------------
# the number of radicals
n_rad = 5

# distribution of radicals
x0_rad = np.array([0.8] * n_rad)


# INITIAL CONDITIONS
#----------------------------------------
# OPTION 1: simulate the initial opinions
# x0 = np.random.uniform(0, 1, n)
# x0.sort()

# OPTION 2: use expected value distribution
x0 = [i/(n+1) for i in range(1, n+1)]
#----------------------------------------


# SIMULATION
#----------------------------------------
res = HK_ode_r (R, n, x0, n_rad, x0_rad, h, stop, result = 'FULL', include_self = False)

# calculate the no of steps taken
steps = len(res)

# PLOT THE RESULTS
#----------------------------------------
figure(figsize=(10, 8), dpi=80)

# define the colour map
colors = plt.cm.gist_rainbow(np.linspace(0, 1, n + n_rad))

# plot each agent separately
for i in range(n + n_rad):
    plt.plot(range(steps), res[:,i], color = colors[i])

plt.xlabel("Timesteps")
plt.ylabel("Opinions")
plt.title("Opinion Dynamics with Radicals")
plt.grid()
plt.show()
