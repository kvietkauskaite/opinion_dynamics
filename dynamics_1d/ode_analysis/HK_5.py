"""
This script compares Euler's and Runge-Kutta method applied for an ODE version of HK opinion dynamics.
The results are plotted on a 'Timesteps vs Opinions' figure. The error norm is provided.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from functions.HK_ode import HK_ode
from functions.dxdt import dxdt
from functions.calc_clusters import calc_clusters
from scipy.integrate import solve_ivp

# PARAMETERS
#----------------------------------------
# the number of agents
n = 50

# the bound
R = 0.23

# the stopping criterion
stop = 10**(-5)

# the step size
h = 0.0035
#----------------------------------------


# INITIAL CONDITIONS
#----------------------------------------
# use expected value distribution
x0 = [i/(n+1) for i in range(1, n+1)]


# EULER'S METHOD
# ------------------------------------------------------------------------------
res_e = HK_ode (R, n, x0, h, stop, result = 'FULL', include_self = True)

# get the number of steps
steps_e = len(res_e)

# get the time array used for both methods
time = np.array(range(steps_e)) * h

# RUNGE-KUTTA METHOD
# ------------------------------------------------------------------------------
res_rk_all = solve_ivp(dxdt, [0, time[-1]], x0, 'RK45', args = (R, n, True), t_eval = time, rtol = 10**(-10), atol = 10**(-10))
res_rk = np.transpose(res_rk_all.y)

# calculate the error norm (L1 vector norm)
err = np.sum(abs(res_e[-1] - res_rk[-1]))

# PLOT THE RESULTS
#----------------------------------------

# define the colour map
colors = plt.cm.gist_rainbow(np.linspace(0, 1, n))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 5))

for j in range(n):

    # Euler
    ax1.plot(time, res_e[:,j], color = colors[j])

    # Runge-Kutta
    ax2.plot(time, res_rk[:,j], color = colors[j])

ax1.set_xlabel("Time", fontsize = 14)
ax1.set_ylabel("Opinions", fontsize = 14)
ax1.set_title("Euler's method", fontsize = 15)
ax1.grid()

ax2.set_xlabel("Time", fontsize = 14)
ax2.set_ylabel("Opinions", fontsize = 14)
ax2.set_title("Runge-Kutta method", fontsize = 15)
ax2.grid()

plt.show()
