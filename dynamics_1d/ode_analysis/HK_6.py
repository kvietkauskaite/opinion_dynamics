"""
This script analyses Euler's convergence to Runge-Kutta method applied for an ODE version of HK opinion dynamics.
The results are plotted on an log-log scale of 'Error norm vs step-size h' figure.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from functions.HK_ode import HK_ode
from functions.dxdt import dxdt
from functions.calc_clusters import calc_clusters
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import pylab
from math import log
import math

# PARAMETERS
#----------------------------------------
# the number of agents
n = 50

# the bound
R = 0.23

# the stopping criterion
stop = 10**(-6)

# the step sizes
h_list = np.linspace(0.1505, 0.0035, num = 50)

#----------------------------------------
# the error norms
errors = []

# INITIAL CONDITIONS
#----------------------------------------
# use expected value distribution
x0 = [i/(n+1) for i in range(1, n+1)]

for h in h_list:

    # EULER'S METHOD
    # ------------------------------------------------------------------------------
    res_e = HK_ode (R, n, x0, h, stop, result = 'FULL', include_self = True)

    # get the number of steps
    steps_e = len(res_e)

    # get the time array used for both methods
    time = np.array(range(steps_e)) * h

    # RUNGE-KUTTA METHOD
    # ------------------------------------------------------------------------------
    res_rk_all = solve_ivp(dxdt, [0, 30], x0, 'RK45', args = (R, n, True), t_eval = time, rtol = 10**(-10), atol = 10**(-10))
    res_rk = np.transpose(res_rk_all.y)

    # calculate the error norm (L1 vector norm)
    err = np.sum(abs(res_e[-1] - res_rk[-1]))
    errors.append(err)

# LEAST SQUARES
#----------------------------------------

# define the function
def f(x, A, B):
    return A*x + B

# define the log of variables
h_list_l = [log(h) for h in h_list]
errors_l = [log(err) for err in errors]

# fit the data
popt, pcov = curve_fit(f, h_list_l, errors_l)

# simulate the line of linear relationship: log(error) = A log(h) + B
lsq_line = popt[0] * np.array(h_list_l) + popt[1]

# define the power law relationship: error = e^B * h^A
lsq_line_2 = h_list**popt[0] * math.e**(popt[1])

# PLOT THE RESULTS
#----------------------------------------
fig = figure(figsize = (15, 5), dpi = 80)
ax = fig.add_subplot(111)

plt.plot(h_list, errors, color = 'grey')
plt.plot(h_list, lsq_line_2, color = 'orange')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('log(h)', fontsize = 14)
plt.ylabel('log(Error norm)', fontsize = 14)
plt.ylim(10**(-3), 10**(-1))
plt.grid(which = 'both')

plt.show()
