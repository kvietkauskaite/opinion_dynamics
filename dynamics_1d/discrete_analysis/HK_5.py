"""
Reproduction of Figure 5 in 'Opinion dynamics under the influence of radical groups, charismatic leaders,
and other constant signals: A simple unifying model' by R. Hegselmann and U. Krause.
The figure depicts the necessary (though not sufficient) condition for numerical corectness: mirror symmetry around opinion 0.5.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from functions.HK_discrete import HK_discrete

# PARAMETERS
#----------------------------------------
# the number of agents
n = 50

# the bounds of interest
R = np.linspace(0, 0.4, 41)

# the stopping criterion
stop = 10**(-5)

# the array of final profiles for all R rounded_end_values
res_all = []

h = 0.1

#----------------------------------------

# INITIAL CONDITIONS
#----------------------------------------
# use expected value distribution
x0 = [i/(n+1) for i in range(1, n+1)]
#----------------------------------------

for i in range(len(R)):

    # SIMULATION
    #----------------------------------------
    # res = HK_discrete (R[i], n, x0, stop, result = 'LAST_STEP', include_self = True)
    res = HK_discrete (R[i], n, x0, stop, result = 'LAST_STEP', include_self = True)
    res_all.append(res)

res_all = np.array(res_all)

# PLOT THE RESULTS
#----------------------------------------
# define the colour map
colors = plt.cm.gist_rainbow(np.linspace(0, 1))
fig, ax = plt.subplots(1, 1, figsize = (15, 5))

# plot each agent separately
for j in range(n):
    ax.plot(R, res_all[:,j], color = colors[j])

# color the 25th and 26th position in black
ax.plot(R, res_all[:,24], color = 'black', lw = 2.5)
ax.plot(R, res_all[:,25], color = 'black', lw = 2.5)

ax.grid()
plt.ylim(0,1)
plt.ylabel("Final position", fontsize = 14)
plt.xlabel("R", fontsize = 14)
plt.show()
