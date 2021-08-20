# opinion_dynamics

This GitHub repository contains code that was used when writing the Master's thesis 'Mathematics of Opinion Formation'. 
The work focuses on the bounded confidence opinion dynamics, in particular, three variants of Hegselmann-Krausse (HK) dynamics. 
The files are located in two folders depending on the dimension of the opinions analysed: 1D or 2D.

___
**In each folder there are four subfolders:**
- functions: holds the functions as tools that are used in the analysis of the discrete-time, ode, or sde case;
- discrete_analysis: holds the scripts for simulating the opinion dynamics and performing the various type of analysis in discrete-time;
- ode_analysis: holds the scripts for simulating the opinion dynamics and performing the various type of analysis in the ODE case using Euler's scheme;
- sde_analysis: holds the scripts for simulating the opinion dynamics and performing the various type of analysis in the SDE case using the Euler-Maruyama scheme.

In the case of the SDE analysis in 1D, the script for analysis of the order parameter uses the multiprocessing package to divide the computational load through 7 processing units.
For a more detailed description, see the write-ups inside the files (scripts or functions).

___
**List of scripts:**

 - HK_1   - the paths of opinion dynamics (time vs opinions);
 - HK_1_r - same as above but with radicals included;
 - HK_2   - clustering patterns and convergence time for varying parameters R and N (R and sigma in SDE case). Data is plotted on a 2D raster (R vs N or R vs sigma);
 - HK_2_r - same as above but with radicals included;
 - HK_3   - the distribution of number of clusters for initial profiles sampled from the uniform distribution (barplots);
 - HK_3_r - same as above but with radicals included;
 - HK_4   - the number of clusters for varying parameter R when n is fixed (R vs number of clusters);
 - HK_5 (discrete) - R-diagram test (R vs opinions at the last step);
 - HK_5 (ODE)- comparison between Euler's and Runge-Kutta methods applied for an ODE version of HK opinion dynamics (2 plots: time vs opinions);
 - HK_6   - Euler's convergence to the Runge-Kutta method (log(h) vs log(error)).
