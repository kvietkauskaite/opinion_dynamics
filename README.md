# opinion_dynamics

This github repository contains code that was used when writing the Master's thesis 'Mathematics of Opinion Formation'.
The work focuses on the bounded confidence opinion dynamics, in particular, three variants of Hegselmann-Krausse (HK) dynamics.
The files are located in two folders depending on the dimension of the opinions analysed: 1D or 2D.

In each folder there are four subfolders: 
- functions: holds the functions as tools that are used in the analysis of discrete-time, ode, or sde case; 
- discrete_analysis: holds the scripts for simulating the opinion dynamics and performing various type of analysis in dicrete-time; 
- ode_analysis: holds the scripts for simulating the opinion dynamics and performing various type of analysis in ODE case using Euler's scheme; 
- sde_analysis: holds the scripts for simulating the opinion dynamics and performing various type of analysis in SDE case using the Euler-Maruyama scheme.

In case of the SDE analysis in 1D, the script for analysis of order parameter uses the multiprocessing package to divide the computational load through 7 processing units.
For a more detailed description, see the write-ups inside the files (scripts or functions).
