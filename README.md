# SteadyState_BH
Code using [AMUSE](https://amuse.readthedocs.io/en/latest/) which looks at simulating intermediate mass black hole (IMBH) clusters, testing their stability in an attempt to investigate whether IMBH mergers could help explain the formation of super massive black holes.


- GRX_Code/ is for local computer run and adopts the GRX formulation. This directory contains all the scripts needed for the numerical simulation. To run, have your environment in the directory and run interface.py
- Data\_Process/ helps plot results. It has scripts for the data analysis, and scripts in the directory data\_extraction to extract the data from the simulation and manipulate them for quicker future reading. To run them, you have to have your environment in this directory and run plot_all_figs.py.

In running the code, keep in mind the various hardcoded values (i.e saved directories, population investigated during data analysis...)

![](https://imgur.com/a/gWdje2c)
