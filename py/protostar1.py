#===============================================================================
"""Import Modules"""
#-------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import auxillary1 as aux
import units as units
import pdb

#===============================================================================
"""Known Constants"""
#-------------------------------------------------------------------------------

C           =   units.C

#===============================================================================
""" Main Problem """
#-------------------------------------------------------------------------------

# mass of model clouds:             solar mass
M_clouds    =   np.array([.7,   .8,     1,      1.5,    2,      3,      5,      9,      15,     25,     60])
# collapse time of model clouds:    Myr
T_bench     =   np.array([100,  68.4,   38.9,   35.4,   23.4,   7.24,   1.15,   .288,   .117,   .0708,  .0282])
# radii of model stars:             solar radius
R_sun       =   np.array([.8,   .93,    1,      1.2,    1.8,    2.2,    2.9,    4.1,    5.2,    10,     13.4])
# radii of model stars:             pc
R_star      =   R_sun * C['r_sun']

M_clouds    =   M_clouds[::-1]
T_bench     =   T_bench[::-1]
R_star      =   R_star[::-1]

def cloud_collapse(i, N_time=1000,N_shell=1000,tol=1e-5,saveA=True):

    data                =   aux.integrate(M_clouds[i],R_star[i],N_time,N_shell,tol)
    return data
