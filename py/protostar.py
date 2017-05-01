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
"""Known Constants and parameters"""
#-------------------------------------------------------------------------------

C           =   units.C

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

# plotting parameters
p           =   {'figsize':(20,15),
                 'fs':20,
                 'style':'-r',
                 'lw':2
                 }

#===============================================================================
""" Main Problem """
#-------------------------------------------------------------------------------

def single_cloud_collapse(i):
    print("\ncalculating and collecting 'cloud_%s'..." % M_clouds[i])
    data    =   aux.integrate(M_clouds[i],R_star[i])
    return data

def write_clouds():
    N_clouds    =   len(M_clouds)
    print("\nstarting cloud writing and compilation sequence...")
    for i in range(N_clouds):
        single_cloud_collapse(i)

def plot_protostars(saveA=True):
    """ plot core temperature vs time for all clouds
    M_clouds:   numpy array of cloud masses
    p:          dictionary of plotting parameters
    """
    N_clouds        =   len(M_clouds)

    def plot_axis(i):
        try:
            print("\nloading 'cloud_%s'..." % M_clouds[i])
            data    =   pd.read_pickle('../data/cloud_%s' % M_clouds[i])
        except:
            print("\ndid not find 'cloud_%s'. Calculating..." % M_clouds[i])
            data    =   integrate(M_clouds[i],R_star[i])

        X       =   data['TIME']
        Y       =   data['T_core']

        ax      =   plt.subplot(4,3,i+1)
        ax.set_title("%s M$_\odot$ Cloud" % M_clouds[i], fontsize=p['fs']+2)
        ax.set_xlable("Time [%s]" % C['time'], fontsize=p['fs'])
        ax.set_ylabel("Temp [%s]" % C['temp'], fontsize=p['fs'])
        ax.plot(X,Y,p['style'], lw=p['lw'])
        return ax

    print("\nstarting plotting sequence...")
    fig = plt.figure(figsize=p['figsize'])
    for i in range(N_clouds):
        plot_axis(i)

    # save and return
    if saveA:
        print("\nsaving 'temp_vs_time' in 'figures' folder.")
        fig.savefig('../figures/temp_vs_time.png', dpi=1000)
        plt.close()
    else:
        plt.show()

def single_cloud_movie(i,saveA=True):
    try:
        print("\nloading 'cloud_%s'..." % M_clouds[i])
        data    =   pd.read_pickle('../data/cloud_%s' % M_clouds[i])
    except:
        print("\ndid not find 'cloud_%s'. Calculating..." % M_clouds[i])
        data    =   integrate(M_clouds[i],R_star[i])

    NotImplemented

    if saveA:
        print("\nsaving 'movie_%s'" % M_clouds[i])
    else:
        plt.show()



def write_protostar_movies(saveA=True):
    N_clouds    =   len(M_clouds)
    print("\nstarting movie sequence...")
    for i in range(N_clouds):
        single_cloud_movie(i)
