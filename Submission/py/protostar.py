#===============================================================================
"""Import Modules"""
#-------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as colors
import matplotlib.cm as cm
import auxillary as aux
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
                 'polar':(15,15),
                 'fs':20,
                 'style':'-r',
                 'lw':2,
                 'N_grid':100
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
        ax.set_xlabel("Time [%s]" % C['time'], fontsize=p['fs'])
        ax.set_ylabel("Temp [%s]" % C['temp'], fontsize=p['fs'])
        ax.set_xlim([min(X),max(X)])
        # pdb.set_trace()
        ax.plot(X,Y,p['style'], lw=p['lw'])
        return ax

    print("\nstarting plotting sequence...")
    fig = plt.figure(figsize=p['figsize'])
    for i in range(N_clouds):
        plot_axis(i)
    plt.tight_layout()

    # save and return
    if saveA:
        print("\nsaving 'temp_vs_time' in 'figures' folder.")
        fig.savefig('../figures/temp_vs_time.png')
        plt.close()
    else:
        plt.show()

def single_cloud_movie(i, degree=5,interval=10,writer='ffmpeg',dpi=400,saveA=True):
    """ acknowledgements: http://matplotlib.org/examples/images_contours_and_fields/pcolormesh_levels.html """
    try:
        print("\nloading 'cloud_%s'..." % M_clouds[i])
        data    =   pd.read_pickle('../data/cloud_%s' % M_clouds[i])
    except:
        print("\ndid not find 'cloud_%s'. Calculating..." % M_clouds[i])
        data    =   integrate(M_clouds[i],R_star[i])

    TIME    =   data['TIME']

    fig     =   plt.figure(figsize=p['polar'])
    # make normalized colorbar

    def animator(i_time):
        # make polar axis
        ax          =   plt.subplot(111)
        ax.set_title("%s M$_\odot$ Cloud: t = %s" % (M_clouds[i],TIME[i_time]), fontsize=p['fs']+2 )
        ax.set_xlabel("X [%s]" % C['length'], fontsize=p['fs'])
        ax.set_ylabel("Y [%s]" % C['length'], fontsize=p['fs'])

        rmax        =   data['R'][i_time,-1]
        # radial temperature profile
        R_data      =   np.hstack(( 0 , data['R'][i_time] ))
        T_data      =   np.hstack(( data['T_core'][i_time] , data['T'][i_time] ))
        fit         =   np.polyfit(R_data,T_data,degree)
        tp          =   np.poly1d(fit)

        # XY
        dx = dy     =   rmax / p['N_grid']
        dA          =   dx*dy
        X,Y         =   np.mgrid[slice(-rmax, rmax+dx, dx),
                                 slice(-rmax, rmax+dy, dy)]

        # luminosity grid: L(x,y) = sigma * T(x,y)**4 * dA
        Z           =   C['sigma'] * tp(np.sqrt(X**2 + Y**2))**4 * dA
        Z           =   Z[:-1, :-1]
        levels      =   MaxNLocator(nbins=50).tick_values( Z.min(),Z.max() )

        cmap        =   plt.get_cmap('hot')
        norm        =   BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        cf          =   ax.contourf(X[:-1,:-1] + dx/2.,
                                    Y[:-1,:-1] + dy/2.,
                                    Z, levels=levels, cmap=cmap)
        return ax

    movie_anim      =   animation.FuncAnimation(fig, animator, frames=len(TIME), blit=False, interval=interval)

    if saveA:
        movie_anim.save('../figures/movie_%s.mp4' % M_clouds[i], writer=writer, dpi=dpi)
    else:
        plt.show()

def write_protostar_movies(saveA=True):
    N_clouds    =   len(M_clouds)
    print("\nstarting movie sequence...")
    for i in range(N_clouds):
        single_cloud_movie(i)
