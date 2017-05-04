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
SI          =   units.SI

# mass of model clouds: solar mass -> kg
M_clouds    =   np.array([.7,   .8,     1,      1.5,    2,      3,      5,      9,      15,     25,     60])
# M_clouds    =   M_clouds / units.solar_mass

# collapse time of model clouds: Myr -> s
T_bench     =   np.array([100,  68.4,   38.9,   35.4,   23.4,   7.24,   1.15,   .288,   .117,   .0708,  .0282])
# T_bench     =   T_bench / units.Myr

# radii of model stars: solar radius -> m
R_star      =   np.array([.8,   .93,    1,      1.2,    1.8,    2.2,    2.9,    4.1,    5.2,    10,     13.4])
R_star      =   R_star * C['r_sun']

N_clouds    =   len(M_clouds)

M_clouds    =   M_clouds[::-1]
T_bench     =   T_bench[::-1]
R_star      =   R_star[::-1]

# plotting parameters
p           =   {'figsize':(20,15),
                 'polar':(15,15),
                 'fs':20,
                 'style':'-r',
                 'lw':2}

m            =   {'N_grid':1000,
                  'writer':'ffmpeg',
                  'interval':int(100),
                  'dpi':int(400),
                  'cmap':cm.hot}

#===============================================================================
""" Main Problem """
#-------------------------------------------------------------------------------

def single_cloud_collapse(i, N_time=1000):
    print("\ncalculating and collecting 'cloud_%s'..." % M_clouds[i])
    return aux.integrate(M_clouds[i],R_star[i],N_time)

def write_clouds(N_time=1000):
    print("\nstarting cloud writing and compilation sequence...")
    for i in range(N_clouds):
        single_cloud_collapse(i, N_time=N_time)

def plot_protostars(N_time=1000,saveA=True):
    """ plot core temperature vs time for all clouds
    M_clouds:   numpy array of cloud masses
    p:          dictionary of plotting parameters
    """

    def plot_axis(i):
        try:
            print("\nloading 'cloud_%s'..." % M_clouds[i])
            data    =   pd.read_pickle('../data/cloud_%s' % M_clouds[i])
        except:
            print("\ndid not find 'cloud_%s'. Calculating..." % M_clouds[i])
            data    =   integrate(M_clouds[i],R_star[i], N_time=N_time)

        X       =   data['TIME']
        Y       =   np.log(data['T'])

        ax      =   plt.subplot(4,3,i+1)
        ax.set_title("%s M$_\odot$ Cloud" % M_clouds[i], fontsize=p['fs']+2)
        ax.set_xlabel("Time [ %s ]" % C['time'], fontsize=p['fs'])
        ax.set_ylabel("ln ( Temp [ %s ] )" % C['temp'], fontsize=p['fs'])
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

def single_cloud_movie(i, N_time=1000,degree=5,saveA=True):
    """ acknowledgements:
    http://matplotlib.org/examples/images_contours_and_fields/pcolormesh_levels.html
    https://matplotlib.org/users/colormapnorms.html"""
    try:
        print("\nloading 'cloud_%s'..." % M_clouds[i])
        data    =   pd.read_pickle('../data/cloud_%s' % M_clouds[i])
    except:
        print("\ndid not find 'cloud_%s'. Calculating..." % M_clouds[i])
        data    =   integrate(M_clouds[i], N_time=N_time)

    # take useful information from cloud data
    N_time      =   data['N_time']
    temp_shells =   data['T']
    R_shells    =   data['R']
    TIME        =   data['TIME']

    # create total cloud Temperature array
    T           =   np.zeros(( N_time , N_shell + 1 ))
    T[:,0]      =   temp_core
    T[:,1:]     =   temp_shells

    # create total cloud radius array assume core is at r = 0
    R           =   np.zeros_like(T)
    R[:,1:]     =   R_shells

    # create flux density in solar luminosity/pc^2
    PHI         =   SI['sigma'] * T**4 * ( units.solar_lum / units.unit_length**2 )
    Pmin,Pmax   =   np.min(PHI), np.max(PHI)
    Plimits     =   Pmin,Pmax
    print("phi shape", PHI.shape)

    # radial flux density profile
    fit         =   np.polyfit(R[99],T[99],degree)
    # fit         =   np.polyfit(R[0],T[0],degree)
    fdp         =   np.poly1d(fit)

    # set up initial figures  coordinates
    rmax0       =   R[99,-1]
    # rmax0       =   data['r_max']
    dx0 = dy0   =   rmax0 / m['N_grid']
    X0,Y0       =   np.mgrid[slice(-rmax0, rmax0+dx0, dx0), slice(-rmax0, rmax0+dy0, dy0)]
    Z0          =   fdp( np.sqrt(X0**2 + Y0**2) )
    Z0          =   Z0[:-1,:-1]

    # initialize figure and make colorbar
    fig         =   plt.figure(figsize=p['figsize'])
    ax          =   plt.subplot(111)
    levels      =   MaxNLocator(nbins=100).tick_values(Z0.min(), Z0.max())
    print(Z0.min(),Z0.max())
    # levels      =   MaxNLocator(nbins=100).tick_values(Pmin, Pmax)
    cmap        =   m['cmap']
    norm        =   BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    ax.set_title("%s M$_\odot$ Cloud: R = %.2f pc , t = %s Myr" % (M_clouds[i],rmax0,TIME[0]), fontsize=p['fs']+2)
    ax.set_xlabel("x [%s]" % C['length'], fontsize=p['fs'])
    ax.set_ylabel("y [%s]" % C['length'], fontsize=p['fs'])
    ax.set_xlim(-rmax0,rmax0)
    ax.set_ylim(-rmax0,rmax0)
    ax.set_aspect(1)
    # plot initial flux density
    im0         =   ax.contourf(X0[:-1,:-1] + dx0/2.,Y0[:-1,:-1] + dy0/2., Z0, levels=levels, cmap=cmap)
    cbar        =   fig.colorbar(im0, ax=ax, pad=0)
    cbar.set_label("L$_\odot$ / pc$^2$", fontsize=p['fs']+2)

    def animator(i_time):
        # radial flux density profile
        fit         =   np.polyfit(R[i_time],T[i_time],degree)
        fdp         =   np.poly1d(fit)

        # set up figures  coordinates
        rmax        =   R[i_time,-1]
        dx = dy     =   rmax0 / m['N_grid']
        X,Y         =   np.mgrid[slice(-rmax, rmax+dx, dx), slice(-rmax, rmax+dy, dy)]
        Z           =   fdp( np.sqrt(X**2 + Y**2) )
        Z           =   Z[:-1,:-1]

        # levels      =   MaxNLocator(nbins=100).tick_values(Z.min(), Z.max())
        # cmap        =   m['cmap']
        # norm        =   BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        # initialize figure and make colorbar
        ax.clear()
        ax.set_title("%s M$_\odot$ Cloud: R = %.4f pc , t = %.4f Myr" % (M_clouds[i],rmax0,TIME[i_time]), fontsize=p['fs']+2)
        ax.set_xlabel("x [%s]" % C['length'], fontsize=p['fs'])
        ax.set_ylabel("y [%s]" % C['length'], fontsize=p['fs'])
        ax.set_xlim(-rmax,rmax)
        ax.set_ylim(-rmax,rmax)
        ax.set_aspect(1)
        # plot initial flux density
        im          =   ax.contourf(X[:-1,:-1] + dx/2.,Y[:-1,:-1] + dy/2., Z, levels=levels, cmap=cmap)

    movie_anim      =   animation.FuncAnimation(fig, animator, frames=10, blit=False, interval=m['interval'])
    # movie_anim      =   animation.FuncAnimation(fig, animator, frames=int(N_time), blit=False, interval=m['interval'])

    if saveA:
        movie_anim.save('../figures/movie_%s.mp4' % M_clouds[i], writer=m['writer'], dpi=m['dpi'])
    else:
        plt.show()

def write_protostar_movies(saveA=True):
    N_clouds    =   len(M_clouds)
    print("\nstarting movie sequence...")
    for i in range(N_clouds):
        single_cloud_movie(i)


def printClouds():
    for i in range(N_clouds):
         data = pd.read_pickle('../data/cloud_%s' % M_clouds[i])
         print("\n M_cloud = %s" % data['M_cloud'])
         print("density0 = %s" % data['density_0'])
         print("R_j = %s" % data['r_max'])
         print("t_ff = %s" % data['t_ff'])
         print("t_collapse = %s" % data['t_collapse'])
         print("t_on = %s" % data['t_on'])
