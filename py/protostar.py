#===============================================================================
"""Import Modules"""
#-------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import auxillary as aux
import pdb

#===============================================================================
""" Imported Parameters """
#-------------------------------------------------------------------------------

const   =   { **aux.const, **aux.MD }

#===============================================================================
""" Main Problem """
#-------------------------------------------------------------------------------

M_clouds    =   np.array([.7,   .8,     1,      1.5,    2,      3,      5,      9,      15,     25,     60])
T_bench     =   np.array([100,  68.4,   38.9,   35.4,   23.4,   7.24,   1.15,   .288,   .117,   .0708,  .0282])
T_model     =   T_bench*100

M_clouds    =   M_clouds[::-1]
T_bench     =   T_bench[::-1]

def cloud_collapse(i, N_time=1000,N_shells=1000,tol=1e-5,saveA=True):
    """ function that models collapsing cloud

    position arguments
    ------------------
    i:          index of M_clouds and T_bench

    keyword arguments
    -----------------
    N_shells:   number of shells        : default = 1000
    N_time:     length of time array    : default = 1000
    tol:        equilibrium tolerance   : default = 1e-5
    """

    """ construct time array: Myr """
    t_max       =   T_model[i]
    dt          =   t_max/N_time
    TIME        =   np.arange(0,t_max+dt,dt)

    """ Jean's radius of cloud: pc """
    r_max       =   aux.Jeans_radius(M_clouds[i])
    dr          =   r_max/N_shells

    """ make a radius array for cloud where initial
    dr < (initial sound speed) x (time increment)
    and r_max = Jean's radius. 'R' is an array of
    out shell radii. shells' inner radii are the
    inner shells' outer radii. """
    v_gas       =   aux.v_sound(const['T'])
    dr_gas      =   v_gas*dt
    assert dr < dr_gas, "shell widths must be less than (sound speed in gas) x (time increment)"
    R           =   np.arange(0,r_max+dr,dr)

    """ initla cloud volume and density """
    Volume_0    =   (4/3) * np.pi * r_max
    density_0   =   M_clouds[i]/Volume_0

    """ number of shells is length of radius array and
    it will stay constant for the whole model. each time
    step will determine a new r_max and dr using
    constant N_shells. N_time is length of TIME """
    N_shells    =   len(R) - 1
    N_time      =   len(TIME)

    """ construct dictionary of arrays """
    d               =   {}                              # data
    d['r']          =   np.zeros((N_time,N_shells))     # shell outer radii
    d['volume']     =   np.zeros_like(d['r'])           # shell volume
    d['area']       =   np.zeros_like(d['r'])           # shell inner area
    d['n']          =   np.zeros_like(d['r'])           # shell particle density
    d['mfp']        =   np.zeros_like(d['r'])           # shell mean free path
    d['temp']       =   np.zeros_like(d['r'])           # shell temperature
    d['acc']        =   np.zeros_like(d['r'])           # shell acceleration (total)
    # d['acc_grav']   =   np.zeros_like(d['r'])           # shell acceleration from gravity alone
    # d['acc_rad']    =   np.zeros_like(d['r'])           # shell acceleration from radiation pressure alone
    # d['acc_gas']    =   np.zeros_like(d['r'])           # shell acceleration from gas pressure alone
    d['vel']        =   np.zeros_like(d['r'])           # shell velocities

    d['mass']       =   np.zeros(N_shells)              # shell mass, constant
    d['mass_r']     =   np.zeros_like(d['mass'])        # shell internal mass, constant

    """ turns data dictionary to panda.Series """
    data            =   pd.Series(d)

    """initialize data"""
    data['r'][0,:]          =   R[1:]
    data['volume'][0,:]     =   np.array([ aux.shell_volume(data,0,j) for j in range(N_shells) ])
    data['area'][0,:]       =   np.array([ aux.shell_inner_area(data,0,j) for j in range(N_shells) ])
    data['mass'][:]         =   density_0 * data['volume'][0]
    data['mass_r'][:]       =   np.array([ np.sum(data['mass'][:j]) for j in range(N_shells) ])
    data['n'][0,:]          =   np.array([ aux.shell_particle_density(data,0,j) for j in range(N_shells) ])
    data['mfp'][0,:]        =   np.array([ aux.shell_mean_free_path(data,0,j) for j in range(N_shells) ])
    data['temp'][0,:]       =   np.ones(N_shells) * const['T']
    data['acc'][0,:]        =   np.array([ aux.acc_total(data,0,j) for j in range(N_shells) ])
    # data['acc_grav'][0,:]   =   np.array([ aux.acc_gravity(data,0,j) for j in range(N_shells) ])
    # data['acc_rad'][0,:]    =   np.array([ aux.acc_pressure_rad(data,0,j) for j in range(N_shells) ])
    # data['acc_gas'][0,:]    =   np.array([ aux.acc_pressure_gas(data,0,j) for j in range(N_shells) ])

    """fill in arrays"""
    data    =   aux.integrate(data,N_time,N_shells,dt)

    """save cloud data frame"""
    if saveA: data.to_pickle('../data/cloud_%s' % str(M_clouds[i]) )
    return data
