#===============================================================================
"""Import Modules"""
#-------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import pdb

#===============================================================================
"""Known Constants"""
#-------------------------------------------------------------------------------

# dictionary of constants and parameters in SI units
SI          =   {}

# dictionary of constants and parameters that are the same between SI and model units
const       =   {}

const['A']  =   6.02214199e23       # Avagadro's number             (#)
SI['m_H']   =   1.6737236e-27       # mass of hydrogen              (kg)
SI['m_He']  =   6.6464764e-27       # mass of helium                (kg)
SI['mol_H'] =   1.00794/1000        # mol mass of hydrogen          (kg/mol)
SI['mol_He']=   4.002602/1000       # mol mass of helium            (kg/mol)
SI['d_H']   =   53e-12              # size of hydrogen atom         (m)
SI['d_He']  =   31e-12              # size of helium atom           (m)
SI['c']     =   2.99792458e8        # speed of light                (m/s)
SI['G']     =   6.67408e-11         # gravitational constant        (N m^2/kg^2)
SI['k']     =   1.3806503e-23       # Bolztmann's constant          (J/K)
SI['sigma'] =   5.670373e-8         # Stefan Bolztmann's constant   W/(m^2 K^4)
SI['a']     =   7.5657e-16          # radiation constant            J/(m^3 K^4)
SI['R']     =   8.314472            # gas constant                  J/(mol K)

# """ pure hydrogen cloud """
# SI['mu']    =   SI['m_H']
# SI['mol_mu']=   SI['mol_H']
# SI['d']     =   SI['d_H']

""" 3/4 hydrogen 1/4 helium cloud """
SI['mu']    =   (3/4)*SI['m_H'] + (1/4)*SI['m_He']      # average particle mass     (kg)
SI['mol_mu']=   (3/4)*SI['mol_H'] + (1/4)*SI['mol_He']  # average particle mol mass (kg/mol)
SI['d']     =   (3/4)*SI['d_H'] + (1/4)*SI['d_He']      # average particle size     (m)

#===============================================================================
"""Model Parameters"""
#-------------------------------------------------------------------------------

const['gamma']  =   5/3     # heat capacity ratio for monatomic ideal gas
const['T']      =   10      # initial cloud temperature (K)

#===============================================================================
"""Conversion factors"""
#-------------------------------------------------------------------------------

unit_length =   1/3.086e16                  # m -> pc
unit_time   =   1/(60*60*24*365.25*1e6)     # s -> Myr
unit_mass   =   1/1.98855e30                # kg -> solar mass
unit_speed  =   unit_length / unit_time
unit_force  =   unit_mass * unit_length / unit_time**2
unit_energy =   unit_force * unit_length
unit_power  =   unit_energy / unit_time

#===============================================================================
""" Auxillary Functions """
#-------------------------------------------------------------------------------

def unit_conversion():
    """ converts SI units to model units and returns dictionary of values"""
    # dictionary of constants and parameters in SI units
    MD = {}

    MD['m_H']   =   SI['m_H'] * unit_mass
    MD['m_He']  =   SI['m_He'] * unit_mass
    MD['mol_H'] =   SI['mol_H'] * unit_mass
    MD['mol_He']=   SI['mol_He'] * unit_mass
    MD['d_H']   =   SI['d_H'] * unit_length
    MD['d_He']  =   SI['d_He'] * unit_length
    MD['c']     =   SI['c'] * unit_speed
    MD['G']     =   SI['G'] * unit_force * unit_length**2 / unit_mass**2
    MD['k']     =   SI['k'] * unit_energy
    MD['sigma'] =   SI['sigma'] * unit_power / unit_length**2
    MD['a']     =   SI['a'] * unit_energy / unit_length**3
    MD['R']     =   SI['R'] * unit_energy
    MD['mu']    =   SI['mu'] * unit_mass
    MD['mol_mu']=   SI['mol_mu'] * unit_mass
    MD['d']     =   SI['d'] * unit_length
    return MD
""" dictionary of constants and parameters
in model units: pc, Myr, solar mass"""
MD          =   unit_conversion()

#===============================================================================
""" Initializing Functions """
#-------------------------------------------------------------------------------

def Jeans_radius(M, units=MD):
    """ Jean's radius

    Parameters
    ----------
    M:      initial cloud mass in desired units
    units:  ** dictionary of units  - default = MD
    """
    return (2/5) * (units['G'] * units['mu'] * M) / (units['k'] * const['T'])

def v_sound(T, units=MD):
    """ sound speed in gas

    Parameters
    ----------
    T:      temperature of uniform cloud or shell (K)
    units:  ** dictionary of units  - default = MD
    """
    assert float(T) > 0.0, "temperature must be > 0"
    return np.sqrt( (const['gamma'] * units['R'] * T) / (units['mol_He']) )

#===============================================================================
""" Shell Functions """
#-------------------------------------------------------------------------------

def shell_volume(data,t,j, units=MD):

    # shell inner radius
    if int(j) == 0:
        r_i = 0
    else:
        r_i =   data['r'][t,j-1]

    # shell outer radius
    r_f     =   data['r'][t,j]

    return (4/3) * np.pi * (r_f**3 - r_i**3)

def shell_inner_area(data,t,j, units=MD):
    if j == 0:
        return 0
    else:
        # inner radius
        r_i     =   data['r'][t,j-1]
        return 4 * np.pi * r_i**2

def shell_particle_density(data,t,j, units=MD):
    return data['mass'][j] / units['mu'] / data['volume'][t,j]

def shell_mean_free_path(data,t,j, units=MD):
    return 1 / (np.sqrt(2) * np.pi * units['d']**2 * data['n'][t,j])

def shell_potential_energy(data,t,j, units=MD):
    # internal mass
    return

#===============================================================================
""" Acceleration Functions """
#-------------------------------------------------------------------------------

def acc_pressure_gas(data,t,j, units=MD):
    # gas pressure from inner shell
    p   =   data['mass'][j-1] * units['k'] * data['temp'][t,j-1] / units['mu'] / shell_volume(data,t,j-1,units=units)

    return data['area'][t,j] * p / data['mass'][j]

def acc_pressure_rad(data,t,j, units=MD):
    # radiation pressure from inner shell
    p   =   (1/3) * units['a'] * data['temp'][t,j-1]

    return data['area'][t,j] * p / data['mass'][j]

def acc_gravity(data,t,j,units=MD):
    # radius of shell
    r_j     =   data['r'][t,j]
    M_r     =   np.sum(data['mass'][:j])
    return - units['G'] * M_r * data['mass'][j] / r_j**2
    # sum of m_i/(r_j-r_i)^2
    # alpha   =   0
    # for i in range(j):
    #     # mass of interior shell
    #     m       =   data['mass'][i]
    #     # radius of interior shell
    #     r_i     =   data['r'][t,i]
    #     r       =   r_j - r_i
    #     alpha   +=  m/r**2
    # return - units['G'] * alpha

def acc_total(data,t,j, units=MD):
    # a_gas   =   acc_pressure_gas(data,t,j, units=MD)
    # a_rad   =   acc_pressure_rad(data,t,j, units=MD)
    a_grav  =   acc_gravity(data,t,j, units=MD)
    # return a_gas + a_rad + a_grav
    return a_grav

#===============================================================================
""" Integration Functions """
#-------------------------------------------------------------------------------

def velocity_verlet(data,t,j,acc_func,dt, units=MD):
    acc1                =   acc_func(data,t,j,units=units)
    v_half              =   data['vel'][t,j] + (dt/2)*acc1
    data['r'][t+1,j]    =   data['r'][t,j] + dt*v_half
    acc2                =   acc_func(data,t+1,j,units=units)
    data['vel'][t+1,j]  =   v_half + (dt/2)*acc2
    data['acc2'][t+1,j] =   acc2
    return data

def integrate(data,N_time,N_shells,dt):

    for t in range(N_time):
        for j in range(N_shells):

            data    =   velocity_verlet(data,t,j,acc_total,dt, units=units)
            # fill in other 2D data (besides 'r', 'vel', and 'acc')
