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

def gas_pressure(m,T,V, units=MD):
    return m * units['k'] * T / units['mu'] / V

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

    # shell volume
    V_tj                =   (4/3) * np.pi * (r_f**3 - r_i**3)
    data['volume'][t,j] =   V_tj
    return data

def shell_inner_area(data,t,j, units=MD):
    if j == 0:
        A       =   0
    else:
        # inner radius
        r_i     =   data['r'][t,j-1]
        A       = 4 * np.pi * r_i**2

    data['area'][t,j]   =   A
    return data

def shell_particle_density(data,t,j, units=MD):
    n               =   data['mass'][j] / units['mu'] / data['volume'][t,j]
    data['n'][t,j]  =   n
    return data

def shell_mean_free_path(data,t,j, units=MD):
    mfp                 =   1 / (np.sqrt(2) * np.pi * units['d']**2 * data['n'][t,j])
    data['mfp'][t,j]    =   mfp
    return data

def shell_potential_energy(data,t,j, units=MD):
    # internal mass
    M_r     =   data['mass_r'][j]
    # shell mass
    m_j     =   data['mass'][j]
    # shell position
    pos     =   data['r'][t,j]

    U_g                 =   units['G'] * M_r * m_j / pos
    data['U_g'][t,j]    =   U_g
    return data

def shell_gas_pressure(data,t,j, units=MD):
    # shell mass
    m_j     =   data['mass'][j]
    # shell temperature
    T_tj    =   data['temp'][t,j]
    # shel volume
    V_tj    =   data['volume'][t,j]

    p_gas               =   gas_pressure(m_j,T_tj,V_tj, units=units)
    data['p_gas'][t,j]  =   p_gas
    return data

def shell_pvt_const(data,j, t=0,units=MD):
    """ since the shell has constant mass
    can use pV/T = const"""
    # shell temperature
    T_tj    =   data['temp'][t,j]
    # shel volume
    V_tj    =   data['volume'][t,j]
    # shell pressure
    P_tj    =   data['p_gas'][t,j]

    pvt_const               =   P_tj * V_tj / T_tj
    data['pvt_const'][j]    =   pvt_const
    return data

def shell_vt_const(data,j, t=0,units=MD):
    # shell temp
    T_tj    =   data['temp'][t,j]
    # shell volume
    V_tj    =   data['volume'][t,j]

    vt_const            =   T_tj * V_tj**(const['gamma']-1)
    data['vt_const'][j] =   vt_const
    return data

def shell_temperature(data,t,j, units=MD):
    const               =   data['vt_const'][j]
    V_tj                =   data['volume'][t,j]
    gamma               =   const['gamma']
    data['temp'][t,j]   =   const / V_tj**(gamma-1)
    return data

def shell_luminosity_flux(data,t,j, units=MD):
    T_tj            =   data['temp'][t,j]
    r_tj            =   data['r'][t,j]
    L_tj            =   4 * np.pi * r_tj**2 * units['sigma'] * T_tj**4
    F_tj            =   units['sigma'] * T_tj**4

    data['L'][t,j]  =   L_tj
    data['F'][t,j]  =   F_tj
    return data



#===============================================================================
""" Acceleration Functions """
#-------------------------------------------------------------------------------

def acc_pressure_gas(data,t,j, units=MD):
    """gas pressure on shell from interior shell"""
    # mass of inner shell
    m_i     =   data['mass'][j-1]
    # temperature of inner shell
    T_ti    =   data['temp'][t,j-1]
    # volume of inner shell
    V_ti    =   data['volume'][t,j-1]
    # gas pressure exerted on shell
    P       =   gas_pressure(m_i,T_ti,V_ti)

    # surface area of shell
    A_tj    =   data['area'][t,j]
    # mass of shell
    m_j     =   data['mass'][j]

    acc_gas                 =   A_tj * P / m_j
    data['acc_gas'][t,j]    =   acc_gas
    return data

def acc_pressure_rad(data,t,j, units=MD):
    # radiation pressure from inner shell
    p   =   (1/3) * units['a'] * data['temp'][t,j-1]

    acc_rad             =   data['area'][t,j] * p / data['mass'][j]
    data['acc_rad']     =   acc_rad
    return data

def acc_gravity(data,t,j,units=MD):
    # radius of shell
    r_tj    =   data['r'][t,j]
    M_r     =   data['mass_r'][j]

    acc_grav                =   - units['G'] * M_r / r_tj**2
    data['acc_grav'][t,j]   =   acc_grav
    return data

def acc_total(data,t,j, units=MD):
    a_gas   =   data['acc_gas'][t,j]
    # a_rad   =   Data['acc_rad'][t,j]
    a_grav  =   data['acc_grav'][t,j]

    acc                 =   a_grav + a_gas
    data['acc'][t,j]    =   acc
    return data

#===============================================================================
""" Integration Functions """
#-------------------------------------------------------------------------------

def velocity_verlet(data,t,j,acc_func,dt, units=MD):
    acc1                =   acc_func(data,t,j,units=units)
    v_half              =   data['vel'][t,j] + (dt/2)*acc1
    data['r'][t+1,j]    =   data['r'][t,j] + dt*v_half
    acc2                =   acc_func(data,t+1,j,units=units)
    data['vel'][t+1,j]  =   v_half + (dt/2)*acc2
    data['acc'][t+1,j]  =   acc2
    return data

def Euler(data,t,j,acc_func,dt, units=MD):
    acc                 =   acc_func(data,t,j, units=MD)
    vel                 =   data['vel'][t,j] + acc
    pos                 =   data['r'][t,j] + vel
    data['acc'][t+1,j]  =   acc
    data['vel'][t+1,j]  =   vel
    data['r'][t+1,j]    =   pos
    return data

def integrate(data,N_time,N_shells,dt, units=MD,integrator=Euler):

    def print_percent(i):
        p_i =   np.linspace(0,N_time,11).astype(int)
        if np.any(i == p_i): print("%s percent" % int( 100 * i / (N_time-1) ))

    for t in range(N_time-1):
        print_percent(t)

        for j in range(N_shells):

            data    =   integrator(data,t,j,acc_total,dt, units=units)
            # fill in other 2D data (besides 'r', 'vel', and 'acc')

    return data
