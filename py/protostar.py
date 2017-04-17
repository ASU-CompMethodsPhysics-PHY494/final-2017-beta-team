#===============================================================================
"""Import Modules"""
#-------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

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

# dictionary of constants and parameters in SI units
# model units: pc, Myr, solar mass
MD          =   unit_conversion()

#===============================================================================
"""Initializing Functions"""
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
"""Auxillary Functions"""
#-------------------------------------------------------------------------------

# not needed?
# def find_nearest_index(array,value):
#     """ find the nearest index of an array
#     that a value has to a value in the array
#
#     Parameters
#     ----------
#     array:  array to search
#     value:  scalar value to match
#     """
#
#     return (np.abs(array-value)).argmin()
#
# def interaction_condition(data,t,i,j,units=MD):
#     """ find the time it would take for
#     innermost shell to send a signal at
#     the speed of light and reach object
#     shell.
#
#     Parameters
#     ----------
#     data:   dictionary of data with dimention (N_time,N_shell)
#     t:      index of time
#     j:      index of object shell
#     units:  ** dictionary of units - default = MD
#     """
#
#     """ 1) find object shell position and agent shell
#     position. 2) find distance between them. 3) calculate
#     interaction time assuming that agent shell hasn't
#     moved after emitting signal (simplification but it might
#     work because signal speed >> collapse speed). 4) create
#     time array from 0 to t. 5) find nearest index for
#     interaction time. 6) return shell positions and time
#     interaction index."""
#     r_j         =   data['r'][t,j]
#     r_i         =   data['r'][t,i]
#     r           =   r_j - r_i
#     assert r > 0, "outer shell position must be greater than inner shell position."
#     time_int    =   r/units['c']
#     TIME1       =   data['r'][:t,0]
#     t_int       =   find_nearest_index(TIME1,time_int)
#     return r_i,r_j,t_int

def area_shell(data,t,j,units=MD):
    """ inner surface area of shell. all pressure contributions
    on object shell come from interior shells because pressure
    contributions from shells outside object shell cancel.

    Parameters
    ----------
    data:   dictionary of data with dimention (N_time,N_shell)
    t:      index of time
    j:      index of object shell
    units:  ** dictionary of units - default = MD
    """
    assert j <= len(data['pos'][0,:]), "'j' is too large"
    assert j >= 0, "j is too small"

    # object shell position and interaction time
    r   =   data['r'][t,j]
    return 4 * np.pi * r**2

def volume_shell(data,t,j,units=MD):
    """ volume of shell. Nothing else to explain.

    Parameters
    ----------
    data:   dictionary of data with dimention (N_time,N_shell)
    t:      index of time
    j:      index of object shell
    units:  ** dictionary of units - default = MD
    """
    assert j <= len(data['pos'][0,:]), "'j' is too large"
    assert j >= 0, "j is too small"

    # shell inner radius and width
    r   =   data['r'][t,j]
    dr  =   data['dr'][t,j]
    return (4/3) * np.pi * ( (r+dr)**3 - r**3 )

def pressure_gas(data,t,j,units=MD):
    """ gas pressure of uniform cloud/shell. Gas pressure on
    object shell comes from the shell directly inside it.
    The shell width is set up to be <= sound speed.
    pressure contributions outside object shell cancel out.

    Parameters
    ----------
    data:   dictionary of data with dimention (N_time,N_shell)
    t:      index of time
    j:      index of object shell
    units:  ** dictionary of units - default = MD
    """
    assert j <= len(data['pos'][0,:]), "'j' is too large"
    assert j >= 0, "j is too small"

    # agent shell: volume, mass, and temperature
    V_i     =   volume_shell(data,t,j-1,units=units)
    m_i     =   data['mass'][t,j-1]
    T_i     =   data['temp'][t,j-1]
    assert float(V) > 0, "Volume must be greater than 0"
    return ( m_i * units['k'] * T_i ) / ( units['mu'] * V_i )

def pressure_rad(data,t,j,units=MD):
    """ radiation pressure of uniform cloud/shell.
    The agent shell is directly inside the object
    shell. radiation pressure travels at speed of
    light. the distnce to travel the whole cloud
    << delta t.

    Parameters
    ----------
    data:   dictionary of data with dimention (N_time,N_shell)
    t:      index of time
    j:      index of object shell
    units:  ** dictionary of units - default = MD
    """
    assert j <= len(data['pos'][0,:]), "'j' is too large"
    assert j >= 0, "j is too small"

    # temperature of agent shell
    T_i     =   data['temp'][t,j-1]
    return (1/3) * units['a'] * T_i**4

def acceleration_grav(data,t,j,units=MD):
    """ gravity pressure of uniform cloud/shell.
    gravit travels at speed of light. Maximum
    interaction time << delta t. Mass contributions
    come from all shells inside object shell.

    Parameters
    ----------
    data:   dictionary of data with dimention (N_time,N_shell)
    t:      index of time
    j:      index of object shell
    units:  ** dictionary of units - default = MD
    """
    assert j <= len(data['pos'][0,:]), "'j' is too large"
    assert j >= 0, "j is too small"

    """ iterate over 'i' and add up contributions of
    m_i/r^2 of all shells inside j."""
    total   =   0
    r_j     =   data['r'][t,j]
    for i in range(j-1):
        m_i     =   data['mass'][t,i]
        r_i     =   data['r'][t,i]
        assert r_j > r_i, "object radius must be larger than agent radius"
        total += m_i / (r_j-r_i)**2
    return - units['G'] * total

def acceleration(data,t,j,units=MD):
    """ find total acceleration of shell

    Parameters
    ----------
    data:   dictionary of data with dimention (N_time,N_shell)
    t:      index of time
    j:      index of object shell
    units:  ** dictionary of units - default = MD
    """
    assert j <= len(data['pos'][0,:]), "'j' is too large"
    assert j >= 0, "j is too small"

    # inner surface area and mass of shell
    A_j     =    area_shell(data,t,j,units=units)
    m_j     =    data['mass'][t,j]

    # acceleration from gravity, radiation, and gas
    a_grav  =   acceleration_grav(data,t,j,units=units)
    a_rad   =   (A_j/m_j) * pressure_rad(data,t,j,units=units)
    a_gas   =   (A_J/m_J) * pressure_gas(data,t,j,units=units)

    return a_grav + a_rad + a_gas

def potential_grav(data,t,j,units=MD):
    """ gravitational potential energy of shell

    Parameters
    ----------
    data:   dictionary of data with dimention (N_time,N_shell)
    t:      index of time
    j:      index of object shell
    units:  ** dictionary of units - default = MD
    """
    assert j <= len(data['pos'][0,:]), "'j' is too large"
    assert j >= 0, "j is too small"

    r_j     =   data['r'][t,j]
    m_j     =   data['mass'][t,j]
    M       =   0
    for i in range(j-1):
        M += data['mass'][t,i]

    assert float(r_j) > 0.0, "shell radius must be > 0"
    return - ( units['G'] * M * m_j ) / r_j

def particle_density(data,t,j,units=MD):
    """particle density in uniform cloud/shell

    Parameters
    ----------
    data:   dictionary of data with dimention (N_time,N_shell)
    t:      index of time
    j:      index of object shell
    units:  ** dictionary of units - default = MD
    """
    assert j <= len(data['pos'][0,:]), "'j' is too large"
    assert j >= 0, "j is too small"

    # mass and volume of shell
    m_j     =   data['mass'][t,j]
    V_j     =   volume_shell(data,t,j, units=units)
    assert float(V) > 0.0, "shell volume must be > 0"
    return m_j / ( units['mu'] * V_j )

def mean_free_path(data,t,j,units=MD):
    """ mean free path in uniform cloud/shell

    Parameters
    ----------
    data:   dictionary of data with dimention (N_time,N_shell)
    t:      index of time
    j:      index of object shell
    units:  ** dictionary of units - default = MD
    """
    assert j <= len(data['pos'][0,:]), "'j' is too large"
    assert j >= 0, "j is too small"

    # particle density in object
    n_j =   particle_density(data,t,j,units=units)
    assert float(n) > 0.0, "particle density must be > 0"
    return 1/ ( np.sqrt(2) * np.pi * units['d']**2 * n_j)

def luminosity(data,t,units=MD):
    """ find luminosity of each shell.
    caculate how much of the luminosity
    shows from that shell and how much
    is transfered to higher shells using
    mean_free_path

    Parameters
    ----------
    data:   dictionary of data with dimention (N_time,N_shell)
    t:      index of time
    units:  ** dictionary of units - default = MD
    """
    

#===============================================================================
"""Main Problem"""
#-------------------------------------------------------------------------------

M_clouds    =   np.array([.7,   .8,     1,      1.5,    2,      3,      5,      9,      15,     25,     60])
T_bench     =   np.array([100,  68.4,   38.9,   35.4,   23.4,   7.24,   1.15,   .288,   .117,   .0708,  .0282])
T_model     =   T_bench*100

def cloud_collapse(i, N_time=1000,tol=1e-5):
    """ function that models collapsing cloud

    position arguments
    ------------------
    i:          index of M_clouds and T_model

    keyword arguments
    -----------------
    N_shells:   number of shells        : default = 1000
    N_time:     length of time array    : default = 1000
    tol:        equilibrium tolerance   : default = 1e-5
    """

    """construct time array: Myr"""
    t_max       =   T_model[i]
    dt          =   t_max/N_time
    TIME        =   np.arange(0,t_max+dv,dv)

    """Jean's radius of cloud: pc"""
    r_max       =   Jeans_radius(M_cloud[i])

    """make a radius array for cloud where initial
    dr = (initial sound speed) x (time increment)
    and r_max = Jean's radius."""
    v_sound     =   v_sound(const['T'])
    dr          =   v_sound*dt
    R           =   np.arange(0,r_max+dr,dr)

    """number of shells is length of radius array and
    it will stay constant for the whole model. each time
    step will determine a new r_max and dr using
    constant N_shells. N_time is length of TIME"""
    N_shells    =   len(R)
    N_time      =   len(T)

    """construct dictionary of arrays"""
    data            =   {}
    data['r']       =   np.zeros((N_time,N_shells))
    data['dr']      =   np.zeros_like(data['r'])
    data['mass']    =   np.zeros_like(data['r'])
    data['U_grav']  =   np.zeros_like(data['r'])
    data['temp']    =   np.zeros_like(data['r'])
    data['acc']     =   np.zeros_like(data['r'])
    data['lum']     =   np.zeros_like(data['r'])
    # not needed?
    # data['a_grav']  =   np.zeros_like(data['r'])
    # data['p_rad']   =   np.zeros_like(data['r'])
    # data['p_gas']   =   np.zeros_like(data['r'])
    # data['chi']     =   np.zeros_like(data['r'])
    # data['lambda']  =   np.zeros_like(data['r'])

    """initialize data"""
    data['r'][0,:]      =   R
    data['dr'][0,:]     =   np.ones_like(R) * dr
    data['mass'][0,:]   =   np.ones_like(R) * M_clouds[i]/N_shells
    data['U_grav'][0,:] =   np.array([ potential_grav(data,0,j) for j in range(N_shells) ])
    data['temp'][0,:]   =   np.ones_like(R) * const['T']
    data['acc'][0,:]    =   np.array([ acceleration(data,0,j) for j in range(N_shells) ])
    data['lum'][0,:]    =   NotImplemented
    # not needed
    # data['a_grav'][0,:] =   NotImplemented
    # data['p_rad'][0,:]  =   NotImplemented
    # data['p_gas'][0,:]  =   NotImplemented
    # data['chi'][0,:]    =   NotImplemented
    # data['lambda'][0,:] =   NotImplemented

    """fill in arrays"""
    # for i_time,time in enumerate(TIME[1:]):

    """save cloud data frame"""
    if saveA:
        pd.to_pickle('../data/cloud_%s' % str(M_clouds[i]) )
    else:
        return cloud
