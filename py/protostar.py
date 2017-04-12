#===============================================================================
# Import Modules
#-------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

#===============================================================================
# Known Constants
#-------------------------------------------------------------------------------

SI          =   {}
SI['m_H']   =   1.6737236e-27       # mass of hydrogen              (kg)
SI['m_He']  =   6.6464764e-27       # mass of helium                (kg)
SI['d_H']   =   53e-12              # size of hydrogen atom         (m)
SI['d_He']  =   31e-12              # size of helium atom           (m)
SI['c']     =   2.99792458e8        # speed of light                (m/s)
SI['G']     =   6.67408e-11         # gravitational constant        (N m^2/kg^2)
SI['k']     =   1.3806503e-23       # Bolztmann's constant          (J/K)
SI['sigma'] =   5.670373e-8         # Stefan Bolztmann's constant   W/(m^2 K^4)
SI['a']     =   7.5657e-16          # radiation constant            J/(m^3 K^4)

#===============================================================================
# Model Parameters
#-------------------------------------------------------------------------------

SI['gamma'] =   5/3                                 # heat capacity ratio for monatomic ideal gas
SI['mu']    =   (3/4)*SI['m_H'] + (1/4)*SI['m_He']  # average particle mass     (kg)
SI['d']     =   (3/4)*SI['d_H'] + (1/4)*SI['d_He']  # average particle size     (m)
SI['T']     =   10                                  # initial cloud temperature (K)

#===============================================================================
# Conversion factors
#-------------------------------------------------------------------------------

unit_length =   1/3.086e16                  # m -> pc
unit_time   =   1/(60*60*24*365.25*1e6)     # s -> MYr
unit_mass   =   1/1.98855e30                # kg -> solar mass

def unit_conversion():
    """ converts SI units to model units and returns dictionary of values"""
    MD = {}
    MD['m_H']   =   SI['m_H'] * unit_mass
    MD['m_He']  =   SI['m_He'] * unit_mass
    MD['d_H']   =   SI['d_H'] * unit_length
    MD['d_He']  =   SI['d_He'] * unit_length
    MD['c']     =   SI['c'] * unit_length / unit_time
    MD['G']     =   SI['G'] * unit_length**3 / unit_mass / unit_time**2
    MD['k']     =   SI['k'] * unit_mass * unit_length**2 / unit_time**2
    MD['sigma'] =   SI['sigma'] * unit_mass / unit_time**3
    MD['a']     =   SI['a'] * unit_mass / unit_time**2 / unit_mass
    MD['gamma'] =   SI['gamma']
    MD['mu']    =   SI['mu'] * unit_mass
    MD['d']     =   SI['d'] * unit_length
    MD['T']     =   SI['T']
    return MD
MD          =   unit_conversion()

#===============================================================================
# Auxillary Functions
#-------------------------------------------------------------------------------

def Jeans_radius(M, units=MD):
    """ Jean's radius

    Parameters
    ----------
    M:      initial cloud mass in desired units
    units:  ** dictionary of units  - default = MD
    """
    return ( units['G'] * units['mu'] * M ) / ( 5 * units['k'] * units['T'] )

def v_gas(T, uits=MD):
    """ sound speed in gas

    Parameters
    ----------
    T:      temperature of uniform cloud or shell (K)
    units:  ** dictionary of units  - default = MD
    """
    assert float(T) > 0.0, "temperature must be > 0"
    return np.sqrt( (units['gamma'] * units['k'] * T) / units['mu'] )

def area_shell(r,dr, units=MD):
    """ outer surface area of shell

    Parameters
    ----------
    r:      shell inner radius
    dr:     shell width
    units:  ** dictionary of units - default = MD"""
    return 4 * np.pi * (r+dr)**2

def volume_shell(r,dr, units=MD):
    """ volume of shell

    Parameters
    ----------
    r:      shell inner radius
    dr:     shell width
    units:  ** dictionary of units - default = MD
    """
    return (4/3) * np.pi * ( (r+dr)**3 - r**3 )

def internal_mass(r):
    """ mass of cloud inside radius 'r'"""
    return NotImplemented

def pressure_gas(m,r,,dr,T, units=MD):
    """ gas pressure of uniform cloud/shell

    Parameters
    ----------
    m:      shell mass
    r:      shell inner radius
    dr:     shell width
    T:      shell temperature (K)
    units:  ** dictionary of units - default = MD
    """
    V   =   volume_shell(r,dr, units=units)
    assert float(V) > 0, "Volume must be greater than 0"
    return ( m * units['k'] * T ) / ( units['mu'] * V )

def pressure_rad(T, units=MD):
    """ radiation pressure of uniform cloud/shell

    Parameters
    ----------
    T:      shell temperature (K)
    units:  ** dictionary of units - default = MD
    """
    return (1/3) * units['a'] * T**4

def pressure_grav(m,r,dr, units=MD):
    """ gravity pressure of uniform cloud/shell

    Parameters
    ----------
    m:      shell mass
    r:      shell radius
    dr:     shell width
    units:  ** dictionary of units - default = MD
    """
    A   =   area_shell(r,dr, units=units)
    M   =   internal_mass(r)
    assert float(r+dr) > 0.0, "shell radius must be > 0"
    assert float(A) > 0.0, "outer surface area of shell must be > 0"
    return - ( units['G'] * M * m ) / ( (r+dr)**2 * A)

def potential_grav(m,r, units=MD):
    """ gravitational potential energy of shell

    Parameters
    ----------
    m:      shell mass
    r:      shell inner radius
    units:  ** dictionary of units - default = MD
    """
    assert float(r) > 0.0, "shell radius must be > 0"
    M   =   internal_mass(r)
    return - ( units['G'] * M * m ) / r

def particle_density(m,r,dr, units=MD):
    """particle density in uniform cloud/shell

    Parameters
    ----------
    m:      shel mass
    r:      shell inner radius
    dr:     shell width
    units:  ** dictionary of units - default = MD
    """
    V   =   volume_shell(r,dr, units=units)
    assert float(V) > 0.0, "shell volume must be > 0"
    return m / ( units['mu'] * V )

def mean_free_path(m,r,dr, units=MD):
    """ mean free path in uniform cloud/shell

    Parameters
    ----------
    m:      shell mass
    r:      shell innder radius
    dr:     shell width
    units:  ** dictionary of values - default = MD
    """
    n   =   particle_density(m,r,dr, units=units)
    assert float(n) > 0.0, "particle density must be > 0"
    return 1/ ( np.sqrt(2) * np.pi * units['d']**2 * n)

#===============================================================================
# Main Problem
#-------------------------------------------------------------------------------
