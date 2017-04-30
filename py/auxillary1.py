#===============================================================================
""" Import Modules"""
#-------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import units as units
import pdb

#===============================================================================
""" Imported Constants"""
#-------------------------------------------------------------------------------

C           =   units.C

#===============================================================================
""" Auxillary Functions """
#-------------------------------------------------------------------------------

def gas_pressure(m,T,V):
    return m * C['k'] * T / C['mu'] / V

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
    return (1/5) * (units['G'] * units['mu'] * M) / (units['k'] * const['T'])

def v_sound(T, units=MD):
    """ sound speed in gas

    Parameters
    ----------
    T:      temperature of uniform cloud or shell (K)
    units:  ** dictionary of units  - default = MD
    """
    assert float(T) > 0.0, "temperature must be > 0"
    return np.sqrt( (const['gamma'] * units['R'] * T) / (units['mol_He']) )
