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

def time_FreeFall(density_0):
    return np.sqrt( (3 * np.pi) / (32 * C['G'] * density_0) )

#===============================================================================
""" Initializing Functions """
#-------------------------------------------------------------------------------

def Jeans_radius(M):
    return (1/5) * (C['G'] * C['mu'] * M) / (C['k'] * C['T'])

def v_sound(T):
    assert float(T) > 0.0, "temperature must be > 0"
    return np.sqrt( (C['gamma'] * C['R'] * T) / (C['mol_He']) )

#===============================================================================
""" Shell Functions """
#-------------------------------------------------------------------------------

def shell_volume(r_i,r_f):
    return (4/3) * np.pi * (r_f**3 - r_i**3)

def shell_vt_const(V0):
    return C['T'] * V0**(C['gamma']-1)

def shell_temperature(V,vt_const):
    gamma   =   C['']

# def shell_volume(data,t,j, units=MD):
#
#     # shell inner radius
#     if int(j) == 0:
#         r_i = 0
#     else:
#         r_i =   data['r'][t,j-1]
#
#     # shell outer radius
#     r_f     =   data['r'][t,j]
#
#     # shell volume
#     V_tj                =   (4/3) * np.pi * (r_f**3 - r_i**3)
#     data['volume'][t,j] =   V_tj
#     return data
#
# def shell_inner_area(data,t,j, units=MD):
#     if j == 0:
#         A       =   0
#     else:
#         # inner radius
#         r_i     =   data['r'][t,j-1]
#         A       = 4 * np.pi * r_i**2
#
#     data['area'][t,j]   =   A
#     return data
#
# def shell_particle_density(data,t,j, units=MD):
#     n               =   data['mass'][j] / units['mu'] / data['volume'][t,j]
#     data['n'][t,j]  =   n
#     return data
#
# def shell_mean_free_path(data,t,j, units=MD):
#     mfp                 =   1 / (np.sqrt(2) * np.pi * units['d']**2 * data['n'][t,j])
#     data['mfp'][t,j]    =   mfp
#     return data
#
# def shell_potential_energy(data,t,j, units=MD):
#     # internal mass
#     M_r     =   data['mass_r'][j]
#     # shell mass
#     m_j     =   data['mass'][j]
#     # shell position
#     pos     =   data['r'][t,j]
#
#     U_g                 =   units['G'] * M_r * m_j / pos
#     data['U_g'][t,j]    =   U_g
#     return data
#
# def shell_gas_pressure(data,t,j, units=MD):
#     # shell mass
#     m_j     =   data['mass'][j]
#     # shell temperature
#     T_tj    =   data['temp'][t,j]
#     # shel volume
#     V_tj    =   data['volume'][t,j]
#
#     p_gas               =   gas_pressure(m_j,T_tj,V_tj, units=units)
#     data['p_gas'][t,j]  =   p_gas
#     return data
#
# def shell_pvt_const(data,j, t=0,units=MD):
#     """ since the shell has constant mass
#     can use pV/T = const"""
#     # shell temperature
#     T_tj    =   data['temp'][t,j]
#     # shel volume
#     V_tj    =   data['volume'][t,j]
#     # shell pressure
#     P_tj    =   data['p_gas'][t,j]
#
#     pvt_const               =   P_tj * V_tj / T_tj
#     data['pvt_const'][j]    =   pvt_const
#     return data
#
# def shell_vt_const(data,j, t=0,units=MD):
#     # shell temp
#     T_tj    =   data['temp'][t,j]
#     # shell volume
#     V_tj    =   data['volume'][t,j]
#
#     vt_const            =   T_tj * V_tj**(const['gamma']-1)
#     data['vt_const'][j] =   vt_const
#     return data
#
# def shell_temperature(data,t,j, units=MD):
#     const               =   data['vt_const'][j]
#     V_tj                =   data['volume'][t,j]
#     gamma               =   const['gamma']
#     data['temp'][t,j]   =   const / V_tj**(gamma-1)
#     return data
#
# def shell_luminosity_flux(data,t,j, units=MD):
#     T_tj            =   data['temp'][t,j]
#     r_tj            =   data['r'][t,j]
#     L_tj            =   4 * np.pi * r_tj**2 * units['sigma'] * T_tj**4
#     F_tj            =   units['sigma'] * T_tj**4
#
#     data['L'][t,j]  =   L_tj
#     data['F'][t,j]  =   F_tj
#     return data

#===============================================================================
""" Acceleration Functions """
#-------------------------------------------------------------------------------

# def acc_pressure_gas(data,t,j, units=MD):
#     """gas pressure on shell from interior shell"""
#     if j == 0:
#         acc_gas     =   0
#     else:
#         # mass of inner shell
#         m_i     =   data['mass'][j-1]
#         # temperature of inner shell
#         T_ti    =   data['temp'][t,j-1]
#         # volume of inner shell
#         V_ti    =   data['volume'][t,j-1]
#         # gas pressure exerted on shell
#         P       =   gas_pressure(m_i,T_ti,V_ti)
#
#         # surface area of shell
#         A_tj    =   data['area'][t,j]
#         # mass of shell
#         m_j     =   data['mass'][j]
#
#         acc_gas                 =   A_tj * P / m_j
#
#     data['acc_gas'][t,j]    =   acc_gas
#     return data
#
# def acc_pressure_rad(data,t,j, units=MD):
#     # radiation pressure from inner shell
#     p   =   (1/3) * units['a'] * data['temp'][t,j-1]
#
#     acc_rad             =   data['area'][t,j] * p / data['mass'][j]
#     data['acc_rad']     =   acc_rad
#     return data

def acc_gravity(M_r,r):
    return - C['G'] * M_r / r**2

def acc_total(M_r,r):
    # a_gas   =   NotImplemented
    # a_rad   =   NotImplemented
    a_grav  =   acc_gravity(M_r,r)
    return a_grav

#===============================================================================
""" Integration Functions """
#-------------------------------------------------------------------------------

def rk4(M_r,r,dt):
    k1  =   dt * acc_total(M_r,r)
    k2  =   dt * acc_total(M_r,r + k1/2)
    k3  =   dt * acc_total(M_r,r + k2/2)
    k4  =   dt * acc_total(M_r,r + k3)
    return r + (1/3)*(k1/2 + k2 + k3 + k4/2)

def integrate(M_cloud,r_star,N_time,N_shell,tol):

    """ Jean's radius of cloud and
    radius increment: pc """
    r_max       =   Jeans_radius(M_cloud)
    dr          =   (r_max-r_star)/N_shell

    """ initial cloud volume and density """
    volume_0    =   (4/3) * np.pi * r_max**3
    volume_star =   (4/3) * np.pi * r_star**3
    density_0   =   M_cloud/volume_0

    """ calculate free fall time: Myr """
    t_ff        =   time_FreeFall(density_0)
    t_max       =   t_ff * .7
    dt          =   t_max/N_time

    # # Only implement if using gas pressure
    # """ if dr_gas > dr, gas pressure can
    # communicate in single time step. """
    # v_gas       =   aux.v_sound(C['T'])
    # dr_gas      =   v_gas*dt
    # assert dr < dr_gas, "shell widths must be less than (v_gas) x (dt)"

    """ if dr_light > r_max, signals that
    travel at light speed can interact
    across cloud in single time step."""
    dr_light    =   C['c']*dt
    assert r_max < dr_light, "cloud radius must be less than (c) x (dt)"

    """ make empty arrays """
    # shell outer radii
    R           =   np.zeros((N_time,N_shell))
    # shell volume
    V           =   np.zeros_like(R)
    # shell temp
    T           =   np.zeros_like(R)
    # core temp
    T_core      =   np.zeros(N_time)

    """ initialize arrays """
    # initial shell outer radii
    R0          =   np.linspace(r_star+dr,r_max,N_shell)
    # initial shell volume
    V0_0        =   shell_volume(r_star,R0[0])
    V0_1_N      =   np.array([ shell_volume(R0[j-1],R0[j]) for j in np.arange(1,N_shell) ])
    V0          =   np.hstack(( V0_0 , V0_1_N ))
    # shell mass
    M           =   density_0 * V0
    # shell internal mass
    Mr          =   np.array([ density_0*volume_star + np.sum(M[:j]) for j in np.arange(1,N_shell+1) ])
    # shell initial temperature
    T0          =   np.ones(N_shell) * C['T']

    # initialize 2D arrays
    R[0,:]      =   R0
    V[0,:]      =   V0
    T[0,:]      =   T0
    T_core[0]   =   C['T']

    # integration
    for i_time in np.arange(1,N_time):

        if R[i_time-1,-1] <= r_star:
            i_terminate         =   i_time # i_time when final cloud shell has collapsed.
            R                   =   np.delete(R, np.s_[i_terminate:],0)
            V                   =   np.delete(V, np.s_[i_terminate:],0)
            T                   =   np.delete(T, np.s_[i_terminate:],0)
            T_core              =   np.delete(T_core, np.s_[i_terminate:])

            R[ R < r_star ]     =   r_star
            V[ V < volume_star] =   volume_star
            print("star collapsed at %s Myr" % (i_time*dt) )
            break

        for i_shell in range(N_shell):

            # old shell outer radius
            r                       =   R[i_time-1,i_shell]

            # update shell outer radius
            rf                      =   rk4(Mr[i_shell],r,dt)       # updated shell radius
            # updated shell inner radius
            if i_shell == 0:
                ri                  =   r_star
            else:
                ri                  =   R[i_time,i_shell-1]

            # update arrays
            R[i_time,i_shell]       =   rf
            V[i_time,i_shell]       =   shell_volume(ri,rf)

            # if T_core[i_time] >= C['T_c']:
            #     i_terminate     =   i_time +1           # i_time of last shell
            #     R               =   np.delete(R, np.s_[i_terminate:],1)
            #     V               =   np.delete(V, np.s_[i_terminate:],1)
            #     T               =   np.delete(T, np.s_[i_terminate:],1)
            #     T_core          =   np.delete(T_core, np.s_[i_terminate:])

    d                   =   {}
    # scalars               # value
    d['M_cloud']        =   M_cloud
    d['r_star']         =   r_star
    d['r_max']          =   r_max
    d['dr']             =   dr
    d['volume_0']       =   volume_0
    d['volume_star']    =   volume_star
    d['density_0']      =   density_0
    d['t_ff']           =   t_ff
    d['t_max']          =   t_max
    d['dt']             =   dt
    d['t_collapse']     =   dt * i_time

    # arrays                array       # ( dimention )
    d['R']              =   R           # ( N_time , N_shell )
    d['V']              =   V           # ( N_time , N_shell )
    d['M']              =   M           # ( N_shell )
    d['Mr']             =   Mr          # ( N_shell )
    d['T']              =   T           # ( N_time , N_shell )
    d['T_core']         =   T_core      # ( N_time )

    return pd.Series(d)
