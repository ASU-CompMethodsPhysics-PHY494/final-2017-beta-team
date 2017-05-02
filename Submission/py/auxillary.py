#===============================================================================
""" Import Modules"""
#-------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import units as units
import pdb

#===============================================================================
""" Imported Constants"""
#-------------------------------------------------------------------------------

C           =   units.C
SI          =   units.SI

#===============================================================================
""" Auxillary Functions """
#-------------------------------------------------------------------------------

def Jeans_radius(M):
    return (1/5) * (C['G'] * C['mu'] * M) / (C['k'] * C['T'])

def v_sound(T):
    assert float(T) > 0.0, "temperature must be > 0"
    return np.sqrt( (C['gamma'] * C['R'] * T) / (C['mol_He']) )

def time_FreeFall(density_0):
    return np.sqrt( (3 * np.pi) / (32 * C['G'] * density_0) )

def shell_volume(r_i,r_f):
    return (4/3) * np.pi * (r_f**3 - r_i**3)

def shell_vt_const(V0):
    return C['T'] * V0**(C['gamma']-1)

def shell_temperature(V,vt_const):
    return vt_const / V**(C['gamma']-1)

def shell_flux_density(T):
    return C['sigma'] * T**4

def cloud_potential_energy(M_cloud,R_cloud):
    return -(3/5) * C['G'] * M_cloud**2 / R_cloud

def core_temperature(E):
    E   =   E / units.unit_energy
    return (2/3) * E * SI['mu'] / SI['k']
    # return (2 * E * C['mu']) / (3 * C['k'] * M_cloud)

def terminator(A,i_terminate,axis=None):
    if axis == None:
        A   =   np.delete(A, np.s_[i_terminate:])
        return A
    else:
        assert type(axis) == int, "axis must be 'None' or 'int'"
        A   =   np.delete(A, np.s_[i_terminate:],axis)
        return A

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

def integrate(M_cloud,r_star, N_time=1000,N_shell=1000,tol=1e-5,saveA=True):

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
    TIME        =   np.arange(0,t_max+dt,dt)

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
    R           =   np.zeros((N_time,N_shell))  # shell outer radii
    V           =   np.zeros_like(R)            # shell volume
    T           =   np.zeros_like(R)            # shell temp
    F           =   np.zeros_like(R)            # shell flux density
    Ug_cloud    =   np.zeros(N_time)            # cloud potential energy
    E_core      =   np.zeros_like(Ug_cloud)     # cloud internal energy
    T_core      =   np.zeros_like(Ug_cloud)     # core temp

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
    # initial shell temperature
    T0          =   np.ones(N_shell) * C['T']
    # vt_const
    VT          =   shell_vt_const(V0)
    # initial shell flux density
    F0          =   shell_flux_density(T0)
    # initial cloud potential energy
    Ug_cloud0   =   cloud_potential_energy(M_cloud,r_max)
    # initial core energy
    E_core0     =   (3/2) * (M_cloud/C['mu']) * C['k'] * C['T']
    # initial core temperature
    T_core0     =   C['T']

        # initialize 2D arrays
    R[0,:]      =   R0
    V[0,:]      =   V0
    T[0,:]      =   T0
    F[0,:]      =   F0
    Ug_cloud[0] =   Ug_cloud0
    E_core[0]   =   E_core0
    T_core[0]   =   T_core0

    # integration
    for i_time in np.arange(1,N_time):

        # collapsing termination condition
        i_terminate             =   'did not terminate'
        if R[i_time-1,-1] <= r_star:
            i_terminate         =   i_time # i_time when final cloud shell has collapsed.
            R                   =   terminator(R,i_terminate,axis=0)
            V                   =   terminator(V,i_terminate,axis=0)
            T                   =   terminator(T,i_terminate,axis=0)
            F                   =   terminator(F,i_terminate,axis=0)
            Ug_cloud            =   terminator(Ug_cloud,i_terminate)
            E_core              =   terminator(E_core,i_terminate)
            T_core              =   terminator(T_core,i_terminate)
            TIME                =   terminator(TIME,i_terminate)

            R[ R < r_star ]     =   r_star
            V[ V < volume_star] =   volume_star
            print("\nstar collapsed at %s Myr" % (i_time*dt) )
            break

        # core temperature termination condition
        i_on                    =   'did not turn on'
        if all(( T_core[i_time-1] >= C['T_c'] , T_core[i_time-2] < C['T_c'] )):
            i_on                =   i_time
            # i_terminate         =   i_time # i_time when final cloud shell has collapsed.
            # R                   =   terminator(R,i_terminate,axis=0)
            # V                   =   terminator(V,i_terminate,axis=0)
            # T                   =   terminator(T,i_terminate,axis=0)
            # F                   =   terminator(F,i_terminate,axis=0)
            # Ug_cloud            =   terminator(Ug_cloud,i_terminate)
            # E_core              =   terminator(E_core,i_terminate)
            # T_core              =   terminator(T_core,i_terminate)
            # TIME                =   terminator(TIME,i_terminate)
            print("\nstar turned on at %s Myr" % (i_time*dt) )

        for i_shell in range(N_shell):

            # update shell values
            r_prev                  =   R[i_time-1,i_shell]         # old shell outer radius
            r_new                   =   rk4(Mr[i_shell],r_prev,dt)  # update shell outer radius
            if i_shell == 0:
                r_i                 =   r_star                      # updated shell inner radius
            else:
                r_i                 =   R[i_time,i_shell-1]         # updated shell inner radius
            v_new                   =   shell_volume(r_i,r_new)     # updated shell volume
            vt                      =   VT[i_shell]                 # vt_const
            t_new                   =   shell_temperature(v_new,vt) # updated shell temperature
            f_new                   =   shell_flux_density(t_new)   # updated shell flux density

            # update shell arrays
            R[i_time,i_shell]       =   r_new
            V[i_time,i_shell]       =   v_new
            T[i_time,i_shell]       =   t_new
            F[i_time,i_shell]       =   f_new

        # update time step values
        r_max_new               =   R[i_time,-1]                                # max cloud radius now
        ug_new                  =   cloud_potential_energy(M_cloud,r_max_new)   # current gravitational potential energy
        # ug_prev                 =   Ug_cloud[i_time-1]                          # previous gravitational potential energy
        # deltaUg_new             =   ug_new - ug_prev                            # new gravitational potential energy of cloud
        # e_core_prev             =   E_core[i_time-1]                            # previous core energy
        # e_core_add              =   abs(deltaUg_new)                            # energy added to core energy
        # e_core_new              =   e_core_prev + e_core_add                    # new core energy
        e_core_new              =   abs(Ug_cloud0 - ug_new)/2                   # new core energy
        t_core_new              =   core_temperature(e_core_new)                # new core tempperature

        # update time step arrays
        Ug_cloud[i_time]        =   ug_new
        E_core[i_time]          =   e_core_new
        T_core[i_time]          =   t_core_new

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
    d['t_collapse']     =   dt * i_terminate
    d['i_collapse']     =   i_terminate
    d['N_shell']        =   N_shell
    d['N_time']         =   i_terminate
    d['t_on']           =   i_on * dt
    d['i_on']           =   i_on

    # arrays                array       # ( dimention )
    d['R']              =   R           # ( N_time , N_shell )
    d['V']              =   V           # ( N_time , N_shell )
    d['M']              =   M           # ( N_shell )
    d['Mr']             =   Mr          # ( N_shell )
    d['T']              =   T           # ( N_time , N_shell )
    d['F']              =   F           # ( N_time , N_shell )
    d['Ug_cloud']       =   Ug_cloud    # ( N_time )
    d['E_core']         =   E_core      # ( N_time )
    d['T_core']         =   T_core      # ( N_time )
    d['TIME']           =   TIME        # ( N_time )

    # save and return
    data                =   pd.Series(d)
    if saveA: data.to_pickle('../data/cloud_%s' % M_cloud)
    return data
