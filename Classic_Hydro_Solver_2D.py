#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 20:16:44 2023

@author: santiago
"""

"""
                                                                              
 This procedure constitutes a 2D finite-volume-like Euler hydrodynamics solver based on operator splitting. In this approach,
 we treat the terms including the pressure as source terms on the right hand side of the Euler conservation equations. Then, 
 we solve the advection for the conserved quantities rho, rho*ux , rho*uy and rho*e_tot and, at the end of each time step, we
 add the source terms to the momentum and energy conservation equations. 
 In order to define the time step, we take into account that information should not be propagated over more than one grid
 cell during one time step, or we lose important information. Hence, we limit the time step by means of the Courant Criterion:
 choose the fastest possible information velocity in the computational domain and set the time step so small that the
 information cannot cross one grid cell during this step, multiplied with a security factor < 1, which is called the
 Courant number.
 For the 2D implementation, a directional Strang splitting approach is used here, more specifically: we do first 1/2 a
 time step sweep in x-direction, then 1 time step sweep in y-direction and finally another 1/2 time step sweep in x-direction.
 This symmetric composition provides second-order accuracy in time with respect to the directional splitting.
 
 
 Execution: classic_hydro_solver_2d(Q1_0, Q2x_0, Q2y_0, Q3_0, Nx, xbeg, xend, Ny, ybeg, yend, t0, tend, ghost, gamma, CFL, viscosity, Xi, bound_cond, scheme, path_save)


 Input:

  Q1_0:              Initial conditions for the conserved quantity Q1 = rho (density).
  Q2x_0:             Initial conditions for the conserved quantity Q2x = rho*ux (density * x-velocity).
  Q2y_0:             Initial conditions for the conserved quantity Q2y = rho*uy (density * y-velocity).
  Q3_0:              Initial conditions for the conserved quantity Q3 = rho*e_tot (density * total_energy).
  Nx:                Number of grid cells in the x-direction.
  xbeg:              Initial value for the computational domain in the x-direction. Defines the size of the grid.
  xend:              Ending value for the computational domain in the x-direction. Defines the size of the grid.
  Ny:                Number of grid cells in the y-direction.
  ybeg:              Initial value for the computational domain in the y-direction. Defines the size of the grid.
  yend:              Ending value for the computational domain in the y-direction. Defines the size of the grid.
  t0:                Initial simulation time.
  tend:              Final simulation time.
  ghost:             Number of ghost cells. Important for imposing the boundary conditions.
  gamma:             Adiabatic exponent. Equation of state: p = (gamma-1)*rho*e, where e is the thermal energy.
  CFL:               Courant number. Security factor <1 used for determining the time step by means of the Courant criterion.
  viscosity:         Boolean parameter. If True, imposes an artificial viscosity. If False, the artificial viscosity is 
                     not added. We want to add artificial viscosity when is desirable to increase the diffusivity near a shock
                     front and only there. It introduces numerical dissipation in compressive regions, allowing shocks to be 
                     captured over a finite number of grid cells while converting kinetic energy into internal energy consistently
                     with shock heating. The most popular artificial viscosity ansatz for handling shocks and the one used here is 
                     the so-called von Neumann-Richtmyer artificial viscosity. It is a bulk viscosity which acts as an additional
                     pressure.
  Xi:                Dimensionless artificial-viscosity coefficient controlling the strength of the von Neumann–Richtmyer viscosity
                     and therefore the numerical width of captured shocks. Values of order unity (typically a few) are commonly used.
  bound_cond:        Type of boundary conditions to be implemented.
                     Options: reflective --- Main advantage is that both mass and energy are globally conserved since no
                                             mass and no energy leave the computational domain, the system is closed.
                                             Major disadvantage is that waves are also reflected back into the computational
                                             domain at closed boundaries which is highly unphysical.
                              periodic --- Useful if you model a real periodic system or if the computational power is not
                                           sufficient to model the whole domain of interest and one can assume that the
                                           expected pattern of the flow have a periodically repeating nature.
                              free_outflow_inflow --- Main advantage is that all waves that are generated in the
                                                      computational domain can leave the grid without being reflected back.
                                                      One major problem with this condition is the following: if the
                                                      velocity in cell 1 gets u1 > 0, then the state in cell 1 will
                                                      determine the influx of material into the domain (same argument holds
                                                      for the boundary at the other side in cell N) and this can eventually
                                                      lead to any arbitrary inflow of matter into the domain.
                              free_outflow --- These boundaries are an attempt to solve the case of arbitrary inflow
                                               in subsonic cases with free outflow/inflow conditions. Obviously, it does
                                               not generate inflow at all, but mass can leave the computational domain.
                                               In other words, the boundary is a sink.
  scheme:            Advection scheme to be used to compute the fluxes. They are written as flux limiters, i.e., for smooth
                     parts of a solution, the scheme do a 2nd order accurate (flux conserved) advection and for regions near
                     a discontinuity the scheme switches to a 1st order (donor cell/upwind) advection.
                     Options: UW --- donor cell / Upwind scheme.
                              LW --- Lax-Wendroff scheme.
                              BW --- Beam-Warming scheme.
                              Fromm --- Fromm scheme.
                              minmod --- minmod scheme.
                              superbee --- superbee scheme.
                              MC --- MC scheme.
                              vanLeer --- van Leer scheme.    
  path_save:         Path where the output of the simulation should be saved.


 Output:    
                                            
  It generates a new folder called Output in the indicated path by 'path_save'. Inside the Output folder, four folders
  called Movie1, Movie2, Movie3,  and Movie4 are created where videos with the obtained simulation are stored. Each
  video corresponds to one important quantity, namely, density, velocity modulus, specific total energy and pressure.                                                   

"""

def classic_hydro_solver_2d(Q1_0, Q2x_0, Q2y_0, Q3_0, Nx, xbeg, xend, Ny, ybeg, yend, t0, tend, ghost=1, gamma=1.4, CFL=0.5, viscosity=True, Xi=3.0, bound_cond='reflective', scheme='UW', path_save='/home/santiago/Documentos'):

    import numpy as np
    import sys
    import subprocess
    import os
    import shutil
    import matplotlib.pyplot as plt
    
    sys.path.append('/home/santiago/Documentos/SHPython/SHhydro')
    from Advection_Schemes import advection_schemes
    
    File_Names = []
    FileNames1 = []
    FileNames2 = []
    FileNames3 = []
    FileNames4 = []
    
    fpath1 = path_save+'/Output'
    fpath2 = fpath1+'/Movie1'
    fpath3 = fpath1+'/Movie2'
    fpath4 = fpath1+'/Movie3'
    fpath5 = fpath1+'/Movie4'
    
    os.mkdir(fpath1)
    os.mkdir(fpath2)
    os.mkdir(fpath3)
    os.mkdir(fpath4)
    os.mkdir(fpath5)
    
    # Computational domain:
    
    Dx = (xend-xbeg)/Nx
    Dy = (yend-ybeg)/Ny
    
    # Ghost cells:
    
    Nx_tot = Nx+2*ghost
    Ibeg_x = ghost
    Iend_x = Nx+ghost-1
    
    Ny_tot = Ny+2*ghost
    Ibeg_y = ghost
    Iend_y = Ny+ghost-1
    
    #Initial conditions:
    
    Q1_values = np.copy(Q1_0)
    Q2x_values = np.copy(Q2x_0)
    Q2y_values = np.copy(Q2y_0)
    Q3_values = np.copy(Q3_0)
    
    #Allocate buffers:
        
    Q1_values_buffer = np.empty_like(Q1_values)
    Q2x_values_buffer = np.empty_like(Q2x_values)
    Q2y_values_buffer = np.empty_like(Q2y_values)
    Q3_values_buffer = np.empty_like(Q3_values)
    
    #Allocate centered velocities:
    
    ux_centered = np.empty_like(Q1_values)
    uy_centered = np.empty_like(Q1_values)
    
    #Get symmetric grid around computational domain:
        
    def make_grid():

        grid_x = xbeg + (np.arange(Nx_tot)-ghost+0.5)*Dx
        grid_y = ybeg + (np.arange(Ny_tot)-ghost+0.5)*Dy
    
        return grid_x, grid_y
            
    #Pressure computation:
        
    def pressure(Q1_values, Q2x_values, Q2y_values, Q3_values, gamma):
        
        ux = Q2x_values/Q1_values
        uy = Q2y_values/Q1_values
        e_tot = Q3_values/Q1_values
        e_kin = (ux**2+uy**2)/2
        e = e_tot-e_kin
        p = (gamma-1)*Q1_values*e
        
        return p
    
    #Interface velocity:
    
    def u_interface(Q1_values, Q2_values, i, j, flag='x-direction'):
        
        if flag == 'x-direction':
        
            u_inter = 1/2*((Q2_values[i, j]/Q1_values[i, j])+(Q2_values[i+1, j]/Q1_values[i+1, j]))
            
        if flag == 'y-direction':
        
            u_inter = 1/2*((Q2_values[i, j]/Q1_values[i, j])+(Q2_values[i, j+1]/Q1_values[i, j+1]))
        
        return u_inter
        
    #Boundary conditions:
    
    def boundary_conditions(values, index, quantity='density', flag1='reflective', flag2='x-direction'):
        
        if flag2 == 'x-direction':
            
            for g in range(ghost):

                #Ghost-cell indices:
                    
                i_left_ghost  = Ibeg_x - 1 - g
                i_right_ghost = Iend_x + 1 + g
    
                #Corresponding interior indices:
                    
                i_left_inner  = Ibeg_x + g
                i_right_inner = Iend_x - g
        
                if flag1 == 'reflective':
            
                    if quantity == 'density' or quantity == 'energy':
                
                        values[i_left_ghost, index] = values[i_left_inner, index]
                    
                        values[i_right_ghost, index] = values[i_right_inner, index]
                        
                    elif quantity == 'x-momentum':
                        
                        values[i_left_ghost, index] = -values[i_left_inner, index]
                    
                        values[i_right_ghost, index] = -values[i_right_inner, index]
                        
                    elif quantity == 'y-momentum':
                        
                        values[i_left_ghost, index] = values[i_left_inner, index]
                    
                        values[i_right_ghost, index] = values[i_right_inner, index]
                    
                elif flag1 == 'periodic':
                    
                    values[i_left_ghost, index] = values[Iend_x-g, index]
                
                    values[i_right_ghost, index] = values[Ibeg_x+g, index]
                    
                elif flag1 == 'free_outflow_inflow':
                    
                    values[i_left_ghost, index] = values[Ibeg_x, index]
                
                    values[i_right_ghost, index] = values[Iend_x, index]
                    
                elif flag1 == 'free_outflow':
                    
                    if quantity == 'density' or quantity == 'energy':
                    
                        values[i_left_ghost, index] = values[Ibeg_x, index]
                    
                        values[i_right_ghost, index] = values[Iend_x, index]
                        
                    elif quantity == 'x-momentum':
    
                        values[i_left_ghost, index] = min(values[Ibeg_x, index], 0.0)
                        values[i_right_ghost, index] = max(values[Iend_x, index], 0.0)
                    
                    elif quantity == 'y-momentum':
                    
                        values[i_left_ghost, index]  = values[Ibeg_x, index]
                        values[i_right_ghost, index] = values[Iend_x, index]
                                        
        elif flag2 == 'y-direction':
            
            for g in range(ghost):

                #Ghost-cell indices:
                    
                j_bottom_ghost = Ibeg_y - 1 - g
                j_top_ghost    = Iend_y + 1 + g
    
                #Corresponding interior indices:
                    
                j_bottom_inner = Ibeg_y + g
                j_top_inner    = Iend_y - g
        
                if flag1 == 'reflective':
                    
                    if quantity == 'density' or quantity == 'energy':
                
                        values[index, j_bottom_ghost] = values[index, j_bottom_inner]
                    
                        values[index, j_top_ghost] = values[index, j_top_inner]
                        
                    elif quantity == 'x-momentum':
            
                        values[index, j_bottom_ghost] = values[index, j_bottom_inner]
                        values[index, j_top_ghost] = values[index, j_top_inner]
            
                    elif quantity == 'y-momentum':
            
                        values[index, j_bottom_ghost] = -values[index, j_bottom_inner]
                        values[index, j_top_ghost] = -values[index, j_top_inner]
                    
                elif flag1 == 'periodic':
                    
                    values[index, j_bottom_ghost] = values[index, Iend_y-g]
                
                    values[index, j_top_ghost] = values[index, Ibeg_y+g]
                    
                elif flag1 == 'free_outflow_inflow':
                    
                    values[index, j_bottom_ghost] = values[index, Ibeg_y]
                
                    values[index, j_top_ghost] = values[index, Iend_y]
                    
                elif flag1 == 'free_outflow':
                    
                    if quantity == 'density' or quantity == 'energy':
                    
                        values[index, j_bottom_ghost] = values[index, Ibeg_y]
                    
                        values[index, j_top_ghost] = values[index, Iend_y]
                        
                    elif quantity == 'x-momentum':
                
                        values[index, j_bottom_ghost]  = values[index, Ibeg_y]
                        values[index, j_top_ghost] = values[index, Iend_y]
                
                    elif quantity == 'y-momentum':
                
                        values[index, j_bottom_ghost] = min(values[index, Ibeg_y], 0.0)
                        values[index, j_top_ghost] = max(values[index, Iend_y], 0.0)
                                    
        return values

    #Von Neumann-Richtmyer artificial viscosity:
        
    def artificial_viscosity(Q1_values, Q2x_values, Q2y_values, p, Xi, Dx, Ibeg_x, Iend_x, Dy, Ibeg_y, Iend_y):

        ux = Q2x_values/Q1_values
        uy = Q2y_values/Q1_values        

        p_eff = np.copy(p)
    
        #Characteristic cell size:
            
        h = min(Dx, Dy)
    
        for i in range(Ibeg_x, Iend_x + 1):
    
            for j in range(Ibeg_y, Iend_y + 1):
    
                #Centered derivatives:
                    
                dux_dx = (ux[i+1, j] - ux[i-1, j]) / (2.0 * Dx)
    
                duy_dy = (uy[i, j+1] - uy[i, j-1]) / (2.0 * Dy)
    
                #Velocity divergence:
                    
                div_u = dux_dx + duy_dy
    
                #Artificial viscosity only in compression:
                    
                if div_u < 0.0:
    
                    Pi = (Q1_values[i, j] * (Xi * h)**2 * div_u**2)
    
                    p_eff[i, j] += Pi
    
        return p_eff
        
    #Time step computation (Courant criterion):
    
    def time_step(Q1_values, Q2x_values, Q2y_values, Q3_values, Dx, Dy, gamma, CFL, viscosity, Xi):
        
        physical = np.s_[Ibeg_x:Iend_x+1, Ibeg_y:Iend_y+1]
        
        if np.any(Q1_values[physical] <= 0.0):
            raise RuntimeError("Non-positive density encountered.")

        ux = Q2x_values/Q1_values
        uy = Q2y_values/Q1_values
        e_tot = Q3_values/Q1_values
        e_kin = (ux**2+uy**2)/2
        e = e_tot-e_kin
        
        if np.any(e[physical] <= 0.0):
            raise RuntimeError("Non-positive internal energy encountered.")
            
        #Physical pressure:

        p = ((gamma - 1.0) * Q1_values * e)

        #Artificial viscosity (for a conservative numerical choice): 
 
        if viscosity:

            p_eff = artificial_viscosity(Q1_values, Q2x_values, Q2y_values, p, Xi, Dx, Ibeg_x, Iend_x, Dy, Ibeg_y, Iend_y)

        else:

            p_eff = p

        #Effective sound speed:

        cs = np.sqrt(gamma * p_eff / Q1_values)
        
        #Maximum characteristic speeds:
            
        vmax_x = np.max((np.abs(ux) + cs)[physical])

        vmax_y = np.max((np.abs(uy) + cs)[physical])

        #CFL timestep:

        dt_x = Dx / vmax_x
        dt_y = Dy / vmax_y

        dt = CFL * min(dt_x, dt_y)
        
        return dt
    
    # Main loop:
        
    grid_x, grid_y = make_grid()
    
    t = t0
    it = 0
    
    output_dt = 0.02 #Amount of simulation time between frames
    next_output = t0
        
    while t < tend:
        
        #Boundary conditions:
        
        for j in range(Ibeg_y, Iend_y + 1):

            boundary_conditions(Q1_values, j, quantity='density', flag1=bound_cond, flag2='x-direction')
            boundary_conditions(Q2x_values, j, quantity='x-momentum', flag1=bound_cond, flag2='x-direction')
            boundary_conditions(Q2y_values, j, quantity='y-momentum', flag1=bound_cond, flag2='x-direction')
            boundary_conditions(Q3_values, j, quantity='energy', flag1=bound_cond, flag2='x-direction')

        for i in range(Ibeg_x, Iend_x + 1):
    
            boundary_conditions(Q1_values, i, quantity='density', flag1=bound_cond, flag2='y-direction')
            boundary_conditions(Q2x_values, i, quantity='x-momentum', flag1=bound_cond, flag2='y-direction')
            boundary_conditions(Q2y_values, i, quantity='y-momentum', flag1=bound_cond, flag2='y-direction')
            boundary_conditions(Q3_values, i, quantity='energy', flag1=bound_cond, flag2='y-direction')
        
        #Time step:
            
        dt = time_step(Q1_values, Q2x_values, Q2y_values, Q3_values, Dx, Dy, gamma, CFL, viscosity, Xi)
        
        #Don't overshoot tend:
            
        dt = min(dt, tend - t)
        
        #1/2 time step sweep in x-direction:
        
        dt_sweep = 0.5*dt
                
        #Buffer:
        
        np.copyto(Q1_values_buffer, Q1_values)
        np.copyto(Q2x_values_buffer, Q2x_values)
        np.copyto(Q2y_values_buffer, Q2y_values)
        np.copyto(Q3_values_buffer, Q3_values)
        
        for j in range(Ibeg_y, Iend_y + 1):
        
            for i in range(Ibeg_x, Iend_x + 1):
                
                #Interface velocities:
                    
                u_inter_R = u_interface(Q1_values_buffer, Q2x_values_buffer, i, j, flag='x-direction')
                u_inter_L = u_interface(Q1_values_buffer, Q2x_values_buffer, i-1, j, flag='x-direction')
        
                #Q1 advection:
                    
                Q1_values[i, j] = Q1_values_buffer[i, j]-(dt_sweep/Dx)*(advection_schemes(Q1_values_buffer, u_inter_R, dt_sweep, Dx, i+1, j, flag1=scheme, flag2='2D', flag3='x-direction')-advection_schemes(Q1_values_buffer, u_inter_L, dt_sweep, Dx, i, j, flag1=scheme, flag2='2D', flag3='x-direction'))
        
                #Q2x advection:
                
                Q2x_values[i, j] = Q2x_values_buffer[i, j]-(dt_sweep/Dx)*(advection_schemes(Q2x_values_buffer, u_inter_R, dt_sweep, Dx, i+1, j, flag1=scheme, flag2='2D', flag3='x-direction')-advection_schemes(Q2x_values_buffer, u_inter_L, dt_sweep, Dx, i, j, flag1=scheme, flag2='2D', flag3='x-direction'))
                
                #Q2y advection:
                
                Q2y_values[i, j] = Q2y_values_buffer[i, j]-(dt_sweep/Dx)*(advection_schemes(Q2y_values_buffer, u_inter_R, dt_sweep, Dx, i+1, j, flag1=scheme, flag2='2D', flag3='x-direction')-advection_schemes(Q2y_values_buffer, u_inter_L, dt_sweep, Dx, i, j, flag1=scheme, flag2='2D', flag3='x-direction'))
                
                #Q3 advection:
            
                Q3_values[i, j] = Q3_values_buffer[i, j]-(dt_sweep/Dx)*(advection_schemes(Q3_values_buffer, u_inter_R, dt_sweep, Dx, i+1, j, flag1=scheme, flag2='2D', flag3='x-direction')-advection_schemes(Q3_values_buffer, u_inter_L, dt_sweep, Dx, i, j, flag1=scheme, flag2='2D', flag3='x-direction'))
    
            #Boundary conditions:
                
            boundary_conditions(Q1_values, j, quantity='density', flag1=bound_cond, flag2='x-direction')
            boundary_conditions(Q2x_values, j, quantity='x-momentum', flag1=bound_cond, flag2='x-direction')
            boundary_conditions(Q2y_values, j, quantity='y-momentum', flag1=bound_cond, flag2='x-direction')
            boundary_conditions(Q3_values, j, quantity='energy', flag1=bound_cond, flag2='x-direction')
                
        #1 time step sweep in y-direction:
                
        dt_sweep = dt
        
        for i in range(Ibeg_x, Iend_x + 1):
                        
            #Boundary conditions:
                
            boundary_conditions(Q1_values, i, quantity='density', flag1=bound_cond, flag2='y-direction')
            boundary_conditions(Q2x_values, i, quantity='x-momentum', flag1=bound_cond, flag2='y-direction')
            boundary_conditions(Q2y_values, i, quantity='y-momentum', flag1=bound_cond, flag2='y-direction')
            boundary_conditions(Q3_values, i, quantity='energy', flag1=bound_cond, flag2='y-direction')
        
        #Buffer:
        
        np.copyto(Q1_values_buffer, Q1_values)
        np.copyto(Q2x_values_buffer, Q2x_values)
        np.copyto(Q2y_values_buffer, Q2y_values)
        np.copyto(Q3_values_buffer, Q3_values)
        
        for i in range(Ibeg_x, Iend_x + 1):
        
            for j in range(Ibeg_y, Iend_y + 1):
                                
                #Interface velocities:
                    
                u_inter_R = u_interface(Q1_values_buffer, Q2y_values_buffer, i, j, flag='y-direction')
                u_inter_L = u_interface(Q1_values_buffer, Q2y_values_buffer, i, j-1, flag='y-direction')
                
                #Q1 advection:
                    
                Q1_values[i, j] = Q1_values_buffer[i, j]-(dt_sweep/Dy)*(advection_schemes(Q1_values_buffer, u_inter_R, dt_sweep, Dy, i, j+1, flag1=scheme, flag2='2D', flag3='y-direction')-advection_schemes(Q1_values_buffer, u_inter_L, dt_sweep, Dy, i, j, flag1=scheme, flag2='2D', flag3='y-direction'))
        
                #Q2x advection:
                
                Q2x_values[i, j] = Q2x_values_buffer[i, j]-(dt_sweep/Dy)*(advection_schemes(Q2x_values_buffer, u_inter_R, dt_sweep, Dy, i, j+1, flag1=scheme, flag2='2D', flag3='y-direction')-advection_schemes(Q2x_values_buffer, u_inter_L, dt_sweep, Dy, i, j, flag1=scheme, flag2='2D', flag3='y-direction'))
                
                #Q2y advection:
                
                Q2y_values[i, j] = Q2y_values_buffer[i, j]-(dt_sweep/Dy)*(advection_schemes(Q2y_values_buffer, u_inter_R, dt_sweep, Dy, i, j+1, flag1=scheme, flag2='2D', flag3='y-direction')-advection_schemes(Q2y_values_buffer, u_inter_L, dt_sweep, Dy, i, j, flag1=scheme, flag2='2D', flag3='y-direction'))
                
                #Q3 advection:
            
                Q3_values[i, j] = Q3_values_buffer[i, j]-(dt_sweep/Dy)*(advection_schemes(Q3_values_buffer, u_inter_R, dt_sweep, Dy, i, j+1, flag1=scheme, flag2='2D', flag3='y-direction')-advection_schemes(Q3_values_buffer, u_inter_L, dt_sweep, Dy, i, j, flag1=scheme, flag2='2D', flag3='y-direction'))
    
            #Boundary conditions:
                
            boundary_conditions(Q1_values, i, quantity='density', flag1=bound_cond, flag2='y-direction')
            boundary_conditions(Q2x_values, i, quantity='x-momentum', flag1=bound_cond, flag2='y-direction')
            boundary_conditions(Q2y_values, i, quantity='y-momentum', flag1=bound_cond, flag2='y-direction')
            boundary_conditions(Q3_values, i, quantity='energy', flag1=bound_cond,flag2='y-direction')
                
        #1/2 time step sweep in x-direction:
        
        dt_sweep = 0.5*dt
        
        for j in range(Ibeg_y, Iend_y + 1):
            
            #Boundary conditions:
                
            boundary_conditions(Q1_values, j, quantity='density', flag1=bound_cond, flag2='x-direction')
            boundary_conditions(Q2x_values, j, quantity='x-momentum', flag1=bound_cond, flag2='x-direction')
            boundary_conditions(Q2y_values, j, quantity='y-momentum', flag1=bound_cond, flag2='x-direction')
            boundary_conditions(Q3_values, j, quantity='energy', flag1=bound_cond, flag2='x-direction')
        
        #Buffer:
        
        np.copyto(Q1_values_buffer, Q1_values)
        np.copyto(Q2x_values_buffer, Q2x_values)
        np.copyto(Q2y_values_buffer, Q2y_values)
        np.copyto(Q3_values_buffer, Q3_values)
        
        for j in range(Ibeg_y, Iend_y + 1):
        
            for i in range(Ibeg_x, Iend_x + 1):
                
                #Interface velocities:
                    
                u_inter_R = u_interface(Q1_values_buffer, Q2x_values_buffer, i, j, flag='x-direction')
                u_inter_L = u_interface(Q1_values_buffer, Q2x_values_buffer, i-1, j, flag='x-direction')
        
                #Q1 advection:
                
                Q1_values[i, j] = Q1_values_buffer[i, j]-(dt_sweep/Dx)*(advection_schemes(Q1_values_buffer, u_inter_R, dt_sweep, Dx, i+1, j, flag1=scheme, flag2='2D', flag3='x-direction')-advection_schemes(Q1_values_buffer, u_inter_L, dt_sweep, Dx, i, j, flag1=scheme, flag2='2D', flag3='x-direction'))
        
                #Q2x advection:
                
                Q2x_values[i, j] = Q2x_values_buffer[i, j]-(dt_sweep/Dx)*(advection_schemes(Q2x_values_buffer, u_inter_R, dt_sweep, Dx, i+1, j, flag1=scheme, flag2='2D', flag3='x-direction')-advection_schemes(Q2x_values_buffer, u_inter_L, dt_sweep, Dx, i, j, flag1=scheme, flag2='2D', flag3='x-direction'))
                
                #Q2y advection:
                
                Q2y_values[i, j] = Q2y_values_buffer[i, j]-(dt_sweep/Dx)*(advection_schemes(Q2y_values_buffer, u_inter_R, dt_sweep, Dx, i+1, j, flag1=scheme, flag2='2D', flag3='x-direction')-advection_schemes(Q2y_values_buffer, u_inter_L, dt_sweep, Dx, i, j, flag1=scheme, flag2='2D', flag3='x-direction'))
                
                #Q3 advection:
            
                Q3_values[i, j] = Q3_values_buffer[i, j]-(dt_sweep/Dx)*(advection_schemes(Q3_values_buffer, u_inter_R, dt_sweep, Dx, i+1, j, flag1=scheme, flag2='2D', flag3='x-direction')-advection_schemes(Q3_values_buffer, u_inter_L, dt_sweep, Dx, i, j, flag1=scheme, flag2='2D', flag3='x-direction'))
    
            #Boundary conditions:
                
            boundary_conditions(Q1_values, j, quantity='density', flag1=bound_cond, flag2='x-direction')
            boundary_conditions(Q2x_values, j, quantity='x-momentum', flag1=bound_cond, flag2='x-direction')
            boundary_conditions(Q2y_values, j, quantity='y-momentum', flag1=bound_cond, flag2='x-direction')
            boundary_conditions(Q3_values, j, quantity='energy', flag1=bound_cond, flag2='x-direction')
        
        #Boundary conditions:
            
        for i in range(Ibeg_x, Iend_x + 1):
        
            boundary_conditions(Q1_values, i, quantity='density', flag1=bound_cond, flag2='y-direction')
            boundary_conditions(Q2x_values, i, quantity='x-momentum', flag1=bound_cond, flag2='y-direction')
            boundary_conditions(Q2y_values, i, quantity='y-momentum', flag1=bound_cond, flag2='y-direction')
            boundary_conditions(Q3_values, i, quantity='energy', flag1=bound_cond, flag2='y-direction')
        
        #Buffer:
        
        np.copyto(Q1_values_buffer, Q1_values)
        np.copyto(Q2x_values_buffer, Q2x_values)
        np.copyto(Q2y_values_buffer, Q2y_values)
        np.copyto(Q3_values_buffer, Q3_values)
        
        #Pressure:
    
        p = pressure(Q1_values, Q2x_values, Q2y_values, Q3_values, gamma)
            
        #Viscosity:
            
        if viscosity == True:
    
            p = artificial_viscosity(Q1_values, Q2x_values, Q2y_values, p, Xi, Dx, Ibeg_x, Iend_x, Dy, Ibeg_y, Iend_y)
        
        
        #Splitting to add source terms in momentum and energy equations:

        #State before the pressure-source update:
            
        Q1_old = Q1_values_buffer
        Q2x_old = Q2x_values_buffer
        Q2y_old = Q2y_values_buffer
        Q3_old = Q3_values_buffer
        
        #Momentum update due to the pressure gradients:
        
        for i in range(Ibeg_x, Iend_x + 1):
        
            for j in range(Ibeg_y, Iend_y + 1):
        
                Q2x_values[i, j] = Q2x_old[i, j] - (dt/(2.0*Dx))*(p[i+1, j] - p[i-1, j])
        
                Q2y_values[i, j] = Q2y_old[i, j] - (dt/(2.0*Dy))*(p[i, j+1] - p[i, j-1])
        
        #Apply boundary conditions to the new momenta before computing the time-centered velocities:
        
        for j in range(Ibeg_y, Iend_y + 1):
        
            boundary_conditions(Q2x_values, j, quantity='x-momentum', flag1=bound_cond, flag2='x-direction')
        
            boundary_conditions(Q2y_values, j, quantity='y-momentum', flag1=bound_cond, flag2='x-direction')
        
        for i in range(Ibeg_x, Iend_x + 1):
        
            boundary_conditions(Q2x_values, i, quantity='x-momentum', flag1=bound_cond, flag2='y-direction')
        
            boundary_conditions(Q2y_values, i, quantity='y-momentum', flag1=bound_cond, flag2='y-direction')
        
        #Time-centered velocities:
                
        np.add(Q2x_old, Q2x_values, out=ux_centered)
        ux_centered *= 0.5
        ux_centered /= Q1_old
        
        np.add(Q2y_old, Q2y_values, out=uy_centered)
        uy_centered *= 0.5
        uy_centered /= Q1_old
        
        #Energy update using the time-centered velocities
        
        for i in range(Ibeg_x, Iend_x + 1):
        
            for j in range(Ibeg_y, Iend_y + 1):
        
                pressure_work_x = (p[i+1, j]*ux_centered[i+1, j] - p[i-1, j]*ux_centered[i-1, j])/(2.0*Dx)
        
                pressure_work_y = (p[i, j+1]*uy_centered[i, j+1] - p[i, j-1]*uy_centered[i, j-1])/(2.0*Dy)
        
                Q3_values[i, j] = (Q3_old[i, j] - dt*(pressure_work_x + pressure_work_y))
                
        t += dt
        
        #Plot writing:
            
        if t >= next_output:
        
            print("Writing plots at time", t)
            
            fig, ax=plt.subplots()
            ax.set_aspect('equal')
            im=ax.pcolormesh(grid_x,grid_y,Q1_values.T,cmap='RdYlBu')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('rho at time='+str(round(t,6)))
            plt.colorbar(im, ax=ax)
            filename1 = "Image_rho_{:07d}.png".format(it)
            FileNames1.append(filename1) 
            fig.savefig(fpath2+'/'+filename1)
            plt.close()
            
            velocity_magnitude = np.sqrt((Q2x_values / Q1_values)**2+(Q2y_values / Q1_values)**2)
            
            fig, ax=plt.subplots()
            ax.set_aspect('equal')
            im=ax.pcolormesh(grid_x,grid_y,velocity_magnitude.T,cmap='RdYlBu')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('|u| at time='+str(round(t,6)))
            plt.colorbar(im, ax=ax)
            filename2 = "Image_u_{:07d}.png".format(it)
            FileNames2.append(filename2) 
            fig.savefig(fpath3+'/'+filename2)
            plt.close()
            
            e_tot = Q3_values / Q1_values
            
            fig, ax=plt.subplots()
            ax.set_aspect('equal')
            im=ax.pcolormesh(grid_x,grid_y,e_tot.T,cmap='RdYlBu')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('e at time='+str(round(t,6)))
            plt.colorbar(im, ax=ax)
            filename3 = "Image_e_{:07d}.png".format(it)
            FileNames3.append(filename3) 
            fig.savefig(fpath4+'/'+filename3)
            plt.close()
            
            # Recompute pressure from the CURRENT state before plotting

            p_plot = pressure(Q1_values, Q2x_values, Q2y_values, Q3_values, gamma)
            
            fig, ax=plt.subplots()
            ax.set_aspect('equal')
            im=ax.pcolormesh(grid_x,grid_y,p_plot.T,cmap='RdYlBu')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('p at time='+str(round(t,6)))
            plt.colorbar(im, ax=ax)
            filename4 = "Image_p_{:07d}.png".format(it)
            FileNames4.append(filename4) 
            fig.savefig(fpath5+'/'+filename4)
            plt.close()
            
            it += 1
            
            while next_output <= t:
                    next_output += output_dt
        
    #Use ffmpeg to combine the images in a movie:
    
    subprocess.run(['ffmpeg','-framerate','30','-pattern_type','glob','-i',fpath2+"/*.png",'-c:v','libx264','-pix_fmt','yuv420p','movie_rho.mp4'])
    shutil.move('movie_rho.mp4', fpath2+'/movie_rho.mp4')
    
    subprocess.run(['ffmpeg','-framerate','30','-pattern_type','glob','-i',fpath3+"/*.png",'-c:v','libx264','-pix_fmt','yuv420p','movie_u.mp4'])
    shutil.move('movie_u.mp4', fpath3+'/movie_u.mp4')
    
    subprocess.run(['ffmpeg','-framerate','30','-pattern_type','glob','-i',fpath4+"/*.png",'-c:v','libx264','-pix_fmt','yuv420p','movie_e.mp4'])
    shutil.move('movie_e.mp4', fpath4+'/movie_e.mp4')
    
    subprocess.run(['ffmpeg','-framerate','30','-pattern_type','glob','-i',fpath5+"/*.png",'-c:v','libx264','-pix_fmt','yuv420p','movie_p.mp4'])
    shutil.move('movie_p.mp4', fpath5+'/movie_p.mp4')

    
    
    for filename1 in FileNames1:  #Delete the image files
    
        os.remove(fpath2+'/'+filename1)
        
    for filename2 in FileNames2:  #Delete the image files
    
        os.remove(fpath3+'/'+filename2)
        
    for filename3 in FileNames3:  #Delete the image files
    
        os.remove(fpath4+'/'+filename3)
        
    for filename4 in FileNames4:  #Delete the image files
     
        os.remove(fpath5+'/'+filename4)
        
        
    return


        
