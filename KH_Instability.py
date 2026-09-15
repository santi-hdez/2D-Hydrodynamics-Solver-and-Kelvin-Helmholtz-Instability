#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 21:46:25 2023

@author: santiago
"""

"""
                                                                              
 This procedure simulates the development of the Kelvin-Helmholtz instability in a two-dimensional compressible
 fluid. The instability develops at shear interfaces separating fluid layers moving with different horizontal
 velocities. A small vertical velocity perturbation is imposed at the interfaces to seed the instability, which
 subsequently grows and produces the characteristic vortical structures of the Kelvin-Helmholtz instability.

 Basic assumptions and setup of the model:

  - The fluid is inviscid apart from the optional numerical artificial viscosity used for shock capturing.
 
  - The fluid obeys an ideal-gas equation of state with a constant adiabatic exponent gamma.
 
  - The initial pressure is spatially uniform, p = 2.5.
 
  - A denser central layer is located between y = -0.5 and y = +0.5 and is surrounded by two lower-density layers.
    The density changes smoothly from rho = 1 in the outer layers to rho = 2 in the central layer using hyperbolic
    tangent transition profiles with a characteristic width of 0.05.
 
  - The central layer initially moves in the positive x-direction with ux approximately +1, while the two outer
    layers move in the negative x-direction with ux approximately -1. The velocity changes smoothly across the same
    shear interfaces as the density.
 
  - A small vertical velocity perturbation with amplitude A = 0.1 is imposed around the two shear interfaces to seed
    the Kelvin-Helmholtz instability. The perturbation is sinusoidal along the x-direction, with two wavelengths across
    the computational domain, and is localized around the interfaces in the y-direction using Gaussian profiles with
    a characteristic width sigma = 0.12.
 
  - By default, periodic boundary conditions are imposed in both spatial directions. The computational domain therefore
    represents a periodically repeating section of the flow.
 
 Here, the Euler equations are solved by means of a 2D classic numerical hydrodynamics solver. 
     
 Recommended to run with the default values for a nice simulation.
 
 
 Execution: kh_instability(Nx, xbeg, xend, Ny, ybeg, yend, t0, tend, ghost, gamma, CFL, viscosity, Xi, bound_cond, scheme, path_save)


 Input:

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
  scheme:            Advection scheme to be used to compute the fluxes to be applied in the Godunov method. They are written
                     as flux limiters, i.e., for smooth parts of a solution, the scheme do a 2nd order accurate
                     (flux conserved) advection and for regions near a discontinuity the scheme switches to a 1st order
                     (donor cell/upwind) advection.
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
  called Movie1, Movie2, Movie3, and Movie4 are created where videos with the obtained simulation are stored. Each
  video corresponds to one important quantity, namely, density, velocity modulus, specific total energy and pressure.                                                    

"""

def kh_instability(Nx=250, xbeg=-1.0, xend=1.0, Ny=250, ybeg=-1.0, yend=1.0, t0=0.0, tend=5.0, ghost=2, gamma=1.4, CFL=0.5, viscosity=True, Xi=3.0, bound_cond='periodic', scheme='vanLeer', path_save='/home/santiago/Documentos'):
            
    import numpy as np
    import sys
    
    sys.path.append('/home/santiago/Documentos/SHPython/SHhydro')
    from Classic_Hydro_Solver_2D import classic_hydro_solver_2d
    
    #Make grid in order to define initial conditions:
        
    Nx_tot = Nx+2*ghost
    Ny_tot = Ny+2*ghost
            
    Dx = (xend-xbeg)/Nx
    Dy = (yend-ybeg)/Ny
    
    def make_grid():

        grid_x = xbeg + (np.arange(Nx_tot)-ghost+0.5)*Dx
        grid_y = ybeg + (np.arange(Ny_tot)-ghost+0.5)*Dy
    
        return grid_x, grid_y
    
    grid_x, grid_y = make_grid()
    
    X, Y = np.meshgrid(grid_x, grid_y, indexing='ij')  
    
    #Initial conditions:
        
    p0 = 2.5 #Uniform pressure

    y_interface = 0.5 #Position of the two shear interfaces
    
    transition_width = 0.05 #Width of transition between layers
    
    u0 = 1.0 #Velocity magnitude

    A = 0.1 #Amplitude of perturbation

    sigma = 0.12 #Width of perturbation around interface
    
    n_mode = 2 #Number of perturbation wavelengths

    #Smooth central layer:

    f = 0.5 * (np.tanh((Y + y_interface) / transition_width)-np.tanh((Y - y_interface) / transition_width))

    rho = 1.0 + f #Density

    ux = -u0 + 2.0*u0*f #Horizontal velocity

    #Vertical perturbation:
        
    Lx = xend - xbeg

    perturbation_x = np.sin(2.0*np.pi*n_mode*(X - xbeg)/Lx)

    perturbation_y = (np.exp(-((Y - y_interface)/sigma)**2)-np.exp(-((Y + y_interface)/sigma)**2))

    uy = A * perturbation_x * perturbation_y

    #Convert to conserved variables:

    Q1_0 = rho

    Q2x_0 = rho * ux

    Q2y_0 = rho * uy
        
    E_internal = p0 / (gamma - 1.0) #Internal energy density

    E_kinetic = 0.5 * rho * (ux**2 + uy**2) #Kinetic energy density

    Q3_0 = E_internal + E_kinetic #Total energy density
    
    classic_hydro_solver_2d(
        Q1_0=Q1_0,
        Q2x_0=Q2x_0,
        Q2y_0=Q2y_0,
        Q3_0=Q3_0,
        Nx=Nx,
        xbeg=xbeg,
        xend=xend,
        Ny=Ny,
        ybeg=ybeg,
        yend=yend,
        t0=t0,
        tend=tend,
        ghost=ghost,
        gamma=gamma,
        CFL=CFL,
        viscosity=viscosity,
        Xi=Xi,
        bound_cond=bound_cond,
        scheme=scheme,
        path_save=path_save
    )
    
    return


kh_instability()








