#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 20:15:39 2023

@author: santiago
"""

"""
                                                                              
 This procedure applies an advection scheme to compute the fluxes to be applied, e.g., in the Godunov method in
 numerical hydro solvers. The advection schemes are written as flux limiters, i.e., for smooth parts of a solution,
 the scheme do a 2nd order accurate (flux conserved) advection and for regions near a discontinuity the scheme switch
 to a 1st order (donor cell/upwind) advection.
 
 
 Execution: F_value = advection_schemes(values, u_interface, Dt, Dx, i, j, flag1, flag2, flag3)

 Input:

  values:          Quantities stored in the cells of the grid for which we want to compute the flux through the interfaces.
  u_interface:     Velocity in the interface of the cell.
  Dt:              Time-step in the discretization of the time coordinate.
  Dx:              Space-step in the discretization of the space coordinate. It's the size of each cell in the grid.
  i:               Index specifying the cell in which we are computing the flux. If flag2 = '2D', it indicates the
                   x-direction index in the grid.
  j:               Index specifying the cell in which we are computing the flux. Only required if flag2 = '2D'.
                   It indicates the y-direction index in the grid. 
  flag1:           Advection scheme to be used to compute the flux.
                   Options: UW --- donor cell / Upwind scheme.
                            LW --- Lax-Wendroff scheme.
                            BW --- Beam-Warming scheme.
                            Fromm --- Fromm scheme.
                            minmod --- minmod scheme.
                            superbee --- superbee scheme.
                            MC --- MC scheme.
                            vanLeer --- van Leer scheme. 
  flag2:           Number of dimensions of the grid in which this procedure is applied.
                   Options: 1D --- 1D grid.
                            2D --- 2D grid.
  flag3:           If flag2 = '2D', this flag indicates the direction in which the flux is computed.
                   Options: x-direction --- Flux through cell interfaces along x-direction.
                            y-direction --- Flux through cell interfaces along y-direction.                                               
    

 Output:    
                                            
  F_value:         Obtained value of the flux in the left/down cell interface (i−1/2 interface for i = i).                                                 

"""

def advection_schemes(values, u_interface, Dt, Dx, i, j=None, flag1='UW', flag2='1D', flag3=None):
    
    import numpy as np
    
    def theta(u_interface):
        
        if u_interface >= 0:
            
            theta = 1.0
            
        else:
            
            theta = -1.0
            
        return theta
    
    def r(values, u_interface, i, j, flag2, flag3):
        
        tinytiny = 1e-20
        
        if flag2 == '1D':
        
            if u_interface >= 0:
    
                rvalue = (values[i-1]-values[i-2])/(values[i]-values[i-1]+tinytiny)
                
            else:
    
                rvalue = (values[i+1]-values[i])/(values[i]-values[i-1]+tinytiny)
                
        if flag2 == '2D':
            
            if flag3 == 'x-direction':
        
                if u_interface >= 0:
        
                    rvalue = (values[i-1, j]-values[i-2, j])/(values[i, j]-values[i-1, j]+tinytiny)
                    
                else:
        
                    rvalue = (values[i+1, j]-values[i, j])/(values[i, j]-values[i-1, j]+tinytiny)
                    
            if flag3 == 'y-direction':
                
                if u_interface >= 0:
        
                    rvalue = (values[i, j-1]-values[i, j-2])/(values[i, j]-values[i, j-1]+tinytiny)
                    
                else:
        
                    rvalue = (values[i, j+1]-values[i, j])/(values[i, j]-values[i, j-1]+tinytiny)

        return rvalue

    def minmodfun(a, b):

        if abs(a) < abs(b) and a*b > 0:

            result = a

        elif abs(b) <= abs(a) and a*b > 0:

            result = b

        elif a*b <= 0:

            result = 0.0

        return result
    
    if flag1 == 'UW':
       
        phi = 0.0
        
    if flag1 == 'LW':
        
        phi = 1.0
        
    if flag1 == 'BW':
        
        phi = r(values, u_interface, i, j, flag2, flag3)
        
    if flag1 == 'Fromm':
        
        phi = (1/2)*(1+r(values, u_interface, i, j, flag2, flag3))
        
    if flag1 == 'minmod':
        
        phi = minmodfun(1, r(values, u_interface, i, j, flag2, flag3))
        
    if flag1 == 'superbee': 
        
        min1 = np.min(np.array([1,2*r(values, u_interface, i, j, flag2, flag3)]))
        
        min2 = np.min(np.array([2,r(values, u_interface, i, j, flag2, flag3)]))
        
        phi = np.max(np.array([0, min1, min2]))
        
    if flag1 == 'MC':
        
        min1 = np.min(np.array([(1+r(values, u_interface, i, j, flag2, flag3))/2, 2, 2*r(values, u_interface, i, j, flag2, flag3)]))
        
        phi = np.max(np.array([0, min1]))
        
    if flag1 == 'vanLeer':
        
        phi = (r(values, u_interface, i, j, flag2, flag3)+abs(r(values, u_interface, i, j, flag2, flag3)))/(1+abs(r(values, u_interface, i, j, flag2, flag3)))
        
    if flag2 == '1D':
      
        F_value = (1/2)*u_interface*((1+theta(u_interface))*values[i-1]+(1-theta(u_interface))*values[i])
        +(1/2)*abs(u_interface)*(1-abs(((u_interface*Dt)/Dx)))*phi*(values[i]-values[i-1])
        
    if flag2 == '2D':
        
        if flag3 == 'x-direction':
            
            F_value = (1/2)*u_interface*((1+theta(u_interface))*values[i-1, j]+(1-theta(u_interface))*values[i, j])
            +(1/2)*abs(u_interface)*(1-abs(((u_interface*Dt)/Dx)))*phi*(values[i, j]-values[i-1, j])
            
        if flag3 == 'y-direction':
            
            F_value = (1/2)*u_interface*((1+theta(u_interface))*values[i, j-1]+(1-theta(u_interface))*values[i, j])
            +(1/2)*abs(u_interface)*(1-abs(((u_interface*Dt)/Dx)))*phi*(values[i, j]-values[i, j-1])
        
    return F_value




