#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes for material properties
"""
import numpy as np

class MaterialElast2D:
    """
    Class for 2D linear elastic materials
    Input
    -----
    Emod : (float) Young's modulus
    nu : (float) Poisson ratio
    plane_type : (string) stress or strain    
        
    """
    def __init__(self, Emod = None, nu=None, plane_type="stress"):
        self.Emod = Emod
        self.nu = nu
        if plane_type=="stress":
            self.Cmat =  Emod/(1-nu**2)*np.array([[1,  nu,  0], [nu,  1,  0], 
                                         [0,  0,  (1-nu)/2]])
        elif plane_type=="strain":
            self.Cmat = Emod/((1+nu)*(1-2*nu))*np.array([[1-nu, nu, 0], [nu, 1-nu, 0],
                                                [0, 0, (1-2*nu)/2]])
        
            
        
class MaterialElast2D_FGM:

    def __init__(self, Emod = None, nu=None, vertices=None, gp=None, N=None, plane_type="stress"):
        self.nu = nu
        self.Emod = Emod
        if plane_type=="stress":
            self.Cmat = 1/(1-nu**2)*np.array([[1,  nu,  0], [nu,  1,  0], 
                                         [0,  0,  (1-nu)/2]])
        elif plane_type=="strain":
            self.Cmat = 1/((1+nu)*(1-2*nu))*np.array([[1-nu, nu, 0], [nu, 1-nu, 0],
                                                [0, 0, (1-2*nu)/2]])
        #b = vertices[3][0]
        h = vertices[3][1]
        self.Y = np.linspace(0, h, N)[:,None].flatten()
        self.gp = gp
        
    
    def elasticity(self, coordinates, mesh=None):

        #xPhys = coordinates[0]
        yPhys = coordinates[1]
        
        self.E = np.interp(yPhys, self.Y, self.gp)

        return self.E  
