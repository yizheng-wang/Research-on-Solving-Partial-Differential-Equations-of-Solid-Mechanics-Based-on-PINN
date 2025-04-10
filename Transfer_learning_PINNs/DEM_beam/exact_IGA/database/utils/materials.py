#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes for material properties
"""
import numpy as np
#import matplotlib.pyplot as plt

# Define RBF kernel
def RBF(x1, x2, lengthscales):    
    diffs = np.expand_dims(x1 / lengthscales, 1) - \
            np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs**2, axis=2)
    return np.exp(-0.5 * r2)

def GRF(N, h, mean=0, variance=1, length_scale = 0.1):
    #N = 128
    jitter = 1e-10
    X = np.linspace(0, h, N)[:,None]
    K = RBF(X, X, length_scale)
    L = np.linalg.cholesky(K + jitter*np.eye(N))
    gp_sample = variance*np.dot(L, np.random.normal(size=N))+mean
    
    #plt.plot(X.flatten(), gp_sample/1000)
    #plt.xlabel('self.Y')
    #plt.ylabel('self.gp')
    #plt.title('Random function for Material')
    #plt.show()
    
    return X.flatten(), gp_sample

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
    """
    Class for 2D linear elastic materials in FGM beam
    Input
    -----
    Emod : (float) Young's modulus
    nu : (float) Poisson ratio
    e0 : (float) Porosity
    plane_type : (string) stress or strain    
        
    """
    def __init__(self, Emod = None, nu=None, e0=None, plane_type="stress"):
        self.nu = nu
        self.Emod = Emod
        self.e0 = e0
        if plane_type=="stress":
            self.Cmat = 1/(1-nu**2)*np.array([[1,  nu,  0], [nu,  1,  0], 
                                         [0,  0,  (1-nu)/2]])
        elif plane_type=="strain":
            self.Cmat = 1/((1+nu)*(1-2*nu))*np.array([[1-nu, nu, 0], [nu, 1-nu, 0],
                                                [0, 0, (1-2*nu)/2]])      
    
    def elasticity(self, coordinates, mesh):
        """
        Retrieves the elasticity modulus for FGM beam based on physical coordinates (xPhys, yPhys) of points in the beam.

        Parameters
        ----------
        coordinates : ndarray
            An array representing the physical coordinates (xPhys, yPhys) of points in the beam.
        mesh : object
            An object for single-patch IGA mesh.
        
        Returns
        -------
        E : float
            The elasticity modulus for the given point in the beam.
        """
        #xPhys = coordinates[0]
        yPhys = coordinates[1]
        #b = np.max(mesh.cpts, axis=1)[0]
        h = np.max(mesh.cpts, axis=1)[0]
        yPhys = yPhys - h/2
        self.E = self.Emod*(1-self.e0*np.cos(np.pi*(yPhys/h)))

        return self.E

class MaterialElast2D_RandomFGM:
    """
    Class for 2D linear elastic materials
    Input
    -----
    Emod : (float) Young's modulus
    nu : (float) Poisson ratio
    plane_type : (string) stress or strain    
        
    """
    def __init__(self, N_material, Emod = None, e0_bot=None, e0_top=None, nu=None, vertices=None, 
                 length_scale = None , plane_type="stress"):
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
        self.Y, self.gp = GRF(N_material, h, length_scale = length_scale)
        Emin = (1-e0_bot)*Emod
        Emax = (1+e0_top)*Emod
        gmax = np.max(self.gp)
        gmin = np.min(self.gp)
        self.gp = (Emax-Emin)/(gmax-gmin)*(self.gp-gmin)+Emin
        
        
    
    def elasticity(self, coordinates, mesh=None):
        """
        Retrieves the elasticity modulus for FGM beam based on physical coordinates (xPhys, yPhys) of points in the beam.

        Parameters
        ----------
        coordinates : ndarray
            An array representing the physical coordinates (xPhys, yPhys) of points in the beam.

        Returns
        -------
        E : float
            The elasticity modulus for the given point in the beam.
        """
        #xPhys = coordinates[0]
        yPhys = coordinates[1]
        
        self.E = np.interp(yPhys, self.Y, self.gp)
        #plt.plot(self.Y, self.gp)
        #plt.xlabel('self.Y')
        #plt.ylabel('self.gp')
        #plt.title('gp with respect to Y')
        #plt.show()
        
        #self.E = self.Emod*np.abs(np.interp(yPhys, self.Y, self.gp))
        #print(np.abs(np.interp(yPhys, self.Y, self.gp)))
        #print(self.E)

        return self.E
    
    