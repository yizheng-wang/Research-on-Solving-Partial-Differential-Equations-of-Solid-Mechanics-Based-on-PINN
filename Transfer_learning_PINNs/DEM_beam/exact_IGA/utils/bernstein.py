#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File for routines related to Bezier extraction
"""
import numpy as np
def bezier_extraction(knot, deg):
    '''
    Bezier extraction
    Based on Algorithm 1, from Borden - Isogeometric finite element data
    structures based on Bezier extraction
    '''
    m = len(knot)-deg-1
    a = deg + 1
    b = a + 1
    # Initialize C with the number of non-zero knotspans in the 3rd dimension
    #nb_final = len(np.unique(knot))-1
    C = []
    nb = 1
    C.append(np.eye(deg + 1))
    while b <= m:        
        C.append(np.eye(deg + 1))
        i = b        
        while (b <= m) and (knot[b] == knot[b-1]):
            b = b+1            
        multiplicity = b-i+1    
        alphas = np.zeros(deg-multiplicity)        
        if (multiplicity < deg):    
            numerator = knot[b-1] - knot[a-1]            
            for j in range(deg,multiplicity,-1):
                alphas[j-multiplicity-1] = numerator/(knot[a+j-1]-knot[a-1])            
            r = deg - multiplicity
            for j in range(1,r+1):
                save = r-j+1
                s = multiplicity + j                          
                for k in range(deg+1,s,-1):                                
                    alpha = alphas[k-s-1]
                    C[nb-1][:,k-1] = alpha*C[nb-1][:,k-1] + (1-alpha)*C[nb-1][:,k-2]  
                if b <= m:                
                    C[nb][save-1:save+j,save-1] = C[nb-1][deg-j:deg+1,deg]  
            nb=nb+1
            if b <= m:
                a=b
                b=b+1    
        elif multiplicity==deg:
            if b <= m:
                nb = nb + 1
                a = b
                b = b + 1                
    #assert(nb==nb_final)
    
    return C, nb

def bernstein_basis(uhat, deg):
    ''' 
    Algorithm A1.3 in Piegl & Tiller
    xi is a 1D array
    Computes the Bernstein polynomial of degree deg at point xi
    
    Parameters
    ----------
    uhat: list of evaluation points on the reference interval [-1, 1]
    deg: degree of Bernstein polynomial (integer)
    
    Returns
    -------
    B: matrix of size num_pts x (deg+1) containing the deg+1 Bernstein polynomials
       evaluated at the points uhat
    '''        
    B = np.zeros((len(uhat),deg+1))
    B[:,0] = 1.0
    u1 = 1-uhat
    u2 = 1+uhat  
    
    for j in range(1,deg+1):
        saved = 0.0
        for k in range(0,j):
            temp = B[:,k].copy()
            B[:,k] = saved + u1*temp        
            saved = u2*temp
        B[:,j] = saved
    B = B/np.power(2,deg)
               
    return B
    
def bernstein_basis_deriv(uhat, deg):
    ''' 
    Algorithm A1.3 in Piegl & Tiller
    xi is a 1D array
    Computes the derivatives of the Bernstein polynomial of degree deg at point
    xi on the interval [-1, 1]
    
    Parameters
    ----------
    uhat: list of evaluation points on the reference interval [-1, 1]
    deg: degree of Bernstein polynomial (integer)
    
    Returns
    -------
    B: matrix of size num_pts x (deg+1) containing the deg+1 Bernstein polynomials
       evaluated at the points uhat
    '''        
    u1 = 1-uhat
    u2 = 1+uhat  
    dB = np.zeros((len(uhat),deg))
    dB[:,0] = 1.0
    for j in range(1,deg):
        saved = 0.0
        for k in range(0,j):
            temp = dB[:,k].copy()
            dB[:,k] = saved + u1*temp
            saved = u2*temp
        dB[:,j] = saved
    dB = dB/np.power(2,deg)
    dB0 = np.transpose(np.array([np.zeros(len(uhat))]))
    dB = np.concatenate((dB0, dB, dB0), axis=1)
    dB = (dB[:,0:-1] - dB[:,1:])*deg
    
    return dB

def form_extended_knot(localKnot, deg):
    """
    Create the extended knot vector (Subsection 4.3.2 in Scott - Isogeometric
    data structures based on the Bézier extraction of T-Splines)
    
    Parameters
    ----------
    localKnots: the local knot vector (list of length deg+2)
    deg: polynomial degree of the basis
    
    Returns
    -------
    extendedKnot: the extended knot vector    
            
    """
    # Repeat the first knot (if needed) so that it appears p+1 times
    firstKnot = localKnot[0]
    tol_comp = 1e-12
    indexFirst = [i for i in localKnot if abs(i-firstKnot)<tol_comp]
    numRep = len(indexFirst)
    numNewRepFirst = deg+1-numRep

    #repeat the last knot (if needed) so that it appears p+1 times
    lastKnot = localKnot[-1]
    indexLast = [i for i in localKnot if abs(i-lastKnot)<tol_comp]
    numRep = len(indexLast)
    numNewRepLast = deg+1-numRep

    #form the extended knot vector
    extendedKnot = np.concatenate((firstKnot*np.ones(numNewRepFirst), localKnot, lastKnot*np.ones(numNewRepLast)))
    indexFun = numNewRepFirst
    return extendedKnot, indexFun+1

def bernstein_basis_2d(pts_u, pts_v, deg):
    """
    Generates the 2D Bernstein polynomials at points (pts_u, pts_v)

    Parameters
    ----------
    pts_u : list of evaluation points in the u direction
    pts_v : list of evaluation points in the v direction
    deg : list of polynomial degrees 

    Returns
    -------
    Buv : values of the basis functions 
    dBdu : values of the derivatives of the basis functions with respect to u
    dBdv : values of the derivatives of the basis functions with respect to v
   
    """
    B_u = bernstein_basis(pts_u, deg[0])
    B_v = bernstein_basis(pts_v, deg[1])
    dB_u = bernstein_basis_deriv(pts_u, deg[0])
    dB_v = bernstein_basis_deriv(pts_v, deg[1])
    
    num_pts_u = len(pts_u)
    num_pts_v = len(pts_v)
    num_basis = (deg[0]+1)*(deg[1]+1)
    basis_counter = 0
    Buv = np.zeros((num_pts_u, num_pts_v, num_basis))
    dBdu = np.zeros((num_pts_u, num_pts_v, num_basis))
    dBdv = np.zeros((num_pts_u, num_pts_v, num_basis))
    
    for j in range(deg[1]+1):
        for i in range(deg[0]+1):
            Buv[:,:,basis_counter] = np.outer(B_u[:,i], B_v[:,j])
            dBdu[:,:,basis_counter] = np.outer(dB_u[:,i], B_v[:,j])
            dBdv[:,:,basis_counter] = np.outer(B_u[:,i], dB_v[:,j])
            basis_counter += 1
    
    return Buv, dBdu, dBdv