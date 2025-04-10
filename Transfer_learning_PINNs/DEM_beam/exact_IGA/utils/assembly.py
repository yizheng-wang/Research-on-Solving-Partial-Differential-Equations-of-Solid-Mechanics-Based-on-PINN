#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File to assembly subroutines 
"""
import numpy as np
from scipy import sparse

from utils.bernstein import bernstein_basis_2d


def gen_gauss_pts(n):
    """
    Generates a dictionary containing the Gauss points and weights in 1D

    Parameters
    ----------
    n : (int) number of Gauss points and weights

    Returns
    -------
    gauss_quad: (dict) with keys
            "nodes" : coordinates of the Gauss points on the reference interval [-1,1]
            "weights" : Gauss points weights for each node
    """
    gpt, gwt = np.polynomial.legendre.leggauss(n)
    gauss_quad = {"nodes": gpt, "weights" : gwt}
    return gauss_quad
    
    

def stiff_2D(mesh_list, a0, gauss_rule):
    """
    Aseembles the stiffness matrix K_ij = ∫ a0(x,y)∇ϕ_i(x,y)∇ϕ_j(x,y) dΩ

    Parameters
    ----------
    mesh_list : (list of IGAMesh2D) multipatch mesh
    a0 : function handle 
    gauss_rule : list of Gauss points and weights in the reference interval [-1,1]
                 (one for each direction)

    Returns
    -------
    stiff : the stiffness matrix
    """
    # Evaluate the basis functions and derivatives at the gauss points
    pts_u = gauss_rule[0]["nodes"]
    pts_v = gauss_rule[1]["nodes"]
    wgts_u = gauss_rule[0]["weights"]
    wgts_v = gauss_rule[1]["weights"]
    Buv, dBdu, dBdv = bernstein_basis_2d(pts_u, pts_v, mesh_list[0].deg)
    num_gauss_u = len(pts_u)
    num_gauss_v = len(pts_v)
    
    # Allocate memory for the triplet arrays
    index_counter = 0
    for i_patch in range(len(mesh_list)):
        for i_elem in range(mesh_list[i_patch].num_elem):
            num_nodes = len(mesh_list[i_patch].elem_node_global[i_elem])
            index_counter += num_nodes**2
        
    II = np.zeros(index_counter, dtype=int)
    JJ = np.zeros(index_counter, dtype=int)
    S = np.zeros(index_counter)
    
    index_counter = 0
    domain_area = 0.
    for i_patch in range(len(mesh_list)):
        for i_elem in range(mesh_list[i_patch].num_elem):
            u_min = mesh_list[i_patch].elem_vertex[i_elem, 0]
            u_max = mesh_list[i_patch].elem_vertex[i_elem, 2]
            v_min = mesh_list[i_patch].elem_vertex[i_elem, 1]
            v_max = mesh_list[i_patch].elem_vertex[i_elem, 3]
            jac_ref_par = (u_max-u_min)*(v_max-v_min)/4
            
            # compute the rational spline basis
            local_nodes = mesh_list[i_patch].elem_node[i_elem]
            global_nodes = mesh_list[i_patch].elem_node_global[i_elem]
            num_nodes = len(local_nodes)
            cpts = mesh_list[i_patch].cpts[0:2, local_nodes]
            wgts = mesh_list[i_patch].wgts[local_nodes]
            local_stiff = np.zeros((num_nodes, num_nodes))
                        
            for j_gauss in range(num_gauss_v):
                for i_gauss in range(num_gauss_u):
                    # compute the B-spline basis functions and derivatives with 
                    # Bezier extraction
                    N_mat = mesh_list[i_patch].C[i_elem]@Buv[i_gauss, j_gauss, :]
                    dN_du = mesh_list[i_patch].C[i_elem]@dBdu[i_gauss, j_gauss,
                                                              :]*2/(u_max-u_min)
                    dN_dv = mesh_list[i_patch].C[i_elem]@dBdv[i_gauss, j_gauss,
                                                              :]*2/(v_max-v_min)
                                        
                    # compute the rational basis
                    RR = N_mat * wgts
                    dRdu = dN_du * wgts
                    dRdv = dN_dv * wgts
                    w_sum = np.sum(RR)
                    dw_xi = np.sum(dRdu)
                    dw_eta = np.sum(dRdv)
                    
                    dRdu = dRdu/w_sum - RR*dw_xi/w_sum**2
                    dRdv = dRdv/w_sum - RR*dw_eta/w_sum**2
                    
                    # compute the solution w.r.t. the physical space
                    dR = np.stack((dRdu, dRdv))
                    dxdxi = dR @ cpts.transpose()
                    dR = np.linalg.solve(dxdxi, dR)
                    jac_par_phys = np.linalg.det(dxdxi)
                    
                    RR /= w_sum
                    phys_pt = cpts@RR
                    #print("phys_pt = ", phys_pt)
                    #plt.scatter(phys_pt[0], phys_pt[1], s=0.5, c="black")
                    
                    local_area = jac_par_phys * jac_ref_par * wgts_u[i_gauss] * \
                                   wgts_v[j_gauss]
                    local_stiff += local_area*a0(phys_pt[0], phys_pt[1]) * \
                                (dR.transpose()@dR)
                    domain_area += local_area
                                
            II[index_counter:index_counter+num_nodes**2] = np.tile(global_nodes, num_nodes)
            JJ[index_counter:index_counter+num_nodes**2] = np.repeat(global_nodes, num_nodes)
            S[index_counter:index_counter+num_nodes**2] = np.reshape(local_stiff, num_nodes**2)
            index_counter += num_nodes**2
    
    print("domain area = ", domain_area)
    stiff = sparse.coo_matrix((S, (II, JJ)))    
    return stiff

def mass_2D(mesh_list, a1, gauss_rule):
    """
    Assembles the mass matrix M_ij = ∫ a1(x)ϕ_i(x,y)ϕ_j(x) dΩ

    

    Parameters
    ----------
    mesh_list : (list of IGAMesh2D) multipatch mesh
    a1 : coefficient function
    gauss_rule : list of Gauss points and weights in the reference interval [-1,1]
                 (one for each direction)

    Returns
    -------
    mass : (sparse COO 2D array)
          the mass matrix

    """
    # Evaluate the basis functions and derivatives at the gauss points
    pts_u = gauss_rule[0]["nodes"]
    pts_v = gauss_rule[1]["nodes"]
    wgts_u = gauss_rule[0]["weights"]
    wgts_v = gauss_rule[1]["weights"]
    Buv, dBdu, dBdv = bernstein_basis_2d(pts_u, pts_v, mesh_list[0].deg)
    num_gauss_u = len(pts_u)
    num_gauss_v = len(pts_v)
    
    # Allocate memory for the triplet arrays
    index_counter = 0
    for i_patch in range(len(mesh_list)):
        for i_elem in range(mesh_list[i_patch].num_elem):
            num_nodes = len(mesh_list[i_patch].elem_node_global[i_elem])
            index_counter += num_nodes**2
        
    II = np.zeros(index_counter, dtype=int)
    JJ = np.zeros(index_counter, dtype=int)
    M = np.zeros(index_counter, dtype=complex)
    
    index_counter = 0
    domain_area = 0.
    for i_patch in range(len(mesh_list)):
        for i_elem in range(mesh_list[i_patch].num_elem):
            u_min = mesh_list[i_patch].elem_vertex[i_elem, 0]
            u_max = mesh_list[i_patch].elem_vertex[i_elem, 2]
            v_min = mesh_list[i_patch].elem_vertex[i_elem, 1]
            v_max = mesh_list[i_patch].elem_vertex[i_elem, 3]
            jac_ref_par = (u_max-u_min)*(v_max-v_min)/4
            
            # compute the rational spline basis
            local_nodes = mesh_list[i_patch].elem_node[i_elem]
            global_nodes = mesh_list[i_patch].elem_node_global[i_elem]
            num_nodes = len(local_nodes)
            cpts = mesh_list[i_patch].cpts[0:2, local_nodes]
            wgts = mesh_list[i_patch].wgts[local_nodes]
            local_mass = np.zeros((num_nodes, num_nodes), dtype=np.complex128)
                        
            for j_gauss in range(num_gauss_v):
                for i_gauss in range(num_gauss_u):
                    # compute the B-spline basis functions and derivatives with 
                    # Bezier extraction
                    N_mat = mesh_list[i_patch].C[i_elem]@Buv[i_gauss, j_gauss, :]
                    dN_du = mesh_list[i_patch].C[i_elem]@dBdu[i_gauss, j_gauss,
                                                              :]*2/(u_max-u_min)
                    dN_dv = mesh_list[i_patch].C[i_elem]@dBdv[i_gauss, j_gauss,
                                                              :]*2/(v_max-v_min)
                                        
                    # compute the rational basis
                    RR = N_mat * wgts
                    dRdu = dN_du * wgts
                    dRdv = dN_dv * wgts
                    w_sum = np.sum(RR)
                    dw_xi = np.sum(dRdu)
                    dw_eta = np.sum(dRdv)
                    
                    dRdu = dRdu/w_sum - RR*dw_xi/w_sum**2
                    dRdv = dRdv/w_sum - RR*dw_eta/w_sum**2
                    
                    # compute the solution w.r.t. the physical space
                    dR = np.stack((dRdu, dRdv))
                    dxdxi = dR @ cpts.transpose()
                    dR = np.linalg.solve(dxdxi, dR)
                    jac_par_phys = np.linalg.det(dxdxi)
                    
                    RR /= w_sum
                    phys_pt = cpts@RR
                    #print("phys_pt = ", phys_pt)
                    #plt.scatter(phys_pt[0], phys_pt[1], s=0.5, c="black")
                    
                    local_area = jac_par_phys * jac_ref_par * wgts_u[i_gauss] * \
                                   wgts_v[j_gauss]
                    local_mass += local_area*a1(phys_pt[0], phys_pt[1]) * \
                                np.outer(RR,RR)
                    domain_area += local_area
                                
            II[index_counter:index_counter+num_nodes**2] = np.tile(global_nodes, num_nodes)
            JJ[index_counter:index_counter+num_nodes**2] = np.repeat(global_nodes, num_nodes)
            M[index_counter:index_counter+num_nodes**2] = np.reshape(local_mass, num_nodes**2)
            index_counter += num_nodes**2
    
    print("domain area = ", domain_area)
    mass = sparse.coo_matrix((M, (II, JJ)))            
    
    return mass
    

def rhs_2D(mesh_list, f, gauss_rule, size_basis):
    """
    Assembles the RHS vector corresponding to the body force
    RHS[i]=∫_Ω ϕ_i(x,y)*f(x,y) dΩ

    Parameters
    ----------
    mesh_list : (list of IGAMesh2D) multipatch mesh
    f : (function handle) RHS function
    gauss_rule : list of Gauss points and weights in the reference interval [-1,1]
                 (one for each direction)
    size_basis : ( int ) dimension of the global basis                
                 

    Returns
    -------
    rhs : RHS vector

    """
    # Evaluate the basis functions and derivatives at the gauss points
    pts_u = gauss_rule[0]["nodes"]
    pts_v = gauss_rule[1]["nodes"]
    wgts_u = gauss_rule[0]["weights"]
    wgts_v = gauss_rule[1]["weights"]
    Buv, dBdu, dBdv = bernstein_basis_2d(pts_u, pts_v, mesh_list[0].deg)
    num_gauss_u = len(pts_u)
    num_gauss_v = len(pts_v)
    
    # Initialize RHS vector and domain area
    rhs =np.zeros(size_basis, dtype=np.complex128)
    domain_area = 0

    for i_patch in range(len(mesh_list)):
        for i_elem in range(mesh_list[i_patch].num_elem):
            u_min = mesh_list[i_patch].elem_vertex[i_elem, 0]
            u_max = mesh_list[i_patch].elem_vertex[i_elem, 2]
            v_min = mesh_list[i_patch].elem_vertex[i_elem, 1]
            v_max = mesh_list[i_patch].elem_vertex[i_elem, 3]
            jac_ref_par = (u_max-u_min)*(v_max-v_min)/4
            
            # compute the rational spline basis
            local_nodes = mesh_list[i_patch].elem_node[i_elem]
            global_nodes = mesh_list[i_patch].elem_node_global[i_elem]
            num_nodes = len(local_nodes)
            cpts = mesh_list[i_patch].cpts[0:2, local_nodes]
            wgts = mesh_list[i_patch].wgts[local_nodes]
            
            local_rhs = np.zeros(num_nodes, dtype=np.complex128)
            for j_gauss in range(num_gauss_v):
                for i_gauss in range(num_gauss_u):
                    # compute the B-spline basis functions and derivatives with 
                    # Bezier extraction
                    N_mat = mesh_list[i_patch].C[i_elem]@Buv[i_gauss, j_gauss, :]
                    dN_du = mesh_list[i_patch].C[i_elem]@dBdu[i_gauss, j_gauss, :]*2/(u_max-u_min)
                    dN_dv = mesh_list[i_patch].C[i_elem]@dBdv[i_gauss, j_gauss, :]*2/(v_max-v_min)
                                                    
                    # compute the rational basis
                    RR = N_mat * wgts
                    dRdu = dN_du * wgts
                    dRdv = dN_dv * wgts
                    w_sum = np.sum(RR)
                    dw_xi = np.sum(dRdu)
                    dw_eta = np.sum(dRdv)
                    
                    dRdu = dRdu/w_sum - RR*dw_xi/w_sum**2
                    dRdv = dRdv/w_sum - RR*dw_eta/w_sum**2
                    
                    # compute the solution w.r.t. the physical space
                    dR = np.stack((dRdu, dRdv))
                    dxdxi = dR @ cpts.transpose()
                    dR = np.linalg.solve(dxdxi, dR)
                    jac_par_phys = np.linalg.det(dxdxi)
                    
                    RR /= w_sum
                    phys_pt = cpts@RR
                    
                    local_area = jac_par_phys * jac_ref_par * wgts_u[i_gauss] * \
                                   wgts_v[j_gauss]
                                   
                    local_rhs += local_area * f(phys_pt[0], phys_pt[1]) * RR                
                    domain_area += local_area
            rhs[global_nodes] += local_rhs 
        
    print("domain area = ", domain_area)
    return rhs


def stiff_elast_2D(mesh_list, Cmat, gauss_rule):
    """
    Aseembles the stiffness matrix Kₑ = ∫ B^T*C*B dΩ

    Parameters
    ----------
    mesh_list : (list of IGAMesh2D) multipatch mesh
    Cmat : (2D array) elasticity matrix
    gauss_rule : list of Gauss points and weights in the reference interval [-1,1]
                 (one for each direction)

    Returns
    -------
    stiff : the stiffness matrix
    """
    # Evaluate the basis functions and derivatives at the gauss points
    pts_u = gauss_rule[0]["nodes"]
    pts_v = gauss_rule[1]["nodes"]
    wgts_u = gauss_rule[0]["weights"]
    wgts_v = gauss_rule[1]["weights"]
    Buv, dBdu, dBdv = bernstein_basis_2d(pts_u, pts_v, mesh_list[0].deg)
    num_gauss_u = len(pts_u)
    num_gauss_v = len(pts_v)
    
    # Allocate memory for the triplet arrays
    index_counter = 0
    for i_patch in range(len(mesh_list)):
        for i_elem in range(mesh_list[i_patch].num_elem):
            num_nodes = len(mesh_list[i_patch].elem_node_global[i_elem])
            index_counter += 4*num_nodes**2
        
    II = np.zeros(index_counter, dtype=int)
    JJ = np.zeros(index_counter, dtype=int)
    S = np.zeros(index_counter)
    
    index_counter = 0
    domain_area = 0.
    for i_patch in range(len(mesh_list)):
        for i_elem in range(mesh_list[i_patch].num_elem):
            u_min = mesh_list[i_patch].elem_vertex[i_elem, 0]
            u_max = mesh_list[i_patch].elem_vertex[i_elem, 2]
            v_min = mesh_list[i_patch].elem_vertex[i_elem, 1]
            v_max = mesh_list[i_patch].elem_vertex[i_elem, 3]
            jac_ref_par = (u_max-u_min)*(v_max-v_min)/4
            
            # compute the rational spline basis
            local_nodes = mesh_list[i_patch].elem_node[i_elem]
            global_nodes = mesh_list[i_patch].elem_node_global[i_elem]            
            num_nodes = len(local_nodes)
            global_nodes_xy = np.reshape(np.stack((2*global_nodes, 
                                        2*global_nodes+1),axis=1),2*num_nodes)
            cpts = mesh_list[i_patch].cpts[0:2, local_nodes]
            wgts = mesh_list[i_patch].wgts[local_nodes]
            local_stiff = np.zeros((2*num_nodes, 2*num_nodes))
            
            B = np.zeros((2*num_nodes,3))           
            for j_gauss in range(num_gauss_v):
                for i_gauss in range(num_gauss_u):
                    # compute the B-spline basis functions and derivatives with 
                    # Bezier extraction
                    N_mat = mesh_list[i_patch].C[i_elem]@Buv[i_gauss, j_gauss, :]
                    dN_du = mesh_list[i_patch].C[i_elem]@dBdu[i_gauss, j_gauss,
                                                              :]*2/(u_max-u_min)
                    dN_dv = mesh_list[i_patch].C[i_elem]@dBdv[i_gauss, j_gauss,
                                                              :]*2/(v_max-v_min)
                                        
                    # compute the rational basis
                    RR = N_mat * wgts
                    dRdu = dN_du * wgts
                    dRdv = dN_dv * wgts
                    w_sum = np.sum(RR)
                    dw_xi = np.sum(dRdu)
                    dw_eta = np.sum(dRdv)
                    
                    dRdu = dRdu/w_sum - RR*dw_xi/w_sum**2
                    dRdv = dRdv/w_sum - RR*dw_eta/w_sum**2
                    
                    # compute the solution w.r.t. the physical space
                    dR = np.stack((dRdu, dRdv))
                    dxdxi = dR @ cpts.transpose()
                    dR = np.linalg.solve(dxdxi, dR)
                    jac_par_phys = np.linalg.det(dxdxi)
                    
                    RR /= w_sum
                    
                    B[0:2*num_nodes-1:2,0] = dR[0,:]
                    B[1:2*num_nodes:2,1] = dR[1,:]
                    B[0:2*num_nodes-1:2,2] = dR[1,:]
                    B[1:2*num_nodes:2,2] = dR[0,:]

                    
                    local_area = jac_par_phys * jac_ref_par * wgts_u[i_gauss] * \
                                   wgts_v[j_gauss]
                    local_stiff += local_area * (B@Cmat@B.transpose())
                    domain_area += local_area
                                
            II[index_counter:index_counter+4*num_nodes**2] = np.tile(global_nodes_xy, 2*num_nodes)
            JJ[index_counter:index_counter+4*num_nodes**2] = np.repeat(global_nodes_xy, 2*num_nodes)
            S[index_counter:index_counter+4*num_nodes**2] = np.reshape(local_stiff, 4*num_nodes**2)
            index_counter += 4*num_nodes**2
    
    print("domain area = ", domain_area)
    stiff = sparse.coo_matrix((S, (II, JJ)))    
    return stiff


def stiff_elast_FGM_2D(mesh_list, material, gauss_rule):
    """
    Aseembles the stiffness matrix Kₑ = ∫ B^T*C*B dΩ

    Parameters
    ----------
    mesh_list : (list of IGAMesh2D) multipatch mesh
    Cmat : (2D array) elasticity matrix
    gauss_rule : list of Gauss points and weights in the reference interval [-1,1]
                 (one for each direction)

    Returns
    -------
    stiff : the stiffness matrix
    """
    # Evaluate the basis functions and derivatives at the gauss points
    pts_u = gauss_rule[0]["nodes"]
    pts_v = gauss_rule[1]["nodes"]
    wgts_u = gauss_rule[0]["weights"]
    wgts_v = gauss_rule[1]["weights"]
    Buv, dBdu, dBdv = bernstein_basis_2d(pts_u, pts_v, mesh_list[0].deg)
    num_gauss_u = len(pts_u)
    num_gauss_v = len(pts_v)
    
    # Allocate memory for the triplet arrays
    index_counter = 0
    for i_patch in range(len(mesh_list)):
        for i_elem in range(mesh_list[i_patch].num_elem):
            num_nodes = len(mesh_list[i_patch].elem_node_global[i_elem])
            index_counter += 4*num_nodes**2
        
    II = np.zeros(index_counter, dtype=int)
    JJ = np.zeros(index_counter, dtype=int)
    S = np.zeros(index_counter)
    
    n = 0
    E_modulus = np.zeros(mesh_list[0].num_elem*num_gauss_u*num_gauss_v)
    coord = np.zeros([mesh_list[0].num_elem*num_gauss_u*num_gauss_v, 2])    
    
    index_counter = 0
    domain_area = 0.
    for i_patch in range(len(mesh_list)):
        for i_elem in range(mesh_list[i_patch].num_elem):
            u_min = mesh_list[i_patch].elem_vertex[i_elem, 0]
            u_max = mesh_list[i_patch].elem_vertex[i_elem, 2]
            v_min = mesh_list[i_patch].elem_vertex[i_elem, 1]
            v_max = mesh_list[i_patch].elem_vertex[i_elem, 3]
            jac_ref_par = (u_max-u_min)*(v_max-v_min)/4
            
            # compute the rational spline basis
            local_nodes = mesh_list[i_patch].elem_node[i_elem]
            global_nodes = mesh_list[i_patch].elem_node_global[i_elem]            
            num_nodes = len(local_nodes)
            global_nodes_xy = np.reshape(np.stack((2*global_nodes, 
                                        2*global_nodes+1),axis=1),2*num_nodes)
            cpts = mesh_list[i_patch].cpts[0:2, local_nodes]
            wgts = mesh_list[i_patch].wgts[local_nodes]
            local_stiff = np.zeros((2*num_nodes, 2*num_nodes))
            
            B = np.zeros((2*num_nodes,3))           
            for j_gauss in range(num_gauss_v):
                for i_gauss in range(num_gauss_u):
                    # compute the B-spline basis functions and derivatives with 
                    # Bezier extraction
                    N_mat = mesh_list[i_patch].C[i_elem]@Buv[i_gauss, j_gauss, :]
                    dN_du = mesh_list[i_patch].C[i_elem]@dBdu[i_gauss, j_gauss,
                                                              :]*2/(u_max-u_min)
                    dN_dv = mesh_list[i_patch].C[i_elem]@dBdv[i_gauss, j_gauss,
                                                              :]*2/(v_max-v_min)
                                        
                    # compute the rational basis
                    RR = N_mat * wgts
                    dRdu = dN_du * wgts
                    dRdv = dN_dv * wgts
                    w_sum = np.sum(RR)
                    dw_xi = np.sum(dRdu)
                    dw_eta = np.sum(dRdv)
                    
                    dRdu = dRdu/w_sum - RR*dw_xi/w_sum**2
                    dRdv = dRdv/w_sum - RR*dw_eta/w_sum**2
                    
                    # compute the solution w.r.t. the physical space
                    dR = np.stack((dRdu, dRdv))
                    dxdxi = dR @ cpts.transpose()
                    dR = np.linalg.solve(dxdxi, dR)
                    jac_par_phys = np.linalg.det(dxdxi)
                    
                    RR /= w_sum
                    
                    phys_pt = cpts@RR
                    
                    E_modulus[n] = material.elasticity(phys_pt, mesh_list[i_patch])
                    Cmat_FGM = E_modulus[n]*material.Cmat
                    
                    coord[n, :] = phys_pt
                    
                    #print(n)
                    #rint(phys_pt)
                    #print(E_modulus[n])
                    n = n+1

                    B[0:2*num_nodes-1:2,0] = dR[0,:]
                    B[1:2*num_nodes:2,1] = dR[1,:]
                    B[0:2*num_nodes-1:2,2] = dR[1,:]
                    B[1:2*num_nodes:2,2] = dR[0,:]

                    
                    local_area = jac_par_phys * jac_ref_par * wgts_u[i_gauss] * \
                                   wgts_v[j_gauss]
                    local_stiff += local_area * (B@Cmat_FGM@B.transpose())
                    domain_area += local_area
                                
            II[index_counter:index_counter+4*num_nodes**2] = np.tile(global_nodes_xy, 2*num_nodes)
            JJ[index_counter:index_counter+4*num_nodes**2] = np.repeat(global_nodes_xy, 2*num_nodes)
            S[index_counter:index_counter+4*num_nodes**2] = np.reshape(local_stiff, 4*num_nodes**2)
            index_counter += 4*num_nodes**2
    
    print("domain area = ", domain_area)
    stiff = sparse.coo_matrix((S, (II, JJ)))    
    return stiff, E_modulus, coord