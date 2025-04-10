#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class and functions for boundary conditions
"""
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import SparseEfficiencyWarning

from utils.bernstein import bernstein_basis_2d



class boundary2D:
    """
    Class for a boundary condition on a single edge in parameter space
    
    Input
    ------
    bnd_type : (string) type of boundary conditions (e.g. "Dirichlet")
    patch_index : (int) index of the patch in a multipatch mesh
    side : (string) side of the parameters space (e.g. "down", "right", "up",
                                                          "left")
    op_value : (function) value of the boundary function
    alpha: (number) parameter value for Robin boundary condition (\alpha u  + u')
    """
    def __init__(self, bnd_type, patch_index, side, op_value, alpha=0.):
        self.type = bnd_type
        self.patch_index = patch_index
        self.side = side
        self.op_value = op_value
        self.alpha = alpha
        

def applyBC2D(mesh_list, bound_cond, lhs, rhs, quad_rule=None):
    """
    Applies the boundary conditions to a linear system for scalar problems
    TODO: At the moment, only homogeneous Dirichlet B.C. are implemented
        
    Parameters
    ----------
    mesh_list :(list of IGAMesh2D) multipatch mesh
    bound_cond : (list of boundary2D) boundary conditions
    lhs : (2D array) stiffness matrix
    rhs : (1D array) rhs vector
    quad_rule : (optional, list of dicts) list of Gauss points and weights in 
                the reference interval [-1,1] (one for each parametric direction)                

    Returns
    -------
    lhs: updated stiffness matrix
    rhs: updated rhs vector
    """
    bcdof = []    
    
    for i in range(len(bound_cond)):
        patch_index = bound_cond[i].patch_index
        side = bound_cond[i].side
        if bound_cond[i].type=="Dirichlet":        
            bcdof += mesh_list[patch_index].bcdof_global[side]
        elif bound_cond[i].type=="Neumann":            
            op_val = bound_cond[i].op_value
            if side=="down" or side=="up":
                quad_rule_side = quad_rule[0]
            elif side=="left" or side=="right":
                quad_rule_side = quad_rule[1]
            rhs = applyNeumannScalar2D(mesh_list[patch_index], rhs, side, 
                                       quad_rule_side, op_val)
        elif bound_cond[i].type=="Robin":
            op_val = bound_cond[i].op_value
            alpha = bound_cond[i].alpha
            if side=="down" or side=="up":
                quad_rule_side = quad_rule[0]
            elif side=="left" or side=="right":
                quad_rule_side = quad_rule[1]
            lhs, rhs = applyRobinScalar2D(mesh_list[patch_index], lhs, rhs, side, 
                                       quad_rule_side, alpha, op_val)
            
    if len(bcdof)>0:        
        bcdof = np.unique(bcdof)
        bcval = np.zeros_like(bcdof)
        rhs = rhs-lhs[:, bcdof]*bcval
        rhs[bcdof] = bcval
        #TODO: fix this warning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',SparseEfficiencyWarning)
            lhs[bcdof, :] = 0
            lhs[:, bcdof] = 0
            lhs[bcdof, bcdof] = 1.
    
    return lhs, rhs

def applyNeumannScalar2D(mesh, rhs, side, quad_rule_side, op_val):
    """
    Applies the Neumann boundary conditins on the given patch and side for
    scalar problems and updates the rhs vector

    Parameters
    ----------
    mesh :(IGAMesh2D) patch mesh
    rhs : (1D array) rhs vector
    side : (string) either "down" (v=0), "right" (u=1), "up" (v=1), "left" (u=0)
    quad_rule_side : (dict) auss points and weights in the reference
                interval [-1,1]
    op_val : (function) function for the flux

    Returns
    -------
    rhs : (1D array)
        updated rhs vector

    """
    index_all = mesh._get_boundary_indices()
    if side=="right":
        pts_u = np.array([1.])
        pts_v = quad_rule_side["nodes"]
    elif side=="left":
        pts_u = np.array([-1.])
        pts_v = quad_rule_side["nodes"]
    elif side=="down":
        pts_u = quad_rule_side["nodes"]
        pts_v = np.array([-1.])
    elif side=="up":
        pts_u = quad_rule_side["nodes"]
        pts_v = np.array([1.])
    sctr = index_all[side]
    # Form the 2D tensor product of the basis functions
    Buv, dBdu, dBdv = bernstein_basis_2d(pts_u, pts_v, mesh.deg)
    
    # Evaluate the Neumann integral on each element
    boundary_length = 0.
    for i_elem in mesh.elem[side]:
        u_min = mesh.elem_vertex[i_elem, 0]
        u_max = mesh.elem_vertex[i_elem, 2]
        v_min = mesh.elem_vertex[i_elem, 1]
        v_max = mesh.elem_vertex[i_elem, 3]
        if side=="right" or side=="left":
            jac_par_ref = (v_max - v_min)/2
        else:
            jac_par_ref = (u_max - u_min)/2
            
        # compute the rational spline basis
        local_nodes = mesh.elem_node[i_elem]
        global_nodes = mesh.elem_node_global[i_elem]
        
        cpts = mesh.cpts[0:2, local_nodes]
        wgts = mesh.wgts[local_nodes]
        local_rhs = np.zeros(len(sctr))
        for i_gauss in range(len(quad_rule_side["nodes"])):
            # compute the (B-)spline basis functions and derivatives with Bezier extraction
            if side=="right" or side=="left":
                N_mat = mesh.C[i_elem]@Buv[0, i_gauss, :]
                dN_du = mesh.C[i_elem]@dBdu[0, i_gauss, :]*2/(u_max-u_min)
                dN_dv = mesh.C[i_elem]@dBdv[0, i_gauss, :]*2/(v_max-v_min)
            else:
                N_mat = mesh.C[i_elem]@Buv[i_gauss, 0, :]
                dN_du = mesh.C[i_elem]@dBdu[i_gauss, 0, :]*2/(u_max-u_min)
                dN_dv = mesh.C[i_elem]@dBdv[i_gauss, 0, :]*2/(v_max-v_min)
            
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
            
            # Jacobian of face mapping
            if side=="right" or side=="left":
                e_jac = dxdxi[1,0]**2 + dxdxi[1,1]**2
            else:
                e_jac = dxdxi[0,0]**2 + dxdxi[0,1]**2
            
            jac_par_phys = np.sqrt(e_jac)
            dR = np.linalg.solve(dxdxi, dR)
            RR /= w_sum
            phys_pt = cpts@RR
            g_func = op_val(phys_pt[0], phys_pt[1])
            local_length = jac_par_phys * jac_par_ref * quad_rule_side["weights"][i_gauss]
            local_rhs += RR[sctr]*g_func*local_length            
            boundary_length += local_length
        rhs[global_nodes[sctr]] += local_rhs
    print("The boundary length is ", boundary_length)
    return rhs


def applyRobinScalar2D(mesh, lhs, rhs, side, quad_rule_side, alpha, op_val):
    """
    Applies the Robin boundary conditins on the given patch and side for
    scalar problems and updates the rhs vector

    Parameters
    ----------
    mesh :(IGAMesh2D) patch mesh
    lhs : (2D array) lhs vector
    rhs : (1D array) rhs vector
    side : (string) either "down" (v=0), "right" (u=1), "up" (v=1), "left" (u=0)
    quad_rule_side : (dict) auss points and weights in the reference
                interval [-1,1]
    alpha: (float) coefficient for Robin boundary conditions
    op_val : (function) function for the flux

    Returns
    -------
    rhs : (1D array)
        updated rhs vector

    """
    index_all = mesh._get_boundary_indices()
    num_nodes_edge = len(quad_rule_side["nodes"])
    if side=="right":
        pts_u = np.array([1.])
        pts_v = quad_rule_side["nodes"]
    elif side=="left":
        pts_u = np.array([-1.])
        pts_v = quad_rule_side["nodes"]
    elif side=="down":
        pts_u = quad_rule_side["nodes"]
        pts_v = np.array([-1.])
    elif side=="up":
        pts_u = quad_rule_side["nodes"]
        pts_v = np.array([1.])
    sctr = index_all[side]
    num_nodes = len(sctr)
    
    # Allocate memory for the triplet arrays
    index_counter = len(mesh.elem[side])*len(sctr)**2
    II = np.zeros(index_counter, dtype=int)
    JJ = np.zeros(index_counter, dtype=int)
    M = np.zeros(index_counter, dtype=np.complex128)
    
    index_counter = 0
    
    # Form the 2D tensor product of the basis functions
    Buv, dBdu, dBdv = bernstein_basis_2d(pts_u, pts_v, mesh.deg)
    
    # Evaluate the Neumann integral on each element
    boundary_length = 0.
    for i_elem in mesh.elem[side]:
        u_min = mesh.elem_vertex[i_elem, 0]
        u_max = mesh.elem_vertex[i_elem, 2]
        v_min = mesh.elem_vertex[i_elem, 1]
        v_max = mesh.elem_vertex[i_elem, 3]
        if side=="right" or side=="left":
            jac_par_ref = (v_max - v_min)/2
        else:
            jac_par_ref = (u_max - u_min)/2
            
        # compute the rational spline basis
        local_nodes = mesh.elem_node[i_elem]
        global_nodes = mesh.elem_node_global[i_elem]
        
        cpts = mesh.cpts[0:2, local_nodes]
        wgts = mesh.wgts[local_nodes]
        local_rhs = np.zeros(len(sctr))
        local_bnd_mass = np.zeros((num_nodes_edge, num_nodes_edge), dtype=np.complex128)
        for i_gauss in range(len(quad_rule_side["nodes"])):
            # compute the (B-)spline basis functions and derivatives with Bezier extraction
            if side=="right" or side=="left":
                N_mat = mesh.C[i_elem]@Buv[0, i_gauss, :]
                dN_du = mesh.C[i_elem]@dBdu[0, i_gauss, :]*2/(u_max-u_min)
                dN_dv = mesh.C[i_elem]@dBdv[0, i_gauss, :]*2/(v_max-v_min)
            else:
                N_mat = mesh.C[i_elem]@Buv[i_gauss, 0, :]
                dN_du = mesh.C[i_elem]@dBdu[i_gauss, 0, :]*2/(u_max-u_min)
                dN_dv = mesh.C[i_elem]@dBdv[i_gauss, 0, :]*2/(v_max-v_min)
            
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
            
            # Jacobian of face mapping
            if side=="right" or side=="left":
                e_jac = dxdxi[1,0]**2 + dxdxi[1,1]**2
            else:
                e_jac = dxdxi[0,0]**2 + dxdxi[0,1]**2
            
            jac_par_phys = np.sqrt(e_jac)
            dR = np.linalg.solve(dxdxi, dR)
            RR /= w_sum
            phys_pt = cpts@RR
            g_func = op_val(phys_pt[0], phys_pt[1])
            local_length = jac_par_phys * jac_par_ref * quad_rule_side["weights"][i_gauss]
            local_rhs += RR[sctr]*g_func*local_length         
            local_bnd_mass += np.outer(RR[sctr], RR[sctr])*alpha*local_length
            boundary_length += local_length
        rhs[global_nodes[sctr]] += local_rhs
        II[index_counter:index_counter+num_nodes**2] = np.tile(global_nodes[sctr], num_nodes)
        JJ[index_counter:index_counter+num_nodes**2] = np.repeat(global_nodes[sctr], num_nodes)
        M[index_counter:index_counter+num_nodes**2] = np.reshape(local_bnd_mass, num_nodes**2)
        index_counter += num_nodes**2        
    size_basis = lhs.shape[0]
    bnd_mass = sparse.coo_matrix((M, (II, JJ)), shape=(size_basis, size_basis))    
    lhs += bnd_mass
    print("The boundary length is ", boundary_length)
    return lhs, rhs
    
    

def applyNeumannElast2D(mesh, rhs, side, quad_rule_side, op_val):
    """
    Applies the Neumann boundary conditions on the given patch and side and 
    updates the rhs vector

    Parameters
    ----------
    mesh : (IGAMesh2D) patch mesh
    rhs : (1D array) rhs vector
    side : (string) either "down" (v=0), "right" (u=1), "up" (v=1), "left" (u=0)
    quad_rule_side : (dict) auss points and weights in the reference
                interval [-1,1]
    op_val : (function) function for the x and y component of the traction

    Returns
    -------
     rhs: updated rhs vector

    """
    index_all = mesh._get_boundary_indices()
    if side=="right":
        pts_u = np.array([1.])
        pts_v = quad_rule_side["nodes"]
    elif side=="left":
        pts_u = np.array([-1.])
        pts_v = quad_rule_side["nodes"]
    elif side=="down":
        pts_u = quad_rule_side["nodes"]
        pts_v = np.array([-1.])
    elif side=="up":
        pts_u = quad_rule_side["nodes"]
        pts_v = np.array([1.])
    sctr = index_all[side]
    #print("pts_u = ", pts_u)
    #print("pts_v = ", pts_v)
    # Form the 2D tensor product of the basis functions
    Buv, dBdu, dBdv = bernstein_basis_2d(pts_u, pts_v, mesh.deg)
    
    # Evaluate the Neumann integral on each element
    boundary_length = 0.
    for i_elem in mesh.elem[side]:
        u_min = mesh.elem_vertex[i_elem, 0]
        u_max = mesh.elem_vertex[i_elem, 2]
        v_min = mesh.elem_vertex[i_elem, 1]
        v_max = mesh.elem_vertex[i_elem, 3]
        if side=="right" or side=="left":
            jac_par_ref = (v_max - v_min)/2
        else:
            jac_par_ref = (u_max - u_min)/2
            
        # compute the rational spline basis
        local_nodes = mesh.elem_node[i_elem]
        global_nodes = mesh.elem_node_global[i_elem]
        global_nodes_xy = np.reshape(np.stack((2*global_nodes[sctr], 
                                        2*global_nodes[sctr]+1),axis=1),2*len(sctr))
        cpts = mesh.cpts[0:2, local_nodes]
        wgts = mesh.wgts[local_nodes]
        local_rhs = np.zeros(2*len(sctr))
        for i_gauss in range(len(quad_rule_side["nodes"])):
            # compute the (B-)spline basis functions and derivatives with Bezier extraction
            if side=="right" or side=="left":
                N_mat = mesh.C[i_elem]@Buv[0, i_gauss, :]
                dN_du = mesh.C[i_elem]@dBdu[0, i_gauss, :]*2/(u_max-u_min)
                dN_dv = mesh.C[i_elem]@dBdv[0, i_gauss, :]*2/(v_max-v_min)
            else:
                N_mat = mesh.C[i_elem]@Buv[i_gauss, 0, :]
                dN_du = mesh.C[i_elem]@dBdu[i_gauss, 0, :]*2/(u_max-u_min)
                dN_dv = mesh.C[i_elem]@dBdv[i_gauss, 0, :]*2/(v_max-v_min)
            
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
            
            # Jacobian of face mapping
            if side=="right" or side=="left":
                e_jac = dxdxi[1,0]**2 + dxdxi[1,1]**2
            else:
                e_jac = dxdxi[0,0]**2 + dxdxi[0,1]**2
                
            # Computation of normal
            if side=="down":
                nor = [dxdxi[0,1], -dxdxi[0,0]]
            elif side=="right":
                nor = [dxdxi[1,1], -dxdxi[1,0]]
            elif side=="up":
                nor = [-dxdxi[0,1], dxdxi[0,0]]
            elif side=="left":
                nor = [-dxdxi[1,1], dxdxi[1,0]]
            tmp = np.hypot(nor[0], nor[1])
            normal = nor/tmp

            jac_par_phys = np.sqrt(e_jac)
            dR = np.linalg.solve(dxdxi, dR)
            RR /= w_sum
            phys_pt = cpts@RR
            g_func = op_val(phys_pt[0], phys_pt[1], normal[0], normal[1])
            local_length = jac_par_phys * jac_par_ref * quad_rule_side["weights"][i_gauss]
            local_rhs[0:-1:2] += RR[sctr]*g_func[0]*local_length
            local_rhs[1::2] += RR[sctr]*g_func[1]*local_length
            boundary_length += local_length
        rhs[global_nodes_xy] += local_rhs
    print("The boundary length is ", boundary_length)
    return rhs
    
                    

def applyBCElast2D(mesh_list, bound_cond, lhs, rhs, quad_rule):
    """
    Applies the boundary conditions to a linear system for 2D elasticity
    TODO: At the moment, only Neumann and homogeneous Dirichlet B.C. are implemented


    Parameters
    ----------
    mesh_list : (list of IGAMesh2D) multipatch mesh
    bound_cond : (list of boundary2D) boundary conditions
    lhs : (2D array) stiffness matrix
    rhs : (1D array) rhs vector
    quad_rule : (list of dicts) list of Gauss points and weights in the reference
                interval [-1,1] (one for each parametric direction)

    Returns
    -------
    lhs: updated stiffness matrix
    rhs: updated rhs vector
    """
    #collect the dofs and values corresponding to the boundary
    bcdof = []    
    eval_pt = [0., 0.]
    for i in range(len(bound_cond)):
        patch_index = bound_cond[i].patch_index
        side = bound_cond[i].side
        if bound_cond[i].type=="Dirichlet":
            # check x-direction
            if bound_cond[i].op_value(eval_pt[0], eval_pt[1])[0]!=None:
                bcdof += [2*j for j in mesh_list[patch_index].bcdof_global[side]]
            # check y-direction
            if bound_cond[i].op_value(eval_pt[0], eval_pt[1])[1]!=None:
                bcdof += [2*j+1 for j in mesh_list[patch_index].bcdof_global[side]]
        elif bound_cond[i].type=="Neumann":
            if side=="down" or side=="up":
                quad_rule_side = quad_rule[0]
            elif side=="left" or side=="right":
                quad_rule_side = quad_rule[1]
            rhs = applyNeumannElast2D(mesh_list[patch_index], rhs, side,
                                      quad_rule_side, bound_cond[i].op_value)
    
    bcdof = np.unique(bcdof)
    bcval = np.zeros_like(bcdof)
    rhs = rhs - lhs[:,bcdof]*bcval
    rhs[bcdof] = bcval
    #TODO: fix this warning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',SparseEfficiencyWarning)
        lhs[bcdof, :] = 0
        lhs[:, bcdof] = 0
        lhs[bcdof, bcdof] = 1.
            
    return lhs, rhs
            
        
    