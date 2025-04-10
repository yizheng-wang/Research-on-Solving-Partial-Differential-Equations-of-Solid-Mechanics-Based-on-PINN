#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for plotting, error norm computations and other post-processing tasks
"""
import numpy as np
import matplotlib.pyplot as plt
from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkQuadraticQuad

from utils.bernstein import bernstein_basis_2d


def plot_sol2D(mesh_list, sol, file_name):
    """
    Plots the computed solution to VTU file using quadratic quad (Q8)
    visualization elements for scalar 2D problems


    Parameters
    ----------
    mesh_list : (list of IGAmesh2D) multipatch mesh
    sol : (1D array) solution vector
    file_name : (string) file name (without extension)

    Returns
    -------
    None.

    """
    # Set the order of points for VTK_QUADRATIC_QUAD  w.r.t. a 3x3 grid from
    # Figure 3 of https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
    VTKOrder = [0, 2, 8, 6, 1, 5, 7, 3]

    num_pts_elem = 3
    eval_pts_u = np.linspace(-1, 1, num_pts_elem)
    eval_pts_v = np.linspace(-1, 1, num_pts_elem)
    Buv, _, _ = bernstein_basis_2d(eval_pts_u, eval_pts_v, mesh_list[0].deg)

    num_elem = 0
    for i_patch in range(len(mesh_list)):
        num_elem += mesh_list[i_patch].num_elem

    points_VTK = np.zeros((3, num_elem * 8))
    point_val_VTK = np.zeros(num_elem * 8)
    local_points = np.zeros((2, num_pts_elem ** 2))
    local_point_val = np.zeros(num_pts_elem ** 2)
    conn = np.zeros(num_elem * 8, dtype=int)
    offsets = np.zeros(num_elem, dtype=int)
    cell_types = np.zeros(num_elem, dtype=int)
    elem_counter = 0
    for i_patch in range(len(mesh_list)):
        for i_elem in range(mesh_list[i_patch].num_elem):
            local_nodes = mesh_list[i_patch].elem_node[i_elem]
            global_nodes = mesh_list[i_patch].elem_node_global[i_elem]
            cpts = mesh_list[i_patch].cpts[0:2, local_nodes]
            wgts = mesh_list[i_patch].wgts[local_nodes]
            local_pt_counter = 0
            for j_pt in range(num_pts_elem):
                for i_pt in range(num_pts_elem):
                    # compute the (B-)spline basis functions and derivatives with
                    # Bezier extraction
                    N_mat = mesh_list[i_patch].C[i_elem] @ Buv[i_pt, j_pt, :]
                    # compute the rational basis
                    RR = N_mat * wgts
                    w_sum = np.sum(RR)
                    RR /= w_sum
                    phys_pt = cpts @ RR
                    local_points[:, local_pt_counter] = phys_pt
                    local_point_val[local_pt_counter] = np.dot(RR, sol[global_nodes])
                    local_pt_counter += 1
            pt_range = list(range(elem_counter * 8, (elem_counter + 1) * 8))
            points_VTK[0:2, pt_range] = local_points[:, VTKOrder]
            point_val_VTK[pt_range] = local_point_val[VTKOrder]
            conn[pt_range] = pt_range
            offsets[elem_counter] = 8 * (elem_counter + 1)
            cell_types[elem_counter] = VtkQuadraticQuad.tid
            elem_counter += 1

    # Output data to VTK
    point_data = {"u": point_val_VTK}
    unstructuredGridToVTK(
        file_name,
        points_VTK[0, :],
        points_VTK[1, :],
        points_VTK[2, :],
        connectivity=conn,
        offsets=offsets,
        cell_types=cell_types,
        pointData=point_data,
    )
    print("Output written to", file_name + ".vtu")


def plot_sol2D_vector(mesh_list, sol, field_names, file_name):
    """
    Plots the computed solution to VTU file using quadratic quad (Q8)
    visualization elements for vector 2D problems. It assumes that 
    the sol contains a vector of the form [u_0, v_0, ..., u_1, v_1, ...]
    where u, v, ... are the different fields


    Parameters
    ----------
    mesh_list : (list of IGAmesh2D) multipatch mesh
    sol : (1D array) solution vector
    field_names : (list of strings) names of each field
    file_name : (string) file name (without extension)

    Returns
    -------
    None.

    """
    # Set the order of points for VTK_QUADRATIC_QUAD  w.r.t. a 3x3 grid from
    # Figure 3 of https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
    VTKOrder = [0, 2, 8, 6, 1, 5, 7, 3]
    num_fields = len(field_names)

    num_pts_elem = 3
    eval_pts_u = np.linspace(-1, 1, num_pts_elem)
    eval_pts_v = np.linspace(-1, 1, num_pts_elem)
    Buv, _, _ = bernstein_basis_2d(eval_pts_u, eval_pts_v, mesh_list[0].deg)

    num_elem = 0
    for i_patch in range(len(mesh_list)):
        num_elem += mesh_list[i_patch].num_elem

    points_VTK = np.zeros((3, num_elem * 8))
    
    local_points = np.zeros((2, num_pts_elem ** 2))
    local_point_val = []
    point_val_VTK = []
    for i_field in range(num_fields):
        local_point_val.append(np.zeros(num_pts_elem ** 2))
        point_val_VTK.append(np.zeros(num_elem * 8))
    conn = np.zeros(num_elem * 8, dtype=int)
    offsets = np.zeros(num_elem, dtype=int)
    cell_types = np.zeros(num_elem, dtype=int)
    elem_counter = 0
    for i_patch in range(len(mesh_list)):
        for i_elem in range(mesh_list[i_patch].num_elem):
            local_nodes = mesh_list[i_patch].elem_node[i_elem]
            global_nodes = mesh_list[i_patch].elem_node_global[i_elem]
            cpts = mesh_list[i_patch].cpts[0:2, local_nodes]
            wgts = mesh_list[i_patch].wgts[local_nodes]
            local_pt_counter = 0
            for j_pt in range(num_pts_elem):
                for i_pt in range(num_pts_elem):
                    # compute the (B-)spline basis functions and derivatives with
                    # Bezier extraction
                    N_mat = mesh_list[i_patch].C[i_elem] @ Buv[i_pt, j_pt, :]
                    # compute the rational basis
                    RR = N_mat * wgts
                    w_sum = np.sum(RR)
                    RR /= w_sum
                    phys_pt = cpts @ RR
                    local_points[:, local_pt_counter] = phys_pt
                    for i_field in range(num_fields):
                        local_point_val[i_field][local_pt_counter] = np.dot(RR,
                                    sol[num_fields*global_nodes+i_field])
                    local_pt_counter += 1
            pt_range = list(range(elem_counter * 8, (elem_counter + 1) * 8))
            points_VTK[0:2, pt_range] = local_points[:, VTKOrder]
            for i_field in range(num_fields):
                point_val_VTK[i_field][pt_range] = local_point_val[i_field][VTKOrder]
            conn[pt_range] = pt_range
            offsets[elem_counter] = 8 * (elem_counter + 1)
            cell_types[elem_counter] = VtkQuadraticQuad.tid
            elem_counter += 1

    # Output data to VTK
    point_data = dict()
    for i_field in range(num_fields):
        point_data[field_names[i_field]] = point_val_VTK[i_field]
    unstructuredGridToVTK(
        file_name,
        points_VTK[0, :],
        points_VTK[1, :],
        points_VTK[2, :],
        connectivity=conn,
        offsets=offsets,
        cell_types=cell_types,
        pointData=point_data,
    )
    print("Output written to", file_name + ".vtu")


def plot_sol2D_elast(mesh_list, Cmat, sol, file_name):
    """
    Plots the computed solution to VTU file using quadratic quad (Q8)
    visualization elements for elasticity 2D problems


    Parameters
    ----------
    mesh_list : (list of IGAmesh2D) multipatch mesh
    Cmat : (2D array) elasticity matrix
    sol : (1D array) solution vector
    file_name : (string) file name (without extension)

    Returns
    -------
    None.

    """
    # Set the order of points for VTK_QUADRATIC_QUAD  w.r.t. a 3x3 grid from
    # Figure 3 of https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
    VTKOrder = [0, 2, 8, 6, 1, 5, 7, 3]

    num_pts_elem = 3
    eval_pts_u = np.linspace(-1, 1, num_pts_elem)
    eval_pts_v = np.linspace(-1, 1, num_pts_elem)
    Buv, dBdu, dBdv = bernstein_basis_2d(eval_pts_u, eval_pts_v, mesh_list[0].deg)

    num_elem = 0
    for i_patch in range(len(mesh_list)):
        num_elem += mesh_list[i_patch].num_elem

    points_VTK = np.zeros((3, num_elem * 8))
    point_disp_VTK = np.zeros((3, num_elem * 8))
    point_stress_VTK = np.zeros((3, num_elem * 8))
    point_stress_VM_VTK = np.zeros(num_elem * 8)
    local_points = np.zeros((2, num_pts_elem ** 2))
    local_points_disp = np.zeros((2, num_pts_elem ** 2))
    local_points_stress = np.zeros((3, num_pts_elem ** 2))
    local_points_stress_VM = np.zeros(num_pts_elem ** 2)
    conn = np.zeros(num_elem * 8, dtype=int)
    offsets = np.zeros(num_elem, dtype=int)
    cell_types = np.zeros(num_elem, dtype=int)
    elem_counter = 0
    for i_patch in range(len(mesh_list)):
        for i_elem in range(mesh_list[i_patch].num_elem):
            u_min = mesh_list[i_patch].elem_vertex[i_elem, 0]
            u_max = mesh_list[i_patch].elem_vertex[i_elem, 2]
            v_min = mesh_list[i_patch].elem_vertex[i_elem, 1]
            v_max = mesh_list[i_patch].elem_vertex[i_elem, 3]
            local_nodes = mesh_list[i_patch].elem_node[i_elem]
            global_nodes = mesh_list[i_patch].elem_node_global[i_elem]
            num_nodes = len(global_nodes)
            global_nodes_xy = np.reshape(
                np.stack((2 * global_nodes, 2 * global_nodes + 1), axis=1),
                2 * num_nodes,
            )
            cpts = mesh_list[i_patch].cpts[0:2, local_nodes]
            wgts = mesh_list[i_patch].wgts[local_nodes]
            local_pt_counter = 0
            B = np.zeros((2 * num_nodes, 3))
            for j_pt in range(num_pts_elem):
                for i_pt in range(num_pts_elem):
                    # compute the (B-)spline basis functions and derivatives with
                    # Bezier extraction
                    N_mat = mesh_list[i_patch].C[i_elem] @ Buv[i_pt, j_pt, :]
                    dN_du = (
                        mesh_list[i_patch].C[i_elem]
                        @ dBdu[i_pt, j_pt, :]
                        * 2
                        / (u_max - u_min)
                    )
                    dN_dv = (
                        mesh_list[i_patch].C[i_elem]
                        @ dBdv[i_pt, j_pt, :]
                        * 2
                        / (v_max - v_min)
                    )
                    # compute the rational basis
                    RR = N_mat * wgts
                    w_sum = np.sum(RR)
                    RR /= w_sum
                    dRdu = dN_du * wgts
                    dRdv = dN_dv * wgts
                    w_sum = np.sum(RR)
                    dw_xi = np.sum(dRdu)
                    dw_eta = np.sum(dRdv)

                    dRdu = dRdu / w_sum - RR * dw_xi / w_sum ** 2
                    dRdv = dRdv / w_sum - RR * dw_eta / w_sum ** 2

                    # compute the solution w.r.t. the physical space
                    dR = np.stack((dRdu, dRdv))
                    dxdxi = dR @ cpts.transpose()
                    phys_pt = cpts @ RR

                    if abs(np.linalg.det(dxdxi)) < 1e-12:
                        print("Warning: Singularity in mapping at ", phys_pt)
                        dR = np.linalg.pinv(dxdxi) @ dR
                    else:
                        dR = np.linalg.solve(dxdxi, dR)

                    B[0 : 2 * num_nodes - 1 : 2, 0] = dR[0, :]
                    B[1 : 2 * num_nodes : 2, 1] = dR[1, :]
                    B[0 : 2 * num_nodes - 1 : 2, 2] = dR[1, :]
                    B[1 : 2 * num_nodes : 2, 2] = dR[0, :]

                    local_points[:, local_pt_counter] = phys_pt
                    sol_val_x = np.dot(RR, sol[2 * global_nodes])
                    sol_val_y = np.dot(RR, sol[2 * global_nodes + 1])
                    local_points_disp[:, local_pt_counter] = [sol_val_x, sol_val_y]
                    stress_vect = Cmat @ B.transpose() @ sol[global_nodes_xy]
                    local_points_stress[:, local_pt_counter] = stress_vect
                    stress_VM = np.sqrt(
                        stress_vect[0] ** 2
                        - stress_vect[0] * stress_vect[1]
                        + stress_vect[1] ** 2
                        + 3 * stress_vect[2] ** 2
                    )
                    local_points_stress_VM[local_pt_counter] = stress_VM
                    local_pt_counter += 1
            pt_range = list(range(elem_counter * 8, (elem_counter + 1) * 8))
            points_VTK[0:2, pt_range] = local_points[:, VTKOrder]
            point_disp_VTK[0:2, pt_range] = local_points_disp[:, VTKOrder]
            point_stress_VTK[:, pt_range] = local_points_stress[:, VTKOrder]
            point_stress_VM_VTK[pt_range] = local_points_stress_VM[VTKOrder]

            conn[pt_range] = pt_range
            offsets[elem_counter] = 8 * (elem_counter + 1)
            cell_types[elem_counter] = VtkQuadraticQuad.tid
            elem_counter += 1

    # Output data to VTK
    point_data = {
        "u": (point_disp_VTK[0, :], point_disp_VTK[1, :], point_disp_VTK[2:]),
        "stress": (
            point_stress_VTK[0, :],
            point_stress_VTK[1, :],
            point_stress_VTK[2, :],
        ),
        "stressVM": point_stress_VM_VTK,
    }

    unstructuredGridToVTK(
        file_name,
        points_VTK[0, :],
        points_VTK[1, :],
        points_VTK[2, :],
        connectivity=conn,
        offsets=offsets,
        cell_types=cell_types,
        pointData=point_data,
    )
    print("Output written to", file_name + ".vtu")


def plot_sol2D_error(mesh_list, sol, u_ex, file_name):
    """
    Plots the error in the computed solution to VTU file using quadratic quad (Q8)
    visualization elements


    Parameters
    ----------
    mesh_list : (list of IGAmesh2D) multipatch mesh
    sol : (1D array) solution vector
    u_ex : (function) exact solution
    file_name : (string) file name (without extension)

    Returns
    -------
    None.

    """
    # Set the order of points for VTK_QUADRATIC_QUAD  w.r.t. a 3x3 grid from
    # Figure 3 of https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
    VTKOrder = [0, 2, 8, 6, 1, 5, 7, 3]

    num_pts_elem = 3
    eval_pts_u = np.linspace(-1, 1, num_pts_elem)
    eval_pts_v = np.linspace(-1, 1, num_pts_elem)
    Buv, _, _ = bernstein_basis_2d(eval_pts_u, eval_pts_v, mesh_list[0].deg)

    num_elem = 0
    for i_patch in range(len(mesh_list)):
        num_elem += mesh_list[i_patch].num_elem

    points_VTK = np.zeros((3, num_elem * 8))
    point_val_VTK = np.zeros(num_elem * 8)
    local_points = np.zeros((2, num_pts_elem ** 2))
    local_point_val = np.zeros(num_pts_elem ** 2)
    conn = np.zeros(num_elem * 8, dtype=int)
    offsets = np.zeros(num_elem, dtype=int)
    cell_types = np.zeros(num_elem, dtype=int)
    elem_counter = 0
    for i_patch in range(len(mesh_list)):
        for i_elem in range(mesh_list[i_patch].num_elem):
            local_nodes = mesh_list[i_patch].elem_node[i_elem]
            global_nodes = mesh_list[i_patch].elem_node_global[i_elem]
            cpts = mesh_list[i_patch].cpts[0:2, local_nodes]
            wgts = mesh_list[i_patch].wgts[local_nodes]
            local_pt_counter = 0
            for j_pt in range(num_pts_elem):
                for i_pt in range(num_pts_elem):
                    # compute the (B-)spline basis functions and derivatives with
                    # Bezier extraction
                    N_mat = mesh_list[i_patch].C[i_elem] @ Buv[i_pt, j_pt, :]
                    # compute the rational basis
                    RR = N_mat * wgts
                    w_sum = np.sum(RR)
                    RR /= w_sum
                    phys_pt = cpts @ RR
                    local_points[:, local_pt_counter] = phys_pt
                    comp_sol_val = np.dot(RR, sol[global_nodes])
                    exact_sol_val = np.real(u_ex(phys_pt[0], phys_pt[1]))
                    local_point_val[local_pt_counter] = exact_sol_val - comp_sol_val
                    local_pt_counter += 1
            pt_range = list(range(elem_counter * 8, (elem_counter + 1) * 8))
            points_VTK[0:2, pt_range] = local_points[:, VTKOrder]
            point_val_VTK[pt_range] = local_point_val[VTKOrder]
            conn[pt_range] = pt_range
            offsets[elem_counter] = 8 * (elem_counter + 1)
            cell_types[elem_counter] = VtkQuadraticQuad.tid
            elem_counter += 1

    # Output data to VTK
    point_data = {"u_err": point_val_VTK}
    unstructuredGridToVTK(
        file_name,
        points_VTK[0, :],
        points_VTK[1, :],
        points_VTK[2, :],
        connectivity=conn,
        offsets=offsets,
        cell_types=cell_types,
        pointData=point_data,
    )
    print("Output written to", file_name + ".vtu")


def plot_sol2D_elast_error(mesh_list, Cmat, sol, ex_disp, ex_stress, file_name):
    """
    Plots the error in the computed solution to VTU file using quadratic quad (Q8)
    visualization elements for 2D elasticity


    Parameters
    ----------
    mesh_list : (list of IGAmesh2D) multipatch mesh
    Cmat : (2D array) elasticity matrix
    sol : (1D array) solution vector
    ex_disp : (function) exact displacement
    ex_stress : (function) exact stress
    file_name : (string) file name (without extension)

    Returns
    -------
    None.

    """
    # Set the order of points for VTK_QUADRATIC_QUAD  w.r.t. a 3x3 grid from
    # Figure 3 of https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
    VTKOrder = [0, 2, 8, 6, 1, 5, 7, 3]

    num_pts_elem = 3
    eval_pts_u = np.linspace(-1, 1, num_pts_elem)
    eval_pts_v = np.linspace(-1, 1, num_pts_elem)
    Buv, dBdu, dBdv = bernstein_basis_2d(eval_pts_u, eval_pts_v, mesh_list[0].deg)

    num_elem = 0
    for i_patch in range(len(mesh_list)):
        num_elem += mesh_list[i_patch].num_elem

    points_VTK = np.zeros((3, num_elem * 8))
    point_disp_VTK = np.zeros((3, num_elem * 8))
    point_stress_VTK = np.zeros((3, num_elem * 8))
    point_stress_VM_VTK = np.zeros(num_elem * 8)
    local_points = np.zeros((2, num_pts_elem ** 2))
    local_points_disp = np.zeros((2, num_pts_elem ** 2))
    local_points_stress = np.zeros((3, num_pts_elem ** 2))
    local_points_stress_VM = np.zeros(num_pts_elem ** 2)
    conn = np.zeros(num_elem * 8, dtype=int)
    offsets = np.zeros(num_elem, dtype=int)
    cell_types = np.zeros(num_elem, dtype=int)
    elem_counter = 0
    for i_patch in range(len(mesh_list)):
        for i_elem in range(mesh_list[i_patch].num_elem):
            u_min = mesh_list[i_patch].elem_vertex[i_elem, 0]
            u_max = mesh_list[i_patch].elem_vertex[i_elem, 2]
            v_min = mesh_list[i_patch].elem_vertex[i_elem, 1]
            v_max = mesh_list[i_patch].elem_vertex[i_elem, 3]
            local_nodes = mesh_list[i_patch].elem_node[i_elem]
            global_nodes = mesh_list[i_patch].elem_node_global[i_elem]
            num_nodes = len(global_nodes)
            global_nodes_xy = np.reshape(
                np.stack((2 * global_nodes, 2 * global_nodes + 1), axis=1),
                2 * num_nodes,
            )
            cpts = mesh_list[i_patch].cpts[0:2, local_nodes]
            wgts = mesh_list[i_patch].wgts[local_nodes]
            local_pt_counter = 0
            B = np.zeros((2 * num_nodes, 3))
            for j_pt in range(num_pts_elem):
                for i_pt in range(num_pts_elem):
                    # compute the (B-)spline basis functions and derivatives with
                    # Bezier extraction
                    N_mat = mesh_list[i_patch].C[i_elem] @ Buv[i_pt, j_pt, :]
                    dN_du = (
                        mesh_list[i_patch].C[i_elem]
                        @ dBdu[i_pt, j_pt, :]
                        * 2
                        / (u_max - u_min)
                    )
                    dN_dv = (
                        mesh_list[i_patch].C[i_elem]
                        @ dBdv[i_pt, j_pt, :]
                        * 2
                        / (v_max - v_min)
                    )
                    # compute the rational basis
                    RR = N_mat * wgts
                    w_sum = np.sum(RR)
                    RR /= w_sum
                    dRdu = dN_du * wgts
                    dRdv = dN_dv * wgts
                    w_sum = np.sum(RR)
                    dw_xi = np.sum(dRdu)
                    dw_eta = np.sum(dRdv)

                    dRdu = dRdu / w_sum - RR * dw_xi / w_sum ** 2
                    dRdv = dRdv / w_sum - RR * dw_eta / w_sum ** 2

                    # compute the solution w.r.t. the physical space
                    dR = np.stack((dRdu, dRdv))
                    dxdxi = dR @ cpts.transpose()
                    phys_pt = cpts @ RR

                    if abs(np.linalg.det(dxdxi)) < 1e-12:
                        print("Warning: Singularity in mapping at ", phys_pt)
                        dR = np.linalg.pinv(dxdxi) * dR
                    else:
                        dR = np.linalg.solve(dxdxi, dR)

                    B[0 : 2 * num_nodes - 1 : 2, 0] = dR[0, :]
                    B[1 : 2 * num_nodes : 2, 1] = dR[1, :]
                    B[0 : 2 * num_nodes - 1 : 2, 2] = dR[1, :]
                    B[1 : 2 * num_nodes : 2, 2] = dR[0, :]

                    local_points[:, local_pt_counter] = phys_pt
                    sol_val_x = np.dot(RR, sol[2 * global_nodes])
                    sol_val_y = np.dot(RR, sol[2 * global_nodes + 1])

                    stress_vect = Cmat @ B.transpose() @ sol[global_nodes_xy]

                    # evaluate the exact displacement and stresses
                    ex_sol_val = ex_disp(phys_pt[0], phys_pt[1])
                    ex_stress_val = ex_stress(phys_pt[0], phys_pt[1])
                    
                    # compute the differences
                    local_points_disp[:, local_pt_counter] = [
                        ex_sol_val[0] - sol_val_x,
                        ex_sol_val[1] - sol_val_y,
                    ]
                    local_points_stress[:, local_pt_counter] = (
                        ex_stress_val[0:3] - stress_vect
                    )
                    stress_VM = comp_von_misses(stress_vect)
                    
                    local_points_stress_VM[local_pt_counter] = ex_stress_val[3] - stress_VM
                    local_pt_counter += 1
            pt_range = list(range(elem_counter * 8, (elem_counter + 1) * 8))
            points_VTK[0:2, pt_range] = local_points[:, VTKOrder]
            point_disp_VTK[0:2, pt_range] = local_points_disp[:, VTKOrder]
            point_stress_VTK[:, pt_range] = local_points_stress[:, VTKOrder]
            point_stress_VM_VTK[pt_range] = local_points_stress_VM[VTKOrder]

            conn[pt_range] = pt_range
            offsets[elem_counter] = 8 * (elem_counter + 1)
            cell_types[elem_counter] = VtkQuadraticQuad.tid
            elem_counter += 1

    # Output data to VTK
    point_data = {
        "u_err": (point_disp_VTK[0, :], point_disp_VTK[1, :], point_disp_VTK[2:]),
        "stress_err": (
            point_stress_VTK[0, :],
            point_stress_VTK[1, :],
            point_stress_VTK[2, :],
        ),
        "stressVM_err": point_stress_VM_VTK,
    }

    unstructuredGridToVTK(
        file_name,
        points_VTK[0, :],
        points_VTK[1, :],
        points_VTK[2, :],
        connectivity=conn,
        offsets=offsets,
        cell_types=cell_types,
        pointData=point_data,
    )
    print("Output written to", file_name + ".vtu")


def comp_error_norm(mesh_list, sol, exact_sol, deriv_exact_sol, a0, gauss_rule):
    """
    Computes the relative errors for Poisson equation

    Parameters
    ----------
    mesh_list : (list of IGAMesh2D) multipatch mesh
    sol : 1D array
        solution vector.
    exact_sol : function
        exact function.
    deriv_exact_sol : function
        derivative of the exact function
    a0 : function
        diffusion term.
    gauss_rule : list
        list of Gauss points and weights in the reference interval [-1,1]
                 (one for each direction)

    Returns
    -------
    rel_l2_err : float
        relative L2-norm error
    rel_h1_err : float
        relative H1-seminorm error
    """
    # Evaluate the basis functions and derivatives at the gauss points
    pts_u = gauss_rule[0]["nodes"]
    pts_v = gauss_rule[1]["nodes"]
    wgts_u = gauss_rule[0]["weights"]
    wgts_v = gauss_rule[1]["weights"]
    Buv, dBdu, dBdv = bernstein_basis_2d(pts_u, pts_v, mesh_list[0].deg)
    num_gauss_u = len(pts_u)
    num_gauss_v = len(pts_v)

    l2_norm_err = 0.0
    l2_norm_sol = 0.0
    h1_norm_err = 0.0
    h1_norm_sol = 0.0
    domain_area = 0.0

    for i_patch in range(len(mesh_list)):
        for i_elem in range(mesh_list[i_patch].num_elem):
            u_min = mesh_list[i_patch].elem_vertex[i_elem, 0]
            u_max = mesh_list[i_patch].elem_vertex[i_elem, 2]
            v_min = mesh_list[i_patch].elem_vertex[i_elem, 1]
            v_max = mesh_list[i_patch].elem_vertex[i_elem, 3]
            jac_ref_par = (u_max - u_min) * (v_max - v_min) / 4

            # compute the rational spline basis
            local_nodes = mesh_list[i_patch].elem_node[i_elem]
            global_nodes = mesh_list[i_patch].elem_node_global[i_elem]
            cpts = mesh_list[i_patch].cpts[0:2, local_nodes]
            wgts = mesh_list[i_patch].wgts[local_nodes]
            for j_gauss in range(num_gauss_v):
                for i_gauss in range(num_gauss_u):
                    # compute the (B-)spline basis functions and derivatives with
                    # Bezier extraction
                    N_mat = mesh_list[i_patch].C[i_elem] @ Buv[i_gauss, j_gauss, :]
                    dN_du = (
                        mesh_list[i_patch].C[i_elem]
                        @ dBdu[i_gauss, j_gauss, :]
                        * 2
                        / (u_max - u_min)
                    )
                    dN_dv = (
                        mesh_list[i_patch].C[i_elem]
                        @ dBdv[i_gauss, j_gauss, :]
                        * 2
                        / (v_max - v_min)
                    )

                    # compute the rational basis
                    RR = N_mat * wgts
                    dRdu = dN_du * wgts
                    dRdv = dN_dv * wgts
                    w_sum = np.sum(RR)
                    dw_xi = np.sum(dRdu)
                    dw_eta = np.sum(dRdv)

                    dRdu = dRdu / w_sum - RR * dw_xi / w_sum ** 2
                    dRdv = dRdv / w_sum - RR * dw_eta / w_sum ** 2

                    # compute the solution w.r.t. the physical space
                    dR = np.stack((dRdu, dRdv))
                    dxdxi = dR @ cpts.transpose()
                    dR = np.linalg.solve(dxdxi, dR)
                    jac_par_phys = np.linalg.det(dxdxi)

                    RR /= w_sum
                    phys_pt = cpts @ RR

                    # evaluate the solution at Gauss points
                    sol_val = np.dot(RR, sol[global_nodes])
                    deriv_sol_val = dR @ sol[global_nodes]
                    ex_sol_val = np.real(exact_sol(phys_pt[0], phys_pt[1])[0])
                    deriv_ex_sol_val = np.real(
                        np.array(deriv_exact_sol(phys_pt[0], phys_pt[1]))
                    )
                    local_area = (
                        jac_par_phys * jac_ref_par * wgts_u[i_gauss] * wgts_v[j_gauss]
                    )
                    domain_area += local_area

                    # compute the norms
                    l2_norm_err += local_area * (ex_sol_val - sol_val) ** 2
                    l2_norm_sol += local_area * ex_sol_val ** 2
                    fun_value = a0(phys_pt[0], phys_pt[1])
                    h1_norm_err += (
                        local_area
                        * fun_value
                        * np.sum((deriv_ex_sol_val - deriv_sol_val) ** 2)
                    )
                    h1_norm_sol += (
                        local_area * fun_value * np.sum(deriv_ex_sol_val ** 2)
                    )

    rel_l2_err = np.sqrt(l2_norm_err / l2_norm_sol)
    rel_h1_err = np.sqrt(h1_norm_err / h1_norm_sol)
    return rel_l2_err, rel_h1_err


def comp_error_norm_elast(mesh_list, Cmat, sol, exact_disp, exact_stress, gauss_rule):
    """
    Computes the relative errors for Poisson equation

    Parameters
    ----------
    mesh_list : (list of IGAMesh2D) multipatch mesh
    Cmat : (2D array) elasticity matrix
    sol : 1D array
        solution vector.
    exact_disp : function
        exact displacement
    exact_stress : function
        exact stress
    gauss_rule : list
        list of Gauss points and weights in the reference interval [-1,1]
                 (one for each direction)

    Returns
    -------
    rel_l2_err : float
        relative L2-norm error
    rel_h1_err : float
        relative H1-seminorm (energy norm) error
    """
    # Evaluate the basis functions and derivatives at the gauss points
    pts_u = gauss_rule[0]["nodes"]
    pts_v = gauss_rule[1]["nodes"]
    wgts_u = gauss_rule[0]["weights"]
    wgts_v = gauss_rule[1]["weights"]
    Buv, dBdu, dBdv = bernstein_basis_2d(pts_u, pts_v, mesh_list[0].deg)
    num_gauss_u = len(pts_u)
    num_gauss_v = len(pts_v)

    l2_norm_err = 0.0
    l2_norm_sol = 0.0
    h1_norm_err = 0.0
    h1_norm_sol = 0.0
    domain_area = 0.0
    invC = np.linalg.inv(Cmat)

    for i_patch in range(len(mesh_list)):
        for i_elem in range(mesh_list[i_patch].num_elem):
            u_min = mesh_list[i_patch].elem_vertex[i_elem, 0]
            u_max = mesh_list[i_patch].elem_vertex[i_elem, 2]
            v_min = mesh_list[i_patch].elem_vertex[i_elem, 1]
            v_max = mesh_list[i_patch].elem_vertex[i_elem, 3]
            jac_ref_par = (u_max - u_min) * (v_max - v_min) / 4

            # compute the rational spline basis
            local_nodes = mesh_list[i_patch].elem_node[i_elem]
            global_nodes = mesh_list[i_patch].elem_node_global[i_elem]
            num_nodes = len(global_nodes)
            global_nodes_xy = np.reshape(
                np.stack((2 * global_nodes, 2 * global_nodes + 1), axis=1),
                2 * num_nodes,
            )
            cpts = mesh_list[i_patch].cpts[0:2, local_nodes]
            wgts = mesh_list[i_patch].wgts[local_nodes]
            B = np.zeros((2 * num_nodes, 3))

            for j_gauss in range(num_gauss_v):
                for i_gauss in range(num_gauss_u):
                    # compute the (B-)spline basis functions and derivatives with
                    # Bezier extraction
                    N_mat = mesh_list[i_patch].C[i_elem] @ Buv[i_gauss, j_gauss, :]
                    dN_du = (
                        mesh_list[i_patch].C[i_elem]
                        @ dBdu[i_gauss, j_gauss, :]
                        * 2
                        / (u_max - u_min)
                    )
                    dN_dv = (
                        mesh_list[i_patch].C[i_elem]
                        @ dBdv[i_gauss, j_gauss, :]
                        * 2
                        / (v_max - v_min)
                    )

                    # compute the rational basis
                    RR = N_mat * wgts
                    dRdu = dN_du * wgts
                    dRdv = dN_dv * wgts
                    w_sum = np.sum(RR)
                    dw_xi = np.sum(dRdu)
                    dw_eta = np.sum(dRdv)

                    dRdu = dRdu / w_sum - RR * dw_xi / w_sum ** 2
                    dRdv = dRdv / w_sum - RR * dw_eta / w_sum ** 2

                    # compute the solution w.r.t. the physical space
                    dR = np.stack((dRdu, dRdv))
                    dxdxi = dR @ cpts.transpose()
                    dR = np.linalg.solve(dxdxi, dR)
                    jac_par_phys = np.linalg.det(dxdxi)

                    RR /= w_sum
                    phys_pt = cpts @ RR

                    B[0 : 2 * num_nodes - 1 : 2, 0] = dR[0, :]
                    B[1 : 2 * num_nodes : 2, 1] = dR[1, :]
                    B[0 : 2 * num_nodes - 1 : 2, 2] = dR[1, :]
                    B[1 : 2 * num_nodes : 2, 2] = dR[0, :]

                    # evaluate the displacement and stresses at Gauss points
                    sol_val_x = np.dot(RR, sol[2 * global_nodes])
                    sol_val_y = np.dot(RR, sol[2 * global_nodes + 1])
                    stress_vect = Cmat @ B.transpose() @ sol[global_nodes_xy]

                    # evaluate the exact displacement and stresses
                    ex_sol_val = exact_disp(phys_pt[0], phys_pt[1])
                    ex_stress_val = exact_stress(phys_pt[0], phys_pt[1])

                    local_area = (
                        jac_par_phys * jac_ref_par * wgts_u[i_gauss] * wgts_v[j_gauss]
                    )
                    domain_area += local_area

                    # compute the norms
                    l2_norm_err += local_area * (
                        (ex_sol_val[0] - sol_val_x) ** 2
                        + (ex_sol_val[1] - sol_val_y) ** 2
                    )
                    l2_norm_sol += local_area * (
                        ex_sol_val[0] ** 2 + ex_sol_val[1] ** 2
                    )

                    h1_norm_err += (
                        local_area
                        * (ex_stress_val[0:3] - stress_vect).transpose()
                        @ invC
                        @ (ex_stress_val[0:3] - stress_vect)
                    )

                    h1_norm_sol += (
                        local_area * np.array(ex_stress_val[0:3]) @ invC @ ex_stress_val[0:3]
                    )

    rel_l2_err = np.sqrt(l2_norm_err / l2_norm_sol)
    rel_h1_err = np.sqrt(h1_norm_err / h1_norm_sol)
    return rel_l2_err, rel_h1_err


def get_measurements_vector(mesh_list, sol, meas_pts_param_xi_eta_i, num_fields):
    """
    Generates values of measurements from a given mesh and solution and a 
    given list measurement points in parameter space for a multi-field solution
    It is assumed that the sol contains a vector of the form 
    [u_0, v_0, ..., u_1, v_1, ...]

    Parameters
    ----------
    mesh_list : (list of IGAMesh2D) multipatch mesh
    sol : 1D array
        solution vector. 
    meas_pts_param_xi_eta_i : (2D array)
        measurements points in the parameter space with one (u,v) coordinate
        and patch index in each row
    num_fields : (int) number of fields in the solution 

    Returns
    -------
    meas_pts_phys_xy : (2D array)
        measurements points in the physical space with one (x,y) coordinate 
        in each row
    meas_val : (list of 1D array)
        the values of the solution computed at each measurement point

    """
    num_pts = len(meas_pts_param_xi_eta_i)
    meas_vals = []
    for _ in range(num_fields):
        meas_vals.append(np.zeros(num_pts))    
    meas_pts_phys_xy = np.zeros((num_pts, 2))
    for i_pt in range(num_pts):
        pt_xi_eta_i = meas_pts_param_xi_eta_i[i_pt]
        xi_coord = pt_xi_eta_i[0]
        eta_coord = pt_xi_eta_i[1]
        patch_index = int(pt_xi_eta_i[2])
        for i in range(len(mesh_list[patch_index].elem_vertex)):
            elem_vertex = mesh_list[patch_index].elem_vertex[i]
            xi_min = elem_vertex[0]
            xi_max = elem_vertex[2]
            eta_min = elem_vertex[1]
            eta_max = elem_vertex[3]
            if xi_min <= xi_coord and eta_min <= eta_coord and xi_coord <= xi_max and eta_coord <= eta_max:
                
                # map point to the reference element (i.e. mapping from 
                # (eta_min, eta_max) and (xi_min, v=xi_max) to (-1, 1)
                local_nodes = mesh_list[patch_index].elem_node[i]
                global_nodes = mesh_list[patch_index].elem_node_global[i]
                cpts = mesh_list[patch_index].cpts[0:2, local_nodes]
                wgts = mesh_list[patch_index].wgts[local_nodes]
                u_coord = 2/(xi_max-xi_min)*(xi_coord-xi_min) - 1
                v_coord = 2/(eta_max - eta_min)*(eta_coord - eta_min) - 1
                Buv, _, _ = bernstein_basis_2d(np.array([u_coord]), np.array([v_coord]), 
                                               mesh_list[patch_index].deg)
                
                # compute the (B-)spline basis functions and derivatives with
                # Bezier extraction
                N_mat = mesh_list[patch_index].C[i] @ Buv[0, 0, :]
                RR = N_mat * wgts
                w_sum = np.sum(RR)
                RR /= w_sum
                meas_pts_phys_xy[i_pt,:] = cpts @ RR
                for i_field in range(num_fields):
                    meas_vals[i_field][i_pt] = np.dot(RR,
                                                      sol[num_fields*global_nodes+i_field])
                break    
    return meas_pts_phys_xy, meas_vals


def get_measurement_stresses(mesh_list, sol, meas_pts_param_xi_eta_i, num_fields, material):
    """
    Generates values of stresses (xx, yy, xy and von Mises) from a given mesh \
    and solution and a given list measurement points in parameter space for a multi-field solution
    It is assumed that the sol contains a vector of the form 
    [u_0, v_0, ..., u_1, v_1, ...]

    Parameters
    ----------
    mesh_list : (list of IGAMesh2D) multipatch mesh
    sol : 1D array
        solution vector. 
    meas_pts_param_xi_eta_i : (2D array)
        measurements points in the parameter space with one (u,v) coordinate
        and patch index in each row
    material : (object) Object containing material properties (nu, Emod) and Cmat.


    Returns
    -------
    meas_pts_phys_xy : (2D array)
        measurements points in the physical space with one (x,y) coordinate 
        in each row
    meas_stress : (list of 1D arrays)
        the values of the stresses computed at each measurement point (one column
                       for the xx, yy, xy and VM stresses)

    """
    num_pts = len(meas_pts_param_xi_eta_i)
    meas_stress = []
    num_fields = 4
    for _ in range(num_fields):
        meas_stress.append(np.zeros(num_pts))    
    meas_pts_phys_xy = np.zeros((num_pts, 2))

    for i_pt in range(num_pts):
        pt_xi_eta_i = meas_pts_param_xi_eta_i[i_pt]
        xi_coord = pt_xi_eta_i[0]
        eta_coord = pt_xi_eta_i[1]
        patch_index = int(pt_xi_eta_i[2])
        for i in range(len(mesh_list[patch_index].elem_vertex)):
            elem_vertex = mesh_list[patch_index].elem_vertex[i]
            xi_min = elem_vertex[0]
            xi_max = elem_vertex[2]
            eta_min = elem_vertex[1]
            eta_max = elem_vertex[3]
            if xi_min <= xi_coord and eta_min <= eta_coord and xi_coord <= xi_max and eta_coord <= eta_max:
                
                # map point to the reference element (i.e. mapping from 
                # (eta_min, eta_max) and (xi_min, v=xi_max) to (-1, 1)
                local_nodes = mesh_list[patch_index].elem_node[i]
                num_nodes = len(local_nodes)
                B = np.zeros((2 * num_nodes, 3))
                global_nodes = mesh_list[patch_index].elem_node_global[i]
                global_nodes_xy = np.reshape(
                    np.stack((2 * global_nodes, 2 * global_nodes + 1), axis=1),
                    2 * num_nodes,
                )
                cpts = mesh_list[patch_index].cpts[0:2, local_nodes]
                wgts = mesh_list[patch_index].wgts[local_nodes]
                u_coord = 2/(xi_max-xi_min)*(xi_coord-xi_min) - 1
                v_coord = 2/(eta_max - eta_min)*(eta_coord - eta_min) - 1
                Buv, dBdu, dBdv = bernstein_basis_2d(np.array([u_coord]), np.array([v_coord]), 
                                               mesh_list[patch_index].deg)
                
                # compute the (B-)spline basis functions and derivatives with
                # Bezier extraction
                N_mat = mesh_list[patch_index].C[i] @ Buv[0, 0, :]
                dN_du = (
                    mesh_list[patch_index].C[i] @ dBdu[0, 0, :] * 2 / (xi_max - xi_min)
                )
                dN_dv = (
                    mesh_list[patch_index].C[i] @ dBdv[0, 0, :] * 2 / (eta_max - eta_min)
                )                
                
                RR = N_mat * wgts
                w_sum = np.sum(RR)
                RR /= w_sum                                
               
                dRdu = dN_du * wgts
                dRdv = dN_dv * wgts
                w_sum = np.sum(RR)
                dw_xi = np.sum(dRdu)
                dw_eta = np.sum(dRdv)

                dRdu = dRdu / w_sum - RR * dw_xi / w_sum ** 2
                dRdv = dRdv / w_sum - RR * dw_eta / w_sum ** 2

                # compute the solution w.r.t. the physical space
                dR = np.stack((dRdu, dRdv))
                dxdxi = dR @ cpts.transpose()
                phys_pt = cpts @ RR

                if abs(np.linalg.det(dxdxi)) < 1e-12:
                    print("Warning: Singularity in mapping at ", phys_pt)
                    dR = np.linalg.pinv(dxdxi) @ dR
                else:
                    dR = np.linalg.solve(dxdxi, dR)

                B[0 : 2 * num_nodes - 1 : 2, 0] = dR[0, :]
                B[1 : 2 * num_nodes : 2, 1] = dR[1, :]
                B[0 : 2 * num_nodes - 1 : 2, 2] = dR[1, :]
                B[1 : 2 * num_nodes : 2, 2] = dR[0, :]
                
                Cmat_FGM = material.elasticity(phys_pt, mesh_list[patch_index])*material.Cmat

                stress_vect = Cmat_FGM @ B.transpose() @ sol[global_nodes_xy]
                stress_VM = np.sqrt(
                    stress_vect[0] ** 2
                    - stress_vect[0] * stress_vect[1]
                    + stress_vect[1] ** 2
                    + 3 * stress_vect[2] ** 2
                )
                
                
                meas_pts_phys_xy[i_pt,:] = phys_pt
                for i in range(3):
                    meas_stress[i][i_pt] = stress_vect[i]
                meas_stress[3][i_pt] = stress_VM
                break    
    return meas_pts_phys_xy, meas_stress


def comp_von_misses(stress_vect):
    return np.sqrt(
        stress_vect[0] ** 2
        - stress_vect[0] * stress_vect[1]
        + stress_vect[1] ** 2
        + 3 * stress_vect[2] ** 2
    )


def exact_stress_vect(xs, ys, exact_stress_fun):
    if isinstance(xs, np.ndarray):
        num_pts = len(xs)
        stress_xx = np.zeros(num_pts)
        stress_yy = np.zeros(num_pts)
        stress_xy = np.zeros(num_pts)
        stress_vm = np.zeros(num_pts)
    
        for i in range(num_pts):
            x = xs[i]
            y = ys[i]
            stress= exact_stress_fun(x, y)
            stress_xx[i] = stress[0]
            stress_yy[i] = stress[1]
            stress_xy[i] = stress[2]
            stress_vm[i] = comp_von_misses(stress)
    else:
        stress= exact_stress_fun(xs, ys)
        stress_xx = stress[0]
        stress_yy = stress[1]
        stress_xy = stress[2]
        stress_vm = comp_von_misses(stress)
    return (stress_xx, stress_yy, stress_xy, stress_vm)


def comp_measurement_values(num_pts_xi, num_pts_eta, mesh_list, sol0, meas_func,
                            num_fields, *params):
    meas_pts_phys_xy_all = []
    meas_vals_all = []
    vals_min = []
    vals_max = []
    for _ in range(num_fields):
        meas_vals_all.append([])
        vals_min.append(float('inf'))
        vals_max.append(float('-inf'))
    for i in range(len(mesh_list)):    
        meas_points_param_xi = np.linspace(0, 1, num_pts_xi)
        meas_points_param_eta = np.linspace(0, 1, num_pts_eta)
        
        meas_pts_param_xi_eta_i = np.zeros((len(meas_points_param_xi)*len(meas_points_param_eta),3))
        row_counter = 0
        for pt_xi in meas_points_param_xi:
            for pt_eta in meas_points_param_eta:
                #meas_pts_param_xi_eta_i.at[row_counter, :].set([pt_xi, pt_eta, i])
                meas_pts_param_xi_eta_i[row_counter, :] = [pt_xi, pt_eta, i]
                row_counter += 1
        
        meas_pts_phys_xy, meas_vals = meas_func(mesh_list, sol0, 
                                                meas_pts_param_xi_eta_i, num_fields, *params)    
        meas_pts_phys_xy_all.append(meas_pts_phys_xy)
        for i_field in range(num_fields):
            meas_vals_all[i_field].append(meas_vals[i_field])
            vals_min[i_field] = np.minimum(vals_min[i_field], np.min(meas_vals[i_field]))
            vals_max[i_field] = np.maximum(vals_max[i_field], np.max(meas_vals[i_field]))
    return meas_vals_all, meas_pts_phys_xy_all, vals_min, vals_max


def plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, mesh_list, 
                meas_pts_phys_xy_all, meas_vals_all, vals_min, vals_max):
    # plot the solution as a contour plot
    num_fields = len(field_names)
    #xy2D = np.zeros([num_pts_xi, num_pts_eta, 2])
    Disp2D = np.zeros([num_pts_xi, num_pts_eta, num_fields])
    for i_field in range(num_fields):
        for i in range(len(mesh_list)):
            xPhysTest2D = np.resize(meas_pts_phys_xy_all[i][:,0], [num_pts_xi, num_pts_eta])
            yPhysTest2D = np.resize(meas_pts_phys_xy_all[i][:,1], [num_pts_xi, num_pts_eta])
            YTest2D = np.resize(meas_vals_all[i_field][i], [num_pts_xi, num_pts_eta])
            
            
            # Plot the real part of the solution and errors
            #plt.contourf(xPhysTest2D, yPhysTest2D, YTest2D, 255, cmap=plt.cm.jet,
            #              vmin=vals_min[i_field], vmax=vals_max[i_field])
        #plt.title(field_title + ' for ' + field_names[i_field])
        #plt.axis('equal')
        #plt.colorbar()    
        #plt.show()
        Disp2D[:, :, i_field] = YTest2D
        
    return Disp2D

def comp_error_2D(num_fields, mesh_list, exact_sol, meas_pts_phys_xy_all, meas_vals_all):
    # compute the error at a set of uniformy spaced points
    err_vals_min = []
    err_vals_max = []
    err_vals_all = []
    exact_vals_all = []
    for _ in range(num_fields):
        err_vals_min.append(float('inf'))
        err_vals_max.append(float('-inf'))
        err_vals_all.append([])
        exact_vals_all.append([])

    for i in range(len(mesh_list)):    
        exact_vals = exact_sol(meas_pts_phys_xy_all[i][:,0], meas_pts_phys_xy_all[i][:,1])
        for i_field in range(num_fields):
            exact_vals_all[i_field].append(exact_vals[i_field])
            err_val = exact_vals[i_field] - meas_vals_all[i_field][i]
            err_vals_all[i_field].append(err_val)
            err_vals_min[i_field] = np.minimum(err_vals_min[i_field], np.min(err_val))
            err_vals_max[i_field] = np.maximum(err_vals_max[i_field], np.max(err_val))
    return err_vals_all, err_vals_min, err_vals_max
