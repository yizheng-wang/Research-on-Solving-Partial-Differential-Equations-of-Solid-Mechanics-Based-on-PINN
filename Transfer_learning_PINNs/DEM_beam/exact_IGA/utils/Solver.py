# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 15:01:22 2023

@author: ms_es
"""

import time
from scipy.sparse.linalg import spsolve
#import matplotlib.pyplot as plt
import numpy as np
#from timeit import default_timer

from utils.Geom_examples import Quadrilateral
from utils.materials import MaterialElast2D_FGM
from utils.IGA import IGAMesh2D
from utils.multipatch import gen_vertex2patch2D, gen_edge_list, zip_conforming
from utils.assembly import gen_gauss_pts, stiff_elast_FGM_2D
from utils.boundary import boundary2D, applyBCElast2D
from utils.postprocessing import (plot_fields_2D,
                                  comp_measurement_values,
                                  get_measurements_vector,
                                  get_measurement_stresses)


np.random.seed(42)

def elastic_beam(traction, elasticity_modulus, model_data):
    # Approximation parameters
    tTotal = 0
    p = q = 3 # Polynomial degree of the approximation
    
    numPtsU = model_data["numPtsU"]
    numPtsV = model_data["numPtsV"]
    Emod = model_data["E"]
    nu = model_data["nu"]
    beam_length = model_data["beam_length"]
    beam_width  = model_data["beam_width"]
    
    #num_refinements = int(math.log(numPtsU, 2) - 4)
    num_refinements = 3
    
    # Step 1: Generate the geometry
    vertices = [[0., 0.], [0., beam_width], [beam_length, 0.], [beam_length, beam_width]]
    patch1 = Quadrilateral(np.array(vertices))
    patch_list = [patch1]

    #Fixed Dirichlet B.C., u_y = 0 and u_x=0 for x=0
    def u_bound_dir_fixed(x,y): return [0., 0.]
    
    X = np.linspace(0, beam_length, numPtsU)[:,None].flatten()
    
    #  Neumann B.C. τ(x,y) = [x, y] on the top boundary
    global n
    n = -1
    def u_bound_neu(x,y,nx,ny):
        global n
        n += 1
        return [0., np.interp(x, X, -traction)]
    
    bound_up = boundary2D("Neumann", 0, "up", u_bound_neu)
    bound_left = boundary2D("Dirichlet", 0, "left", u_bound_dir_fixed)
    bound_right = boundary2D("Dirichlet", 0, "right", u_bound_dir_fixed)
    
    bound_all = [bound_up, bound_left, bound_right]
    
    # Step 0: Define the material properties
    material = MaterialElast2D_FGM(Emod=Emod, nu=nu, vertices=vertices, gp=elasticity_modulus, N=numPtsV, plane_type="stress")
        
    # Step 2: Degree elevate and refine the geometry
    t = time.time()
    for patch in patch_list:
        patch.degreeElev(p-1, q-1)
    elapsed = time.time() - t
    print("Degree elevation took ", elapsed, " seconds")
    tTotal += elapsed
    
    t = time.time()
    # Refine the mesh in the horizontal direction first two times to get square elements
    for i in range(2):
        for patch in patch_list:
            patch.refine_knotvectors(True, False)
    
    for i in range(num_refinements):
        for patch in patch_list:
            patch.refine_knotvectors(True, True)
    elapsed = time.time() - t
    print("Knot insertion took ", elapsed, " seconds")
    tTotal += elapsed
    
    #_, ax = plt.subplots()
    #for patch in patch_list:
    #    patch.plotKntSurf(ax)
    #plt.show()
    t = time.time()
    mesh_list = []
    for patch in patch_list:
        mesh_list.append(IGAMesh2D(patch))
    elapsed = time.time() - t
    print("Mesh initialization took ", elapsed, " seconds")
    tTotal += elapsed
    
    for mesh in mesh_list:
        mesh.classify_boundary()
    
    vertices, vertex2patch, patch2vertex = gen_vertex2patch2D(mesh_list)
    edge_list = gen_edge_list(patch2vertex)
    size_basis = zip_conforming(mesh_list, vertex2patch, edge_list)
    
    # Step 3. Assemble linear system
    gauss_quad_u = gen_gauss_pts(p+1)
    gauss_quad_v = gen_gauss_pts(q+1)
    gauss_rule = [gauss_quad_u, gauss_quad_v]
    t = time.time()
    stiff, E_modulus, coord = stiff_elast_FGM_2D(mesh_list, material, gauss_rule)
    stiff = stiff.tocsr()
    elapsed = time.time() - t
    print("Stiffness assembly took ", elapsed, " seconds")
    tTotal += elapsed
    
    # Step 4. Apply boundary conditions
    t = time.time()
    # Assume no volume force
    rhs = np.zeros(2*size_basis)
    stiff, rhs = applyBCElast2D(mesh_list, bound_all, stiff, rhs, gauss_rule)
    elapsed = time.time() - t
    print("Applying B.C.s took ", elapsed, " seconds")
    tTotal += elapsed
    
    # Step 6. Solve the linear system
    t = time.time()
    sol0 = spsolve(stiff, rhs)
    elapsed = time.time() - t
    print("Linear sparse solver took ", elapsed, " seconds")
    tTotal += elapsed
        
    # Step 7a. Plot the solution in matplotlib
    t = time.time()
    # compute the displacements at a set of uniformly spaced points
    num_pts_xi = numPtsU#2**(num_refinements+4)+2
    num_pts_eta = numPtsV#2**(num_refinements+4)//S+2
    num_fields = 2
    meas_vals_all, meas_pts_phys_xy_all, vals_disp_min, vals_disp_max = comp_measurement_values(num_pts_xi,
                                                                  num_pts_eta,
                                                                  mesh_list,
                                                                  sol0,
                                                                  get_measurements_vector,
                                                                  num_fields)
    elapsed = time.time() - t
    print("Computing the displacement values at measurement points took ", elapsed, " seconds")
    tTotal += elapsed
    
    t = time.time()
    # compute the stresses at a set of uniformly spaced points
    num_pts_xi = numPtsU#2**(num_refinements+4)+2
    num_output_fields = 4
    meas_stress_all, meas_pts_phys_xy_all, vals_stress_min, vals_stress_max = comp_measurement_values(num_pts_xi,
                                                                  num_pts_eta,
                                                                  mesh_list,
                                                                  sol0,
                                                                  get_measurement_stresses,
                                                                  num_output_fields,
                                                                  material)
    elapsed = time.time() - t
    print("Computing the stress values at measurement points took ", elapsed, " seconds")
    
    
    t = time.time()
    field_title = "Computed solution"
    field_names = ['x-disp', 'y-disp']
    disp2D = plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, mesh_list, 
                    meas_pts_phys_xy_all, meas_vals_all, vals_disp_min, vals_disp_max)
    elapsed = time.time() - t
    print("Plotting the solution (matplotlib) took ", elapsed, " seconds")
    
    #t = time.time()
    field_title = "Computed solution"
    field_names = ['xx-stress', 'yy-stress', 'xy-stress', 'VM-stress']
    stress2D = plot_fields_2D(num_pts_xi, num_pts_eta, field_title, field_names, mesh_list, 
                    meas_pts_phys_xy_all, meas_stress_all, vals_stress_min, vals_stress_max)
    #elapsed = time.time() - t
    #print("Plotting the solution (matplotlib) took ", elapsed, " seconds")
    return disp2D, stress2D, tTotal
