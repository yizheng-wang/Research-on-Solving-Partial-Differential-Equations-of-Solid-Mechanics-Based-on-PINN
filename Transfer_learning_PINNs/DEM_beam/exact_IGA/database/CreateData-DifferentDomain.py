#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solves a 2D elasticity problem on a FGM beam with random distribution of Elasticity modulus under different random 
tensions and creates a database.

This script defines a function `elastic_beam` that solves a 2D elasticity problem for a beam structure subjected to
different random tensions. It utilizes isogeometric analysis (IGA) techniques and boundary element methods (BEM) to 
model the beam's behavior.

\Omega = (0,0.1)x(0,2) 
Fixed BC: x = 0 and x = 2
Traction \tau = GRF at y=0.1 in the vertical direction
"""

import time
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer
import random

from utils.Geom_examples import Quadrilateral
from utils.materials import MaterialElast2D_RandomFGM
from utils.IGA import IGAMesh2D
from utils.multipatch import gen_vertex2patch2D, gen_edge_list, zip_conforming
from utils.assembly import gen_gauss_pts, stiff_elast_FGM_2D
from utils.boundary import boundary2D, applyBCElast2D
from utils.postprocessing import (plot_fields_2D,
                                  comp_measurement_values,
                                  get_measurements_vector,
                                  get_measurement_stresses)

def RBF(x1, x2, lengthscales):    
    diffs = np.expand_dims(x1 / lengthscales, 1) - \
            np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs**2, axis=2)
    return np.exp(-0.5 * r2)

def GRF(N, h, mean=0, variance=1, length_scale = 0.1):
    #N = 128#512
    jitter = 1e-10
    X = np.linspace(0, h, N)[:,None]
    K = RBF(X, X, length_scale)
    L = np.linalg.cholesky(K + jitter*np.eye(N))
    gp_sample = variance*np.dot(L, np.random.normal(size=N))+mean
    return X.flatten(), gp_sample


def createDatabase(num_refinements, model_data):
    # Approximation parameters
    
    p = q = 3 # Polynomial degree of the approximation
    num_refinements = num_refinements # Number of uniform refinements
    #output_filename = "elast2d_fixed_plate"
    
    # Step 0: Generate the geometry
    beam_length = model_data["beam_length"]
    beam_width  = model_data["beam_width"]
    vertices = [[0., 0.], [0., beam_width], [beam_length, 0.], [beam_length, beam_width]]
    patch1 = Quadrilateral(np.array(vertices))
    patch_list = [patch1]
        
    numPtsU = model_data["numPtsU"]
    numPtsV = model_data["numPtsV"]
    
    #Fixed Dirichlet B.C., u_y = 0 and u_x=0 for x=0
    def u_bound_dir_fixed(x,y): return [0., 0.]
    
    #  Neumann B.C. τ(x,y) = [x, y] on the top boundary
    
    bound_trac = model_data["trac_mean"]
    trac_var = model_data["trac_var"]
    trac_scale = model_data["trac_scale"]
    
    N_traction = model_data["N_traction"]
    N_material = model_data["N_material"]

    X, gp = GRF(N_traction, beam_length, mean=bound_trac, variance=trac_var*bound_trac, length_scale = trac_scale*beam_length)
    traction = np.zeros(numPtsU)
    
    global n
    n = -1
    
    def u_bound_neu(x,y,nx,ny):
        global n
        n = n+1
        traction[n] = np.interp(x, X, gp)
        #print(n)
        #print(x)
        #print(traction[n])
        return [0., -traction[n]]
    
    bound_up = boundary2D("Neumann", 0, "up", u_bound_neu)
    bound_left = boundary2D("Dirichlet", 0, "left", u_bound_dir_fixed)
    bound_right = boundary2D("Dirichlet", 0, "right", u_bound_dir_fixed)

    bound_all = [bound_up, bound_left, bound_right]
    
    # Step 1: Define the material properties
    Emod = model_data["E"]
    nu = model_data["nu"]
    e0_min = model_data["e0_min"]
    e0_max = model_data["e0_max"]
    e0_bot = random.uniform(e0_min, e0_max)
    e0_top = random.uniform(e0_min, e0_max)
    Emod_scale = model_data["Emod_scale"]
    
    material = MaterialElast2D_RandomFGM(N_material, Emod=Emod, e0_bot=e0_bot, e0_top=e0_top, nu=nu, vertices=vertices, length_scale = Emod_scale*beam_width, plane_type="stress")
   
    # Step 2: Degree elevate and refine the geometry
    t = time.time()
    for patch in patch_list:
        patch.degreeElev(p-1, q-1)
    elapsed = time.time() - t
    print("Degree elevation took ", elapsed, " seconds")
    
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
    
    """
    # test of random function for material

    x_coords = coord[:,0]
    y_coords = coord[:,1]

    constant_x = x_coords[201]
    indices = np.where((x_coords <= 1.001 * constant_x) & (x_coords >= 0.99 * constant_x))[0]
    print(indices.shape)
    plt.plot(y_coords[indices], E_modulus[indices])
    plt.xlabel('y Coordinates')
    plt.ylabel('E')
    plt.title('E with respect to X Coordinates where Y is constant')
    plt.show()
    
    constant_x = x_coords[100]
    indices = np.where((x_coords <= 1.001 * constant_x) & (x_coords >= 0.99 * constant_x))[0]
    plt.plot(y_coords[indices], E_modulus[indices])
    plt.xlabel('y Coordinates')
    plt.ylabel('E')
    plt.title('E with respect to X Coordinates where Y is constant')
    plt.show()

    constant_y = y_coords[10]
    indices = np.where(y_coords == constant_y)[0]
    plt.plot(x_coords[indices], E_modulus[indices])
    plt.xlabel('y Coordinates')
    plt.ylabel('E')
    plt.title('E with respect to X Coordinates where Y is constant')
    plt.show()
    """
    
    # Step 4. Apply boundary conditions
    t = time.time()
    # Assume no volume force
    rhs = np.zeros(2*size_basis)
    stiff, rhs = applyBCElast2D(mesh_list, bound_all, stiff, rhs, gauss_rule)
    elapsed = time.time() - t
    print("Applying B.C.s took ", elapsed, " seconds")
    
    """
    # test of random function for Traction 
    plt.figure()
    plt.plot(X, gp)
    plt.xlabel('X')
    plt.ylabel('gp')
    plt.title('Random function for Traction')
    plt.show()

    #plt.figure()
    #plt.plot(traction)
    #plt.xlabel('Index')
    #plt.ylabel('Values')
    #plt.title('Plotting Traction')
    #plt.show()
    """
    
    # Step 6. Solve the linear system
    t = time.time()
    sol0 = spsolve(stiff, rhs)
    elapsed = time.time() - t
    print("Linear sparse solver took ", elapsed, " seconds")
    
    # Step 7. Plot the solution to VTK
    #t = time.time()
    #plot_sol2D_elast(mesh_list, material.Cmat, sol0, output_filename)
    #elapsed = time.time() - t
    #print("Plotting the solution took ", elapsed, " seconds")
    
    # Step 7a. Plot the solution in matplotlib
    t = time.time()
    # compute the displacements at a set of uniformly spaced points
    num_pts_xi = numPtsU
    num_pts_eta = numPtsV
    num_fields = 2
    meas_vals_all, meas_pts_phys_xy_all, vals_disp_min, vals_disp_max = comp_measurement_values(num_pts_xi,
                                                                  num_pts_eta,
                                                                  mesh_list,
                                                                  sol0,
                                                                  get_measurements_vector,
                                                                  num_fields)
    elapsed = time.time() - t
    print("Computing the displacement values at measurement points took ", elapsed, " seconds")
    
    t = time.time()
    # compute the stresses at a set of uniformly spaced points
    num_pts_xi = numPtsU
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
    
    return gp, material.gp, disp2D


num_refinements = 3

nlength = 101
nwidth = 101
beam_lengths = np.linspace(1,11,nlength)
beam_widths = np.linspace(0.05,2.1,nwidth)

ndata = nlength*nwidth

model_data = dict()


model_data["trac_mean"] = 2e-1
model_data["trac_var"] = 0.75
model_data["trac_scale"] = 0.1
    
model_data["E"] = 200
model_data["nu"] =  0.333
model_data["e0_min"] = 0.0
model_data["e0_max"] = 0.9
model_data["Emod_scale"] = 0.25

#len_ratio = int(model_data["beam_length"]/model_data["beam_width"]/5)
# = len_ratio if len_ratio > 0 else 1

numPtsU = 128 #2**(num_refinements+4)
numPtsV = 32 #2**(num_refinements+4)//len_ratio

N_traction = 128 #numPtsU
N_material = 32 #numPtsV

model_data["numPtsU"] = numPtsU
model_data["numPtsV"] = numPtsV
model_data["N_traction"] = N_traction
model_data["N_material"] = N_material

domain = np.zeros([ndata, 2])
traction = np.zeros([ndata, N_traction])
material = np.zeros([ndata, N_material])
disp2D = np.zeros([ndata, numPtsU, numPtsV, 2])

ij = 0

for beam_length in beam_lengths:
    for beam_width in beam_widths:
        model_data["beam_length"] = beam_length
        model_data["beam_width"] = beam_width        
        t1 = default_timer()
        traction[ij,:], material[ij,:], disp2D[ij,:,:,:] = \
                            createDatabase(num_refinements, model_data)
        domain[ij,:] = [beam_length, beam_width]
        
        #plt.figure()
        #plt.plot(traction[i,:])
        #plt.xlabel('Index')
        #plt.ylabel('Values')
        #plt.title('Plotting a 1D Array')
        #plt.show()
        t2 = default_timer()
        print(ij, t2-t1)
        print('-------------------------------')
        ij += 1
    

np.save("traction_" f"{ndata}" "x" f"{N_traction}" ".npy", traction)
np.save("material_" f"{ndata}" "x" f"{N_material}" ".npy", material)
np.save("disp2D_" f"{ndata}" "x" f"{numPtsU}" "x" f"{numPtsV}" "x2" ".npy", disp2D)
np.save("domain_" f"{ndata}" "x2" ".npy", domain)




