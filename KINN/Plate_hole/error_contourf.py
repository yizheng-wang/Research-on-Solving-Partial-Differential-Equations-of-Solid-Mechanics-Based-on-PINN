# -*- coding: utf-8 -*-
"""
输出相应算法的云图
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pyevtk.hl import gridToVTK
from pyevtk.hl import pointsToVTK
import matplotlib.tri as tri

# 把有限元解输出一下
def write_vtk_v2p_fem(filename, dom, U_mag,  SVonMises): #已经将输出的感兴趣场进行了分类VTK导出
    xx = np.ascontiguousarray(dom[:, 0])
    yy = np.ascontiguousarray(dom[:, 1])
    zz = np.ascontiguousarray(dom[:, 2]) # Points for VTK
    U_mag_contiguous = np.ascontiguousarray(U_mag)
    SVonMises_contiguous = np.ascontiguousarray(SVonMises)
    pointsToVTK(filename, xx, yy, zz, data={"S-VonMises": SVonMises, "U-mag": U_mag})

        
def generate_training_data(dom_num = 200, boundary_num = 1000):
    plate_size = Length
    hole_radius = Radius
    points = []
    for i in range(dom_num):
        for j in range(dom_num):
            x = i * plate_size / (dom_num - 1)
            y = j * plate_size / (dom_num - 1)
            if x**2 + y**2 >= hole_radius**2:
                points.append((x, y))
    points = np.array(points).astype(np.float32)
    
    # Create a Delaunay triangulation
    triangulation = tri.Triangulation(points[:, 0], points[:, 1])
    triangles = points[triangulation.triangles]
    # Sum of all triangle areas
    dom_point = triangles.mean(1)

    
    # 删除大于radius的点
    distances = np.sqrt(np.sum(dom_point**2, axis=1))
    filtered_points = dom_point[distances >= hole_radius]



    # Generate points along the vertical and horizontal boundaries
    left_wall = np.array([[0, y] for y in np.linspace(0, plate_size, boundary_num)]).astype(np.float32)
    bottom_wall = np.array([[x, 0] for x in np.linspace(hole_radius, plate_size, boundary_num)]).astype(np.float32)
    right_wall = np.array([[Length, y] for y in np.linspace(0, plate_size, boundary_num)]).astype(np.float32)
    up_wall = np.array([[x, Length] for x in np.linspace(0, plate_size, boundary_num)]).astype(np.float32)
    # Generate points along the circular arc
    theta = np.linspace(0, np.pi/2, boundary_num)
    arc = np.array([[hole_radius * np.cos(t), hole_radius* np.sin(t)] for t in theta]).astype(np.float32)

    # Combine all points
    boundary_points = {'l': left_wall, 'd': bottom_wall, 'r': right_wall, 'u': up_wall, 'c': arc}
    
    return filtered_points, boundary_points

Radius = 5.
Length = 20.
dom_p, boundary_p = generate_training_data()
z = np.zeros((dom_p.shape[0], 1))
datatest = np.concatenate((dom_p, z), 1)

U_mag_exact = np.load('./abaqus_reference/onehole/U_mag.npy')[:,1]
MISES_exact = np.load('./abaqus_reference/onehole/MISES.npy')[:,1]
write_vtk_v2p_fem('./abaqus_reference/onehole/Plate_hole_FEM', datatest, U_mag_exact, MISES_exact)


