#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 16:38:19 2021

@author: sg
"""
import numpy as np

# ------------------------------ network settings ---------------------------------------------------
iteration = 800
D_in = 2
H = 30
D_out = 2
lr = 0.0001
# ------------------------------ material parameter -------------------------------------------------
model_energy = 'NeoHookean2D'
E = 1000
nu = 0.3
rou = 1000 # 密度
param_c1 = 630
param_c2 = -1.2
param_c = 100
# ----------------------------- define structural parameters ---------------------------------------
Length = 4.0
Height = 1.0
Depth = 1.0
Period = 10.0 # 周期，计算的时间
# ------------------------------ define the value of neumal and dirichilet boundary condition -------------------------------
known_left_ux = 0 
known_left_uy = 0
bc_left_penalty = 1.0

known_down_tx = 0.
known_down_ty = 0.
bc_down_penalty = 1.0
bc_down_style = 'fix'

known_right_tx = 0
known_right_ty = -0.5 
bc_right_penalty = 1.0
bc_right_style = 'sin'

known_up_tx = 0.
known_up_ty = 0. 
bc_up_penalty = 1.0
bc_up_style = 'fix'
# ------------------------------ define the value of initial condition -------------------------------
initial_d = 0 # 初始位移全场为0
initial_v = 0 # 初始速度全场为0
# ------------------------------ define domain and collocation points -------------------------------
Nx = 200 # 120  # 120
Ny = 50 # 30  # 60
Nt = 100 # 时间间隔数目
# ------------------------------ define normal direction of the boundary -------------------------------
bc_left_normal2d = np.zeros((Ny, 2)) # left boundary
bc_left_normal2d[:, 0] = -1 

bc_down_normal2d = np.zeros((Nx, 2)) # left boundary
bc_down_normal2d[:, 1] = -1 

bc_right_normal2d = np.zeros((Ny, 2)) # left boundary
bc_right_normal2d[:, 0] = 1 

bc_up_normal2d = np.zeros((Nx, 2)) # left boundary
bc_up_normal2d[:, 1] = 1 


x_min, y_min, t_min = (0.0, 0.0, 0.0)
hx = Length / (Nx - 1)
hy = Height / (Ny - 1)
ht = Period / (Nt - 1)
shape = [Nx, Ny]
dxdy = [hx, hy]

# ------------------------------ data testing -------------------------------------------------------
num_test_x = 200
num_test_y = 50
num_test_t = 100

# ------------------------------ particular and distance tol -------------------------------------------------------
particular_tol = 10
distance_tol = 10