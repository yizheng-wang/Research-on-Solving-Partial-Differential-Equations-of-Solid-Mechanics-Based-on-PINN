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
param_c1 = 630
param_c2 = -1.2
param_c = 100
# ----------------------------- define structural parameters ---------------------------------------
Length = 20.
Height = 20.
Depth = 1.0
known_left_ux = 0
known_left_uy = 0 # 将左边界的y方向的位移先给定为0，实际上这里是力边界条件，只不过力边界条件是0
bc_left_penalty = 1.0

known_down_ux = 0 # 将下边界的x方向的位移先给定为0，实际上这里是力边界条件，只不过力边界条件是0
known_down_uy = 0
bc_down_penalty = 1.0

known_right_tx = 100.0
known_right_ty = 0.
bc_right_penalty = 1.0
# ------------------------------ define domain and collocation points -------------------------------
r = 5
Nx = 401 # 120  # 120
Ny = 401 # 30  # 60
bc_left_normal2d = np.zeros((Ny, 2)) # define 2D dimensional normal direction 
bc_left_normal2d[:, 0] = -1 #------------- specify the normal direction at the dir condition for hw functional
bc_down_normal2d = np.zeros((Nx, 2)) # define 2D dimensional normal direction 
bc_down_normal2d[:, 1] = -1 #------------- specify the normal direction at the dir condition for hw functional

x_min, y_min = (0.0, 0.0)
hx = Length / (Nx - 1)
hy = Height / (Ny - 1)
shape = [Nx, Ny]
dxdy = [hx, hy]

# ------------------------------ data testing -------------------------------------------------------
num_test_x = 201
num_test_y = 201