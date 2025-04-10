#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fourier Neural Operator for a 2D elasticity problem on a FGM beam with 
random distribution of Elasticity modulus under different random tensions

Problem statement:
    \Omega = (0,0.1)x(0,2) 
    Fixed BC: x = 0 and x = 2
    Traction \tau = GRF at y=0.1 in the vertical direction   
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from timeit import default_timer
import time
import os
import math

from utils.postprocessing_NO import plot_pred_timo
from utils.fno_2d  import FNO2d
from utils.fno_utils import count_params, LpLoss, train_fno
from utils.Solver import elastic_beam


torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device name:")
print(device)
if torch.cuda.is_available():
    torch.cuda.set_device(0) 
################################################################
#  configurations
################################################################
beam_length = 4.
beam_width = 1.0

numPtsU = 401
numPtsV = 101

N_traction = numPtsU
N_material = numPtsV

model_data = dict()
model_data["E"] = 200#200e3
model_data["nu"] = 0.333
model_data["beam_length"] = beam_length
model_data["beam_width"] = beam_width


model_data["numPtsU"] = numPtsU
model_data["numPtsV"] = numPtsV
model_data["N_traction"] = numPtsU
model_data["N_material"] = numPtsV

x_test_plot = numPtsU
y_test_plot = numPtsV

def test_result(f_name, f, Emod, nu, e0, beam_length, beam_width, porositydist):
    f_tensor = f.reshape(numPtsU, 1, 1)
    f_tensor = torch.from_numpy(f_tensor).float().to(device).unsqueeze(0)

    x_test_plot = np.linspace(0, beam_length, numPtsU).astype('float64')
    y_test_plot = np.linspace(0, beam_width, numPtsV).astype('float64')
    x_plot_grid, y_plot_grid = np.meshgrid(x_test_plot, y_test_plot)
    x_plot_grid = x_plot_grid.transpose()
    y_plot_grid = y_plot_grid.transpose()

    new_model_data = dict()
    new_model_data["E"] = Emod
    new_model_data["nu"] = nu
    new_model_data["beam_length"] = beam_length
    new_model_data["beam_width"] = beam_width
    new_model_data["numPtsU"] = numPtsU
    new_model_data["numPtsV"] = numPtsV
    new_model_data["N_traction"] = N_traction
    new_model_data["N_material"] = N_material

    yPhys = y_test_plot - beam_width/2
    if porositydist=="dist1":
        elasticity_modulus = Emod*(1-e0*np.cos(np.pi*(yPhys/beam_width)))
    elif porositydist =="dist2":
        elasticity_modulus = Emod*(1-e0*np.cos((np.pi/2.0*(yPhys/beam_width)) + np.pi/4))

    
    E_tensor = elasticity_modulus.reshape(1, numPtsV, 1)
    E_tensor = torch.from_numpy(E_tensor).float().to(device).unsqueeze(0)

    disp_IGA, stress_IGA, t_IGA = elastic_beam(f, elasticity_modulus, new_model_data)

    t = time.time()
    t_net = time.time() - t

    u_exact = disp_IGA[:, :, 0]
    v_exact = disp_IGA[:, :, 1]
    
    von_mise = stress_IGA[:, :, 3]
    if not os.path.exists(f"results/{f_name}/"):
        os.makedirs(f"results/{f_name}/")

    np.save(f"results/{f_name}/traction.npy", f)
    np.save(f"results/{f_name}/u_exact.npy", u_exact)
    np.save(f"results/{f_name}/v_exact.npy", v_exact)
    np.save(f"results/{f_name}/von_mise.npy", von_mise)
    np.save(f"results/{f_name}/x_plot_grid.npy", x_plot_grid)
    np.save(f"results/{f_name}/y_plot_grid.npy", y_plot_grid)


# testing f=const.

f_name = "const-dist1"
Emod = 200
nu =  1/3
e0 = 0.5

x_test_plot = np.linspace(0, beam_length, numPtsU).astype('float64')
y_test_plot = np.linspace(0, beam_width, numPtsV).astype('float64')


# ---------------------------
# testing f=(1/4)sin((4*pi/l)*x)+1/4

# ---------------------------
# testing f=linear
f_name = "const-dist1"
f = np.ones(x_test_plot.shape)*1.0
test_result(f_name, f, Emod, nu, e0, beam_length, beam_width, "dist1")


f_name = "const-dist2"
test_result(f_name, f, Emod, nu, e0, beam_length, beam_width, "dist2")

# ---------------------------