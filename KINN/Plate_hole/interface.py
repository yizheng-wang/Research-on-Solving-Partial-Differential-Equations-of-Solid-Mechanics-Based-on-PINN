# -*- coding: utf-8 -*-
"""
Created on Sat May 25 18:06:48 2024

@author: admin
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# 定义文件夹路径和精确解文件路径
folders = [ "BINN_MLP", "KINN_BINN", "PINNs_MLP_penalty", "KINN_PINN_penalty", "DEM_MLP_rbf", "KINN_DEM_rbf"]
folders_in = [ 'BINN', 'KINN_BINN',  "PINN", "KINN_PINN", "DEM", "KINN_DEM",]
labels = ['y', 'y', 'x']
base_path = "./results/"  # 替换为实际的基础路径
exact_files = {
    "x0disy": os.path.join(base_path, "DEM_MLP_rbf", "exact_x0disy.npy"),
    "x0mise": os.path.join(base_path, "DEM_MLP_rbf", "exact_x0mise.npy"),
    "y0disx": os.path.join(base_path, "DEM_MLP_rbf", "exact_y0disx.npy")
}


def sparse_data(data, step=2):
    return data[:, ::step]

# 读取精确解
exact_solutions = {key: np.load(path) for key, path in exact_files.items()}

# 定义一个函数来读取数据
def load_data(folder, filename):
    path = os.path.join(base_path, folder, filename)
    return np.load(path) if os.path.exists(path) else None

# 比较并绘制图像
def plot_comparison(variable, exact_data, method_data, labels, title, xlabel):
    
    plt.figure(figsize=(10, 8))
    plt.plot(exact_data[0,:], exact_data[1,:], 'r', label="Reference", linewidth=8)
    colors = [ 'green', 'green' , 'blue', 'blue', 'cyan', 'cyan',]
    linestyles = ['--', '-', '--', '-','--', '-']
    # markers = ['o', 's', 'o', 's', 'o', 's']
    
    for data, label, color, linestyle in zip(method_data, labels, colors, linestyles):
        data = sparse_data(data)
        plt.plot(data[0, :], data[1, :], color=color, label=label, linestyle=linestyle)
    
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(variable, fontsize=20)
    plt.title(title, fontsize=20)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.savefig('./results/' + title + '.pdf', dpi=300)
    plt.show()

# 读取各个方法的数据并比较
for variable, xlabel in zip(["x0disy", "x0mise", "y0disx"], labels):
    exact_data = exact_solutions[variable]
    method_data = [load_data(folders[i], folders_in[i]+'_'+variable+'.npy') for i in range(len(folders))]
    valid_data = [data for data in method_data if data is not None]
    valid_labels = [folder for data, folder in zip(method_data, folders) if data is not None]
    plot_comparison(variable, exact_data, valid_data, valid_labels, f"Comparison of {variable}", xlabel)

