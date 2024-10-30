import numpy as np
import matplotlib.pyplot as plt
import os

# 请将路径修改为您的文件存放路径
directory = './output/dem/'

# 定义文件名
files_L2 = [
    "NeoHook_MLP_trap_L2_norm.npy",
    "NeoHook_MLP_simp_L2_norm.npy",
    "NeoHook_MLP_mont_L2_norm.npy",
    "NeoHook_KAN_trap_L2_norm.npy",
    "NeoHook_KAN_simp_L2_norm.npy",
    "NeoHook_KAN_mont_L2_norm.npy"
]

files_H1 = [
    "NeoHook_MLP_trap_H1_norm.npy",
    "NeoHook_MLP_simp_H1_norm.npy",
    "NeoHook_MLP_mont_H1_norm.npy",
    "NeoHook_KAN_trap_H1_norm.npy",
    "NeoHook_KAN_simp_H1_norm.npy",
    "NeoHook_KAN_mont_H1_norm.npy"
]

# 读取数据
data_L2 = [np.load(os.path.join(directory, file)) for file in files_L2]
data_H1 = [np.load(os.path.join(directory, file)) for file in files_H1]

# 定义绘图函数
def plot_error_evolution(data, title, ylabel):
    plt.figure(figsize=(10, 6))
    
    # 定义样式
    styles = ['--', '--', '--', '-', '-', '-']
    colors = ['r', 'g', 'b', 'r', 'g', 'b']
    labels = [
        "MLP Trap", "MLP Simp", "MLP Mont",
        "KAN Trap", "KAN Simp", "KAN Mont"
    ]
    
    # 绘制每条曲线
    for i, d in enumerate(data):
        plt.plot(d, linestyle=styles[i], color=colors[i], label=labels[i])
    
    plt.title(title, fontsize = 20)
    plt.xlabel('Iteration',fontsize = 20)
    plt.ylabel(ylabel, fontsize = 20)
    plt.yscale('log')
    plt.legend(fontsize = 15)
    plt.grid(True)
    plt.savefig('./output/'+title+'.pdf', dpi = 300)
    plt.show()

# 绘制L2误差演化图
plot_error_evolution(data_L2, "L2 Norm Error Evolution", "L2 Norm Error")

# 绘制H1误差演化图
plot_error_evolution(data_H1, "H1 Norm Error Evolution", "H1 Norm Error")
