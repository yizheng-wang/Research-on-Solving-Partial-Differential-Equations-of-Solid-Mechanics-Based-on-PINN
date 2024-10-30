import numpy as np
import matplotlib.pyplot as plt
import os

# 定义文件路径
directory = './output/dem/'

# 定义文件名
files_y05 = {
    "KAN_mont": "NeoHook_KAN_mont_y05.npy",
    "KAN_simp": "NeoHook_KAN_simp_y05.npy",
    "KAN_trap": "NeoHook_KAN_trap_y05.npy",
    "MLP_mont": "NeoHook_MLP_mont_y05.npy",
    "MLP_simp": "NeoHook_MLP_simp_y05.npy",
    "MLP_trap": "NeoHook_MLP_trap_y05.npy" # Assuming there is a file for FEM data
}

files_x2 = {
    "KAN_mont": "NeoHook_KAN_mont_x2.npy",
    "KAN_simp": "NeoHook_KAN_simp_x2.npy",
    "KAN_trap": "NeoHook_KAN_trap_x2.npy",
    "MLP_mont": "NeoHook_MLP_mont_x2.npy",
    "MLP_simp": "NeoHook_MLP_simp_x2.npy",
    "MLP_trap": "NeoHook_MLP_trap_x2.npy"  # Assuming there is a file for FEM data
}

# 读取数据
def load_data(files_dict):
    data_dict = {}
    for key, file in files_dict.items():
        data_dict[key] = np.load(os.path.join(directory, file), allow_pickle=True).item()
    return data_dict

data_y05 = load_data(files_y05)
data_x2 = load_data(files_x2)

# 定义绘图函数
def plot_data_y05(data_dict, title, ylabel, pred_key, fem_key="Dis_fem"):
    plt.figure(figsize=(10, 6))
    
    # 定义样式
    styles = {
        "KAN_mont": ("-", "purple"),
        "KAN_simp": ("-", "g"),
        "KAN_trap": ("-", "b"),
        "MLP_mont": ("--", "purple"),
        "MLP_simp": ("--", "g"),
        "MLP_trap": ("--", "b"),
        "FEM": ("-", "m")
    }
    index = 0
    # 绘制每条曲线
    for key, data in data_dict.items():
        x = data['X'][:, 0]
        if pred_key =='Dis_pred' and index==0:
            y = data['Dis_fem'].flatten()
            plt.plot(x, y, color='r', label='FEM', linewidth = 5)
            index = index + 1
        if pred_key =='Von_pred' and index==0:
            y = data['Von_exact'].flatten()
            plt.plot(x, y, color='r', label='FEM', linewidth = 5)
            index = index + 1
        y = data[pred_key].flatten()
        linestyle, color = styles[key]
        label = f"{key} {pred_key}" 
        plt.plot(x, y, linestyle=linestyle, color=color, label=label)
 
    # 画精确解

    plt.title(title, fontsize = 18)
    plt.xlabel('X', fontsize = 15)
    plt.ylabel(ylabel, fontsize = 15)
    plt.legend(fontsize = 15)
    plt.grid(True)
    plt.savefig('./output/'+title+'.pdf')
    plt.show()


def plot_data_x2(data_dict, title, ylabel, pred_key, fem_key="Dis_fem"):
    plt.figure(figsize=(10, 6))
    
    # 定义样式
    styles = {
        "KAN_mont": ("-", "purple"),
        "KAN_simp": ("-", "g"),
        "KAN_trap": ("-", "b"),
        "MLP_mont": ("--", "purple"),
        "MLP_simp": ("--", "g"),
        "MLP_trap": ("--", "b"),
        "FEM": ("-", "m")
    }
    
    # 绘制每条曲线
    index = 0
    for key, data in data_dict.items():
        x = data['X'][:, 1]
        if pred_key =='Dis_pred' and index==0:
            y = data['Dis_fem'].flatten()
            plt.plot(x, y, color='r', label='FEM', linewidth = 5)
            index = index + 1
        if pred_key =='Von_pred' and index==0:
            y = data['Von_exact'].flatten()
            plt.plot(x, y, color='r', label='FEM', linewidth = 5)
            index = index + 1
        y = data[pred_key].flatten()
        linestyle, color = styles[key]
        label = f"{key} {pred_key}"

        plt.plot(x, y, linestyle=linestyle, color=color, label=label)
     
    plt.title(title, fontsize = 18)
    plt.xlabel('X', fontsize = 15)
    plt.ylabel(ylabel, fontsize = 15)
    plt.legend(fontsize = 15)
    plt.grid(True)
    plt.savefig('./output/'+title+'.pdf')
    plt.show()
# 绘制图像
plot_data_y05(data_y05, "Displacement Magnitude (y05)", "Displacement Magnitude", "Dis_pred")
plot_data_y05(data_y05, "Von Mises Stress (y05)", "Von Mises Stress", "Von_pred")
plot_data_x2(data_x2, "Displacement Magnitude (x2)", "Displacement Magnitude", "Dis_pred")
plot_data_x2(data_x2, "Von Mises Stress (x2)", "Von Mises Stress", "Von_pred")
