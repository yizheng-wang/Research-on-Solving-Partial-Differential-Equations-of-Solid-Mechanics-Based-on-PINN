import os
import numpy as np
import matplotlib.pyplot as plt

# 定义文件夹路径和文件名
folders = [ "DEM_MLP_rbf", "KINN_DEM_rbf", "KINN_PINN_penalty", "PINNs_MLP_penalty"]
base_path = "./results"  # 替换为实际的基础路径
files = ["U_mag_loss_array.npy", "Mise_loss_array.npy"]
labels = ["U_mag_loss_array", "Mise_loss_array"]

# 定义颜色和标记样式（不包括红色）
colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
markers = ['o', 's', 'D', '^', 'v', '*']

# 定义一个函数来读取数据
def load_data(folder, filename):
    path = os.path.join(base_path, folder, filename)
    return np.load(path) if os.path.exists(path) else None

# 比较并绘制图像
def plot_loss_comparison(filenames, title, y_label):
    plt.figure(figsize=(14, 8))
    for filename, color, marker, folder in zip(filenames, colors, markers, folders):
        data = load_data(folder, filename)
        print('Error ' + folder + str(data[-1]))
        if data is not None:
            plt.plot(data, label=folder, color=color, marker=marker, linestyle='-', markersize=6)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.yscale('log')  # y轴使用对数刻度
    plt.title(title, fontsize=25)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--")
    plt.savefig('./results/' + title + '.png', dpi=100)
    plt.show()

# 绘制U_mag_loss_array和Mise_loss_array的比较图
plot_loss_comparison(["U_mag_loss_array.npy"] * len(folders), "Comparison of U_mag_loss_array", "U_mag_loss (log scale)")
plot_loss_comparison(["Mise_loss_array.npy"] * len(folders), "Comparison of Mise_loss_array", "Mise_loss (log scale)")

