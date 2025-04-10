import os
import numpy as np
import matplotlib.pyplot as plt

# 定义文件夹路径和文件名
folders = ["circle", "elipse"]
base_path = "./results"  # 替换为实际的基础路径


# 定义颜色和标记样式（不包括红色）
colors = ['blue', 'green']
markers = ['o', 's', 'D', '^', 'v', '*']

# 定义滑动平均函数
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# 定义一个函数来读取数据
def load_data(folder, filename):
    path = os.path.join(base_path, folder, filename)
    return np.load(path) if os.path.exists(path) else None

# 比较并绘制图像（带平滑处理）
def plot_loss_comparison_smoothed(filenames, title, y_label, smooth_window=500):
    plt.figure(figsize=(14, 8))
    for filename, color, marker, folder in zip(filenames, colors, markers, folders):
        data = load_data(folder, filename)
        if data is not None:
            smoothed_data = moving_average(data, window_size=smooth_window)  # 平滑数据
            print(f'Error {folder}: Original={data[-1]:.4f}, Smoothed={smoothed_data[-1]:.4f}')
            plt.plot(
                range(len(smoothed_data)),  # 确保x轴对齐
                smoothed_data,
                label=folder,
                color=color,
                marker=marker,
                linestyle='-',
                markersize=6
            )
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.yscale('log')  # y轴使用对数刻度
    plt.title(title, fontsize=25)
    plt.legend(fontsize=30)
    plt.grid(True, which="both", ls="--")
    plt.savefig('./pic/' + title + '.png', dpi=100)
    plt.show()

# 绘制U_mag_loss_array和Mise_loss_array的比较图（带平滑处理）
plot_loss_comparison_smoothed(["U_mag_loss_array.npy"] * len(folders), "Comparison of U_mag_loss_array", "U_mag_loss (log scale)")


def plot_loss_comparison_smoothed(filenames, title, y_label, smooth_window=500):
    plt.figure(figsize=(14, 8))
    for filename, color, marker, folder in zip(filenames, colors, markers, folders):
        data = load_data(folder, filename)
        if data is not None:
            smoothed_data = moving_average(data, window_size=smooth_window)  # 平滑数据
            print(f'Error {folder}: Original={data[-1]:.4f}, Smoothed={smoothed_data[-1]:.4f}')
            plt.plot(
                range(len(smoothed_data)),  # 确保x轴对齐
                smoothed_data,
                label=folder,
                color=color,
                marker=marker,
                linestyle='-',
                markersize=6
            )
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.yscale('log')  # y轴使用对数刻度
    plt.title(title, fontsize=25)
    plt.legend(fontsize=30)
    plt.grid(True, which="both", ls="--")
    plt.savefig('./pic/' + title + '.png', dpi=100)
    plt.show()

# 绘制U_mag_loss_array和Mise_loss_array的比较图（带平滑处理）
plot_loss_comparison_smoothed(["Mise_loss_array.npy"] * len(folders), "Comparison of Mise_loss_array", "Mise_loss (log scale)")
