import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


base_path = "./results/"  # 替换为实际的基础路径
# 定义文件路径
file_paths = {
    'flower_BINN_KAN_error': base_path + 'flower_BINN_KAN_error.npy',
    'flower_BINN_MLP_error': base_path + 'flower_BINN_MLP_error.npy',
    'koch_DEM_KAN_error': base_path + 'koch_DEM_KAN_error.npy',
    'koch_DEM_MLP_error': base_path + 'koch_DEM_MLP_error.npy',
    'koch_PINN_KAN_error': base_path + 'koch_PINN_KAN_error.npy',
    'koch_PINN_MLP_error': base_path + 'koch_PINN_MLP_error.npy'
}

# 加载数据
data = {name: np.load(path) for name, path in file_paths.items()}

# 提取数据
dem_kan = data['koch_DEM_KAN_error']
dem_mlp = data['koch_DEM_MLP_error']
pinn_kan = data['koch_PINN_KAN_error']
pinn_mlp = data['koch_PINN_MLP_error']
binn_mlp = data['flower_BINN_MLP_error']
binn_kan = data['flower_BINN_KAN_error']

# 绘制DEM和PINN数组
plt.figure(figsize=(10, 6))
plt.plot(dem_kan, label='DEM KAN', color='blue', linestyle='-')
plt.plot(dem_mlp, label='DEM MLP', color='blue', linestyle='--')
plt.plot(pinn_kan, label='PINN KAN', color='red', linestyle='-')
plt.plot(pinn_mlp, label='PINN MLP', color='red', linestyle='--')
plt.legend(fontsize=20)
plt.title('DEM and PINN Error Evolution', fontsize=25)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Relative Error', fontsize=14)
plt.yscale('log')  # y轴使用对数刻度
plt.grid(True)
plt.savefig('./pic/DEM_PINN_error_evolution.pdf', dpi = 300)
plt.show()


# 移动平均法
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# 使用 Savitzky-Golay 滤波器平滑数据
def smooth_data(data, window_size, poly_order):
    return savgol_filter(data, window_size, poly_order)

# 设置参数
window_size = 11  # 窗口大小
poly_order = 1    # 多项式阶数
binn_kan_smooth = smooth_data(binn_kan, window_size, poly_order)
binn_mlp_smooth = smooth_data(binn_mlp, window_size, poly_order)
# 绘制BINN数组
plt.figure(figsize=(10, 6))
plt.plot(binn_kan_smooth, label='BINN KAN', color='r', linestyle='-')
plt.plot(binn_mlp_smooth, label='BINN MLP', color='b', linestyle='--')
plt.legend(fontsize=20)
plt.title('BINN Error Evolution', fontsize=25)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Relative Error', fontsize=14)
plt.yscale('log')  # y轴使用对数刻度
plt.yticks([1e-0, 1e-1, 1e-2, 1e-3, 1e-4])  # 设置 y 轴刻度
plt.grid(True)
plt.savefig('./pic/BINN_error_evolution.pdf', dpi = 300)
plt.show()
