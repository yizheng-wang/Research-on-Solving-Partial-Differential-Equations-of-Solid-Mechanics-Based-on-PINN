import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # 导入平滑函数

# 加载文件A、B和C
A = np.load('results/PINN/loss_error_results_w_3.14.npy', allow_pickle=True).item()
B = np.load('results/PINN/loss_error_results_w_6.28.npy', allow_pickle=True).item()
C = np.load('results/PINN/loss_error_results_w_9.42.npy', allow_pickle=True).item()

# 提取psi_error和omega_error
psi_error_A = A['psi_error_t']
psi_error_B = B['psi_error_t']
psi_error_C = C['psi_error_t']
omega_error_A = A['omega_error_t']
omega_error_B = B['omega_error_t']
omega_error_C = C['omega_error_t']

# 平滑曲线（Savitzky-Golay滤波器，窗口大小=51，平滑度=3）
psi_error_A_smooth = savgol_filter(psi_error_A, window_length=50, polyorder=5)
psi_error_B_smooth = savgol_filter(psi_error_B, window_length=50, polyorder=5)
psi_error_C_smooth = savgol_filter(psi_error_C, window_length=50, polyorder=5)

omega_error_A_smooth = savgol_filter(omega_error_A, window_length=10, polyorder=5)
omega_error_B_smooth = savgol_filter(omega_error_B, window_length=10, polyorder=5)
omega_error_C_smooth = savgol_filter(omega_error_C, window_length=10, polyorder=5)

# 创建一个图形，绘制psi_error的对比
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)  # 左侧图
plt.yscale('log')  # 设置y轴为对数坐标
plt.plot(psi_error_A_smooth, label='w=1.0*3.14', color='blue')
plt.plot(psi_error_B_smooth, label='w=2.0*3.14', color='red')
plt.plot(psi_error_C_smooth, label='w=3.0*3.14', color='green')
plt.xlabel('Iteration/Index')
plt.ylabel('psi_error')
plt.legend()
plt.title('Comparison of psi_error')

# 绘制omega_error的对比
plt.subplot(1, 2, 2)  # 右侧图
plt.yscale('log')  # 设置y轴为对数坐标
plt.plot(omega_error_A_smooth, label='w=1.0*3.14', color='blue')
plt.plot(omega_error_B_smooth, label='w=2.0*3.14', color='red')
plt.plot(omega_error_C_smooth, label='w=3.0*3.14', color='green')
plt.xlabel('Iteration/Index')
plt.ylabel('omega_error')
plt.legend()
plt.title('Comparison of omega_error')

# 显示图形
plt.tight_layout()

# 保存图形为PNG文件
plt.savefig(f'./pic/PINN/error_comparision.png')

# 显示图形
plt.show()




import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # 导入平滑函数

# 加载文件A、B和C
A = np.load('results/KINN/loss_error_results_w_3.14.npy', allow_pickle=True).item()
B = np.load('results/KINN/loss_error_results_w_6.28.npy', allow_pickle=True).item()
C = np.load('results/KINN/loss_error_results_w_9.42.npy', allow_pickle=True).item()

# 提取psi_error和omega_error
psi_error_A = A['psi_error_t']
psi_error_B = B['psi_error_t']
psi_error_C = C['psi_error_t']
omega_error_A = A['omega_error_t']
omega_error_B = B['omega_error_t']
omega_error_C = C['omega_error_t']

# 平滑曲线（Savitzky-Golay滤波器，窗口大小=51，平滑度=3）
psi_error_A_smooth = savgol_filter(psi_error_A, window_length=10, polyorder=5)
psi_error_B_smooth = savgol_filter(psi_error_B, window_length=10, polyorder=5)
psi_error_C_smooth = savgol_filter(psi_error_C, window_length=10, polyorder=5)

omega_error_A_smooth = savgol_filter(omega_error_A, window_length=10, polyorder=5)
omega_error_B_smooth = savgol_filter(omega_error_B, window_length=10, polyorder=5)
omega_error_C_smooth = savgol_filter(omega_error_C, window_length=10, polyorder=5)

# 创建一个图形，绘制psi_error的对比
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)  # 左侧图
plt.yscale('log')  # 设置y轴为对数坐标
plt.plot(psi_error_A_smooth, label='w=1.0*3.14', color='blue')
plt.plot(psi_error_B_smooth, label='w=2.0*3.14', color='red')
plt.plot(psi_error_C_smooth, label='w=3.0*3.14', color='green')
plt.xlabel('Iteration/Index')
plt.ylabel('psi_error')
plt.legend()
plt.title('Comparison of psi_error')

# 绘制omega_error的对比
plt.subplot(1, 2, 2)  # 右侧图
plt.yscale('log')  # 设置y轴为对数坐标
plt.plot(omega_error_A_smooth, label='w=1.0*3.14', color='blue')
plt.plot(omega_error_B_smooth, label='w=2.0*3.14', color='red')
plt.plot(omega_error_C_smooth, label='w=3.0*3.14', color='green')
plt.xlabel('Iteration/Index')
plt.ylabel('omega_error')
plt.legend()
plt.title('Comparison of omega_error')

# 显示图形
plt.tight_layout()

# 保存图形为PNG文件
plt.savefig(f'./pic/KINN/error_comparision.png')

# 显示图形
plt.show()
