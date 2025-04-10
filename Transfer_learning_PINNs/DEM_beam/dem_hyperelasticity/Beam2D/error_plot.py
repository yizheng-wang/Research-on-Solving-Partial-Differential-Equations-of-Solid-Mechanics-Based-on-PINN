import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # 导入平滑函数

# 加载文件A、B和C
FGM1_l2 = np.load('output/dem/FGM1_MLP_trap_L2_norm.npy')
FGM2_l2 = np.load('output/dem/FGM2_MLP_trap_L2_norm.npy')

FGM1_h1 = np.load('output/dem/FGM1_MLP_trap_H1_norm.npy')
FGM2_h1 = np.load('output/dem/FGM2_MLP_trap_H1_norm.npy')
# 提取psi_error和omega_error


# # 平滑曲线（Savitzky-Golay滤波器，窗口大小=51，平滑度=3）
FGM1_l2_smooth = savgol_filter(FGM1_l2, window_length=10, polyorder=5)
FGM2_l2_smooth = savgol_filter(FGM2_l2, window_length=10, polyorder=5)

FGM1_h1_smooth = savgol_filter(FGM1_h1, window_length=10, polyorder=5)
FGM2_h1_smooth = savgol_filter(FGM2_h1, window_length=10, polyorder=5)

# 创建一个图形，绘制psi_error的对比
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)  # 左侧图
plt.yscale('log')  # 设置y轴为对数坐标
plt.plot(FGM1_l2_smooth, label='Symmetric', color='blue')
plt.plot(FGM2_l2_smooth, label='Asymmetric', color='red')
plt.xlabel('Iteration')
plt.ylabel(r'Relative error: $\mathcal{L}_{2}$', fontsize=20)
plt.legend()
# plt.title('Comparison of psi_error')

# 绘制omega_error的对比
plt.subplot(1, 2, 2)  # 右侧图
plt.yscale('log')  # 设置y轴为对数坐标
plt.plot(FGM1_h1_smooth, label='Symmetric', color='blue')
plt.plot(FGM2_h1_smooth, label='Asymmetric', color='red')
plt.xlabel('Iteration')
plt.ylabel(r'Relative error: $\mathcal{H}_{1}$', fontsize=20)
plt.legend()
# plt.title('Comparison of omega_error')

# 显示图形
plt.tight_layout()

# 保存图形为PNG文件
plt.savefig(f'./pic/FGM_DEM_error_comparision.pdf', dpi = 500)

# 显示图形
plt.show()


