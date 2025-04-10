import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # 导入平滑函数

# 加载文件A、B和C
FGM1_l2 = np.load('output/dem_e20W/FGM1_MLP_trap_L2_norm.npy')
FGM2_l2 = np.load('output/dem_e20W/FGM2_MLP_trap_L2_norm.npy')

FGM1_h1 = np.load('output/dem_e20W/FGM1_MLP_trap_H1_norm.npy')
FGM2_h1 = np.load('output/dem_e20W/FGM2_MLP_trap_H1_norm.npy')
# 提取psi_error和omega_error

# LoRA data
FGM1to2_MLP_trap_L2_norm_lora_r4 = np.load('output/dem_lora/r=4/FGM1to2_MLP_trap_L2_norm.npy')
# LoRA data
FGM2to1_MLP_trap_L2_norm_lora_r4 = np.load('output/dem_lora/r=4/FGM2to1_MLP_trap_L2_norm.npy')

FGM1to2_MLP_trap_H1_norm_lora_r4 = np.load('output/dem_lora/r=4/FGM1to2_MLP_trap_H1_norm.npy')
# LoRA data
FGM2to1_MLP_trap_H1_norm_lora_r4 = np.load('output/dem_lora/r=4/FGM2to1_MLP_trap_H1_norm.npy')


# FGM1to2_MLP_trap_L2_norm_lora_r4 = np.hstack([FGM2_l2[:100000], FGM1to2_MLP_trap_L2_norm_lora_r4])
# FGM2to1_MLP_trap_L2_norm_lora_r4 = np.hstack([FGM1_l2[:100000], FGM2to1_MLP_trap_L2_norm_lora_r4])
# FGM1to2_MLP_trap_H1_norm_lora_r4 = np.hstack([FGM2_h1[:100000], FGM1to2_MLP_trap_H1_norm_lora_r4])
# FGM2to1_MLP_trap_H1_norm_lora_r4 = np.hstack([FGM1_h1[:100000], FGM2to1_MLP_trap_H1_norm_lora_r4])

sm = 31
# # 平滑曲线（Savitzky-Golay滤波器，窗口大小=51，平滑度=3）
FGM1_l2_smooth = savgol_filter(FGM1_l2, window_length=sm, polyorder=5)
FGM2_l2_smooth = savgol_filter(FGM2_l2, window_length=sm, polyorder=5)

FGM1_h1_smooth = savgol_filter(FGM1_h1, window_length=sm, polyorder=5)
FGM2_h1_smooth = savgol_filter(FGM2_h1, window_length=sm, polyorder=5)

# # 平滑曲线（Savitzky-Golay滤波器，窗口大小=51，平滑度=3）
FGM1to2_l2_lora_smooth = savgol_filter(FGM1to2_MLP_trap_L2_norm_lora_r4, window_length=sm, polyorder=5)
FGM2to1_l2_lora_smooth = savgol_filter(FGM2to1_MLP_trap_L2_norm_lora_r4, window_length=sm, polyorder=5)

FGM1to2_h1_lora_smooth = savgol_filter(FGM1to2_MLP_trap_H1_norm_lora_r4, window_length=sm, polyorder=5)
FGM2to1_h1_lora_smooth = savgol_filter(FGM2to1_MLP_trap_H1_norm_lora_r4, window_length=sm, polyorder=5)


epoch = np.linspace(100000, 200000-1, 100000)
# 创建一个图形，绘制psi_error的对比
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)  # 左侧图
plt.yscale('log')  # 设置y轴为对数坐标
plt.plot(FGM1_l2_smooth, label='Symmetric', color='blue')
plt.plot(epoch, FGM2to1_l2_lora_smooth, label='Asy->Sym', color='red')
plt.xlabel('Iteration')
plt.ylabel(r'Relative error: $\mathcal{L}_{2}$', fontsize=20)
plt.legend()
# plt.title('Comparison of psi_error')

# 绘制omega_error的对比
plt.subplot(1, 2, 2)  # 右侧图
plt.yscale('log')  # 设置y轴为对数坐标
plt.plot(FGM1_h1_smooth, label='Symmetric', color='blue')
plt.plot(epoch, FGM2to1_h1_lora_smooth, label='Asy->Sym', color='red')
plt.xlabel('Iteration')
plt.ylabel(r'Relative error: $\mathcal{H}_{1}$', fontsize=20)
plt.legend()
# plt.title('Comparison of omega_error')

# 显示图形
plt.tight_layout()

# 保存图形为PNG文件
plt.savefig(f'./pic/FGM_DEM_error_comparision_scenario_fused.pdf', dpi = 500)

# 显示图形
plt.show()


