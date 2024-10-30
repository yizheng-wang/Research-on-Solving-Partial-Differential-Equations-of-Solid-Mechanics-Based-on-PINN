import numpy as np
import matplotlib.pyplot as plt

# 定义文件路径
folders = [
    'CPINNs_MLP_penalty',
    'BINN_MLP',
    'DEM_MLP_rbf',
    'KINN_CPINN_penalty',
    'BINN_KAN',
    'KINN_DEM_rbf'
]
base_path = './results/'  # 替换为你的文件夹路径

colors = ['b', 'g', 'r', 'b', 'g', 'r']  # 蓝，绿，红，青，洋红，黄
linestyles = ['--', '--', '--', '-', '-', '-']


# 加载数据并绘图
plt.figure(figsize=(10, 6))
for i, folder in enumerate(folders):
    error_path = f"{base_path}{folder}/error.npy"
    error_data = np.load(error_path)
    plt.semilogy(error_data, label=folder, color=colors[i], linestyle=linestyles[i], markersize=5)



plt.title('Comparison of Relative Errors by Different Algorithms')
plt.xlabel('Index')
plt.ylabel('Relative Error')
plt.legend()
plt.savefig('./results/Crack_error.pdf', dpi = 300)
plt.show()



# 定义文件路径
folders = [
    'DEM_MLP_rbf',
    'KINN_DEM_rbf'
]
base_path = './results/'  # 替换为你的文件夹路径

colors = ['b', 'g']  # 蓝，绿，红，青，洋红，黄
linestyles = ['--', '-']


# 加载数据并绘图
plt.figure(figsize=(10, 6))
for i, folder in enumerate(folders):
    error_path = f"{base_path}{folder}/error.npy"
    error_data = np.load(error_path)
    plt.semilogy(error_data, label=folder+'_uni', color=colors[i], linestyle=linestyles[0], markersize=5)

    error_path = f"{base_path}{folder}/error_tri.npy"
    error_data = np.load(error_path)
    plt.semilogy(error_data, label=folder+'_tri', color=colors[i], linestyle=linestyles[1], markersize=5)

plt.title('Comparison of Different Numerical Integration in DEM and KINN-DEM')
plt.xlabel('Index')
plt.ylabel('Relative Error')
plt.legend()
plt.savefig('./results/Crack_error_tri.pdf', dpi = 300)
plt.show()