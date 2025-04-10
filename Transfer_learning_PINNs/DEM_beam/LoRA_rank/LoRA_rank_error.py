import os
import numpy as np
import matplotlib.pyplot as plt

# 设置基础路径和 r 的取值范围
base_path = "./"
r_values = [1] + list(range(4, 101, 4))

# 需要提取的文件名
file_names = [
    "FGM1to2_MLP_trap_H1_norm.npy",
    "FGM1to2_MLP_trap_L2_norm.npy",
    "FGM2to1_MLP_trap_H1_norm.npy",
    "FGM2to1_MLP_trap_L2_norm.npy"
]

# 存储不同文件的数据
data_dict = {name: [] for name in file_names}

# 遍历文件夹并提取数据
for r in r_values:
    folder_path = os.path.join(base_path, f"r={r}")
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            data = np.load(file_path, allow_pickle=True)
            data_dict[file_name].append((r, np.mean(data[-1000:])))  # 取最后一个值

# 绘制图像
plt.figure(figsize=(12, 8))

for i, file_name in enumerate(file_names):
    data_array = np.array(data_dict[file_name])
    plt.subplot(2, 2, i+1)
    plt.plot(data_array[:, 0], data_array[:, 1], marker="o", linestyle="-", label=file_name)
    plt.xlabel("r values")
    plt.ylabel("Last value of data")
    plt.title(file_name)
    plt.grid(True)
    plt.yscale("log")
    plt.legend()

plt.tight_layout()
plt.savefig("FGM_norm_vs_r_values.png", dpi=300)
plt.show()
