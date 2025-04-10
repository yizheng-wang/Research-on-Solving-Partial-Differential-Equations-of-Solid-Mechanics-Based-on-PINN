import os
import numpy as np
import matplotlib.pyplot as plt

# 设置基础路径和 r 的取值范围
base_paths = ["./elipse2circle_lora", "./circle2elipse_lora"]
r_values = [1] + list(range(4, 101, 4))

# 需要提取的文件名
file_names = [
    "U_mag_loss_array.npy",
    "Mise_loss_array.npy"
]

# 存储不同文件的数据
data_dict = {name: {path: [] for path in base_paths} for name in file_names}

# 遍历文件夹并提取数据
for base_path in base_paths:
    for r in r_values:
        folder_path = os.path.join(base_path, f"r={r}")
        for file_name in file_names:
            file_path = os.path.join(folder_path, file_name)
            if os.path.exists(file_path):
                data = np.load(file_path, allow_pickle=True)
                data_dict[file_name][base_path].append((r, np.mean(data[-1000:])))  # 取最后一个值

# 绘制图像
plt.figure(figsize=(12, 10))

for i, file_name in enumerate(file_names):
    for j, base_path in enumerate(base_paths):
        data_array = np.array(data_dict[file_name][base_path])
        plt.subplot(2, 2, i * 2 + j + 1)
        plt.plot(data_array[:, 0], data_array[:, 1], marker="o", linestyle="-", label=f"{file_name} ({base_path})")
        plt.xlabel("r values")
        plt.ylabel("Last value of data")
        plt.title(f"{file_name} ({base_path})")
        plt.grid(True)
        plt.yscale("log")
        plt.legend()

plt.tight_layout()
plt.savefig("Loss_vs_r_values.png", dpi=300)
plt.show()
