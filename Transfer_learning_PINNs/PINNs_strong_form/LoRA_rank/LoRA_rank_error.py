import os
import numpy as np
import matplotlib.pyplot as plt

# 设置基础路径和 r 的取值范围
base_path = "./"
r_values = [1] + list(range(4, 101, 4))

# 定义 wi 和 wo 的组合
wi_values = [3.14, 6.28, 9.42]
wo_values = [3.14, 6.28, 9.42]
combinations = [(wi, wo) for wi in wi_values for wo in wo_values if wi != wo]
fs = 20
# 遍历所有组合并绘图
for wi, wo in combinations:
    # 需要提取的数据文件名
    file_pattern = f"loss_error_results_wi_{wi}_wo_{wo}.npy"

    # 存储 r 和 psi_error_t 最后一个值
    psi_error_last_values = []

    # 遍历文件夹并提取数据
    for r in r_values:
        file_path = os.path.join(base_path, f"r={r}", file_pattern)
        if os.path.exists(file_path):
            data = np.load(file_path, allow_pickle=True).item()
            if "psi_error_t" in data:
                psi_error_last_values.append((r, np.mean(data["psi_error_t"][-1000:])))

    # 如果有数据，则绘制图像
    if psi_error_last_values:
        psi_error_last_values = np.array(psi_error_last_values)

        plt.figure(figsize=(10, 6))
        plt.plot(psi_error_last_values[:, 0], psi_error_last_values[:, 1], marker="o", linestyle="-", color="b")

        plt.xlabel("r values", fontsize = fs)
        plt.ylabel(r'Relative error: $\mathcal{L}_{2}$', fontsize = fs)
        plt.title(f"Psi Error vs r values (w: {wi}->{wo})", fontsize = fs)
        plt.grid(True)
        plt.xscale("linear")  # 可以改成 'log' 如果适用
        plt.yscale("log")

        # 保存并展示图像
        plt.savefig(f"psi_error_vs_r_values_wi_{wi}_wo_{wo}.png", dpi=300)
        plt.show()


# 遍历所有组合并绘图
for wi, wo in combinations:
    # 需要提取的数据文件名
    file_pattern = f"loss_error_results_wi_{wi}_wo_{wo}.npy"

    # 存储 r 和 psi_error_t 最后一个值
    psi_error_last_values = []

    # 遍历文件夹并提取数据
    for r in r_values:
        file_path = os.path.join(base_path, f"r={r}", file_pattern)
        if os.path.exists(file_path):
            data = np.load(file_path, allow_pickle=True).item()
            if "psi_error_t" in data:
                psi_error_last_values.append((r, np.mean(data["omega_error_t"][-1000:])))

    # 如果有数据，则绘制图像
    if psi_error_last_values:
        psi_error_last_values = np.array(psi_error_last_values)

        plt.figure(figsize=(10, 6))
        plt.plot(psi_error_last_values[:, 0], psi_error_last_values[:, 1], marker="o", linestyle="-", color="b")

        plt.xlabel("r values", fontsize = fs)
        plt.ylabel(r'Relative error: $\mathcal{L}_{2}$', fontsize = fs)
        plt.title(f"Omega Error vs r values (w: {wi}->{wo})", fontsize = fs)
        plt.grid(True)
        plt.xscale("linear")  # 可以改成 'log' 如果适用
        plt.yscale("log")

        # 保存并展示图像
        plt.savefig(f"omega_error_vs_r_values_wi_{wi}_wo_{wo}.png", dpi=300)
        plt.show()
