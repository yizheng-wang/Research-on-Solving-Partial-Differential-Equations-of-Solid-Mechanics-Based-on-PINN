import numpy as np
import matplotlib.pyplot as plt

# 设置字体大小
font_size = 10

# 加载数据

picasuo_kan_l2 = np.load('PINNs_results/picasuo_kan_l2.npy')
salvator_kan_l2 = np.load('PINNs_results/salvator_kan_l2.npy')
sky_kan_l2 = np.load('PINNs_results/sky_kan_l2.npy')

# 创建图表
#plt.figure(figsize=(14, 8), dpi=300)

# 第一个图表：picasuo
plt.plot(picasuo_kan_l2, label='KAN')
plt.xlabel('Iteration', fontsize=font_size)
plt.ylabel('Loss', fontsize=font_size*1.5)
plt.yscale('log')  # y轴使用对数刻度
plt.legend(fontsize=font_size)
plt.savefig('./PINNs_results/picasuo_loss.pdf', dpi=300)
plt.show()
# 第二个图表：salvator
plt.plot(salvator_kan_l2, label='KAN')
plt.xlabel('Iteration', fontsize=font_size)
plt.ylabel('Loss', fontsize=font_size*1.5)
plt.yscale('log')  # y轴使用对数刻度
plt.legend(fontsize=font_size)
plt.savefig('./PINNs_results/salvator_loss.pdf', dpi=300)
plt.show()
# 第三个图表：sky
plt.plot(sky_kan_l2, label='KAN')
plt.xlabel('Iteration', fontsize=font_size)
plt.ylabel('Loss', fontsize=font_size*1.5)
plt.yscale('log')  # y轴使用对数刻度
plt.legend(fontsize=font_size)
plt.savefig('./PINNs_results/sky_loss.pdf', dpi=300)
plt.show()
