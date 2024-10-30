import numpy as np
import matplotlib.pyplot as plt

# 设置字体大小
font_size = 10

# 加载数据
picasuo_mlp_l2 = np.load('results/picasuo_mlp_loss.npy')
picasuo_kan_l2 = np.load('results/picasuo_kan_loss.npy')
salvator_mlp_l2 = np.load('results/salvator_mlp_loss.npy')
salvator_kan_l2 = np.load('results/salvator_kan_loss.npy')
sky_mlp_l2 = np.load('results/sky_mlp_loss.npy')
sky_kan_l2 = np.load('results/sky_kan_loss.npy')

# 创建图表
#plt.figure(figsize=(14, 8), dpi=300)

# 第一个图表：picasuo
plt.plot(picasuo_mlp_l2, label='MLP')
plt.plot(picasuo_kan_l2, label='KAN')
plt.xlabel('Iteration', fontsize=font_size)
plt.ylabel('Loss', fontsize=font_size*1.5)
plt.yscale('log')  # y轴使用对数刻度
plt.legend(fontsize=font_size)
plt.savefig('./results/picasuo_loss.pdf', dpi=300)
plt.show()
# 第二个图表：salvator
plt.plot(salvator_mlp_l2, label='MLP')
plt.plot(salvator_kan_l2, label='KAN')
plt.xlabel('Iteration', fontsize=font_size)
plt.ylabel('Loss', fontsize=font_size*1.5)
plt.yscale('log')  # y轴使用对数刻度
plt.legend(fontsize=font_size)
plt.savefig('./results/salvator_loss.pdf', dpi=300)
plt.show()
# 第三个图表：sky
plt.plot(sky_mlp_l2, label='MLP')
plt.plot(sky_kan_l2, label='KAN')
plt.xlabel('Iteration', fontsize=font_size)
plt.ylabel('Loss', fontsize=font_size*1.5)
plt.yscale('log')  # y轴使用对数刻度
plt.legend(fontsize=font_size)
plt.savefig('./results/sky_loss.pdf', dpi=300)
plt.show()
