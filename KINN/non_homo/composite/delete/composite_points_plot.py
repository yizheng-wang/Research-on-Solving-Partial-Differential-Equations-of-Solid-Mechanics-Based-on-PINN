import numpy as np
import matplotlib.pyplot as plt
import torch

def interface(Ni=1000):
    theta = np.random.rand(Ni)*2*np.pi
    r = r0
    x = np.cos(theta) * r0
    y = np.sin(theta) * r0
    xi = np.hstack([x, y])
    return xi

def boundary(Ni=1000):
    # 生成从-1到1的1000个点
    points = np.linspace(-1, 1, Ni)
    
    # 上边界 (-1, 1) 到 (1, 1)
    top_edge = np.vstack((points, np.ones(Ni))).T
    
    # 下边界 (-1, -1) 到 (1, -1)
    bottom_edge = np.vstack((points, -np.ones(Ni))).T
    
    # 左边界 (-1, -1) 到 (-1, 1)
    left_edge = np.vstack((-np.ones(Ni), points)).T
    
    # 右边界 (1, -1) 到 (1, 1)
    right_edge = np.vstack((np.ones(Ni), points)).T
    
    # 合并所有边界点
    all_edges = np.vstack((top_edge, bottom_edge, left_edge, right_edge))
    return all_edges

def domain(Ni=100):
    num_points_per_side = Ni
    x = np.linspace(-1, 1, num_points_per_side)
    y = np.linspace(-1, 1, num_points_per_side)
    
    # 使用meshgrid生成正方形内部的均匀点
    xx, yy = np.meshgrid(x, y)
    internal_points = np.vstack([xx.flatten(), yy.flatten()]).T
    return internal_points

r0 = 0.5
Xb = boundary()
dom_koch_n = domain() # 获得n_test个koch的随机分布点
interface = interface()
dom_koch_n1 = dom_koch_n[np.linalg.norm(dom_koch_n, axis=1)<r0]
dom_koch_n2 = dom_koch_n[np.linalg.norm(dom_koch_n, axis=1)>=r0]


plt.cla()

plt.figure(dpi = 1000)
ax = plt.gca()
ax.set_aspect(1)
plt.scatter(dom_koch_n1[:, 0], dom_koch_n1[:, 1], c='b', s=0.1)
plt.scatter(dom_koch_n2[:, 0], dom_koch_n2[:, 1], c='g', s=0.1)
plt.scatter(Xb[:, 0], Xb[:, 1], c='r', s=0.1)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('./pic/scatter_points_heterogeneous.pdf', bbox_inches = 'tight', dpi = 300)
plt.show()
