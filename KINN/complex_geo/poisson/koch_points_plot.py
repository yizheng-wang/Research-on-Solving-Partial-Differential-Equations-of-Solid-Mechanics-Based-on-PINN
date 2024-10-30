import koch 
import koch_points
import numpy as np
import matplotlib.pyplot as plt

def interface(Ni):
    '''
     生成交界面的随机点
    '''
    theta = np.random.rand(Ni)*2*np.pi
    r = r0
    x = np.cos(theta) * r0
    y = np.sin(theta) * r0
    xi = np.stack([x, y], 1)
    return xi

r0 = 0.5
Xb = koch.point_bound_rand(10)
dom_koch_n = koch_points.get_koch_points(10000) # 获得n_test个koch的随机分布点
plt.cla()

plt.figure(dpi = 1000)
ax = plt.gca()
ax.set_aspect(1)
plt.scatter(dom_koch_n[:, 0], dom_koch_n[:, 1], c='b', s=0.1)
plt.scatter(Xb[:, 0], Xb[:, 1], c='r', s=0.1)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('./pic/scatter_points_koch.pdf', bbox_inches = 'tight', dpi = 300)
plt.show()
