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
dom_koch_n1 = dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)<r0**2]#定义内部的dom点，即是r<r0的点
dom_koch_n2 = dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)>=r0**2]#定义内部的dom点，即是r<r0的点
Xi = interface(100)
plt.cla()

plt.figure(dpi = 1000)
ax = plt.gca()
ax.set_aspect(1)
plt.scatter(dom_koch_n1[:, 0], dom_koch_n1[:, 1], c='g', s=0.1)
plt.scatter(dom_koch_n2[:, 0], dom_koch_n2[:, 1], c='b', s=0.1)
plt.scatter(Xi[:, 0], Xi[:, 1], c='y', s=0.1)
plt.scatter(Xb[:, 0], Xb[:, 1], c='r', s=0.1)
plt.xlabel('x')
plt.ylabel('y')
#plt.savefig('../../picture/koch/配点方案.pdf', bbox_inches = 'tight')
plt.show()
