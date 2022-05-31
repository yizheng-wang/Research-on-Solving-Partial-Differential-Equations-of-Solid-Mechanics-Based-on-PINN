# 计算科赫雪花的精确积分，通过MC方法计算，因为边界条件的复杂性
import numpy as np
import torch
import matplotlib.pyplot as plt
from  torch.autograd import grad
from sklearn.neighbors import KDTree
import time
from itertools import chain
import koch 
import koch_points

r0 = 0.5
a1 = 1/15
a2 = 1
# 获得两类点 分别是域内1点，以及域内2点
J_l = []
J1_l = []
J2_l = []
for i in range(10):
    dom_koch_n = koch_points.get_koch_points(10000000) # 获得n_test个koch的随机分布点
    dom_koch_n1 = dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)<r0**2]#定义内部的dom点，即是r<r0的点
    dom_koch_n2 = dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)>=r0**2]#定义内部的dom点，即是r<r0的点
    
    J1 = (0.5 * a1 * np.sum((4/a1*np.linalg.norm(dom_koch_n1, axis =1)**3)**2) + np.sum(16/a1*np.linalg.norm(dom_koch_n1, axis =1)**6)) * (r0**2*np.pi)/len(dom_koch_n1)
    
    J2 = (0.5 * a2 * np.sum((4/a2*np.linalg.norm(dom_koch_n2, axis =1)**3)**2) + np.sum((16*np.linalg.norm(dom_koch_n2, axis =1)**2)*(np.linalg.norm(dom_koch_n2, axis =1)**4+r0**4*(1/a1-1/a2)))) * (10*np.sqrt(3)/3-r0**2*np.pi)/len(dom_koch_n2) 

    J = J1 + J2
    
    J1_l.append(J1)
    J2_l.append(J2)
    J_l.append(J)
print('J1 : %f' % np.array(J1_l).mean())
print('J2 : %f' % np.array(J2_l).mean())
print('J : %f' % (np.array(J1_l).mean()+np.array(J2_l).mean()))