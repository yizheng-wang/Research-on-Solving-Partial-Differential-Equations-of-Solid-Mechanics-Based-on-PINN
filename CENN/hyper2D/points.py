import numpy as np
import torch
import matplotlib.pyplot as plt
from  torch.autograd import grad
from sklearn.neighbors import KDTree
import time
from itertools import chain
from pyevtk.hl import gridToVTK
from pyevtk.hl import pointsToVTK
import meshio

def interface(Ni): # 交界面点
    '''
     生成裂纹尖端上半圆的点，为了多分配点
    '''
    x = 4*np.random.rand(Ni)
    y = 0.5*np.ones(Ni)
    xi = np.stack([x, y], 1)
    return xi

def essential_bound(Ni): # 本质边界点
    '''
     生成裂纹尖端上半圆的点，为了多分配点
    '''
    x = np.zeros(Ni)
    y = np.random.rand(Ni)
    xeb = np.stack([x, y], 1)
    return xeb

def train_data(Nb, Nf): # 上下两个域以及力边界条件点
    '''
    生成强制边界点，四周以及裂纹处
    生成上下的内部点
    '''
    

    x = 4*np.ones(Nb)
    y = np.random.rand(Nb)
    xnb = np.stack([x, y], 1)
    
    x = 4*np.random.rand(Nf)
    y = np.random.rand(Nf)
    xf = np.stack([x, y], 1)
    return xnb, xf

def train_data_cenn(Nb, Nf): # 上下两个域以及力边界条件点
    '''
    生成强制边界点，四周以及裂纹处
    生成上下的内部点
    '''
    

    x = 4*np.ones(Nb)
    y = np.random.rand(Nb)
    xnb = np.stack([x, y], 1)
    
    
    x = 4*np.random.rand(Nf)
    y = np.random.rand(Nf)
    xf = np.stack([x, y], 1)
    

    xf1 = xf[(xf[:, 1]>0.5)] # 上区域点，去除内部多配的点
    xf2 = xf[(xf[:, 1]<0.5)]  # 下区域点，去除内部多配的点

    
    return xnb, xf1, xf2

plt.figure(dpi=1000, figsize = (6,6))
Xn, Xf = train_data(256, 4096)
Xeb = essential_bound(256)
ax = plt.gca()
ax.set_aspect(1)
plt.scatter(Xeb[:, 0], Xeb[:, 1], c='r', s = 0.1)
plt.scatter(Xn[:, 0], Xn[:, 1], c='c', s = 0.1)
plt.scatter(Xf[:, 0], Xf[:, 1], c='b', s = 0.1)
plt.xlabel('x')
plt.ylabel('y')
#plt.title('DEM distribution points')

plt.figure(dpi=1000, figsize = (6,6))
Xn, Xf1, Xf2 = train_data_cenn(256, 4096)
Xeb = essential_bound(256)
Xi = interface(3000)
ax = plt.gca()
ax.set_aspect(1)
plt.scatter(Xeb[:, 0], Xeb[:, 1], c='r', s = 0.1)
plt.scatter(Xn[:, 0], Xn[:, 1], c='c', s = 0.1)
plt.scatter(Xi[:, 0], Xi[:, 1], c='y', s = 0.1)
plt.scatter(Xf1[:, 0], Xf1[:, 1], c='b', s = 0.1)
plt.scatter(Xf2[:, 0], Xf2[:, 1], c='g', s = 0.1)
plt.xlabel('x')
plt.ylabel('y')
#plt.title('CENN distribution points')
#plt.savefig('../../图片/超弹性/points.pdf', bbox_inches = 'tight')
plt.show()