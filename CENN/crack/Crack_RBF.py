# 这是构造可能位移场的程序，不再用距离神经网络，而是用RBF来解析的构造，虽然不是精确满足，但是精度已经非常的好了
import numpy as np
import torch
import matplotlib.pyplot as plt
from  torch.autograd import grad
from sklearn.neighbors import KDTree
import time
from itertools import chain

#torch.set_default_tensor_type(torch.DoubleTensor) # 将tensor的类型变成默认的double
#torch.set_default_tensor_type(torch.cuda.DoubleTensor) # 将cuda的tensor的类型变成默认的double
penalty = 1000
delta = 0.0
train_p = 0
nepoch_u0 = 2500# the float type
a = 1
elasp = 0.001

# learning the boundary solution
# step 1: get the boundary train data and target for boundary condition



# learning the distance neural network
def RBF(x):
    d_total_t = torch.from_numpy(d_total).unsqueeze(1)#.cuda()
    w_t = torch.from_numpy(w)#.cuda()
    x_l = x.unsqueeze(0).repeat(len(d_total_t), 1, 1) # 创立一个足够大的x矩阵
    # 得到R大矩阵
    
    R = torch.norm(d_total_t - x_l, dim=2)
    #Rn = -(x_l[:, :, 0]-d_total_t[:, :, 0]) # 试一试我自己创建的theta的径向基函数
    y = torch.mm(torch.exp(-gama*R.T), w_t)
    #y = torch.mm(torch.sqrt(0.5*(R.T-Rn.T)), w_t)# 试一试我自己创建的theta的径向基函数
    #y = torch.mm(torch.sqrt(R.T), w_t)
    return y

n_d = 10 # 一条边上的本质边界条件的配点数
n_dom = 5
gama = 0.5
ep = np.linspace(-1, 1, n_d) # 从-1到1均匀配点
# 获得4条边界上的点
ep1 = np.zeros((n_d, 2))
ep1[:, 0], ep1[:, 1] = ep, 1
ep2 = np.zeros((n_d, 2))
ep2[:, 0], ep2[:, 1] = ep, -1
ep3 = np.zeros((n_d, 2))
ep3[:, 0], ep3[:, 1] = -1, ep
ep4 = np.zeros((n_d, 2))
ep4[:, 0], ep4[:, 1] = 1, ep
ep5 = np.zeros((n_d, 2))
ep5[:, 0], ep5[:, 1] = ep/2-0.5, 0
points_d = np.concatenate((ep1, ep2, ep3, ep4, ep5)) # 获得本质边界条件点
points_d = np.unique(points_d, axis=0) # 去除重复点
kdt = KDTree(points_d, metric='euclidean') # 将本质边界条件封装成一个对象


domx = np.linspace(-1, 1, n_dom)[1:-1]
domy = np.linspace(-1, 1, n_dom)[1:-1]
domx, domy = np.meshgrid(domx, domy)
domxy = np.stack((domx.flatten(), domy.flatten()), 1)
domxy = domxy[(domxy[:, 1]!=0)|(domxy[:, 0]>0)]
#domxy = np.unique(domxy, axis=0)
d_dir, _ = kdt.query(points_d, k=1, return_distance = True)
d_dom, _ = kdt.query(domxy, k=1, return_distance = True)
# 将本质边界条件和内部点拼接起来
d_total = np.concatenate((points_d, domxy))
#d_total = np.unique(d_total, axis=0)
# 获得距离矩阵，这是获得K（用来求RBF的权重矩阵的关键）的前提
dx = d_total[:, 0][:, np.newaxis] - d_total[:, 0][np.newaxis, :]
dy = d_total[:, 1][:, np.newaxis] - d_total[:, 1][np.newaxis, :]
#Rn = d_total[:,0][np.newaxis, :]-d_total[:,0][:, np.newaxis] # 试一试我创建的径向基函数
R = np.sqrt(dx**2+dy**2)
K = np.exp(-gama*R)
# K = np.sqrt(R)
#K = np.sqrt(0.5*(R-Rn))# 试一试我创建的径向基函数
b = np.concatenate((d_dir, d_dom))
w = np.dot(np.linalg.inv(K),b)



n_test = 201 # 获得测试点 n_test**2个
domx_t = np.linspace(-1, 1, n_test)
domy_t = np.linspace(-1, 1, n_test)
domx_t, domy_t = np.meshgrid(domx_t, domy_t)
domxy_t = np.stack((domx_t.flatten(), domy_t.flatten()), 1)
domxy_t= torch.from_numpy(domxy_t).requires_grad_(True) # 获得了两列的测试点，放入RBF中，这是为了和其他神经网络结构保持一致

dis = RBF(domxy_t)
# plot 配点图以及RBF距离函数

#for paper plot
fig = plt.figure(dpi=1000,figsize = (3,3))
fig.tight_layout()
plt.subplots_adjust(wspace = 0.4, hspace=0)


plt.scatter(d_total[:, 0], d_total[:, 1])
ax = plt.gca()
ax.set_aspect(1)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

dis_plot = dis.data.cpu().numpy().reshape(n_test, n_test)
fig = plt.figure(dpi=1000,figsize = (4,3))
h1 = plt.contourf(domx_t, domy_t, dis_plot,  cmap = 'jet', levels = 100)
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h1)
plt.xlabel('x')
plt.ylabel('y')
#plt.savefig('../../图片/裂纹/RBF.pdf', bbox_inches = 'tight')
plt.show()

    

    