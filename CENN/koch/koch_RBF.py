# 这是构造可能位移场的程序，不再用距离神经网络，而是用RBF来解析的构造，虽然不是精确满足，但是精度已经非常的好了
# 发现在裂纹尖端处效果不好，这是因为距离神经网络的影响，接下来我们尝试用里兹法，不用距离神经网络
# 注意这个程序是可能位移场
import numpy as np
import torch
import matplotlib.pyplot as plt
from  torch.autograd import grad
from sklearn.neighbors import KDTree
import time
from itertools import chain
import koch 
import koch_points
from pyevtk.hl import gridToVTK
from pyevtk.hl import pointsToVTK

# learning the distance neural network
def write_arr2DVTK(filename, coordinates, values, name):
    '''
    这是一个将tensor转化为VTK数据可视化的函数，输入coordinates坐标，是一个二维的二列数据，
    value是相应的值，是一个列向量
    name是转化的名称，比如RBF距离函数，以及特解网路等等
    '''
    # displacement = np.concatenate((values[:, 0:1], values[:, 1:2], values[:, 0:1]), axis=1)
    x = np.array(coordinates[:, 0].flatten(), dtype='float32')
    y = np.array(coordinates[:, 1].flatten(), dtype='float32')
    z = np.zeros(len(coordinates), dtype='float32')
    disX = np.array(values[:, 0].flatten(), dtype='float32')
    pointsToVTK(filename, x, y, z, data={name: disX}) # 二维数据的VTK文件导出
def RBF(x):
    
    d_total_t = torch.tensor(d_total, device='cuda').unsqueeze(1)
    
    w_t = torch.tensor(w,  device='cuda')

    x_l = x.unsqueeze(0).repeat(len(d_total_t), 1, 1) # 创立一个足够大的x矩阵
    # 得到R2大矩阵
    R2 = torch.norm(d_total_t - x_l, dim=2)
    y = torch.mm(torch.exp(-gama*R2.T), w_t)
    return y

torch.set_default_tensor_type(torch.DoubleTensor) # 将tensor的类型变成默认的double
torch.set_default_tensor_type(torch.cuda.DoubleTensor) # 将cuda的tensor的类型变成默认的double
penalty = 100
delta = 0.0
train_p = 0
nepoch_u0 = 101 # the float type
a = 1
tol_p = 0.00001
gama = 0.3
Q = np.array([[1/2, -0.5*3**0.5],[0.5*3**0.5, 0.5]])

points_d = koch.point_bound(5) # 获得本质边界条件点
kdt = KDTree(points_d, metric='euclidean') # 将本质边界条件封装成一个对象

# 由于边界上所有点都是本质边界条件，且都为0，所以我们这里弄一个内部点来计算非零距离
domxy1 = koch_points.get_koch_points(0) # 弄一些随机点
domxy2 = np.array([[0., 0.],[0, 5*3**0.5],[7.5, 2.5*3**0.5],\
                  [7.5, -2.5*3**0.5],[0, -5*3**0.5],[-7.5, -2.5*3**0.5],[-7.5, 2.5*3**0.5]])
domxy3 = domxy2 * 16/9
domxy4 = domxy2 * 0.5
domxy5 = domxy2 * 1.5
domxy6 = np.array([[-10/3, -70/9*3**0.5], [10/3, -70/9*3**0.5], [20/6, 20/6*3**0.5], [-20/6, 20/6*3**0.5]])
domxy7 = np.concatenate((np.dot(domxy6, Q), np.dot(domxy6, Q@Q), np.dot(domxy6, Q@Q@Q), np.dot(domxy6, Q@Q@Q@Q), np.dot(domxy6, Q@Q@Q@Q@Q)))
domxy = np.concatenate((domxy1, domxy2, domxy3, domxy4, domxy5, domxy6, domxy7))/10 # 本来横跨是30，所以要除一个比例
domxy = np.unique(domxy, axis=0)
#domxy = np.unique(domxy, axis=0)
d_dir, _ = kdt.query(points_d, k=1, return_distance = True)
d_dom, _ = kdt.query(domxy, k=1, return_distance = True)
# 将本质边界条件和内部点拼接起来
d_total = np.concatenate((points_d, domxy))
#d_total = np.unique(d_total, axis=0)
# 获得距离矩阵，这是获得K（用来求RBF的权重矩阵的关键）的前提
dx = d_total[:, 0][:, np.newaxis] - d_total[:, 0][np.newaxis, :]
dy = d_total[:, 1][:, np.newaxis] - d_total[:, 1][np.newaxis, :]
R2 = np.sqrt(dx**2+dy**2)
K = np.exp(-gama*R2)
b = np.concatenate((d_dir, d_dom))
w = np.dot(np.linalg.inv(K),b)

num_bound = 10 # 获得边界测试点5个
num_int = 10000
point_b_t = koch.point_bound(5) # 边界上弄5个测试点,为了判断边界上的最大误差
point_b_t = torch.tensor(point_b_t, device='cuda')
print('The mean abs error on the esstial bound: %f, max:%f' % (RBF(point_b_t).mean().data.cpu(), RBF(point_b_t).max().data.cpu()))



n_test = 100 # 获得测试点 n_test**2个
domx_t = np.linspace(-15, 15, n_test)/10 # 本来横跨是30，所以要除一个比例
domy_t = np.linspace(-10*3**0.5, 10*3**0.5, n_test)/10 # 本来横跨是30，所以要除一个比例
domx_t, domy_t = np.meshgrid(domx_t, domy_t)
domxy_t_n = np.stack((domx_t.flatten(), domy_t.flatten()), 1)
domxy_t= torch.tensor(domxy_t_n,  requires_grad=True, device = 'cuda') # 获得了两列的测试点，放入RBF中，这是为了和其他神经网络结构保持一致
dis = torch.zeros((len(domxy_t), 1)).cuda() # 弄一个相应储存dis的矩阵，如果不是内部点，就默认是0
label_koch = koch_points.whether_koch(domxy_t_n)  # 输入相应一个方点，然而判断哪些点是属于koch的
dis[label_koch] = RBF(domxy_t[label_koch]) # 将属于koch的点输出得到相应的dis，其余点则是0



#fig = plt.figure(dpi=1000,figsize = (8,3))
#fig.tight_layout()
#plt.subplots_adjust(wspace = 0.4, hspace=0)
plt.show()
plt.figure(dpi = 1000)
# plt.subplot(1, 2, 1)
plt.scatter(d_total[:, 0], d_total[:, 1], s=0.2)
# plt.xlim(-1.5, 1.5)
# plt.ylim(-3**0.5, 3**0.5)
ax = plt.gca()
ax.set_aspect(1)
#plt.title('the RBF points')
#plt.xlabel('x')
#plt.ylabel('y')


# plt.subplot(1, 2, 2) # RBF距离函数没法用matplotlib处理，所以我们转化为VTK处理
# dis_plot = dis.data.cpu().reshape(domx_t.shape)
# h1 = plt.contourf(domx_t, domy_t, dis_plot, levels = 50)
# plt.colorbar(h1)
# plt.title('the RBF distance function')
# plt.xlabel('x')
# plt.ylabel('y')
#plt.savefig('../../picture/koch/plane_RBF.pdf', bbox_inches = 'tight')
plt.show()

n_test = 300
dom_koch_n = koch_points.get_koch_points_lin(n_test) # 获得n_test个koch的随机分布点
dom_koch_t= torch.tensor(dom_koch_n,  requires_grad=True, device = 'cuda') # 将numpy数据转化为tensor
dis = RBF(dom_koch_t)
dis_plot = dis.data
# write_arr2DVTK('./output_ntk/plane_RBF', dom_koch_n, dis_plot.data.cpu(), 'RBF distance')