# PINN可能位移场的能量形式，不进行分片
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
import sys
sys.path.append("../") 
from kan_efficiency import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(2024)

train_p = 0
tol_p = 0.00001
a = 1

nepoch = 2500

hyper_d = 1



    
# def NTK(pred):
#     paranum = sum(p.numel() for p in model_h.parameters())
#     A = torch.zeros((len(pred), paranum)).cuda()
#     for index,pred_e in enumerate(pred):
#         grad_e = grad(pred_e, model_h.parameters(),retain_graph=True,  create_graph=True) # 获得每一个预测对参数的梯度
#         grad_one = torch.tensor([]).cuda() # 创建一个空的tensor
#         for i in grad_e:
#             grad_one = torch.cat((grad_one, i.flatten())) # 将每一个预测对参数的梯度预测都放入grad_one 中
#         A[index] = grad_one # 存入A中，A的行是样本个数，列是梯度个数
#     K = torch.mm(A, A.T) # 矩阵的乘法组成K矩阵，这个K矩阵的获得是利用了矩阵的乘法，矩阵的乘法可以加速获得K矩阵，这里要避免使用for循环
#     L,_ = torch.eig(K.data.cpu()) # torch不能对gpu求特征值
#     eig = torch.norm(L, dim=1)
#     return eig


# for particular solution 
class particular(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(particular, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, D_out)
        
        #self.a1 = torch.nn.Parameter(torch.Tensor([0.2]))
        

        self.a1 = torch.Tensor([0.1]).cuda()
        self.a2 = torch.Tensor([0.1])
        self.a3 = torch.Tensor([0.1])
        self.n = 1/self.a1.data.cuda()


        torch.nn.init.normal_(self.linear1.weight, mean=0, std=np.sqrt(2/(D_in+H)))
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear3.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear4.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear5.weight, mean=0, std=np.sqrt(2/(H+D_out)))        
        
        torch.nn.init.normal_(self.linear1.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear2.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear3.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear4.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear5.bias, mean=0, std=1)
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        y1 = torch.tanh(self.n*self.a1*self.linear1(x))
        y2 = torch.tanh(self.n*self.a1*self.linear2(y1))
        y3 = torch.tanh(self.n*self.a1*self.linear3(y2))
        y4 = torch.tanh(self.n*self.a1*self.linear4(y3))
        y = self.n*self.a1*self.linear5(y4)
        return y
class MultiLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MultiLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, D_out)

        torch.nn.init.normal_(self.linear1.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear2.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear3.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear4.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear5.bias, mean=0, std=1)

        torch.nn.init.normal_(self.linear1.weight, mean=0, std=np.sqrt(2/(D_in+H)))
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear3.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear4.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear5.weight, mean=0, std=np.sqrt(2/(H+D_out)))

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        yt = x
        y1 = torch.tanh(self.linear1(yt))
        y2 = torch.tanh(self.linear2(y1))
        y3 = torch.tanh(self.linear3(y2)) + y1
        y4 = torch.tanh(self.linear4(y3)) + y2
        y =  self.linear5(y4)
        return y    
    
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
    
    

def pred(xy):
    '''
    

    Parameters
    ----------
    输入xytensor，然而输出位移，都是GPU的张量
    '''
    xy_scale = xy/1.73# 2是特征尺寸长度
    pred = model_p(xy) + RBF(xy) * model(xy_scale)
    return pred    
    
def evaluate():
    N_test = 100
    dom_koch_n = koch_points.get_koch_points_lin(N_test).astype(np.float32) # 均匀在科赫雪花内部步点
    dom_koch_t = torch.tensor(dom_koch_n, device = 'cuda')
    
    u_pred = pred(dom_koch_t)
    u_pred = u_pred.data.cpu()
    

    x = dom_koch_n[:, 0][:, np.newaxis]
    y = dom_koch_n[:, 1][:, np.newaxis]
    
    u_exact = 1/a*(np.sin(x)*np.sinh(y)+np.cos(x)*np.cosh(y))

    u_exact = torch.from_numpy(u_exact)
    error = torch.abs(u_pred - u_exact) # get the error in every points
    error_t = torch.norm(error)/torch.norm(u_exact) # get the total relative L2 error
    return error_t

if train_p == 1:
    start = time.time()
    model_p = particular(2, 20, 1).cuda() # 定义两个特解，这是因为在裂纹处是间断值

    
    loss_bn = 100
    epoch_b = 0
    criterion = torch.nn.MSELoss()
    optimp = torch.optim.Adam(params=model_p.parameters(), lr= 0.0005)

    loss_bn_array = []
    # training the particular network for particular solution on the bonudaray
    while loss_bn>tol_p:
        
        if epoch_b%10 == 0:
            Xb = koch.point_bound_rand(100).astype(np.float32)
            Xb = torch.tensor(Xb).cuda()
            x = Xb[:, 0].unsqueeze(1)
            y = Xb[:, 1].unsqueeze(1)
            target_b = 1/a*(torch.sin(x)*torch.sinh(y)+torch.cos(x)*torch.cosh(y))

        epoch_b = epoch_b + 1
        def closure():  
            pred_b = model_p(Xb) # predict the boundary condition
            loss_bn = criterion(pred_b, target_b)  
           
            optimp.zero_grad()
            loss_bn.backward()
            loss_bn_array.append(loss_bn.data)
            if epoch_b%10==0:
                print('trianing particular network : the number of epoch is %i, the loss is %f' % (epoch_b, loss_bn.data))
            return loss_bn
        optimp.step(closure)
        loss_bn = loss_bn_array[-1]
    torch.save(model_p, './particular_nn')

model_p = torch.load('particular_nn')

# learning the distance neural network
def RBF(x):
    d_total_t = torch.from_numpy(d_total.astype(np.float32)).unsqueeze(1).cuda()
    w_t = torch.from_numpy(w.astype(np.float32)).cuda()
    x_l = x.unsqueeze(0).repeat(len(d_total_t), 1, 1) # 创立一个足够大的x矩阵
    # 得到R大矩阵
    
    R = torch.norm(d_total_t - x_l, dim=2)
    #Rn = -(x_l[:, :, 0]-d_total_t[:, :, 0]) # 试一试我自己创建的theta的径向基函数
    y = torch.mm(torch.exp(-gama*R.T), w_t)
    #y = torch.mm(torch.sqrt(0.5*(R.T-Rn.T)), w_t)# 试一试我自己创建的theta的径向基函数
    #y = torch.mm(torch.sqrt(R.T), w_t)
    return y

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
domxy6 = np.array([[-10/3, -70/9*3**0.5], [10/3, -70/9*3**0.5]])
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


model = KAN([2, 5,5,5, 1], base_activation=torch.nn.SiLU, grid_size=10, grid_range=[-1.0, 1.0], spline_order=2).cuda()
criterion = torch.nn.MSELoss()
optim = torch.optim.Adam(params=model.parameters(), lr= 0.001) # 0.001比较好，两个神经网络

scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[1000, 3000, 5000], gamma = 0.1)

loss_array = []
loss_d_array = [] # 内部r<ro
loss_b_array = [] # 内部加上边界
error_array = []


start = time.time()

for epoch in range(nepoch): # cpinn先优化1
    if epoch ==1000:
        end = time.time()
        consume_time = end-start
        print('time is %f' % consume_time)
    if epoch%100 == 0:
        dom_koch_n = koch_points.get_koch_points(1000).astype(np.float32) # 获得n_test个koch的随机分布点
        dom_koch_t= torch.tensor(dom_koch_n,  requires_grad=True, device = 'cuda') # 将numpy数据转化为tensor
    
        # f = -16 * torch.norm(dom_koch_t, dim = 1, keepdim=True).data**2 # 定义体力 

        Xb = koch.point_bound_rand(1000).astype(np.float32) # 获得边界点，用来优化网络2的外部
        Xb = torch.tensor(Xb).cuda() # 变成张量
        x = Xb[:,0].unsqueeze(1)
        y = Xb[:,1].unsqueeze(1)
        target_b = 1/a*(torch.sin(x)*torch.sinh(y)+torch.cos(x)*torch.cosh(y)) # 得到相应的标签     
    def closure():  

        # 构造可能位移场
        u_pred = pred(dom_koch_t)
        
        dudxy = grad(u_pred, dom_koch_t, torch.ones(dom_koch_t.size()[0], 1).cuda(), create_graph=True)[0]
        dudx = dudxy[:, 0].unsqueeze(1)
        dudy = dudxy[:, 1].unsqueeze(1)

        dudxxy = grad(dudx, dom_koch_t, torch.ones(dom_koch_t.size()[0], 1).cuda(), create_graph=True)[0]
        dudxx = dudxxy[:, 0].unsqueeze(1)
        dudxy = dudxxy[:, 1].unsqueeze(1)

        dudyxy = grad(dudy, dom_koch_t, torch.ones(dom_koch_t.size()[0], 1).cuda(), create_graph=True)[0]
        dudyx = dudyxy[:, 0].unsqueeze(1)
        dudyy = dudyxy[:, 1].unsqueeze(1)

        J =  torch.sum((a*(dudxx + dudyy))**2)/len(dudxx)
        
        u_b = pred(Xb)
        Jb = criterion(u_b, target_b)
        
        loss = J
        
        error_t = evaluate()
        optim.zero_grad()
        loss.backward()
        loss_array.append(loss.data.cpu())
        loss_d_array.append(J.data.cpu())
        loss_b_array.append(Jb.data.cpu())
        error_array.append(error_t.data.cpu())

        if epoch%10==0:
            print(' epoch : %i, the loss : %f , loss_d : %f, loss_b : %f, error : %f' % (epoch, loss.data, J.data,  Jb.data, error_t.data))
        return loss
    optim.step(closure)
    scheduler.step()
    # 网络1损失函数下不去，所以我们用交界面来作为本质边界条件训练网络1
        

# %%   
n_test = 300
dom_koch_n = koch_points.get_koch_points_lin(n_test).astype(np.float32) # 获得n_test个koch的随机分布点
xy_test= torch.tensor(dom_koch_n,  requires_grad=True, device = 'cuda') # 将numpy数据转化为tensor


u_pred = pred(xy_test)
u_pred = u_pred.data.cpu()

x = dom_koch_n[:, 0][:, np.newaxis]
y = dom_koch_n[:, 1][:, np.newaxis]
u_exact = 1/a*(np.sin(x)*np.sinh(y)+np.cos(x)*np.cosh(y)) # 现在网格里面赋值 

u_exact = torch.from_numpy(u_exact) # 将精确解从array变成tensor

error = torch.abs(u_pred - u_exact) # get the error in every points
error_t = torch.norm(error)/torch.norm(u_exact) # get the total relative L2 error

# plot the prediction solution
fig = plt.figure(figsize=(20, 20))

plt.subplot(3, 3, 1)
Xb =  koch.point_bound(5).astype(np.float32)
Xf = koch_points.get_koch_points(10000).astype(np.float32)
 # 将边界和内部点画图
plt.scatter(Xb[:, 0], Xb[:, 1], c='r', s = 0.1)
plt.scatter(Xf[:, 0], Xf[:, 1], c='b', s = 0.1)

write_arr2DVTK('./output_ntk/pred_pinn_kan', dom_koch_n, u_pred, 'pred_pinn')

write_arr2DVTK('./output_ntk/error_pinn_kan', dom_koch_n, error, 'error_pinn')

plt.subplot(3, 3, 2)
loss_d_array = np.array(loss_d_array)
loss_b_array = np.array(loss_b_array)
plt.yscale('log')
plt.plot(loss_d_array)
plt.plot(loss_b_array)
plt.legend(['loss_d', 'loss_b'])
plt.xlabel('the iteration')
plt.ylabel('loss')
plt.title('loss evolution') 
plt.subplot(3, 3, 3)
error_array = np.array(error_array)
np.save('./results/koch_PINN_KAN_error.npy', error_array)
plt.yscale('log')
plt.plot(error_array)
plt.xlabel('the iteration')
plt.ylabel('error')
plt.title('relative total error evolution') 

plt.subplot(3, 3, 4) # 做x=0的位移预测
n_test = 100
x0 = np.zeros((n_test, 2)).astype(np.float32)
x0[:, 1] = np.linspace(-np.sqrt(3), np.sqrt(3), 100).astype(np.float32)
exactx0 = np.zeros((n_test, 1)).astype(np.float32)
x = x0[:, 0][:, np.newaxis]
y = x0[:, 1][:, np.newaxis]
exactx0 =  1/a*(np.sin(x)*np.sinh(y)+np.cos(x)*np.cosh(y)) # 不同的区域进行不同的解析解赋予
x0t = torch.tensor(x0).cuda()
predx0 = pred(x0t).data.cpu().numpy() # 预测x=0的原函数
plt.plot(x0[:, 1], exactx0.flatten())
plt.plot(x0[:, 1], predx0.flatten())
plt.xlabel('y')
plt.ylabel('u')
plt.legend(['exact', 'pred'])
plt.title('x=0 u') 

plt.subplot(3, 3, 5) # 做y=0的位移预测
n_test = 100
y0 = np.zeros((n_test, 2)).astype(np.float32)
y0[:, 0] = np.linspace(-1, 1, 100).astype(np.float32)
exacty0 = np.zeros((n_test, 1)).astype(np.float32)
x = y0[:, 0][:, np.newaxis]
y = y0[:, 1][:, np.newaxis]
exacty0 =  1/a*(np.sin(x)*np.sinh(y)+np.cos(x)*np.cosh(y)) # 不同的区域进行不同的解析解赋予
y0t = torch.tensor(y0).cuda()
predy0 = pred(y0t).data.cpu().numpy() # 预测x=0的原函数
plt.plot(y0[:, 0], exacty0.flatten())
plt.plot(y0[:, 0], predy0.flatten())
plt.xlabel('x')
plt.ylabel('u')
plt.legend(['exact', 'pred'])
plt.title('y=0 u') 

plt.subplot(3, 3, 6) # 做x=0的位移导数预测 dudy
n_test = 100
x0 = np.zeros((n_test, 2)).astype(np.float32)
x0[:, 1] = np.linspace(-np.sqrt(3), np.sqrt(3), 100).astype(np.float32)
x = x0[:, 0][:, np.newaxis]
y = x0[:, 1][:, np.newaxis]
exactdx0 =  1/a*(np.sin(x)*np.cosh(y)+np.cos(x)*np.sinh(y)) # 不同的区域进行不同的解析解赋予
x0t = torch.tensor(x0, requires_grad=True).cuda() # 将numpy 变成 tensor
predx0 = pred(x0t) # 预测一下
dudxy = grad(predx0, x0t, torch.ones(x0t.size()[0], 1).cuda(), create_graph=True)[0]
dudx = dudxy[:, 0].unsqueeze(1).data.cpu().numpy()
dudy = dudxy[:, 1].unsqueeze(1).data.cpu().numpy()
plt.plot(x0[:, 1], exactdx0.flatten())
plt.plot(x0[:, 1], dudy.flatten())
plt.xlabel('y')
plt.ylabel('u')
plt.legend(['exact', 'pred'])
plt.title('x=0 dudy')

plt.subplot(3, 3, 7) # 做y=0的位移导数预测 dudy
n_test = 100
y0 = np.zeros((n_test, 2)).astype(np.float32)
y0[:, 0] = np.linspace(-1, 1, 100).astype(np.float32)
x = y0[:, 0][:, np.newaxis]
y = y0[:, 1][:, np.newaxis]
exactdy0 =  1/a*(np.cos(x)*np.sinh(y) - np.sin(x)*np.cosh(y)) # 不同的区域进行不同的解析解赋予
y0t = torch.tensor(y0, requires_grad=True).cuda()
predy0 = pred(y0t) # 预测x=0的原函数
dudxy = grad(predy0, y0t, torch.ones(y0t.size()[0], 1).cuda(), create_graph=True)[0]
dudx = dudxy[:, 0].unsqueeze(1).data.cpu().numpy()
dudy = dudxy[:, 1].unsqueeze(1).data.cpu().numpy()
plt.plot(y0[:, 0], exactdy0.flatten())
plt.plot(y0[:, 0], dudx.flatten())
plt.xlabel('x')
plt.ylabel('u')
plt.legend(['exact', 'pred'])
plt.title('y=0 dudx') 


plt.suptitle("pinn")
plt.savefig('./pic/pinn_kan_plane.png')
plt.show()
print('the relative error is %f' % error_t.data)   
    
    
    