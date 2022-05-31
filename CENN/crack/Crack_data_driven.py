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

torch.set_default_tensor_type(torch.DoubleTensor) # 将tensor的类型变成默认的double
torch.set_default_tensor_type(torch.cuda.DoubleTensor) # 将cuda的tensor的类型变成默认的double
penalty = 100
delta = 0.1
train_p = 0
nepoch_u0 = 500 # the float type
a = 1
def NTK(pred):
    paranum = sum(p.numel() for p in model_h.parameters())
    A = torch.zeros((len(pred), paranum)).cuda()
    for index,pred_e in enumerate(pred):
        grad_e = grad(pred_e, model_h.parameters(),retain_graph=True,  create_graph=True) # 获得每一个预测对参数的梯度
        grad_one = torch.tensor([]).cuda() # 创建一个空的tensor
        for i in grad_e:
            grad_one = torch.cat((grad_one, i.flatten())) # 将每一个预测对参数的梯度预测都放入grad_one 中
        A[index] = grad_one # 存入A中，A的行是样本个数，列是梯度个数
    K = torch.mm(A, A.T) # 矩阵的乘法组成K矩阵，这个K矩阵的获得是利用了矩阵的乘法，矩阵的乘法可以加速获得K矩阵，这里要避免使用for循环
    L,_ = torch.eig(K.data.cpu()) # torch不能对gpu求特征值
    eig = torch.norm(L, dim=1)
    return eig
def interface1(Ni1): # 内部的边界点,由于交界面附近的梯度很大，所以多配一些点
    '''
     生成裂纹尖端上半圆的点，为了多分配点
    '''
    
    theta = np.pi*np.random.rand(Ni1) # 角度0到pi
    rp = delta*np.random.rand(Ni1) # 半径0到delta
    x = np.cos(theta) * rp
    y = np.sin(theta) * rp
    xi = np.stack([x, y], 1)
    xi = torch.tensor(xi, requires_grad=True, device='cuda')
    return xi
def interface2(Ni1): # 外部的边界点
    '''
     生成裂纹尖端下半圆的点，为了多分配点
    '''
    
    theta = -np.pi*np.random.rand(Ni1) # 角度0到-pi
    rp = delta*np.random.rand(Ni1) # 半径0到delta
    x = np.cos(theta) * rp
    y = np.sin(theta) * rp
    xi = np.stack([x, y], 1)
    xi = torch.tensor(xi, requires_grad=True, device='cuda')
    return xi
def interface(Ni):
    '''
     生成交界面的随机点
    '''
    re = np.random.rand(Ni)
    theta = 0
    x = np.cos(theta) * re
    y = np.sin(theta) * re
    xi = np.stack([x, y], 1)
    xi = torch.tensor(xi,   requires_grad=True, device='cuda')
    return xi
def train_data(Nb, Nf):
    '''
    生成强制边界点，四周以及裂纹处
    生成上下的内部点
    '''
    

    xu = np.hstack([np.random.rand(int(Nb/4),1)*2-1,  np.ones([int(Nb/4),1])])
    xd = np.hstack([np.random.rand(int(Nb/4),1)*2-1,  -np.ones([int(Nb/4),1])])
    xl = np.hstack([-np.ones([int(Nb/4),1]), np.random.rand(int(Nb/4),1)*2-1])
    xr = np.hstack([np.ones([int(Nb/4),1]), np.random.rand(int(Nb/4),1)*2-1])
    xcrack = np.hstack([-np.random.rand(int(Nb/4),1), np.zeros([int(Nb/4),1])])
    # 随机撒点时候，边界没有角点，这里要加上角点
    xc1 = np.array([[-1., 1.], [1., 1.], [1., -1.], [-1., -1.]]) # 边界点效果不好，增加训练点
    xc2 = np.array([[-1., 1.], [1., 1.], [1., -1.], [-1., -1.]])
    xc3 = np.array([[-1., 1.], [1., 1.], [1., -1.], [-1., -1.]])
    xc4 = np.stack((np.zeros(Nb) ,np.random.rand(Nb)*2-1), 1)
    xc4 = torch.tensor(xc4, device='cuda')
    Xb = np.concatenate((xu, xd, xl, xr, xcrack, xc1, xc2, xc3)) # 上下左右四个边界组装起来

    Xb = torch.tensor(Xb, device='cuda') # 转化成tensor
    
    Xf = torch.rand(Nf, 2)*2-1

    Xf1 = Xf[(Xf[:, 1]>0) & (torch.norm(Xf, dim=1)>=delta)] # 上区域点，去除内部多配的点
    Xf2 = Xf[(Xf[:, 1]<0) & (torch.norm(Xf, dim=1)>=delta)]  # 下区域点，去除内部多配的点
    
    Xf1 = torch.tensor(Xf1, requires_grad=True, device='cuda')
    Xf2 = torch.tensor(Xf2,  requires_grad=True, device='cuda')
    
    return Xb, Xf1, Xf2

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
        self.linear4 = torch.nn.Linear(H, D_out)
        
        #self.a1 = torch.nn.Parameter(torch.Tensor([0.2]))
        

        self.a1 = torch.Tensor([0.1]).cuda()
        self.a2 = torch.Tensor([0.1])
        self.a3 = torch.Tensor([0.1])
        self.n = 1/self.a1.data.cuda()


        torch.nn.init.normal_(self.linear1.weight, mean=0, std=np.sqrt(2/(D_in+H)))
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear3.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear4.weight, mean=0, std=np.sqrt(2/(H+D_out)))
        
        
        torch.nn.init.normal_(self.linear1.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear2.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear3.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear4.bias, mean=0, std=1)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        y1 = torch.tanh(self.n*self.a1*self.linear1(x))
        y2 = torch.tanh(self.n*self.a1*self.linear2(y1))
        y3 = torch.tanh(self.n*self.a1*self.linear3(y2))
        y = self.n*self.a1*self.linear4(y3)
        return y

# for the u0 solution satisfying the homogenous boundary condition
class homo(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(homo, self).__init__()
        self.linear1 = torch.nn.Linear(H, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, D_out)
        #self.trans = torch.nn.Linear(1, H)
        # self.a1 = torch.nn.Parameter(torch.Tensor([0.2]))
        

        self.a1 = torch.Tensor([0.1]).cuda()
        self.a2 = torch.Tensor([0.1])
        self.a3 = torch.Tensor([0.1])
        self.n = 1/self.a1.data
        
        self.fre1 = torch.ones(25).cuda()
        self.fre2 = torch.ones(25).cuda()
        
        torch.nn.init.normal_(self.linear1.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear2.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear3.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear4.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear5.bias, mean=0, std=1)

        torch.nn.init.normal_(self.fre1, mean=0, std=a)
        torch.nn.init.normal_(self.fre2, mean=0, std=a)

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
        yt1 = torch.cat((torch.cos(self.fre1*x[:, 0].unsqueeze(1)), torch.sin(self.fre1*x[:, 0].unsqueeze(1))), 1)
        yt2 = torch.cat((torch.cos(self.fre2*x[:, 0].unsqueeze(1)), torch.sin(self.fre2*x[:, 0].unsqueeze(1))), 1)
        yt3 = torch.cat((torch.cos(self.fre1*x[:, 1].unsqueeze(1)), torch.sin(self.fre1*x[:, 1].unsqueeze(1))), 1)
        yt4 = torch.cat((torch.cos(self.fre2*x[:, 1].unsqueeze(1)), torch.sin(self.fre2*x[:, 1].unsqueeze(1))), 1)        
        yt = torch.cat((yt1, yt2, yt3, yt4), 1)
        y1 = torch.tanh(self.n*self.a1*self.linear1(yt))
        y2 = torch.tanh(self.n*self.a1*self.linear2(y1))
        y3 = torch.tanh(self.n*self.a1*self.linear3(y2)) + y1
        y4 = torch.tanh(self.n*self.a1*self.linear4(y3)) + y2
        y = self.n*self.a1*self.linear5(y4)
        return y
    
def pred(xy):
    '''
    

    Parameters
    ----------
    Nb : int
        the  number of boundary point.
    Nf : int
        the  number of internal point.

    Returns
    -------
    Xb : tensor
        The boundary points coordinates.
    Xf1 : tensor
        interior hole.
    Xf2 : tensor
        exterior region.
    '''
    pred = torch.zeros((len(xy), 1), device = 'cuda')
    # 上区域的预测，由于要用到不同的特解网络
    pred[(xy[:, 1]>0) ] = model_p1(xy[xy[:, 1]>0]) + \
        RBF(xy[xy[:, 1]>0]) * model_h(xy[xy[:, 1]>0])
    # 下区域的预测，由于要用到不同的特解网络
    pred[xy[:, 1]<0] = model_p2(xy[xy[:, 1]<0]) + \
        RBF(xy[xy[:, 1]<0]) * model_h(xy[xy[:, 1]<0])
    # 裂纹右端的预测，用到不同的特解网络，所以去了平局值，即是分片的交界面, 下面最好不要用，因为如果用下面的代码，就要用到裂纹处的判断，然而在裂纹处的判断是会出现paradox的
    pred[(xy[:, 1]==0) & (xy[:, 0]>0)] = (model_p1(xy[(xy[:, 1]==0) & (xy[:, 0]>0)]) + model_p2(xy[(xy[:, 1]==0) & (xy[:, 0]>0)]))/2  + \
        RBF(xy[(xy[:, 1]==0) & (xy[:, 0]>0)]) * model_h(xy[(xy[:, 1]==0) & (xy[:, 0]>0)])

    pred[(xy[:, 1]==0) & (xy[:, 0]==0)] = (model_p1(xy[(xy[:, 1]==0) & (xy[:, 0]==0)]) + model_p2(xy[(xy[:, 1]==0) & (xy[:, 0]==0)]))/2  + \
        RBF(xy[(xy[:, 1]==0) & (xy[:, 0]==0)]) * model_h(xy[(xy[:, 1]==0) & (xy[:, 0]==0)])

    return pred    
    
def evaluate():
    N_test = 100
    x = np.linspace(-1, 1, N_test)
    y = np.linspace(-1, 1, N_test)
    x, y = np.meshgrid(x, y)
    xy_test = np.stack((x.flatten(), y.flatten()),1)
    xy_test = torch.from_numpy(xy_test).cuda() # to the model for the prediction
    
    u_pred = pred(xy_test)
    u_pred = u_pred.data
    u_pred = u_pred.reshape(N_test, N_test)
    

    u_exact = np.zeros(x.shape)
    u_exact[y>0] = np.sqrt(np.sqrt(x[y>0]**2+y[y>0]**2))*np.sqrt((1-x[y>0]/np.sqrt(x[y>0]**2+y[y>0]**2))/2)
    u_exact[y<0] = -np.sqrt(np.sqrt(x[y<0]**2+y[y<0]**2))*np.sqrt((1-x[y<0]/np.sqrt(x[y<0]**2+y[y<0]**2))/2)

    u_exact = u_exact.reshape(x.shape)
    u_exact = torch.tensor(u_exact)
    error = torch.abs(u_pred - u_exact) # get the error in every points
    error_t = torch.norm(error)/torch.norm(u_exact) # get the total relative L2 error
    return error_t


# learning the boundary solution
# step 1: get the boundary train data and target for boundary condition

if train_p == 1:
    model_p1 = particular(2, 10, 1).cuda() # 定义两个特解，这是因为在裂纹处是间断值
    model_p2 = particular(2, 10, 1).cuda()
    tol_p = 0.000001
    loss_bn = 100
    epoch_b = 0
    criterion = torch.nn.MSELoss()
    optimp = torch.optim.Adam(params=chain(model_p1.parameters(), model_p2.parameters()), lr= 0.0005)
    loss_b_array = []
    loss_b1_array = []
    loss_b2_array = []
    loss_bi_array = []
    loss_bn_array = []
    # training the particular network for particular solution on the bonudaray
    while loss_bn>tol_p:
        
        if epoch_b%10 == 0:
            Xb, Xf1, Xf2 = train_data(256, 4096)
            Xi = interface(1000)
            Xb1 = Xb[Xb[:, 1]>=0] # 上边界点
            Xb2 = Xb[Xb[:, 1]<=0] # 下边界点
            target_b1 = torch.sqrt(torch.sqrt(Xb1[:,0]**2+Xb1[:, 1]**2))*torch.sqrt((1-Xb1[:,0]/torch.sqrt(Xb1[:,0]**2+Xb1[:,1]**2))/2)
            target_b1 = target_b1.unsqueeze(1)
            target_b2 = -torch.sqrt(torch.sqrt(Xb2[:,0]**2+Xb2[:, 1]**2))*torch.sqrt((1-Xb2[:,0]/torch.sqrt(Xb2[:,0]**2+Xb2[:,1]**2))/2)
            target_b2 = target_b2.unsqueeze(1)
        epoch_b = epoch_b + 1
        def closure():  
            pred_b1 = model_p1(Xb1) # predict the boundary condition
            loss_b1 = criterion(pred_b1, target_b1)  
            pred_b2 = model_p2(Xb2) # predict the boundary condition
            loss_b2 = criterion(pred_b2, target_b2) 
            loss_b = loss_b1 + loss_b2# 边界的损失函数
            # 求交界面的损失函数，因为这里是两个特解网络
            pred_bi1 = model_p1(Xi) # predict the boundary condition
            pred_bi2 = model_p2(Xi) # predict the boundary condition
            loss_bi = criterion(pred_bi1, pred_bi2)             
            
            optimp.zero_grad()
            loss_bn = loss_b + loss_bi # 本质边界和交界面损失的总和
            loss_bn.backward()
            loss_b_array.append(loss_b.data)
            loss_b1_array.append(loss_b1.data)
            loss_b2_array.append(loss_b2.data)
            loss_bi_array.append(loss_bi.data)
            loss_bn_array.append(loss_bn.data)
            if epoch_b%10==0:
                print('trianing particular network : the number of epoch is %i, the loss1 is %f, the loss2 is %f, the lossi is %f' % (epoch_b, loss_b1.data, loss_b2.data, loss_bi.data))
            return loss_bn
        optimp.step(closure)
        loss_bn = loss_bn_array[-1]
    torch.save(model_p1, './particular1_nn')
    torch.save(model_p2, './particular2_nn')
model_p1 = torch.load('particular1_nn')
model_p2 = torch.load('particular2_nn')

# learning the distance neural network
def RBF(x):
    
    d_total_t = torch.tensor(d_total, device='cuda').unsqueeze(1)
    
    w_t = torch.tensor(w,  device='cuda')

    x_l = x.unsqueeze(0).repeat(len(d_total_t), 1, 1) # 创立一个足够大的x矩阵
    # 得到R2大矩阵
    R2 = torch.norm(d_total_t - x_l, dim=2)
    y = torch.mm(torch.exp(-gama*R2.T), w_t)
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

# 由于边界上所有点都是本质边界条件，且都为0，所以我们这里弄一个内部点来计算非零距离
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
R2 = np.sqrt(dx**2+dy**2)
K = np.exp(-0.5*R2)
b = np.concatenate((d_dir, d_dom))
w = np.dot(np.linalg.inv(K),b)

n_test = 101 # 获得测试点 n_test**2个
domx_t = np.linspace(-1, 1, n_test)
domy_t = np.linspace(-1, 1, n_test)
domx_t, domy_t = np.meshgrid(domx_t, domy_t)
domxy_t = np.stack((domx_t.flatten(), domy_t.flatten()), 1)
domxy_t= torch.tensor(domxy_t,  requires_grad=True, device = 'cuda') # 获得了两列的测试点，放入RBF中，这是为了和其他神经网络结构保持一致

dis = RBF(domxy_t)
# plot 配点图以及RBF距离函数





# get f_p from model_p 

# learning the homogenous network



model_h = homo(2, 200, 1).cuda()
criterion = torch.nn.MSELoss()
optim_h = torch.optim.Adam(params=chain(model_h.parameters(), model_p1.parameters(), model_p2.parameters()), lr= 0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optim_h, milestones=[1000, 3000, 5000], gamma = 0.1)
loss_array = []
error_array = []
loss1_array = []
loss2_array = []  
lossii1_array = []  
lossii2_array = []   
lossi_array = []
nepoch_u0 = int(nepoch_u0)
eigvalues = []
start = time.time()
for epoch in range(nepoch_u0):
    if epoch ==1000:
        end = time.time()
        consume_time = end-start
        print('time is %f' % consume_time)
    if epoch%100 == 0:        
        Xb, Xf1, Xf2 = train_data(256, 101)
        Xi1 = interface1(5000)
        Xi2 = interface2(5000)
        Xi = interface(1000)
        u_exact1 = torch.zeros((len(Xf1), 1))
        u_exact1 = torch.sqrt(torch.sqrt(Xf1[:, 0]**2+Xf1[:, 1]**2))*torch.sqrt((1-Xf1[:, 0]/torch.sqrt(Xf1[:, 0]**2+Xf1[:, 1]**2))/2)
        u_exact1 = u_exact1.unsqueeze(1)
        u_exacti1 = torch.zeros((len(Xi1), 1))
        u_exacti1 = torch.sqrt(torch.sqrt(Xi1[:, 0]**2+Xi1[:, 1]**2))*torch.sqrt((1-Xi1[:, 0]/torch.sqrt(Xi1[:, 0]**2+Xi1[:, 1]**2))/2)
        u_exacti1 = u_exacti1.unsqueeze(1)
        
        u_exact2 = torch.zeros((len(Xf2), 1))
        u_exact2 = -torch.sqrt(torch.sqrt(Xf2[:, 0]**2+Xf2[:, 1]**2))*torch.sqrt((1-Xf2[:, 0]/torch.sqrt(Xf2[:, 0]**2+Xf2[:, 1]**2))/2)
        u_exact2 = u_exact2.unsqueeze(1)
        u_exacti2 = torch.zeros((len(Xi2), 1))
        u_exacti2 = -torch.sqrt(torch.sqrt(Xi2[:, 0]**2+Xi2[:, 1]**2))*torch.sqrt((1-Xi2[:, 0]/torch.sqrt(Xi2[:, 0]**2+Xi2[:, 1]**2))/2)
        u_exacti2 = u_exacti2.unsqueeze(1)
        
        Xb1 = Xb[Xb[:, 1]>=0] # 上边界点
        Xb2 = Xb[Xb[:, 1]<=0] # 下边界点
        target_b1 = torch.sqrt(torch.sqrt(Xb1[:,0]**2+Xb1[:, 1]**2))*torch.sqrt((1-Xb1[:,0]/torch.sqrt(Xb1[:,0]**2+Xb1[:,1]**2))/2)
        target_b1 = target_b1.unsqueeze(1)
        target_b2 = -torch.sqrt(torch.sqrt(Xb2[:,0]**2+Xb2[:, 1]**2))*torch.sqrt((1-Xb2[:,0]/torch.sqrt(Xb2[:,0]**2+Xb2[:,1]**2))/2)
        target_b2 = target_b2.unsqueeze(1)
    def closure():  

        u_h1 = model_h(Xf1) 
        u_h2 = model_h(Xf2)
        
        u_p1 = model_p1(Xf1)  
        u_p2 = model_p2(Xf2) # 获得特解
        # 构造可能位移场
        u_pred1 = u_p1 + RBF(Xf1)*u_h1
        u_pred2 = u_p2 + RBF(Xf2)*u_h2

        J1 = criterion(u_pred1, u_exact1)
        
        J2 = criterion(u_pred2, u_exact2)

        # 添加上交界面的能量
        
        u_i1 = model_p1(Xi1)  + RBF(Xi1)*model_h(Xi1)  # 分别得到两个网络的交界面预测
        u_i2 = model_p2(Xi2)  + RBF(Xi2)*model_h(Xi2)         


        # 计算交界面能量
        Jie1 = criterion(u_i1, u_exacti1) 
        Jie2 = criterion(u_i2, u_exacti2) 
        if delta != 0:
            J = J1 + J2 + Jie1 + Jie2
        if delta ==0:
            J = J1 + J2
        
        # 算一算积分值是多少，解析积分是0.8814
        du1dxy = grad(u_pred1, Xf1, torch.ones(Xf1.size()[0], 1).cuda(), create_graph=True)[0]
        du1dx = du1dxy[:, 0].unsqueeze(1)
        du1dy = du1dxy[:, 1].unsqueeze(1)

        du2dxy = grad(u_pred2, Xf2, torch.ones(Xf2.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        du2dx = du2dxy[:, 0].unsqueeze(1)
        du2dy = du2dxy[:, 1].unsqueeze(1)

        J1e = (0.5 * torch.sum(du1dx**2 + du1dy**2)) * (2- 0.5*(delta)**2*np.pi)/len(Xf1) 
        
        J2e = (0.5 * torch.sum(du2dx**2 + du2dy**2)) * (2- 0.5*(delta)**2*np.pi)/len(Xf2) 

        # 添加上交界面的能量       

        du1dxyi = grad(u_i1, Xi1, torch.ones(Xi1.size()[0], 1).cuda(), create_graph=True)[0]
        du1dxi = du1dxyi[:, 0].unsqueeze(1)
        du1dyi = du1dxyi[:, 1].unsqueeze(1)

        du2dxyi = grad(u_i2, Xi2, torch.ones(Xi2.size()[0], 1).cuda(), create_graph=True)[0]
        du2dxi = du2dxyi[:, 0].unsqueeze(1)
        du2dyi = du2dxyi[:, 1].unsqueeze(1)     
        # 计算交界面能量
        Jie1e = (0.5 * torch.sum(du1dxi**2 + du1dyi**2)) * 0.5 * (delta**2)*np.pi/len(Xi1) 
        Jie2e = (0.5 * torch.sum(du2dxi**2 + du2dyi**2)) * 0.5 * (delta**2)*np.pi/len(Xi2) 
        if delta != 0:
            Je = J1e + J2e + Jie1e + Jie2e
        if delta ==0:
            Je = J1e + J2e
        
        # # 特解网络再次训练，因为在y=0，距离网络的导数为0
        # pred_b1 = model_p1(Xb1) # predict the boundary condition
        # loss_b1 = criterion(pred_b1, target_b1)  
        # pred_b2 = model_p2(Xb2) # predict the boundary condition
        # loss_b2 = criterion(pred_b2, target_b2) 
        # loss_b = loss_b1 + loss_b2# 边界的损失函数
        # 求交界面的损失函数，因为这里是两个特解网络
        pred_bi1 = model_p1(Xi) # predict the boundary condition
        pred_bi2 = model_p2(Xi) # predict the boundary condition
        loss_bi = criterion(pred_bi1, pred_bi2)  
        
        loss = J + penalty * ( loss_bi)
        
        error_t = evaluate()
        optim_h.zero_grad()
        loss.backward(retain_graph=True)
        loss_array.append(loss.data.cpu())
        error_array.append(error_t.data.cpu())
        loss1_array.append(J1.data.cpu())
        loss2_array.append(J2.data.cpu())
        lossii1_array.append(Jie1.data.cpu())
        lossii2_array.append(Jie2.data.cpu())
        if epoch%10==0:
            print(' epoch : %i, the loss : %f ,  J1 : %f , J2 : %f , Jie1 : %f, Jie2 : %f, J : %f ' % (epoch, loss.data, J1.data, J2.data, Jie1.data, Jie2.data, Je.data))
        if epoch%500==0:
            x_ntk = torch.cat((Xf1[0:50], Xf2[0:50])).data
            pred_ntk = pred(x_ntk)
            eigvalue = NTK(pred_ntk)
            eigvalues.append(eigvalue)
            print('the NTK is done')
        return loss
    optim_h.step(closure)
    scheduler.step()
    
    
# %%
N_test = 101
x = np.linspace(-1, 1, N_test)
y = np.linspace(-1, 1, N_test)
x, y = np.meshgrid(x, y)
xy_test = np.stack((x.flatten(), y.flatten()),1)
xy_test = torch.from_numpy(xy_test).cuda() # to the model for the prediction

u_pred = pred(xy_test)
u_pred = u_pred.data
u_pred = u_pred.reshape(N_test, N_test).cpu()

u_exact = np.zeros(x.shape)
u_exact[y>0] = np.sqrt(np.sqrt(x[y>0]**2+y[y>0]**2))*np.sqrt((1-x[y>0]/np.sqrt(x[y>0]**2+y[y>0]**2))/2)
u_exact[y<0] = -np.sqrt(np.sqrt(x[y<0]**2+y[y<0]**2))*np.sqrt((1-x[y<0]/np.sqrt(x[y<0]**2+y[y<0]**2))/2)
u_exact = torch.from_numpy(u_exact)
error = torch.abs(u_pred - u_exact) # get the error in every points
error_t = torch.norm(error)/torch.norm(u_exact) # get the total relative L2 error

# plot the prediction solution
fig = plt.figure(figsize=(20, 20))

plt.subplot(4, 3, 1)
Xb, Xf1, Xf2 = train_data(256, 4096)
Xi1 = interface1(1000)
Xi2 = interface2(1000)
plt.scatter(Xb.detach().cpu().numpy()[:, 0], Xb.detach().cpu().numpy()[:, 1], c='r', s = 0.1)
plt.scatter(Xf1.detach().cpu().numpy()[:, 0], Xf1.detach().cpu().numpy()[:, 1], c='b', s = 0.1)
plt.scatter(Xf2.detach().cpu().numpy()[:, 0], Xf2.detach().cpu().numpy()[:, 1], c='g', s = 0.1)
plt.scatter(Xi1.detach().cpu().numpy()[:, 0], Xi1.detach().cpu().numpy()[:, 1], c='m', s = 0.1)
plt.scatter(Xi2.detach().cpu().numpy()[:, 0], Xi2.detach().cpu().numpy()[:, 1], c='y', s = 0.1)

plt.subplot(4, 3, 2)
plt.scatter(d_total[:, 0], d_total[:, 1])
plt.title('the RBF points')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(4, 3, 3)
dis_plot = dis.data.cpu().numpy().reshape(n_test, n_test)
h1 = plt.contourf(domx_t, domy_t, dis_plot, levels = 50)
plt.colorbar(h1)
plt.title('the RBF distance function')
plt.xlabel('x')
plt.ylabel('y')


plt.subplot(4, 3, 4)
h2 = plt.contourf(x, y, u_pred.detach().numpy(), levels=100 ,cmap = 'jet')
plt.title('datadriven prediction') 
plt.colorbar(h2)
plt.subplot(4, 3, 5)
h3 = plt.contourf(x, y, u_exact, levels=100 ,cmap = 'jet')
plt.title('exact solution') 
plt.colorbar(h3)
plt.subplot(4, 3, 6)
h4 = plt.contourf(x, y, error.detach().numpy(), levels=100 ,cmap = 'jet')
plt.title('absolute error') 
plt.colorbar(h4)
plt.subplot(4, 3, 7)
loss_array = np.array(loss_array)
loss_array = loss_array[loss_array<50]
plt.yscale('log')
plt.plot(loss_array)
plt.xlabel('the iteration')
plt.ylabel('loss')
plt.title('loss evolution') 
plt.subplot(4, 3, 8)
error_array = np.array(error_array)
error_array = error_array[error_array<1]
plt.yscale('log')
plt.plot(error_array)
plt.xlabel('the iteration')
plt.ylabel('error')
plt.title('relative total error evolution') 
plt.subplot(4, 3, 9)
interx = torch.linspace(0, 1, N_test)[1:-1] # 获得interface的测试点
intery = torch.zeros(N_test)[1:-1]
inter = torch.stack((interx, intery),1)
inter = inter.requires_grad_(True).cuda() # 可以求导并且放入cuda中
pred_inter = pred(inter)
dudxyi = grad(pred_inter, inter, torch.ones(inter.size()[0], 1).cuda())[0]
dudyi = dudxyi[:, 1].unsqueeze(1) # 获得了交界面奇异应变
dudyi_e = 0.5/torch.sqrt(torch.norm(inter, dim=1))# 获得交界面的应变的精确值 
plt.plot(interx.cpu(), dudyi_e.detach().cpu(), label = 'exact')
plt.plot(interx.cpu(), dudyi.detach().cpu(), label = 'predict')
plt.xlabel('coordinate x')
plt.ylabel('strain e32')
plt.title('cpinn about e32 on interface')
plt.legend()


plt.subplot(4, 3, 10) # NTK分析
for index, i in enumerate(eigvalues):
    num = 500*index
    plt.yscale('log')
    plt.plot(i, label = '%i' % num)
    plt.title('Spectrum NTK')
plt.legend()
plt.subplot(4, 3, 11) # 精确解的二维频域图
fre = np.abs(np.fft.fftshift(np.fft.fft2(u_exact.cpu().numpy())))
plt.imshow(fre,                         #numpy array generating the image
            #color map used to specify colors
           interpolation='nearest'    #algorithm used to blend square colors; with 'nearest' colors will not be blended
          )
plt.colorbar()
plt.title('the frequency of the exact solution') 

plt.subplot(4, 3, 12) # 预测解的二维频域图
fre_p = np.abs(np.fft.fftshift(np.fft.fft2(u_pred.cpu().numpy())))
plt.imshow(fre_p,                         #numpy array generating the image
            #color map used to specify colors
           interpolation='nearest'    #algorithm used to blend square colors; with 'nearest' colors will not be blended
          )
plt.colorbar()
plt.title('the frequency of the pred solution') 
plt.suptitle("data driven")
plt.savefig('./picture/data_driven.png')
plt.show()
print('the relative error is %f' % error_t.data)   
    
    
#for paper plot
fig = plt.figure(dpi=1000,figsize = (8,3))
fig.tight_layout()
plt.subplots_adjust(wspace = 0.4, hspace=0)

plt.subplot(1, 2, 1)
plt.scatter(d_total[:, 0], d_total[:, 1])
plt.title('the RBF points')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(1, 2, 2)
dis_plot = dis.data.cpu().numpy().reshape(n_test, n_test)
h1 = plt.contourf(domx_t, domy_t, dis_plot, levels = 50)
plt.colorbar(h1)
plt.title('the RBF distance function')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('../../图片/裂纹/RBF.pdf', bbox_inches = 'tight')
plt.show()

# 画x=0中线的结果，发现这个结果不好，可能是因为训练点不包含x=0的原因
# %%
y = torch.linspace(-1,1,N_test)
x= torch.zeros(N_test)
xy_test = torch.stack((x, y),1).cuda()
u_pred_mid_data = torch.zeros((N_test,1)).cuda()
u_pred_mid_data[y>0] = model_p1(xy_test[y>0]) + RBF(xy_test[y>0])*model_h(xy_test[y>0]) # 用上半部分网络获得预测解,尽量test不要包括中点
u_pred_mid_data[y<0] = model_p2(xy_test[y<0]) + RBF(xy_test[y<0])*model_h(xy_test[y<0]) # 用下半部分网络获得预测解
u_pred_mid_data[y==0] = (model_p1(xy_test[y==0]) + model_p2(xy_test[y==0]))/2 + RBF(xy_test[y==0])*model_h(xy_test[y==0]) # 用下半部分网络获得预测解
u_pred_mid_data = u_pred_mid_data.flatten().cpu()
u_exact_mid = torch.zeros(N_test) # 获得中线的精确解
u_exact_mid[y>0] = torch.norm(xy_test[y>0], dim=1)**0.5*np.sqrt(2)/2
u_exact_mid[y<0] = -torch.norm(xy_test[y<0], dim=1)**0.5*np.sqrt(2)/2
u_exact_mid = u_exact_mid.cpu()
error_mid_data = torch.abs(u_pred_mid_data - u_exact_mid) # get the error in every points
error_mid_data = torch.norm(error_mid_data)/torch.norm(u_exact_mid) # get the total relative L2 error
plt.plot(y.cpu(), u_exact_mid)
plt.plot(y.cpu(), u_pred_mid_data.detach(), linestyle=':')
plt.legend(['exact_mid', 'data_driven_mid'], loc='upper right')
plt.xlabel('y')
plt.ylabel('u')
plt.show()













