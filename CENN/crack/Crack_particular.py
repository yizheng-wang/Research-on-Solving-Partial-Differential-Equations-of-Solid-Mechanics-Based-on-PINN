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

import matplotlib as mpl
mpl.rcParams['font.sans-serif']=['SimHei'] # 为了显示中文matplotlib

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     #random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

torch.set_default_tensor_type(torch.DoubleTensor) # 将tensor的类型变成默认的double
torch.set_default_tensor_type(torch.cuda.DoubleTensor) # 将cuda的tensor的类型变成默认的double
penalty = 100
delta = 0.0
train_p = 1
nepoch_u0 = 101 # the float type
a = 1
tol_p = 0.000001

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


# learning the boundary solution
# step 1: get the boundary train data and target for boundary condition

if train_p == 1:
    start = time.time()
    model_p1 = particular(2, 10, 1).cuda() # 定义两个特解，这是因为在裂纹处是间断值
    model_p2 = particular(2, 10, 1).cuda()
    
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
            loss_b1_array.append(loss_b1.data.cpu())
            loss_b2_array.append(loss_b2.data.cpu())
            loss_bi_array.append(loss_bi.data.cpu())
            loss_bn_array.append(loss_bn.data)
            if epoch_b%10==0:
                print('trianing particular network : the number of epoch is %i, the loss1 is %f, the loss2 is %f, the lossi is %f' % (epoch_b, loss_b1.data, loss_b2.data, loss_bi.data))
            return loss_bn
        optimp.step(closure)
        loss_bn = loss_bn_array[-1]
#     torch.save(model_p1, './particular1_nn')
#     torch.save(model_p2, './particular2_nn')
model_p1 = torch.load('particular1_nn')
model_p2 = torch.load('particular2_nn')
end = time.time()        
#consume_time = end-start
#print('time is %f' % consume_time)
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
    pred[(xy[:, 1]>0) ] = model_p1(xy[xy[:, 1]>0]) 
    # 下区域的预测，由于要用到不同的特解网络
    pred[xy[:, 1]<0] = model_p2(xy[xy[:, 1]<0]) 
    # 裂纹右端的预测，用到不同的特解网络，所以去了平局值，即是分片的交界面, 下面最好不要用，因为如果用下面的代码，就要用到裂纹处的判断，然而在裂纹处的判断是会出现paradox的
    return pred   

# %%
N_test = 100
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

#for paper plot
#fig = plt.figure(dpi=1000,figsize = (20,5))
#fig = 
#fig.tight_layout()
# plot the prediction solution
#plt.subplots_adjust(wspace = 0.4, hspace=0)
fig = plt.figure(dpi=1000,figsize = (6.6,5))
h3 = plt.contourf(x, y, u_exact, levels=100 ,cmap = 'jet')
#plt.title('Exact solution') 
plt.colorbar(h3, ticks = np.linspace(-1, 1.2, 12)).ax.set_title('U')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

fig = plt.figure(dpi=1000,figsize = (6.6,5))
dis_plot = u_pred.data.cpu().numpy().reshape(N_test, N_test)
h1 = plt.contourf(x, y, dis_plot, cmap='jet',  levels=100)
plt.colorbar(h1, ticks = np.linspace(-1, 1.2, 12)).ax.set_title('U')
#plt.title('Particular network')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

fig = plt.figure(dpi=1000,figsize = (6.6,5))
plt.ticklabel_format(style='sci', scilimits=(-1,2), axis = 'x')
plt.yscale('log')
plt.plot(loss_b1_array)
plt.plot(loss_b2_array)
plt.plot(loss_bi_array)
#plt.xticks(fontsize=10)
plt.legend(['ParticularNN_1', 'ParticularNN_2', 'Interface'])
#plt.title('Loss evolution')
plt.xlabel('Iteration')
plt.ylabel('Loss')
#plt.savefig('../../picture/crack/particularnn.pdf', bbox_inches = 'tight')
plt.show()

# 画一个轨道比较
# 先画上轨道，即是上面的区域
# %%
fig = plt.figure(dpi=1000,figsize = (6.6,5))
x1 = np.stack((np.linspace(0, -1, 11), np.zeros(11)), 1)
x2 = np.stack((-np.ones(6), np.linspace(0, 1, 6)), 1)
x3 = np.stack((np.linspace(-1, 1, 11), np.ones(11)), 1)
x4 = np.stack((np.ones(6), np.linspace(1, 0, 6)), 1)
xstack1 = np.concatenate((x1, x2, x3, x4))
_, idx = np.unique(xstack1, return_index=True,axis =0)
xstack1 = xstack1[np.sort(idx)][:-1]  # 抛去最后一个点，因为是重复的
exact_part1 = np.sqrt((np.sqrt(xstack1[:, 0]**2+xstack1[:, 1]**2)-xstack1[:, 0])/2)

x5 = np.stack((np.ones(6), np.linspace(0, -1, 6)), 1)
x6 = np.stack((np.linspace(1, -1, 11), -np.ones(11)), 1)
x7 = np.stack((-np.ones(6), np.linspace(-1, 0, 6)), 1)
x8 = np.stack((np.linspace(-1, 0, 11), np.zeros(11)), 1)
xstack2 = np.concatenate((x5, x6, x7, x8))
_, idx = np.unique(xstack2, return_index=True,axis =0)
xstack2 = xstack2[np.sort(idx)][:-1] # 抛去最后一个点，因为是重复的
exact_part2 = -np.sqrt((np.sqrt(xstack2[:, 0]**2+xstack2[:, 1]**2)-xstack2[:, 0])/2)

exact_part = np.concatenate((exact_part1, exact_part2)) # 将两个精确解拼起来
xstack = np.concatenate((xstack1, xstack2))

pred_part1 = model_p1(torch.tensor(xstack1).cuda())
pred_part2 = model_p2(torch.tensor(xstack2).cuda())
pred_part = torch.cat((pred_part1.data.cpu(), pred_part2.data.cpu())).numpy().flatten()

love = np.linspace(0, 10, 61)[:-1]  # 抛去最后一个点，因为是重复的
plt.scatter(love, exact_part, marker = '*')
plt.plot(love, pred_part)
plt.legend(['Exact', 'ParticularNN'])
#plt.title('Essential conditon')
plt.xlabel('r')
plt.ylabel('u')
#plt.savefig('../../picture/crack/particular_circle.pdf', bbox_inches = 'tight')
plt.show()

