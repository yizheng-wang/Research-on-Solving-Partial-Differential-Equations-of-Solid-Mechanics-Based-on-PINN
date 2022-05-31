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

torch.set_default_tensor_type(torch.DoubleTensor) # 将tensor的类型变成默认的double
torch.set_default_tensor_type(torch.cuda.DoubleTensor) # 将cuda的tensor的类型变成默认的double
train_p = 0
tol_p = 0.00001
a = 1
a1 = 1/15
a2 = 1
r0 = 0.5
nepoch_u0 = 2500
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
    
    
# for the u0 solution satisfying the homogenous boundary condition
class homo(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(homo, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, D_out)
#        self.trans = torch.nn.Linear(1, H)
        # self.a1 = torch.nn.Parameter(torch.Tensor([0.2]))
        

        self.a1 = torch.nn.Parameter(torch.Tensor([0.1])).cuda()
        self.a2 = torch.Tensor([0.1])
        self.a3 = torch.Tensor([0.1])
        self.n = 1/self.a1.data
        
        self.fre1 = torch.ones(25).cuda()
        self.fre2 = torch.ones(25).cuda()
        self.fre3 = torch.ones(25).cuda()
        self.fre4 = torch.ones(25).cuda()
        self.fre5 = torch.ones(25).cuda()
        self.fre6 = torch.ones(25).cuda()
        self.fre7 = torch.ones(25).cuda()
        self.fre8 = torch.ones(25).cuda()
        
        torch.nn.init.normal_(self.linear1.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear2.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear3.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear4.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear5.bias, mean=0, std=1)

        torch.nn.init.normal_(self.fre1, mean=0, std=a)
        torch.nn.init.normal_(self.fre2, mean=0, std=a*2)
        torch.nn.init.normal_(self.fre1, mean=0, std=a*3)
        torch.nn.init.normal_(self.fre2, mean=0, std=a*4)
        torch.nn.init.normal_(self.fre1, mean=0, std=a*5)
        torch.nn.init.normal_(self.fre2, mean=0, std=a*6)
        torch.nn.init.normal_(self.fre1, mean=0, std=a*7)
        torch.nn.init.normal_(self.fre2, mean=0, std=a*8)
        
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
        #特征变换
        # yt1 = torch.cat((torch.cos(self.fre1*x[:, 0].unsqueeze(1)), torch.sin(self.fre1*x[:, 0].unsqueeze(1))), 1)
        # yt2 = torch.cat((torch.cos(self.fre2*x[:, 0].unsqueeze(1)), torch.sin(self.fre2*x[:, 0].unsqueeze(1))), 1)
        # yt3 = torch.cat((torch.cos(self.fre3*x[:, 1].unsqueeze(1)), torch.sin(self.fre3*x[:, 1].unsqueeze(1))), 1)
        # yt4 = torch.cat((torch.cos(self.fre4*x[:, 1].unsqueeze(1)), torch.sin(self.fre4*x[:, 1].unsqueeze(1))), 1)      
        # yt5 = torch.cat((torch.cos(self.fre5*x[:, 1].unsqueeze(1)), torch.sin(self.fre5*x[:, 1].unsqueeze(1))), 1)
        # yt6 = torch.cat((torch.cos(self.fre6*x[:, 1].unsqueeze(1)), torch.sin(self.fre6*x[:, 1].unsqueeze(1))), 1)        
        # yt7 = torch.cat((torch.cos(self.fre7*x[:, 1].unsqueeze(1)), torch.sin(self.fre7*x[:, 1].unsqueeze(1))), 1)
        # yt8 = torch.cat((torch.cos(self.fre8*x[:, 1].unsqueeze(1)), torch.sin(self.fre8*x[:, 1].unsqueeze(1))), 1)        
        # yt = torch.cat((yt1, yt2, yt3, yt4, yt5, yt6, yt7, yt8), 1)
        yt = x
        y1 = torch.tanh(self.n*self.a1*self.linear1(yt))
        y2 = torch.tanh(self.n*self.a1*self.linear2(y1))
        y3 = torch.tanh(self.n*self.a1*self.linear3(y2)) + y1
        y4 = torch.tanh(self.n*self.a1*self.linear4(y3)) + y2
        y =  self.linear5(y4)
        return y
    
def pred(xy):
    '''
    

    Parameters
    ----------
    输入xytensor，然而输出位移，都是GPU的张量
    '''
    pred = model_p(xy) + RBF(xy) * model_h(xy)
    return pred    
    
def evaluate():
    N_test = 100
    dom_koch_n = koch_points.get_koch_points_lin(N_test) # 均匀在科赫雪花内部步点
    dom_koch_t = torch.tensor(dom_koch_n, device = 'cuda')
    
    u_pred = pred(dom_koch_t)
    u_pred = u_pred.data.cpu()
    

    u_exact = np.zeros((len(dom_koch_n) ,1))
    u_exact[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)<r0**2] = 1/a1*np.linalg.norm(dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)<r0**2], axis=1, keepdims=True)**4 # 现在网格里面赋值
    u_exact[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)>=r0**2] = 1/a2*np.linalg.norm(dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)>=r0**2], axis=1, keepdims=True)**4 + r0**4*(1/a1-1/a2) # 现在网格外面赋值

    u_exact = torch.from_numpy(u_exact)
    error = torch.abs(u_pred - u_exact) # get the error in every points
    error_t = torch.norm(error)/torch.norm(u_exact) # get the total relative L2 error
    return error_t


# learning the boundary solution
# step 1: get the boundary train data and target for boundary condition

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
            Xb = koch.point_bound_rand(100)
            Xb = torch.tensor(Xb).cuda()
            target_b = 1/a2*torch.norm(Xb, dim=1, keepdim=True)**4+(1/a1-1/a2)*r0**4

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
    d_total_t = torch.from_numpy(d_total).unsqueeze(1).cuda()
    w_t = torch.from_numpy(w).cuda()
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


# get f_p from model_p 

# learning the homogenous network



model_h = homo(2, 20, 1).cuda()
criterion = torch.nn.MSELoss()
optim_h = torch.optim.Adam([{'params':model_h.parameters()}], lr= 0.0001) # 0.001比较好
scheduler = torch.optim.lr_scheduler.MultiStepLR(optim_h, milestones=[1000, 3000, 5000], gamma = 0.1)
loss_array = []
error_array = []
eigvalues = []
nepoch_u0 = int(nepoch_u0)
start = time.time()
for epoch in range(nepoch_u0):
    if epoch ==1000:
        end = time.time()
        consume_time = end-start
        print('time is %f' % consume_time)
    if epoch%100 == 0:
        dom_koch_n = koch_points.get_koch_points(10000) # 获得n_test个koch的随机分布点
        dom_koch_n1 = dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)<r0**2]#定义内部的dom点，即是r<r0的点
        dom_koch_n2 = dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)>=r0**2]#定义内部的dom点，即是r<r0的点
        dom_koch_t1= torch.tensor(dom_koch_n1,  requires_grad=True, device = 'cuda') # 将numpy数据转化为tensor
        dom_koch_t2= torch.tensor(dom_koch_n2,  requires_grad=True, device = 'cuda') # 将numpy数据转化为tensor
        f1 = -16 * torch.norm(dom_koch_t1, dim = 1, keepdim=True).data**2 # 定义体力 
        f2 = -16 * torch.norm(dom_koch_t2, dim = 1, keepdim=True).data**2 # 定义体力
    def closure():  

        # 构造可能位移场
        u_pred1 = pred(dom_koch_t1)
        u_pred2 = pred(dom_koch_t2)
        
        du1dxy = grad(u_pred1, dom_koch_t1, torch.ones(dom_koch_t1.size()[0], 1).cuda(), create_graph=True)[0]
        du1dx = du1dxy[:, 0].unsqueeze(1)
        du1dy = du1dxy[:, 1].unsqueeze(1)

        du2dxy = grad(u_pred2, dom_koch_t2, torch.ones(dom_koch_t2.size()[0], 1).cuda(), create_graph=True)[0]
        du2dx = du2dxy[:, 0].unsqueeze(1)
        du2dy = du2dxy[:, 1].unsqueeze(1)

        J1 = ((0.5 * a1 * torch.sum(du1dx**2 + du1dy**2)) - torch.sum(f1*u_pred1)) * (r0**2*np.pi)/len(dom_koch_t1) 
        
        J2 = ((0.5 * a2 * torch.sum(du2dx**2 + du2dy**2)) - torch.sum(f2*u_pred2)) * (10*np.sqrt(3)/3-r0**2*np.pi)/len(dom_koch_t2) 

        J = J1 + J2

        loss = J 
        error_t = evaluate()
        optim_h.zero_grad()
        loss.backward()
        loss_array.append(loss.data.cpu())
        error_array.append(error_t.data.cpu())

        if epoch%10==0:
            print(' epoch : %i, the loss : %f ,  error : %f' % (epoch, loss.data, error_t.data))
        if epoch%500==0:
            x_ntk = torch.cat((dom_koch_t1[0:50], dom_koch_t2[0:50])).data
            pred_ntk = pred(x_ntk)
            eigvalue = NTK(pred_ntk)
            eigvalues.append(eigvalue)
            print('the NTK is done')
        return loss
    optim_h.step(closure)
    scheduler.step()
    
    
    
n_test = 100
dom_koch_n = koch_points.get_koch_points_lin(n_test) # 获得n_test个koch的随机分布点
xy_test= torch.tensor(dom_koch_n,  requires_grad=True, device = 'cuda') # 将numpy数据转化为tensor


u_pred = pred(xy_test)
u_pred = u_pred.data.cpu()

u_exact = np.zeros((len(dom_koch_n) ,1))
u_exact[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)<r0**2] = 1/a1*np.linalg.norm(dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)<r0**2], axis=1, keepdims=True)**4 # 现在网格里面赋值
u_exact[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)>=r0**2] = 1/a2*np.linalg.norm(dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)>=r0**2], axis=1, keepdims=True)**4 + r0**4*(1/a1-1/a2) # 现在网格外面赋值

u_exact = torch.from_numpy(u_exact) # 将精确解从array变成tensor

error = torch.abs(u_pred - u_exact) # get the error in every points
error_t = torch.norm(error)/torch.norm(u_exact) # get the total relative L2 error

# plot the prediction solution
fig = plt.figure(figsize=(20, 20))

plt.subplot(3, 3, 1)
Xb =  koch.point_bound(5)
Xf = koch_points.get_koch_points(10000)
 # 将边界和内部点画图
plt.scatter(Xb[:, 0], Xb[:, 1], c='r', s = 0.1)
plt.scatter(Xf[:, 0], Xf[:, 1], c='b', s = 0.1)

write_arr2DVTK('./output_ntk/pred_energy', dom_koch_n, u_pred, 'pred_energy')

write_arr2DVTK('./output_ntk/error_energy', dom_koch_n, error, 'error_energy')

plt.subplot(3, 3, 2)
loss_array = np.array(loss_array)
plt.yscale('log')
plt.plot(loss_array)
plt.xlabel('the iteration')
plt.ylabel('loss')
plt.title('loss evolution') 
plt.subplot(3, 3, 3)
error_array = np.array(error_array)
plt.yscale('log')
plt.plot(error_array)
plt.xlabel('the iteration')
plt.ylabel('error')
plt.title('relative total error evolution') 

plt.subplot(3, 3, 4) # 做x=0的位移预测
n_test = 100
x0 = np.zeros((n_test, 2))
x0[:, 1] = np.linspace(-np.sqrt(3), np.sqrt(3), 100)
exactx0 = np.zeros((n_test, 1))
exactx0[np.linalg.norm(x0, axis=1)<r0] =  1/a1*np.linalg.norm(x0[np.linalg.norm(x0, axis=1)<r0], axis=1, keepdims=True)**4 # 不同的区域进行不同的解析解赋予
exactx0[np.linalg.norm(x0, axis=1)>=r0] =  1/a2*np.linalg.norm(x0[np.linalg.norm(x0, axis=1)>=r0], axis=1, keepdims=True)**4 + (1/a1-1/a2)*r0**4
x0t = torch.tensor(x0)
predx0 = pred(x0t).data.cpu().numpy() # 预测x=0的原函数
plt.plot(x0[:, 1], exactx0.flatten())
plt.plot(x0[:, 1], predx0.flatten())
plt.xlabel('y')
plt.ylabel('u')
plt.legend(['exact', 'pred'])
plt.title('x=0 u') 

plt.subplot(3, 3, 5) # 做y=0的位移预测
n_test = 100
y0 = np.zeros((n_test, 2))
y0[:, 0] = np.linspace(-1, 1, 100)
exacty0 = np.zeros((n_test, 1))
exacty0[np.linalg.norm(y0, axis=1)<r0] =  1/a1*np.linalg.norm(y0[np.linalg.norm(y0, axis=1)<r0], axis=1, keepdims=True)**4 # 不同的区域进行不同的解析解赋予
exacty0[np.linalg.norm(y0, axis=1)>=r0] =  1/a2*np.linalg.norm(y0[np.linalg.norm(y0, axis=1)>=r0], axis=1, keepdims=True)**4 + (1/a1-1/a2)*r0**4
y0t = torch.tensor(y0)
predy0 = pred(y0t).data.cpu().numpy() # 预测x=0的原函数
plt.plot(y0[:, 0], exacty0.flatten())
plt.plot(y0[:, 0], predy0.flatten())
plt.xlabel('x')
plt.ylabel('u')
plt.legend(['exact', 'pred'])
plt.title('y=0 u') 

plt.subplot(3, 3, 6) # 做x=0的位移导数预测 dudy
n_test = 100
x0 = np.zeros((n_test, 2))
x0[:, 1] = np.linspace(-np.sqrt(3), np.sqrt(3), 100)
exactdx0 = np.zeros((n_test, 1))
exactdx0[np.linalg.norm(x0, axis=1)<r0] =  4/a1*np.linalg.norm(x0[np.linalg.norm(x0, axis=1)<r0], axis=1, keepdims=True)**3 # 不同的区域进行不同的解析解赋予
exactdx0[np.linalg.norm(x0, axis=1)>=r0] =  4/a2*np.linalg.norm(x0[np.linalg.norm(x0, axis=1)>=r0], axis=1, keepdims=True)**3
x0t = torch.tensor(x0, requires_grad=True) # 将numpy 变成 tensor
predx0 = pred(x0t) # 预测一下
dudxy = grad(predx0, x0t, torch.ones(x0t.size()[0], 1).cuda(), create_graph=True)[0]
dudx = dudxy[:, 0].unsqueeze(1).data.cpu().numpy()
dudy = dudxy[:, 1].unsqueeze(1).data.cpu().numpy()
dudy[x0[:, 1]<0] = -dudy[x0[:, 1]<0] # y小于0的导数添加负号
plt.plot(x0[:, 1], exactdx0.flatten())
plt.plot(x0[:, 1], dudy.flatten())
plt.xlabel('y')
plt.ylabel('u')
plt.legend(['exact', 'pred'])
plt.title('x=0 dudy')

plt.subplot(3, 3, 7) # 做y=0的位移导数预测 dudy
n_test = 100
y0 = np.zeros((n_test, 2))
y0[:, 0] = np.linspace(-1, 1, 100)
exactdy0 = np.zeros((n_test, 1))
exactdy0[np.linalg.norm(y0, axis=1)<r0] =  4/a1*np.linalg.norm(y0[np.linalg.norm(y0, axis=1)<r0], axis=1, keepdims=True)**3 # 不同的区域进行不同的解析解赋予
exactdy0[np.linalg.norm(y0, axis=1)>=r0] =  4/a2*np.linalg.norm(y0[np.linalg.norm(y0, axis=1)>=r0], axis=1, keepdims=True)**3
y0t = torch.tensor(y0, requires_grad=True)
predy0 = pred(y0t) # 预测x=0的原函数
dudxy = grad(predy0, y0t, torch.ones(y0t.size()[0], 1).cuda(), create_graph=True)[0]
dudx = dudxy[:, 0].unsqueeze(1).data.cpu().numpy()
dudy = dudxy[:, 1].unsqueeze(1).data.cpu().numpy()
dudx[y0[:, 0]<0] = -dudx[y0[:, 0]<0] # y小于0的导数添加负号
plt.plot(y0[:, 0], exactdy0.flatten())
plt.plot(y0[:, 0], dudx.flatten())
plt.xlabel('x')
plt.ylabel('u')
plt.legend(['exact', 'pred'])
plt.title('y=0 dudx') 

plt.suptitle("energy")
plt.savefig('./picture/energy_plane.png')
plt.show()
print('the relative error is %f' % error_t.data)   
    
    
    