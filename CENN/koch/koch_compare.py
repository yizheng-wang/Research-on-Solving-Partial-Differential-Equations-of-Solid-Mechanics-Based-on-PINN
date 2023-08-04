# 这个程序用来比较CPINN，能量不分片，以及CENN（本方法）的预测以及误差结果
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
from matplotlib.pyplot import MultipleLocator

import matplotlib as mpl
mpl.rcParams['font.sans-serif']=['SimHei'] # 为了显示中文matplotlib

def settick():
    '''
    对刻度字体进行设置，让上标的符号显示正常
    :return: None
    '''
    ax1 = plt.gca()  # 获取当前图像的坐标轴
 
    # 更改坐标轴字体，避免出现指数为负的情况
    tick_font = mpl.font_manager.FontProperties(family='DejaVu Sans', size=7.0)
    for labelx  in ax1.get_xticklabels():
        labelx.set_fontproperties(tick_font) #设置 x轴刻度字体
    for labely in ax1.get_yticklabels():
        labely.set_fontproperties(tick_font) #设置 y轴刻度字体
    ax1.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))  # x轴刻度设置为整数
    plt.tight_layout()  
    
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子


torch.set_default_tensor_type(torch.DoubleTensor) # 将tensor的类型变成默认的double
torch.set_default_tensor_type(torch.cuda.DoubleTensor) # 将cuda的tensor的类型变成默认的double
train_p = 0
tol_p = 0.00001
a = 1
a1 = 1/15
a2 = 1
r0 = 0.5
penalty = 100
error_total = {}
dd = 102
setup_seed(dd)
nepoch_inter = 125
nepoch_u1 = 20
nepoch_u2 = 20
# nepoch_inter = 2
# nepoch_u1 = 2
# nepoch_u2 = 2
b1 = 60
b2 = 1
b3 = 1
b4 = 30
def interface(Ni):
    '''
     生成交界面的随机点
    '''
    theta = np.random.rand(Ni)*2*np.pi
    r = r0
    x = np.cos(theta) * r0
    y = np.sin(theta) * r0
    xi = np.stack([x, y], 1)
    xi = torch.tensor(xi, requires_grad=True, device='cuda')
    return xi
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
    out = torch.zeros((len(xy), 1))
    out[torch.norm(xy, dim=1)<r0] = model_p1(xy[torch.norm(xy, dim=1)<r0]) # 内部的位移场，不需要RBF距离函数以及特解，只需要一个神经网络即可 
    out[torch.norm(xy, dim=1)>=r0] = model_p2(xy[torch.norm(xy, dim=1)>=r0])# 外部的神经网络
    return out
    
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


model_p1 = particular(2, 20, 1).cuda()
model_p2 = particular(2, 20, 1).cuda()
criterion = torch.nn.MSELoss()
optim_h1 = torch.optim.Adam(params=model_p1.parameters(), lr= 0.001) # 0.001比较好，两个神经网络
optim_h2 = torch.optim.Adam(params=model_p2.parameters(), lr= 0.001) # 0.001比较好，两个神经网络
scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optim_h1, milestones=[1000, 3000, 5000], gamma = 0.1)
scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optim_h2, milestones=[1000, 3000, 5000], gamma = 0.1)
loss_array = []
loss1_array = [] # 内部加上边界
loss2_array = [] # 内部加上边界
error_array = []

nepoch_u1 = int(nepoch_u1)
nepoch_u2 = int(nepoch_u2)
start = time.time()
for epoch_inter in range(nepoch_inter): # 两个神经网络交错优化
    print('the process : %i' % epoch_inter) # 交错优化的步骤数
    for epoch in range(nepoch_u1): # cpinn先优化1
        if epoch ==1000:
            end = time.time()
            consume_time = end-start
            print('time is %f' % consume_time)
        if epoch%100 == 0:
            dom_koch_n = koch_points.get_koch_points(10000) # 获得n_test个koch的随机分布点
            dom_koch_n1 = dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)<r0**2]#定义内部的dom点，即是r<r0的点
            dom_koch_n2 = dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)>=r0**2]#定义外部的dom点，即是r>=r0的点
            dom_koch_t1= torch.tensor(dom_koch_n1,  requires_grad=True, device = 'cuda') # 将numpy数据转化为tensor
            dom_koch_t2= torch.tensor(dom_koch_n2,  requires_grad=True, device = 'cuda') # 将numpy数据转化为tensor
            f1 = -16 * torch.norm(dom_koch_t1, dim = 1, keepdim=True).data**2 # 定义体力 
            f2 = -16 * torch.norm(dom_koch_t2, dim = 1, keepdim=True).data**2 # 定义体力
            Xi = interface(1000)
            Xb = koch.point_bound_rand(1000) # 获得边界点，用来优化网络2的外部
            Xb = torch.tensor(Xb).cuda() # 变成张量
            target_b = 1/a2*torch.norm(Xb, dim=1, keepdim=True)**4+(1/a1-1/a2)*r0**4 # 得到相应的标签   
            
        def closure():  
    
            # 构造可能位移场
            u_pred1 = pred(dom_koch_t1)
            u_pred2 = pred(dom_koch_t2)
            
            du1dxy = grad(u_pred1, dom_koch_t1, torch.ones(dom_koch_t1.size()[0], 1).cuda(), create_graph=True)[0]
            du1dx = du1dxy[:, 0].unsqueeze(1)
            du1dy = du1dxy[:, 1].unsqueeze(1)

            du1dxxy = grad(du1dx, dom_koch_t1, torch.ones(dom_koch_t1.size()[0], 1).cuda(), create_graph=True)[0]
            du1dxx = du1dxxy[:, 0].unsqueeze(1)
            du1dxy = du1dxxy[:, 1].unsqueeze(1)

            du1dyxy = grad(du1dy, dom_koch_t1, torch.ones(dom_koch_t1.size()[0], 1).cuda(), create_graph=True)[0]
            du1dyx = du1dyxy[:, 0].unsqueeze(1)
            du1dyy = du1dyxy[:, 1].unsqueeze(1)
    
            du2dxy = grad(u_pred2, dom_koch_t2, torch.ones(dom_koch_t2.size()[0], 1).cuda(), create_graph=True)[0]
            du2dx = du2dxy[:, 0].unsqueeze(1)
            du2dy = du2dxy[:, 1].unsqueeze(1)

            du2dxy = grad(u_pred2, dom_koch_t2, torch.ones(dom_koch_t2.size()[0], 1).cuda(), create_graph=True)[0]
            du2dx = du2dxy[:, 0].unsqueeze(1)
            du2dy = du2dxy[:, 1].unsqueeze(1)

            du2dxxy = grad(du2dx, dom_koch_t2, torch.ones(dom_koch_t2.size()[0], 1).cuda(), create_graph=True)[0]
            du2dxx = du2dxxy[:, 0].unsqueeze(1)
            du2dxy = du2dxxy[:, 1].unsqueeze(1)

            du2dyxy = grad(du2dy, dom_koch_t2, torch.ones(dom_koch_t2.size()[0], 1).cuda(), create_graph=True)[0]
            du2dyx = du2dyxy[:, 0].unsqueeze(1)
            du2dyy = du2dyxy[:, 1].unsqueeze(1)
    
            J1 =  torch.sum((a1 * (du1dxx + du1dyy)+f1)**2)/len(du1dxx)
            
            J2 =  torch.sum((a2 * (du2dxx + du2dyy)+f2)**2)/len(du2dxx)
    

            # 添加交界面的损失函数
            u_i1 = model_p1(Xi)  # 内部网络的交界面预测
            u_i2 = model_p2(Xi)  # 外部网络的交界面预测
            Ji = criterion(u_i1, u_i2)
            
            # 添加交界面的导数的预测
            du1dxyi = grad(u_i1, Xi, torch.ones(Xi.size()[0], 1).cuda(), create_graph=True)[0]
            du1dxi = du1dxyi[:, 0].unsqueeze(1)
            du1dyi = du1dxyi[:, 1].unsqueeze(1)
   
            du2dxyi = grad(u_i2, Xi, torch.ones(Xi.size()[0], 1).cuda(), create_graph=True)[0]
            du2dxi = du2dxyi[:, 0].unsqueeze(1)
            du2dyi = du2dxyi[:, 1].unsqueeze(1)
            
            di1 = a1 * (Xi[:, 0].unsqueeze(1)*du1dxi + Xi[:, 1].unsqueeze(1)*du1dyi)/torch.norm(Xi, dim=1, keepdim=True)
            di2 = a2 * (Xi[:, 0].unsqueeze(1)*du2dxi + Xi[:, 1].unsqueeze(1)*du2dyi)/torch.norm(Xi, dim=1, keepdim=True)
            Jdi = criterion(di1, di2)
            
            u_b = model_p2(Xb)
            Jb = criterion(u_b, target_b)
            
            loss = b1*J1 + b3*(Ji+Jdi)
            error_t = evaluate()
            optim_h1.zero_grad()
            loss.backward()
            loss1_array.append(( b1*J1 + b3*Ji).data.cpu())
            loss_array.append(loss.data.cpu())
            error_array.append(error_t.data.cpu())
    
            if epoch%10==0:
                print(' epoch : %i, the loss : %f , loss1 : %f, loss2 : %f, inter : %f, bound : %f, error : %f' % (epoch, loss.data, J1.data, J2.data, Ji.data,  Jb.data, error_t.data))
            return loss
        optim_h1.step(closure)
        scheduler1.step()
        # 网络1损失函数下不去，所以我们用交界面来作为本质边界条件训练网络1
        
    for epoch in range(nepoch_u2): #  对第二个网络进行优化
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
            Xi = interface(1000)
            Xb = koch.point_bound_rand(1000) # 获得边界点，用来优化网络2的外部
            Xb = torch.tensor(Xb).cuda() # 变成张量
            target_b = 1/a2*torch.norm(Xb, dim=1, keepdim=True)**4+(1/a1-1/a2)*r0**4 # 得到相应的标签   
            
        def closure():  
    
            # 构造可能位移场
            u_pred1 = pred(dom_koch_t1)
            u_pred2 = pred(dom_koch_t2)
            
            du1dxy = grad(u_pred1, dom_koch_t1, torch.ones(dom_koch_t1.size()[0], 1).cuda(), create_graph=True)[0]
            du1dx = du1dxy[:, 0].unsqueeze(1)
            du1dy = du1dxy[:, 1].unsqueeze(1)

            du1dxxy = grad(du1dx, dom_koch_t1, torch.ones(dom_koch_t1.size()[0], 1).cuda(), create_graph=True)[0]
            du1dxx = du1dxxy[:, 0].unsqueeze(1)
            du1dxy = du1dxxy[:, 1].unsqueeze(1)

            du1dyxy = grad(du1dy, dom_koch_t1, torch.ones(dom_koch_t1.size()[0], 1).cuda(), create_graph=True)[0]
            du1dyx = du1dyxy[:, 0].unsqueeze(1)
            du1dyy = du1dyxy[:, 1].unsqueeze(1)
    
            du2dxy = grad(u_pred2, dom_koch_t2, torch.ones(dom_koch_t2.size()[0], 1).cuda(), create_graph=True)[0]
            du2dx = du2dxy[:, 0].unsqueeze(1)
            du2dy = du2dxy[:, 1].unsqueeze(1)

            du2dxy = grad(u_pred2, dom_koch_t2, torch.ones(dom_koch_t2.size()[0], 1).cuda(), create_graph=True)[0]
            du2dx = du2dxy[:, 0].unsqueeze(1)
            du2dy = du2dxy[:, 1].unsqueeze(1)

            du2dxxy = grad(du2dx, dom_koch_t2, torch.ones(dom_koch_t2.size()[0], 1).cuda(), create_graph=True)[0]
            du2dxx = du2dxxy[:, 0].unsqueeze(1)
            du2dxy = du2dxxy[:, 1].unsqueeze(1)

            du2dyxy = grad(du2dy, dom_koch_t2, torch.ones(dom_koch_t2.size()[0], 1).cuda(), create_graph=True)[0]
            du2dyx = du2dyxy[:, 0].unsqueeze(1)
            du2dyy = du2dyxy[:, 1].unsqueeze(1)
    
            J1 =  torch.sum((a1 * (du1dxx + du1dyy)+f1)**2)/len(du1dxx)
            
            J2 =  torch.sum((a2 * (du2dxx + du2dyy)+f2)**2)/len(du2dxx)
    

            # 添加交界面的损失函数
            u_i1 = model_p1(Xi)  # 内部网络的交界面预测
            u_i2 = model_p2(Xi)  # 外部网络的交界面预测
            Ji = criterion(u_i1, u_i2)
            
            # 添加交界面的导数的预测
            du1dxyi = grad(u_i1, Xi, torch.ones(Xi.size()[0], 1).cuda(), create_graph=True)[0]
            du1dxi = du1dxyi[:, 0].unsqueeze(1)
            du1dyi = du1dxyi[:, 1].unsqueeze(1)
   
            du2dxyi = grad(u_i2, Xi, torch.ones(Xi.size()[0], 1).cuda(), create_graph=True)[0]
            du2dxi = du2dxyi[:, 0].unsqueeze(1)
            du2dyi = du2dxyi[:, 1].unsqueeze(1)
            
            di1 = a1 * (Xi[:, 0].unsqueeze(1)*du1dxi + Xi[:, 1].unsqueeze(1)*du1dyi)/torch.norm(Xi, dim=1, keepdim=True)
            di2 = a2 * (Xi[:, 0].unsqueeze(1)*du2dxi + Xi[:, 1].unsqueeze(1)*du2dyi)/torch.norm(Xi, dim=1, keepdim=True)
            Jdi = criterion(di1, di2)
            
            u_b = model_p2(Xb)
            Jb = criterion(u_b, target_b)
            
            loss = b2*J2 + b3*(Ji+Jdi) + b4*Jb 
            error_t = evaluate()
            optim_h2.zero_grad()
            loss.backward()
            loss2_array.append(( b2*J2 + b3*Ji + b4*Jb).data.cpu())
            loss_array.append(loss.data.cpu())
            error_array.append(error_t.data.cpu())
    
            if epoch%10==0:
                print(' epoch : %i, the loss : %f , loss1 : %f, loss2 : %f, inter : %f, bound : %f, error : %f' % (epoch, loss.data, J1.data, J2.data, Ji.data,  Jb.data, error_t.data))
            return loss
        optim_h2.step(closure)
        scheduler2.step()

n_test = 300
dom_koch_n = koch_points.get_koch_points_lin(n_test) # 获得n_test个koch的随机分布点
xy_test= torch.tensor(dom_koch_n,  requires_grad=True, device = 'cuda') # 将numpy数据转化为tensor


u_pred = pred(xy_test)
u_pred_cpinn = u_pred.data.cpu()

u_exact = np.zeros((len(dom_koch_n) ,1))
u_exact[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)<r0**2] = 1/a1*np.linalg.norm(dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)<r0**2], axis=1, keepdims=True)**4 # 现在网格里面赋值
u_exact[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)>=r0**2] = 1/a2*np.linalg.norm(dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)>=r0**2], axis=1, keepdims=True)**4 + r0**4*(1/a1-1/a2) # 现在网格外面赋值

u_exact = torch.from_numpy(u_exact) # 将精确解从array变成tensor

error_cpinn = torch.abs(u_pred_cpinn - u_exact) # get the error in every points
error_t_cpinn = torch.norm(error_cpinn)/torch.norm(u_exact) # get the total relative L2 error


plt.subplot(3, 3, 1)
Xb =  koch.point_bound(5)
Xf = koch_points.get_koch_points(10000)
 # 将边界和内部点画图

write_arr2DVTK('./output_ntk/pred_cpinn%i' % dd, dom_koch_n, u_pred_cpinn, 'pred_cpinn')

write_arr2DVTK('./output_ntk/error_cpinn%i' % dd, dom_koch_n, error_cpinn, 'error_cpinn')


loss1_array_cpinn = np.array(loss1_array)
loss2_array_cpinn = np.array(loss2_array)

error_array_cpinn = np.array(error_array)
 

n_test = 100
x0 = np.zeros((n_test, 2))
x0[:, 1] = np.linspace(0, np.sqrt(3), 100) # 利用对称性，比较y>=0即可
exactx0 = np.zeros((n_test, 1))
exactx0[np.linalg.norm(x0, axis=1)<r0] =  1/a1*np.linalg.norm(x0[np.linalg.norm(x0, axis=1)<r0], axis=1, keepdims=True)**4 # 不同的区域进行不同的解析解赋予
exactx0[np.linalg.norm(x0, axis=1)>=r0] =  1/a2*np.linalg.norm(x0[np.linalg.norm(x0, axis=1)>=r0], axis=1, keepdims=True)**4 + (1/a1-1/a2)*r0**4
x0t = torch.tensor(x0)
predx0_cpinn = pred(x0t).data.cpu().numpy() # 预测x=0的原函数

n_test = 100
y0 = np.zeros((n_test, 2))
y0[:, 0] = np.linspace(0, 1, 100) # 利用对称性，比较x>=0即可
exacty0 = np.zeros((n_test, 1))
exacty0[np.linalg.norm(y0, axis=1)<r0] =  1/a1*np.linalg.norm(y0[np.linalg.norm(y0, axis=1)<r0], axis=1, keepdims=True)**4 # 不同的区域进行不同的解析解赋予
exacty0[np.linalg.norm(y0, axis=1)>=r0] =  1/a2*np.linalg.norm(y0[np.linalg.norm(y0, axis=1)>=r0], axis=1, keepdims=True)**4 + (1/a1-1/a2)*r0**4
y0t = torch.tensor(y0)
predy0_cpinn = pred(y0t).data.cpu().numpy() # 预测x=0的原函数

n_test = 100
x0 = np.zeros((n_test, 2))
x0[:, 1] = np.linspace(0, np.sqrt(3), 100) # 利用对称性，比较y>=0即可
exactdx0 = np.zeros((n_test, 1))
exactdx0[np.linalg.norm(x0, axis=1)<r0] =  4/a1*np.linalg.norm(x0[np.linalg.norm(x0, axis=1)<r0], axis=1, keepdims=True)**3 # 不同的区域进行不同的解析解赋予
exactdx0[np.linalg.norm(x0, axis=1)>=r0] =  4/a2*np.linalg.norm(x0[np.linalg.norm(x0, axis=1)>=r0], axis=1, keepdims=True)**3
x0t = torch.tensor(x0, requires_grad=True) # 将numpy 变成 tensor
predx0 = pred(x0t) # 预测一下
dudxy = grad(predx0, x0t, torch.ones(x0t.size()[0], 1).cuda(), create_graph=True)[0]
dudx = dudxy[:, 0].unsqueeze(1).data.cpu().numpy()
dudy_cpinn = dudxy[:, 1].unsqueeze(1).data.cpu().numpy()
# dudy_cpinn[x0[:, 1]<0] = -dudy_cpinn[x0[:, 1]<0] # y小于0的导数添加负号

n_test = 100
y0 = np.zeros((n_test, 2))
y0[:, 0] = np.linspace(0, 1, 100) # 利用对称性，比较x>=0即可
exactdy0 = np.zeros((n_test, 1))
exactdy0[np.linalg.norm(y0, axis=1)<r0] =  4/a1*np.linalg.norm(y0[np.linalg.norm(y0, axis=1)<r0], axis=1, keepdims=True)**3 # 不同的区域进行不同的解析解赋予
exactdy0[np.linalg.norm(y0, axis=1)>=r0] =  4/a2*np.linalg.norm(y0[np.linalg.norm(y0, axis=1)>=r0], axis=1, keepdims=True)**3
y0t = torch.tensor(y0, requires_grad=True)
predy0 = pred(y0t) # 预测x=0的原函数
dudxy = grad(predy0, y0t, torch.ones(y0t.size()[0], 1).cuda(), create_graph=True)[0]
dudx_cpinn = dudxy[:, 0].unsqueeze(1).data.cpu().numpy()
dudy = dudxy[:, 1].unsqueeze(1).data.cpu().numpy()
# dudx_cpinn[y0[:, 0]<0] = -dudx_cpinn[y0[:, 0]<0] # y小于0的导数添加负号,利用对称性，比较
 

# =============================================================================
# # 能量法，不分片
# =============================================================================

train_p = 0
tol_p = 0.00001
a = 1
a1 = 1/15
a2 = 1
r0 = 0.5
nepoch_u0 = 5000
# nepoch_u0 = 10
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
loss1_array = []
loss2_array = []
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
        loss1_array.append(J1.data.cpu())
        loss2_array.append(J2.data.cpu())
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
    
    
    
n_test = 300
dom_koch_n = koch_points.get_koch_points_lin(n_test) # 获得n_test个koch的随机分布点
xy_test= torch.tensor(dom_koch_n,  requires_grad=True, device = 'cuda') # 将numpy数据转化为tensor


u_pred = pred(xy_test)
u_pred_energy = u_pred.data.cpu()

u_exact = np.zeros((len(dom_koch_n) ,1))
u_exact[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)<r0**2] = 1/a1*np.linalg.norm(dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)<r0**2], axis=1, keepdims=True)**4 # 现在网格里面赋值
u_exact[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)>=r0**2] = 1/a2*np.linalg.norm(dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)>=r0**2], axis=1, keepdims=True)**4 + r0**4*(1/a1-1/a2) # 现在网格外面赋值

u_exact = torch.from_numpy(u_exact) # 将精确解从array变成tensor

error_energy = torch.abs(u_pred_energy - u_exact) # get the error in every points
error_t_energy = torch.norm(error_energy)/torch.norm(u_exact) # get the total relative L2 error

Xb =  koch.point_bound(5)
Xf = koch_points.get_koch_points(10000)


write_arr2DVTK('./output_ntk/pred_energy%i' % dd, dom_koch_n, u_pred_energy, 'pred_energy')

write_arr2DVTK('./output_ntk/error_energy%i' % dd, dom_koch_n, error_energy, 'error_energy')

loss1_array_energy = np.array(loss1_array)
loss2_array_energy = np.array(loss2_array)

error_array_energy = np.array(error_array)
 


n_test = 100
x0 = np.zeros((n_test, 2))
x0[:, 1] = np.linspace(0, np.sqrt(3), 100)
exactx0 = np.zeros((n_test, 1))
exactx0[np.linalg.norm(x0, axis=1)<r0] =  1/a1*np.linalg.norm(x0[np.linalg.norm(x0, axis=1)<r0], axis=1, keepdims=True)**4 # 不同的区域进行不同的解析解赋予
exactx0[np.linalg.norm(x0, axis=1)>=r0] =  1/a2*np.linalg.norm(x0[np.linalg.norm(x0, axis=1)>=r0], axis=1, keepdims=True)**4 + (1/a1-1/a2)*r0**4
x0t = torch.tensor(x0)
predx0_energy = pred(x0t).data.cpu().numpy() # 预测x=0的原函数

n_test = 100
y0 = np.zeros((n_test, 2))
y0[:, 0] = np.linspace(0, 1, 100)
exacty0 = np.zeros((n_test, 1))
exacty0[np.linalg.norm(y0, axis=1)<r0] =  1/a1*np.linalg.norm(y0[np.linalg.norm(y0, axis=1)<r0], axis=1, keepdims=True)**4 # 不同的区域进行不同的解析解赋予
exacty0[np.linalg.norm(y0, axis=1)>=r0] =  1/a2*np.linalg.norm(y0[np.linalg.norm(y0, axis=1)>=r0], axis=1, keepdims=True)**4 + (1/a1-1/a2)*r0**4
y0t = torch.tensor(y0)
predy0_energy = pred(y0t).data.cpu().numpy() # 预测x=0的原函数

n_test = 100
x0 = np.zeros((n_test, 2))
x0[:, 1] = np.linspace(0, np.sqrt(3), 100)
exactdx0 = np.zeros((n_test, 1))
exactdx0[np.linalg.norm(x0, axis=1)<r0] =  4/a1*np.linalg.norm(x0[np.linalg.norm(x0, axis=1)<r0], axis=1, keepdims=True)**3 # 不同的区域进行不同的解析解赋予
exactdx0[np.linalg.norm(x0, axis=1)>=r0] =  4/a2*np.linalg.norm(x0[np.linalg.norm(x0, axis=1)>=r0], axis=1, keepdims=True)**3
x0t = torch.tensor(x0, requires_grad=True) # 将numpy 变成 tensor
predx0 = pred(x0t) # 预测一下
dudxy = grad(predx0, x0t, torch.ones(x0t.size()[0], 1).cuda(), create_graph=True)[0]
dudx = dudxy[:, 0].unsqueeze(1).data.cpu().numpy()
dudy_energy = dudxy[:, 1].unsqueeze(1).data.cpu().numpy()
# dudy_energy[x0[:, 1]<0] = -dudy_energy[x0[:, 1]<0] # y小于0的导数添加负号

n_test = 100
y0 = np.zeros((n_test, 2))
y0[:, 0] = np.linspace(0, 1, 100)
exactdy0 = np.zeros((n_test, 1))
exactdy0[np.linalg.norm(y0, axis=1)<r0] =  4/a1*np.linalg.norm(y0[np.linalg.norm(y0, axis=1)<r0], axis=1, keepdims=True)**3 # 不同的区域进行不同的解析解赋予
exactdy0[np.linalg.norm(y0, axis=1)>=r0] =  4/a2*np.linalg.norm(y0[np.linalg.norm(y0, axis=1)>=r0], axis=1, keepdims=True)**3
y0t = torch.tensor(y0, requires_grad=True)
predy0 = pred(y0t) # 预测x=0的原函数
dudxy = grad(predy0, y0t, torch.ones(y0t.size()[0], 1).cuda(), create_graph=True)[0]
dudx_energy = dudxy[:, 0].unsqueeze(1).data.cpu().numpy()
dudy = dudxy[:, 1].unsqueeze(1).data.cpu().numpy()
# dudx_energy[y0[:, 0]<0] = -dudx_energy[y0[:, 0]<0] # y小于0的导数添加负号

# =============================================================================
# # CENN
# =============================================================================
# %%
train_p = 0
tol_p = 0.00001
a = 1
a1 = 1/15
a2 = 1
r0 = 0.5
nepoch_inter = 1
nepoch_u0 = 2500
nepoch_u1 = 2500
# nepoch_inter = 2
# nepoch_u0 = 2
# nepoch_u1 = 2
# penalty = 100
def interface(Ni):
    '''
     生成交界面的随机点
    '''
    theta = np.random.rand(Ni)*2*np.pi
    x = np.cos(theta) * r0
    y = np.sin(theta) * r0
    xi = np.stack([x, y], 1)
    xi = torch.tensor(xi, requires_grad=True, device='cuda')
    return xi
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
    out = torch.zeros((len(xy), 1))
    out[torch.norm(xy, dim=1)<r0] = model_h1(xy[torch.norm(xy, dim=1)<r0]) # 内部的位移场，不需要RBF距离函数以及特解，只需要一个神经网络即可 
    out[torch.norm(xy, dim=1)>=r0] = model_p(xy[torch.norm(xy, dim=1)>=r0]) + RBF(xy[torch.norm(xy, dim=1)>=r0]) * model_h2(xy[torch.norm(xy, dim=1)>=r0]) # 外部的神经网络
    return out
    
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



model_h1 = homo(2, 20, 1).cuda()
model_h2 = homo(2, 20, 1).cuda()
criterion = torch.nn.MSELoss()
optim_h = torch.optim.Adam(params=chain(model_h1.parameters(), model_h2.parameters()), lr= 0.001) # 0.001比较好，两个神经网络
optim_h1 = torch.optim.Adam(params=model_h1.parameters(), lr= 0.001) # 0.001比较好，神经网络1
scheduler = torch.optim.lr_scheduler.MultiStepLR(optim_h, milestones=[1000, 3000, 5000], gamma = 0.1)
scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optim_h1, milestones=[1000, 3000, 5000], gamma = 0.1)
loss_array = []
loss1_array = []
loss2_array = []
error_array = []
eigvalues = []
nepoch_u0 = int(nepoch_u0)
time_CENN_start = time.time()   
for epoch_inter in range(nepoch_inter): # 两个神经网络交错优化
    print('the process : %i' % epoch_inter) # 交错优化的步骤数
    for epoch in range(nepoch_u0): # 两个一起优化
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
            Xi = interface(1000)
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
            # 添加交界面的损失函数
            u_i1 = model_h1(Xi)  # 内部网络的交界面预测
            u_i2 = model_p(Xi)  + RBF(Xi)*model_h2(Xi)  # 外部网络的交界面预测
            Ji = criterion(u_i1, u_i2)
            
            loss = J + penalty * Ji
            error_t = evaluate()
            optim_h.zero_grad()
            loss.backward()
            loss1_array.append(J1.data.cpu())
            loss2_array.append(J2.data.cpu())
            loss_array.append(loss.data.cpu())
            error_array.append(error_t.data.cpu())
    
            if epoch%10==0:
                print(' epoch : %i, the loss : %f , loss1 : %f, loss2 : %f, inter : %f, error : %f' % (epoch, loss.data, J1.data, J2.data, Ji.data, error_t.data))
            return loss
        optim_h.step(closure)
        scheduler.step()
        # 网络1损失函数下不去，所以我们用交界面来作为本质边界条件训练网络1
    for epoch in range(nepoch_u1): #固定网络2，优化网络2
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
            Xi = interface(1000)
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
            
            J = J1 
            # 添加交界面的损失函数
            u_i1 = model_h1(Xi)  # 内部网络的交界面预测
            u_i2 = model_p(Xi)  + RBF(Xi)*model_h2(Xi)  # 外部网络的交界面预测
            Ji = criterion(u_i1, u_i2)
            
            loss = J + penalty * Ji
            error_t = evaluate()
            optim_h1.zero_grad()
            loss.backward()
            loss1_array.append(J1.data.cpu())
            loss2_array.append(J2.data.cpu())
            loss_array.append(loss.data.cpu())
            error_array.append(error_t.data.cpu())
    
            if epoch%10==0:
                print(' epoch : %i, the loss : %f , loss1 : %f, inter : %f, error : %f' % (epoch, loss.data, J1.data, Ji.data, error_t.data))
            return loss
        optim_h1.step(closure)
        scheduler1.step() 
time_CENN_end = time.time()   
time_CENN = time_CENN_end - time_CENN_start  
n_test = 300
dom_koch_n = koch_points.get_koch_points_lin(n_test) # 获得n_test个koch的随机分布点
xy_test= torch.tensor(dom_koch_n,  requires_grad=True, device = 'cuda') # 将numpy数据转化为tensor


u_pred = pred(xy_test)
u_pred_cenn = u_pred.data.cpu()

u_exact = np.zeros((len(dom_koch_n) ,1))
u_exact[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)<r0**2] = 1/a1*np.linalg.norm(dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)<r0**2], axis=1, keepdims=True)**4 # 现在网格里面赋值
u_exact[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)>=r0**2] = 1/a2*np.linalg.norm(dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)>=r0**2], axis=1, keepdims=True)**4 + r0**4*(1/a1-1/a2) # 现在网格外面赋值

u_exact = torch.from_numpy(u_exact) # 将精确解从array变成tensor

error_cenn = torch.abs(u_pred_cenn - u_exact) # get the error in every points
error_t_cenn = torch.norm(error_cenn)/torch.norm(u_exact) # get the total relative L2 error

Xb =  koch.point_bound(5)
Xf = koch_points.get_koch_points(10000)

write_arr2DVTK('./output_ntk/pred_energy_cenn%i' % dd, dom_koch_n, u_pred_cenn, 'pred_energy_cenn')

write_arr2DVTK('./output_ntk/error_energy_cenn%i' % dd, dom_koch_n, error_cenn, 'error_energy_cenn')

loss1_array_cenn = np.array(loss1_array)
loss2_array_cenn = np.array(loss2_array)

error_array_cenn = np.array(error_array)

n_test = 100
x0 = np.zeros((n_test, 2))
x0[:, 1] = np.linspace(0, np.sqrt(3), 100)
exactx0 = np.zeros((n_test, 1))
exactx0[np.linalg.norm(x0, axis=1)<r0] =  1/a1*np.linalg.norm(x0[np.linalg.norm(x0, axis=1)<r0], axis=1, keepdims=True)**4 # 不同的区域进行不同的解析解赋予
exactx0[np.linalg.norm(x0, axis=1)>=r0] =  1/a2*np.linalg.norm(x0[np.linalg.norm(x0, axis=1)>=r0], axis=1, keepdims=True)**4 + (1/a1-1/a2)*r0**4
x0t = torch.tensor(x0)
predx0_cenn = pred(x0t).data.cpu().numpy() # 预测x=0的原函数

n_test = 100
y0 = np.zeros((n_test, 2))
y0[:, 0] = np.linspace(0, 1, 100)
exacty0 = np.zeros((n_test, 1))
exacty0[np.linalg.norm(y0, axis=1)<r0] =  1/a1*np.linalg.norm(y0[np.linalg.norm(y0, axis=1)<r0], axis=1, keepdims=True)**4 # 不同的区域进行不同的解析解赋予
exacty0[np.linalg.norm(y0, axis=1)>=r0] =  1/a2*np.linalg.norm(y0[np.linalg.norm(y0, axis=1)>=r0], axis=1, keepdims=True)**4 + (1/a1-1/a2)*r0**4
y0t = torch.tensor(y0)
predy0_cenn = pred(y0t).data.cpu().numpy() # 预测x=0的原函数

x0 = np.zeros((n_test, 2))
x0[:, 1] = np.linspace(0, np.sqrt(3), 100)
exactdx0 = np.zeros((n_test, 1))
exactdx0[np.linalg.norm(x0, axis=1)<r0] =  4/a1*np.linalg.norm(x0[np.linalg.norm(x0, axis=1)<r0], axis=1, keepdims=True)**3 # 不同的区域进行不同的解析解赋予
exactdx0[np.linalg.norm(x0, axis=1)>=r0] =  4/a2*np.linalg.norm(x0[np.linalg.norm(x0, axis=1)>=r0], axis=1, keepdims=True)**3
x0t = torch.tensor(x0, requires_grad=True) # 将numpy 变成 tensor
predx0 = pred(x0t) # 预测一下
dudxy = grad(predx0, x0t, torch.ones(x0t.size()[0], 1).cuda(), create_graph=True)[0]
dudx = dudxy[:, 0].unsqueeze(1).data.cpu().numpy()
dudy_cenn = dudxy[:, 1].unsqueeze(1).data.cpu().numpy()
# dudy_cenn[x0[:, 1]<0] = -dudy_cenn[x0[:, 1]<0] # y小于0的导数添加负号

n_test = 100
y0 = np.zeros((n_test, 2))
y0[:, 0] = np.linspace(0, 1, 100)
exactdy0 = np.zeros((n_test, 1))
exactdy0[np.linalg.norm(y0, axis=1)<r0] =  4/a1*np.linalg.norm(y0[np.linalg.norm(y0, axis=1)<r0], axis=1, keepdims=True)**3 # 不同的区域进行不同的解析解赋予
exactdy0[np.linalg.norm(y0, axis=1)>=r0] =  4/a2*np.linalg.norm(y0[np.linalg.norm(y0, axis=1)>=r0], axis=1, keepdims=True)**3
y0t = torch.tensor(y0, requires_grad=True)
predy0 = pred(y0t) # 预测x=0的原函数
dudxy = grad(predy0, y0t, torch.ones(y0t.size()[0], 1).cuda(), create_graph=True)[0]
dudx_cenn = dudxy[:, 0].unsqueeze(1).data.cpu().numpy()
dudy = dudxy[:, 1].unsqueeze(1).data.cpu().numpy()
# dudx_cenn[y0[:, 0]<0] = -dudx_cenn[y0[:, 0]<0] # y小于0的导数添加负号

# =============================================================================
# CPINN_RBF
# =============================================================================
model_h1 = homo(2, 20, 1).cuda()
model_h2 = homo(2, 20, 1).cuda()
criterion = torch.nn.MSELoss()
optim_h = torch.optim.Adam(params=chain(model_h1.parameters(), model_h2.parameters()), lr= 0.001) # 0.001比较好，两个神经网络
optim_h1 = torch.optim.Adam(params=model_h1.parameters(), lr= 0.001) # 0.001比较好，神经网络1
scheduler = torch.optim.lr_scheduler.MultiStepLR(optim_h, milestones=[1000, 3000, 5000], gamma = 0.1)
scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optim_h1, milestones=[1000, 3000, 5000], gamma = 0.1)
loss_array = []
loss1_array = []
loss2_array = []
error_array = []
eigvalues = []
nepoch_u0 = int(nepoch_u0)
time_CPINN_RBF_start = time.time()   
for epoch_inter in range(nepoch_inter): # 两个神经网络交错优化
    print('the process : %i' % epoch_inter) # 交错优化的步骤数
    for epoch in range(nepoch_u0): # 两个一起优化
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
            Xi = interface(1000)
        def closure():  
    
            # 构造可能位移场
            u_pred1 = pred(dom_koch_t1)
            u_pred2 = pred(dom_koch_t2)
            
            du1dxy = grad(u_pred1, dom_koch_t1, torch.ones(dom_koch_t1.size()[0], 1).cuda(), create_graph=True)[0]
            du1dx = du1dxy[:, 0].unsqueeze(1)
            du1dy = du1dxy[:, 1].unsqueeze(1)

            du1dxxy = grad(du1dx, dom_koch_t1, torch.ones(dom_koch_t1.size()[0], 1).cuda(), create_graph=True)[0]
            du1dxx = du1dxxy[:, 0].unsqueeze(1)
            du1dxy = du1dxxy[:, 1].unsqueeze(1)

            du1dyxy = grad(du1dy, dom_koch_t1, torch.ones(dom_koch_t1.size()[0], 1).cuda(), create_graph=True)[0]
            du1dyx = du1dyxy[:, 0].unsqueeze(1)
            du1dyy = du1dyxy[:, 1].unsqueeze(1)
    
            du2dxy = grad(u_pred2, dom_koch_t2, torch.ones(dom_koch_t2.size()[0], 1).cuda(), create_graph=True)[0]
            du2dx = du2dxy[:, 0].unsqueeze(1)
            du2dy = du2dxy[:, 1].unsqueeze(1)

            du2dxy = grad(u_pred2, dom_koch_t2, torch.ones(dom_koch_t2.size()[0], 1).cuda(), create_graph=True)[0]
            du2dx = du2dxy[:, 0].unsqueeze(1)
            du2dy = du2dxy[:, 1].unsqueeze(1)

            du2dxxy = grad(du2dx, dom_koch_t2, torch.ones(dom_koch_t2.size()[0], 1).cuda(), create_graph=True)[0]
            du2dxx = du2dxxy[:, 0].unsqueeze(1)
            du2dxy = du2dxxy[:, 1].unsqueeze(1)

            du2dyxy = grad(du2dy, dom_koch_t2, torch.ones(dom_koch_t2.size()[0], 1).cuda(), create_graph=True)[0]
            du2dyx = du2dyxy[:, 0].unsqueeze(1)
            du2dyy = du2dyxy[:, 1].unsqueeze(1)
    
            J1 =  torch.sum((a1 * (du1dxx + du1dyy)+f1)**2)/len(du1dxx)
            
            J2 =  torch.sum((a2 * (du2dxx + du2dyy)+f2)**2)/len(du2dxx)
            J = J2
            # 添加交界面的损失函数
            u_i1 = model_h1(Xi)  # 内部网络的交界面预测
            u_i2 = model_p(Xi)  + RBF(Xi)*model_h2(Xi)  # 外部网络的交界面预测
            Ji = criterion(u_i1, u_i2)

            # 添加交界面的导数的预测
            du1dxyi = grad(u_i1, Xi, torch.ones(Xi.size()[0], 1).cuda(), create_graph=True)[0]
            du1dxi = du1dxyi[:, 0].unsqueeze(1)
            du1dyi = du1dxyi[:, 1].unsqueeze(1)
   
            du2dxyi = grad(u_i2, Xi, torch.ones(Xi.size()[0], 1).cuda(), create_graph=True)[0]
            du2dxi = du2dxyi[:, 0].unsqueeze(1)
            du2dyi = du2dxyi[:, 1].unsqueeze(1)
            
            di1 = a1 * (Xi[:, 0].unsqueeze(1)*du1dxi + Xi[:, 1].unsqueeze(1)*du1dyi)/torch.norm(Xi, dim=1, keepdim=True)
            di2 = a2 * (Xi[:, 0].unsqueeze(1)*du2dxi + Xi[:, 1].unsqueeze(1)*du2dyi)/torch.norm(Xi, dim=1, keepdim=True)
            Jdi = criterion(di1, di2)
                         
            loss = J + b3*(Ji+Jdi)
            error_t = evaluate()
            optim_h.zero_grad()
            loss.backward()
            loss1_array.append(J1.data.cpu())
            loss2_array.append(J2.data.cpu())
            loss_array.append(loss.data.cpu())
            error_array.append(error_t.data.cpu())
    
            if epoch%10==0:
                print(' epoch : %i, the loss : %f , loss1 : %f, loss2 : %f, inter : %f, error : %f' % (epoch, loss.data, J1.data, J2.data, Ji.data, error_t.data))
            return loss
        optim_h.step(closure)
        scheduler.step()
        # 网络1损失函数下不去，所以我们用交界面来作为本质边界条件训练网络1
    for epoch in range(nepoch_u1): #固定网络2，优化网络2
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
            Xi = interface(1000)
        def closure():  
    
    
            # 构造可能位移场
            u_pred1 = pred(dom_koch_t1)
            u_pred2 = pred(dom_koch_t2)
            
            du1dxy = grad(u_pred1, dom_koch_t1, torch.ones(dom_koch_t1.size()[0], 1).cuda(), create_graph=True)[0]
            du1dx = du1dxy[:, 0].unsqueeze(1)
            du1dy = du1dxy[:, 1].unsqueeze(1)

            du1dxxy = grad(du1dx, dom_koch_t1, torch.ones(dom_koch_t1.size()[0], 1).cuda(), create_graph=True)[0]
            du1dxx = du1dxxy[:, 0].unsqueeze(1)
            du1dxy = du1dxxy[:, 1].unsqueeze(1)

            du1dyxy = grad(du1dy, dom_koch_t1, torch.ones(dom_koch_t1.size()[0], 1).cuda(), create_graph=True)[0]
            du1dyx = du1dyxy[:, 0].unsqueeze(1)
            du1dyy = du1dyxy[:, 1].unsqueeze(1)
    
            du2dxy = grad(u_pred2, dom_koch_t2, torch.ones(dom_koch_t2.size()[0], 1).cuda(), create_graph=True)[0]
            du2dx = du2dxy[:, 0].unsqueeze(1)
            du2dy = du2dxy[:, 1].unsqueeze(1)

            du2dxy = grad(u_pred2, dom_koch_t2, torch.ones(dom_koch_t2.size()[0], 1).cuda(), create_graph=True)[0]
            du2dx = du2dxy[:, 0].unsqueeze(1)
            du2dy = du2dxy[:, 1].unsqueeze(1)

            du2dxxy = grad(du2dx, dom_koch_t2, torch.ones(dom_koch_t2.size()[0], 1).cuda(), create_graph=True)[0]
            du2dxx = du2dxxy[:, 0].unsqueeze(1)
            du2dxy = du2dxxy[:, 1].unsqueeze(1)

            du2dyxy = grad(du2dy, dom_koch_t2, torch.ones(dom_koch_t2.size()[0], 1).cuda(), create_graph=True)[0]
            du2dyx = du2dyxy[:, 0].unsqueeze(1)
            du2dyy = du2dyxy[:, 1].unsqueeze(1)
    
            J1 =  torch.sum((a1 * (du1dxx + du1dyy)+f1)**2)/len(du1dxx)
            
            J2 =  torch.sum((a2 * (du2dxx + du2dyy)+f2)**2)/len(du2dxx)
            J = J1
            # 添加交界面的损失函数
            u_i1 = model_h1(Xi)  # 内部网络的交界面预测
            u_i2 = model_p(Xi)  + RBF(Xi)*model_h2(Xi)  # 外部网络的交界面预测
            Ji = criterion(u_i1, u_i2)

            # 添加交界面的导数的预测
            du1dxyi = grad(u_i1, Xi, torch.ones(Xi.size()[0], 1).cuda(), create_graph=True)[0]
            du1dxi = du1dxyi[:, 0].unsqueeze(1)
            du1dyi = du1dxyi[:, 1].unsqueeze(1)
   
            du2dxyi = grad(u_i2, Xi, torch.ones(Xi.size()[0], 1).cuda(), create_graph=True)[0]
            du2dxi = du2dxyi[:, 0].unsqueeze(1)
            du2dyi = du2dxyi[:, 1].unsqueeze(1)
            
            di1 = a1 * (Xi[:, 0].unsqueeze(1)*du1dxi + Xi[:, 1].unsqueeze(1)*du1dyi)/torch.norm(Xi, dim=1, keepdim=True)
            di2 = a2 * (Xi[:, 0].unsqueeze(1)*du2dxi + Xi[:, 1].unsqueeze(1)*du2dyi)/torch.norm(Xi, dim=1, keepdim=True)
            Jdi = criterion(di1, di2)
                         
            loss = J + b3*(Ji+Jdi)
            error_t = evaluate()
            optim_h.zero_grad()
            loss.backward()
            loss1_array.append(J1.data.cpu())
            loss2_array.append(J2.data.cpu())
            loss_array.append(loss.data.cpu())
            error_array.append(error_t.data.cpu())
    
            if epoch%10==0:
                print(' epoch : %i, the loss : %f , loss1 : %f, loss2 : %f, inter : %f, error : %f' % (epoch, loss.data, J1.data, J2.data, Ji.data, error_t.data))
            return loss
        optim_h1.step(closure)
        scheduler1.step() 
time_CPINN_RBF_end = time.time()    
time_CPINN_RBF = time_CPINN_RBF_end - time_CPINN_RBF_start    
n_test = 300
dom_koch_n = koch_points.get_koch_points_lin(n_test) # 获得n_test个koch的随机分布点
xy_test= torch.tensor(dom_koch_n,  requires_grad=True, device = 'cuda') # 将numpy数据转化为tensor


u_pred = pred(xy_test)
u_pred_rao = u_pred.data.cpu()

u_exact = np.zeros((len(dom_koch_n) ,1))
u_exact[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)<r0**2] = 1/a1*np.linalg.norm(dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)<r0**2], axis=1, keepdims=True)**4 # 现在网格里面赋值
u_exact[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)>=r0**2] = 1/a2*np.linalg.norm(dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)>=r0**2], axis=1, keepdims=True)**4 + r0**4*(1/a1-1/a2) # 现在网格外面赋值

u_exact = torch.from_numpy(u_exact) # 将精确解从array变成tensor

error_rao = torch.abs(u_pred_rao - u_exact) # get the error in every points
error_t_rao = torch.norm(error_rao)/torch.norm(u_exact) # get the total relative L2 error

Xb =  koch.point_bound(5)
Xf = koch_points.get_koch_points(10000)

write_arr2DVTK('./output_ntk/pred_energy_rao%i' % dd, dom_koch_n, u_pred_rao, 'pred_energy_rao')

write_arr2DVTK('./output_ntk/error_energy_rao%i' % dd, dom_koch_n, error_rao, 'error_energy_rao')

loss1_array_rao = np.array(loss1_array)
loss2_array_rao = np.array(loss2_array)

error_array_rao = np.array(error_array)

n_test = 100
x0 = np.zeros((n_test, 2))
x0[:, 1] = np.linspace(0, np.sqrt(3), 100)
exactx0 = np.zeros((n_test, 1))
exactx0[np.linalg.norm(x0, axis=1)<r0] =  1/a1*np.linalg.norm(x0[np.linalg.norm(x0, axis=1)<r0], axis=1, keepdims=True)**4 # 不同的区域进行不同的解析解赋予
exactx0[np.linalg.norm(x0, axis=1)>=r0] =  1/a2*np.linalg.norm(x0[np.linalg.norm(x0, axis=1)>=r0], axis=1, keepdims=True)**4 + (1/a1-1/a2)*r0**4
x0t = torch.tensor(x0)
predx0_rao = pred(x0t).data.cpu().numpy() # 预测x=0的原函数

n_test = 100
y0 = np.zeros((n_test, 2))
y0[:, 0] = np.linspace(0, 1, 100)
exacty0 = np.zeros((n_test, 1))
exacty0[np.linalg.norm(y0, axis=1)<r0] =  1/a1*np.linalg.norm(y0[np.linalg.norm(y0, axis=1)<r0], axis=1, keepdims=True)**4 # 不同的区域进行不同的解析解赋予
exacty0[np.linalg.norm(y0, axis=1)>=r0] =  1/a2*np.linalg.norm(y0[np.linalg.norm(y0, axis=1)>=r0], axis=1, keepdims=True)**4 + (1/a1-1/a2)*r0**4
y0t = torch.tensor(y0)
predy0_rao = pred(y0t).data.cpu().numpy() # 预测x=0的原函数

x0 = np.zeros((n_test, 2))
x0[:, 1] = np.linspace(0, np.sqrt(3), 100)
exactdx0 = np.zeros((n_test, 1))
exactdx0[np.linalg.norm(x0, axis=1)<r0] =  4/a1*np.linalg.norm(x0[np.linalg.norm(x0, axis=1)<r0], axis=1, keepdims=True)**3 # 不同的区域进行不同的解析解赋予
exactdx0[np.linalg.norm(x0, axis=1)>=r0] =  4/a2*np.linalg.norm(x0[np.linalg.norm(x0, axis=1)>=r0], axis=1, keepdims=True)**3
x0t = torch.tensor(x0, requires_grad=True) # 将numpy 变成 tensor
predx0 = pred(x0t) # 预测一下
dudxy = grad(predx0, x0t, torch.ones(x0t.size()[0], 1).cuda(), create_graph=True)[0]
dudx = dudxy[:, 0].unsqueeze(1).data.cpu().numpy()
dudy_rao = dudxy[:, 1].unsqueeze(1).data.cpu().numpy()
# dudy_cenn[x0[:, 1]<0] = -dudy_cenn[x0[:, 1]<0] # y小于0的导数添加负号

n_test = 100
y0 = np.zeros((n_test, 2))
y0[:, 0] = np.linspace(0, 1, 100)
exactdy0 = np.zeros((n_test, 1))
exactdy0[np.linalg.norm(y0, axis=1)<r0] =  4/a1*np.linalg.norm(y0[np.linalg.norm(y0, axis=1)<r0], axis=1, keepdims=True)**3 # 不同的区域进行不同的解析解赋予
exactdy0[np.linalg.norm(y0, axis=1)>=r0] =  4/a2*np.linalg.norm(y0[np.linalg.norm(y0, axis=1)>=r0], axis=1, keepdims=True)**3
y0t = torch.tensor(y0, requires_grad=True)
predy0 = pred(y0t) # 预测x=0的原函数
dudxy = grad(predy0, y0t, torch.ones(y0t.size()[0], 1).cuda(), create_graph=True)[0]
dudx_rao = dudxy[:, 0].unsqueeze(1).data.cpu().numpy()
dudy = dudxy[:, 1].unsqueeze(1).data.cpu().numpy()
# dudx_cenn[y0[:, 0]<0] = -dudx_cenn[y0[:, 0]<0] # y小于0的导数添加负号

# =============================================================================
# 画图
# =============================================================================
 
# 首先画损失函数以及误差图
# %%        
fig = plt.figure(dpi=1000, figsize=(12, 11)) #接下来画损失函数的三种方法的比较以及相对误差的比较
plt.subplot(2, 2, 1) # cpinn的损失函数
plt.yscale('log')
plt.plot(loss1_array_cpinn, '--')
plt.plot(loss2_array_cpinn, '-.')
plt.legend([ 'SUB-CPINN-1', 'SUB-CPINN_2'], loc = 'upper right')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('CPINN', fontsize = 10) 
settick()
plt.subplot(2, 2, 2) # 能量法以及cenn的损失函数，比较J1，需要画一条精确的线
#plt.grid(axis='y')
exactJ1=1.10
plt.axhline(y=1.0, color='r', ls = ':')
plt.plot(loss1_array_energy/exactJ1, ls = '--')
plt.plot(loss1_array_cenn/exactJ1, ls = '-.')
plt.ylim(bottom=0.5)
plt.legend(['Exact', 'DEM', 'CENN'], loc = 'upper right')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('DEM and CENN internal', fontsize = 10) 
settick()
plt.subplot(2, 2, 3)  # 能量法以及cenn的损失函数，比较J2，需要画一条精确的线
#plt.grid(axis='y')
exactJ2 = 436.45
plt.axhline(y=1.0, color='r', ls = ':') # 画出精确解
plt.plot(loss2_array_energy/exactJ2, ls = '--')
plt.plot(loss2_array_cenn/exactJ2, ls = '-.')
y_major_locator=MultipleLocator(0.5)
ax=plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
plt.ylim(bottom=0.0, top = 2.0)
plt.legend(['Exact', 'DEM', 'CENN'], loc = 'upper right')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('DEM and CENN external', fontsize = 10) 
settick()

plt.subplot(2, 2, 4)
plt.yscale('log')
plt.plot(error_array_cpinn)
plt.plot(error_array_rao, ':')
plt.plot(error_array_energy, '--')
plt.plot(error_array_cenn, '-.')
plt.legend(['CPINN', 'CPINN_RBF', 'DEM', 'CENN'], loc = 'upper right')
plt.xlabel('Iteration')
plt.ylabel('${\mathcal{L}_2}$ Error')
plt.title('Error comparision', fontsize = 10) 
settick()
plt.savefig('./pic/koch_compare_loss_error.pdf' , bbox_inches = 'tight')
plt.show()





#%%
######################################################################################################################################  
# 再画不同线的比较
axs = plt.figure(dpi=1000, figsize=(14, 11)).subplots(2,2)
plt.subplot(2, 2, 1)
# plt.figure(dpi=1000, figsize=(6, 5.5))
plt.plot(x0[:, 1], exactx0.flatten())
plt.plot(x0[:, 1], predx0_cpinn.flatten(), linestyle=':')
plt.plot(x0[:, 1], predx0_energy.flatten(), linestyle='--')
plt.plot(x0[:, 1], predx0_cenn.flatten(), linestyle='-.')
plt.legend(['Exact', 'CPINN', 'DEM', 'CENN'], loc='upper left')
plt.xlabel('Y')
plt.ylabel('U')
plt.title('X=0', size = 10)
#axs[0, 0].set_title('X=0', size = 10)
# plt.savefig('./pic/x=0_u_%i.png' % dd)
# plt.show()

plt.subplot(2, 2, 2)
# plt.figure(dpi=1000, figsize=(6, 5.5))
plt.plot(y0[:, 0], exacty0.flatten())
plt.plot(y0[:, 0], predy0_cpinn.flatten(), linestyle=':')
plt.plot(y0[:, 0], predy0_energy.flatten(), linestyle='--')
plt.plot(y0[:, 0], predy0_cenn.flatten(), linestyle='-.')
plt.legend(['Exact', 'CPINN', 'DEM', 'CENN'], loc='upper left')
plt.xlabel('X')
plt.ylabel('U')
plt.title('Y=0', size = 10)
# plt.savefig('./pic/y=0_u_%i.png' % dd)
# plt.show()

plt.subplot(2, 2, 3)
# plt.figure(dpi=1000, figsize=(6, 5.5))
plt.plot(x0[:-1, 1], exactdx0.flatten()[:-1]) # 删除最后一个边界点
plt.plot(x0[:-1, 1], dudy_cpinn.flatten()[:-1], linestyle=':')
plt.plot(x0[:-1, 1], dudy_energy.flatten()[:-1], linestyle='--')
plt.plot(x0[:-1, 1], dudy_cenn.flatten()[:-1], linestyle='-.')
plt.legend(['Exact', 'CPINN', 'DEM', 'CENN'], loc='upper left')
plt.xlabel('Y')
plt.ylabel('$\partial u/\partial y$')
# plt.savefig('./pic/x=0_dudy_%i.png' % dd)
plt.title('X=0', size = 10)

# plt.show()

plt.subplot(2, 2, 4)
# plt.figure(dpi=1000, figsize=(6, 5.5))
plt.plot(y0[:, 0], exactdy0.flatten())
plt.plot(y0[:, 0], dudx_cpinn.flatten(), linestyle=':')
plt.plot(y0[:, 0], dudx_energy.flatten(), linestyle='--')
plt.plot(y0[:, 0], dudx_cenn.flatten(), linestyle='-.')
plt.legend(['Exact', 'CPINN', 'DEM', 'CENN'], loc='upper left')
plt.xlabel('X')
plt.ylabel('$\partial u/\partial x$')
plt.title('Y=0', size = 10)
# plt.savefig('./pic/y=0_dudx_%i.png' % dd)
plt.savefig('./pic/koch_compare_cross%i.pdf' % dd, bbox_inches = 'tight')
plt.show()


print('the relative error is %f' % error_t_cpinn.data)   # 输出cpinn的整体误差，用来画图
print('the relative error is %f' % error_t_energy.data)   # 输出能量法不分片的整体误差，用来画图
print('the relative error is %f' % error_t_cenn.data)  
error_total[dd] = {'cpinn':float(error_t_cpinn.data.cpu()), 'energy':float(error_t_energy.data.cpu()), 'cenn':float(error_t_cenn.data.cpu())}
ft = open('./error_record.txt', 'a')
ft.write(str(error_total))
ft.close()








