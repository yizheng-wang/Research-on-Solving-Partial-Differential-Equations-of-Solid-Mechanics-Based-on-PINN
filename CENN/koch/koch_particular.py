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

import matplotlib as mpl
mpl.rcParams['font.sans-serif']=['SimHei'] # 为了显示中文matplotlib
mpl.rcParams['figure.dpi'] = 1000


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
setup_seed(55)
torch.set_default_tensor_type(torch.DoubleTensor) # 将tensor的类型变成默认的double
torch.set_default_tensor_type(torch.cuda.DoubleTensor) # 将cuda的tensor的类型变成默认的double
train_p = 1
#tol_p = 4
tol_p = 0.00001
a1 = 1/15
a2 = 1
r0 = 0.5
# for particular solution 
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
            loss_bn_array.append(loss_bn.data.cpu())
            if epoch_b%10==0:
                print('trianing particular network : the number of epoch is %i, the loss is %f' % (epoch_b, loss_bn.data))
            return loss_bn
        optimp.step(closure)
        loss_bn = loss_bn_array[-1]
    torch.save(model_p, './particular_nn')

model_p = torch.load('particular_nn')

 # 获得测试点 n_test**2个


# %%
# N_test = 101
# x = np.linspace(-15, 15, N_test)
# y = np.linspace(-10*3**0.5, 10*3**0.5, N_test)
# x, y = np.meshgrid(x/10, y/10)
# xy_test = np.stack((x.flatten(), y.flatten()),1)
# xy_test_t = torch.from_numpy(xy_test).cuda() # to the model for the prediction
# u_pred = torch.zeros((len(xy_test), 1)).cuda()
# label_koch = koch_points.whether_koch(xy_test) # 获得内部点的标签



# u_pred[label_koch] = model_p(xy_test_t[label_koch])
# u_pred = u_pred.data
# u_pred = u_pred.reshape(N_test, N_test).cpu()

# u_exact = np.zeros(x.shape)
# # 精确解有2个，一个内部一个外部，内部设为1，外部设为2
# u_exact[(x**2+y**2)<r0**2] = 1/a1*np.sqrt(x[(x**2+y**2)<=r0**2]**2+y[(x**2+y**2)<=r0**2]**2)**4 # 现在网格里面赋值
# label_koch_row = label_koch.reshape(N_test, N_test) # 获得网络的科赫雪花label
# u_exact[((x**2+y**2)>=r0**2) & (label_koch_row)] = 1/a2*np.sqrt(x[((x**2+y**2)>=r0**2) & (label_koch_row)]**2+y[((x**2+y**2)>=r0**2) & (label_koch_row)]**2)**4+(1/a1 - 1/a2)*r0**4
# u_exact = torch.from_numpy(u_exact)

# #for paper plot
# fig = plt.figure(dpi=1000,figsize = (20,5))
# #fig = 
# fig.tight_layout()
# # plot the prediction solution
# #plt.subplots_adjust(wspace = 0.4, hspace=0)
# plt.subplot(1, 3, 1)
# h3 = plt.contourf(x, y, u_exact, levels=100 ,cmap = 'jet')
# plt.title('exact solution') 
# plt.colorbar(h3)




# plt.subplot(1, 3, 2)
# dis_plot = u_pred.data.cpu().numpy().reshape(N_test, N_test)
# h1 = plt.contourf(x, y, dis_plot, cmap='jet', levels = 50)
# plt.colorbar(h1)
# plt.title('the particular network')
# plt.xlabel('x')
# plt.ylabel('y')


# plt.subplot(1, 3, 3)
plt.cla()
plt.ticklabel_format(style='sci', scilimits=(-1,2), axis = 'x')
plt.yscale('log')
plt.plot(loss_bn_array)
plt.legend(['Particular'])
#plt.title('损失函数')
plt.xlabel('Iteration')
plt.ylabel('Loss')
#plt.savefig('../../picture/koch/loss_particular.pdf', bbox_inches = 'tight')
settick()
plt.show()

n_test = 300
dom_koch_n = koch_points.get_koch_points_lin(n_test) # 获得n_test个koch的随机分布点
dom_koch_t= torch.tensor(dom_koch_n,  requires_grad=True, device = 'cuda') # 将numpy数据转化为tensor

u_exact = np.zeros((len(dom_koch_n), 1))
# 精确解有2个，一个内部一个外部，内部设为1，外部设为2
u_exact[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)<r0**2] = 1/a1*np.linalg.norm(dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)<r0**2], axis=1, keepdims=True)**4 # 现在网格里面赋值
u_exact[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)>=r0**2] = 1/a2*np.linalg.norm(dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)>=r0**2], axis=1, keepdims=True)**4 + r0**4*(1/a1-1/a2) # 现在网格外面赋值
#write_arr2DVTK('./output_ntk/exact_plane', dom_koch_n, u_exact, 'exact')

u_pred = model_p(dom_koch_t)
u_pred = u_pred.data.cpu()
#write_arr2DVTK('./output_ntk/pred_particular_plane', dom_koch_n, u_pred, 'particular')







