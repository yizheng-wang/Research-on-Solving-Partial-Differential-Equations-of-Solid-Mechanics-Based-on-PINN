# 添加Airy应力函数解析项，增加精度。

import numpy as np
import torch
import matplotlib.pyplot as plt
from  torch.autograd import grad
import time
import matplotlib as mpl
torch.set_default_tensor_type(torch.DoubleTensor) # 将tensor的类型变成默认的double
torch.set_default_tensor_type(torch.cuda.DoubleTensor) # 将cuda的tensor的类型变成默认的double
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
    
setup_seed(0)

Po = 10
Pi = 5
a = 0.5
b = 1.0

nepoch = 2000

dom_num = 10000
N_test = 100
E = 1000
nu = 0.3
Ura = 1/E*((1-nu)*(a**2*Pi-b**2*Po)/(b**2-a**2)*a + (1+nu)*(a**2*b**2)*(Pi-Po)/(b**2-a**2)/a)
Urb = 1/E*((1-nu)*(a**2*Pi-b**2*Po)/(b**2-a**2)*b + (1+nu)*(a**2*b**2)*(Pi-Po)/(b**2-a**2)/b)

r1 = torch.tensor([0], requires_grad=True,  dtype=torch.float64, device='cuda')
r2 = torch.tensor([0], requires_grad=True,  dtype=torch.float64, device='cuda')
r3 = torch.tensor([0], requires_grad=True,  dtype=torch.float64, device='cuda')
r4 = torch.tensor([0], requires_grad=True,  dtype=torch.float64, device='cuda')
r5 = torch.tensor([0], requires_grad=True,  dtype=torch.float64, device='cuda')
r6 = torch.tensor([0], requires_grad=True,  dtype=torch.float64, device='cuda')
r7 = torch.tensor([0], requires_grad=True,  dtype=torch.float64, device='cuda')
r8 = torch.tensor([0], requires_grad=True,  dtype=torch.float64, device='cuda')



def dom_data(Nf):
    '''
    生成内部点，极坐标形式生成
    '''
    
    # r = (b-a)*np.random.rand(Nf)+a
    # theta = 2 * np.pi * np.random.rand(Nf) # 角度0到2*pi
    r = np.linspace(a,b, Nf)
    xy_dom = np.stack([r], 1)
    xy_dom = torch.tensor(xy_dom,  requires_grad=True, device='cuda')
    
    return xy_dom


class FNN(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(FNN, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, D_out)
        
        #self.a1 = torch.nn.Parameter(torch.Tensor([0.2]))
        
        # 有可能用到加速收敛技术
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

def pred(xy):
    '''
    

    Parameters
    ----------
    xy : tensor 
        the coordinate of the input tensor.

    Returns
    the prediction of fai

    '''

    pred_fai = model(xy) + r2*xy**2 +  r3*torch.log(xy) + r4*xy**2*torch.log(xy)
    return pred_fai

def evaluate_sigma(N_test):# calculate the prediction of the stress rr and theta
# 分析sigma应力，输入坐标是极坐标r和theta
    r = np.linspace(a, b, N_test)
    theta = np.linspace(0, 2*np.pi, N_test) # y方向N_test个点
    r_mesh, theta_mesh = np.meshgrid(r, theta)
    xy = np.stack((r_mesh.flatten(), theta_mesh.flatten()), 1)
    X_test = torch.tensor(xy,  requires_grad=True, device='cuda')
    r = X_test[:, 0].unsqueeze(1)
    # 将r输入到pred中，输出应力函数
    fai = pred(r)
    dfaidr = grad(fai, r, torch.ones(r.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    dfaidrr = grad(dfaidr, r, torch.ones(r.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    pred_sigma_rr = (1/r)*dfaidr # rr方向的正应力
    pred_sigma_theta = dfaidrr # theta方向的正应力
    
    pred_sigma_rr = pred_sigma_rr.data.cpu().numpy().reshape(N_test, N_test)
    pred_sigma_theta = pred_sigma_theta.data.cpu().numpy().reshape(N_test, N_test)
    # sigma_r = a**2/(b**2-a**2)*(1-b**2/r**2)*Pi - b**2/(b**2-a**2)*(1-a**2/r**2)*Po
    # sigma_theta = a**2/(b**2-a**2)*(1+b**2/r**2)*Pi - b**2/(b**2-a**2)*(1+a**2/r**2)*Po

    return r_mesh, theta_mesh, pred_sigma_rr, pred_sigma_theta

def evaluate_sigma_line(N_test):# output the prediction of the stress rr and theta along radius in direction of r without theta
    r_numpy = np.linspace(a, b, N_test) # generate the coordinate of r by numpy 
    r = torch.tensor(r_numpy,  requires_grad=True, device='cuda').unsqueeze(1) # change the type of array to tensor used to be AD
    fai = pred(r) # Input r to the pred function to get the necessary predition stress function
    dfaidr = grad(fai, r, torch.ones(r.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    dfaidrr = grad(dfaidr, r, torch.ones(r.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    pred_sigma_rr = (1/r)*dfaidr # rr方向的正应力
    pred_sigma_theta = dfaidrr # theta方向的正应力
    pred_sigma_rr = pred_sigma_rr.data.cpu().numpy() # change the type of tensor to the array for the use of plotting
    pred_sigma_theta = pred_sigma_theta.data.cpu().numpy()

    return r_numpy, pred_sigma_rr, pred_sigma_theta
# learning the homogenous network

model = FNN(1, 20, 1).cuda() # input: r; output: Airy stress function
optim = torch.optim.Adam([{'params' : model.parameters()}, \
                             {'params' : r1}, {'params' : r2}, {'params' : r3}, {'params' : r4}, \
                             {'params' : r5}, {'params' : r6}, {'params' : r7}, {'params' : r8}],\
                          lr= 0.001)
loss_array = []
loss_dom_array = []
loss_ex_array = []
error_sigma_rr_array = []
error_sigma_theta_array = []
nepoch = int(nepoch)
start = time.time()

for epoch in range(nepoch):
    if epoch == nepoch-1:
        end = time.time()
        consume_time = end-start
        print('time is %f' % consume_time)
    if epoch%100 == 0: # 重新分配点   
        Xf = dom_data(dom_num)
        
    def closure():  
        # 区域内部损失
        r = Xf
        fai = pred(r)  
        dfaidr = grad(fai, r, torch.ones(r.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        dfaidrr = grad(dfaidr, r, torch.ones(r.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]   

        J_dom = torch.mean(0.5/E*(1/r**2*dfaidr**2 + dfaidrr**2 - 2*nu/r*dfaidr*dfaidrr)*2*np.pi*r) * (b-a) # 计算余能
        # calculate the complementary external energy
        ra = torch.tensor([a],  requires_grad=True, device='cuda').unsqueeze(1)
        rb = torch.tensor([b],  requires_grad=True, device='cuda').unsqueeze(1)
        faia = pred(ra)
        faib = pred(rb)
        dfaidra = grad(faia, ra, retain_graph=True, create_graph=True)[0]
        dfaidrb = grad(faib, rb, retain_graph=True, create_graph=True)[0]
        J_ex = 2*np.pi*(-Ura*dfaidra + Urb*dfaidrb) # 计算外力余势
   
        loss = J_dom - J_ex 
        optim.zero_grad()
        loss.backward(retain_graph=True)
        loss_array.append(loss.data.cpu())
        loss_dom_array.append(J_dom.data.cpu())
        loss_ex_array.append(J_ex.data.cpu())
        if epoch == nepoch-1:
            print()
        r_numpy, pred_sigma_rr, pred_sigma_theta = evaluate_sigma_line(N_test)  
        exact_sigma_rr = a**2/(b**2-a**2)*(1-b**2/r_numpy**2)*Pi - b**2/(b**2-a**2)*(1-a**2/r_numpy**2)*Po
        exact_sigma_theta = a**2/(b**2-a**2)*(1+b**2/r_numpy**2)*Pi - b**2/(b**2-a**2)*(1+a**2/r_numpy**2)*Po        
        L2_rr_error = np.linalg.norm(pred_sigma_rr.flatten() - exact_sigma_rr.flatten())/np.linalg.norm(exact_sigma_rr.flatten())
        L2_theta_error = np.linalg.norm(pred_sigma_theta.flatten() - exact_sigma_theta.flatten())/np.linalg.norm(exact_sigma_theta.flatten())
        error_sigma_rr_array.append(L2_rr_error)
        error_sigma_theta_array.append(L2_theta_error)
        if epoch%10==0:
            print(' epoch : %i, the loss : %f, loss dom : %f, loss ex : %f' % \
                  (epoch, loss.data, J_dom.data, J_ex.data))
        return loss
    optim.step(closure)
    
    
    
r_mesh, theta_mesh, pred_sigma_rr, pred_sigma_theta = evaluate_sigma(N_test) 
exact_sigma_rr = a**2/(b**2-a**2)*(1-b**2/r_mesh**2)*Pi - b**2/(b**2-a**2)*(1-a**2/r_mesh**2)*Po
exact_sigma_theta = a**2/(b**2-a**2)*(1+b**2/r_mesh**2)*Pi - b**2/(b**2-a**2)*(1+a**2/r_mesh**2)*Po  

# =============================================================================
# plot the contouf of the sigma
# =============================================================================

# rr精确剪应力
x_mesh = r_mesh * np.cos(theta_mesh)
y_mesh = r_mesh * np.sin(theta_mesh)
h1 = plt.contourf(x_mesh, y_mesh, exact_sigma_rr,  cmap = 'jet', levels = 100)
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h1).ax.set_title(r'$\sigma_{rr}$')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# zy精确剪应力
h2 = plt.contourf(x_mesh, y_mesh, exact_sigma_theta,  cmap = 'jet', levels = 100)
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h2).ax.set_title(r'$\sigma_{\theta\theta}$')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# zx最小余能预测
h3 = plt.contourf(x_mesh, y_mesh, pred_sigma_rr,  cmap = 'jet', levels = 100)
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h3).ax.set_title(r'$\sigma_{rr}$')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# zy最小余能预测
h4 = plt.contourf(x_mesh, y_mesh, pred_sigma_theta,  cmap = 'jet', levels = 100)
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h4).ax.set_title(r'$\sigma_{\theta\theta}$')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# zx最小余能误差
h5 = plt.contourf(x_mesh, y_mesh, np.abs(pred_sigma_rr - exact_sigma_rr),  cmap = 'jet', levels = 100)
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h5).ax.set_title(r'$\sigma_{rr}$')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# zy最小余能误差
h6 = plt.contourf(x_mesh, y_mesh, np.abs(pred_sigma_theta - exact_sigma_theta),  cmap = 'jet', levels = 100)
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h6).ax.set_title(r'$\sigma_{\theta\theta}$')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# =============================================================================
# plot line sigma_rr and theta
# =============================================================================
r_numpy, predline_sigma_rr, predline_sigma_theta = evaluate_sigma_line(N_test)
exactline_sigma_rr = a**2/(b**2-a**2)*(1-b**2/r_numpy**2)*Pi - b**2/(b**2-a**2)*(1-a**2/r_numpy**2)*Po
exactline_sigma_theta = a**2/(b**2-a**2)*(1+b**2/r_numpy**2)*Pi - b**2/(b**2-a**2)*(1+a**2/r_numpy**2)*Po
plt.plot(r_numpy, exactline_sigma_rr, color = 'r')
plt.scatter(r_numpy[::2], predline_sigma_rr[::2], marker = '*', linestyle='-')
plt.xlabel('r')
plt.ylabel(r'$\sigma_{rr}$')
plt.legend(['Exact', 'DCM'])
settick()
plt.show()

plt.plot(r_numpy, exactline_sigma_theta, color = 'r')
plt.scatter(r_numpy[::2], predline_sigma_theta[::2], marker = '*', linestyle='-')
plt.xlabel('r')
plt.ylabel(r'$\sigma_{\theta\theta}$')
plt.legend(['Exact', 'DCM'])
settick()
plt.show()

# =============================================================================
# plot the loss function
# =============================================================================

loss_array = np.array(loss_array)
loss_dom_array = np.array(loss_dom_array)
loss_ex_array = np.array(loss_ex_array)



plt.plot(loss_array) 
plt.plot(loss_dom_array, linestyle='-.')
plt.plot(loss_ex_array, linestyle=':')
plt.legend(['余能', '余应变能', '余势'])
plt.xlabel('迭代数')
plt.ylabel('损失函数')
plt.show()
# plt.title('loss evolution')

plt.yscale('log')
plt.plot(error_sigma_rr_array) 
plt.plot(error_sigma_theta_array, linestyle='-.')
plt.legend([r'$\sigma_{rr}$', r'$\sigma_{\theta\theta}$'])
plt.xlabel('迭代数')
plt.ylabel('误差')
plt.show()
print('The total relative error of sigma rr is ' + str(error_sigma_rr_array[-1]))
print('The total relative error of sigma theta is ' + str(error_sigma_theta_array[-1]))
    
    