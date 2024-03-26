import numpy as np
import torch
import matplotlib.pyplot as plt
from  torch.autograd import grad
import time
import matplotlib as mpl
torch.set_default_tensor_type(torch.DoubleTensor) # 将tensor的类型变成默认的double
torch.set_default_tensor_type(torch.cuda.DoubleTensor) # 将cuda的tensor的类型变成默认的double

plt.rcParams['font.family'] = ['sans-serif'] # 用来正常显示负号
mpl.rcParams['figure.dpi'] = 1000


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
setup_seed(1)

Po = 10
Pi = 5
a = 0.5
b = 1.0

nepoch = 10000

dom_num = 101
N_test = 101
E = 1000
nu = 0.3
Ura = 1/E*((1-nu)*(a**2*Pi-b**2*Po)/(b**2-a**2)*a + (1+nu)*(a**2*b**2)*(Pi-Po)/(b**2-a**2)/a)
Urb = 1/E*((1-nu)*(a**2*Pi-b**2*Po)/(b**2-a**2)*b + (1+nu)*(a**2*b**2)*(Pi-Po)/(b**2-a**2)/b)


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
def simpson_int(y, x,  nx = N_test):
    '''
    Simpson integration for 1D

    Parameters
    ----------
    y : tensor
        The value of the x.
    x : tensor
        Coordinate of the input.
    nx : int, optional
        The grid node number of x axis. The default is N_test.
        
    Returns
    -------
    result : tensor
        the result of the integration.

    '''
    weightx = [4, 2] * int((nx-1)/2)
    weightx = [1] + weightx
    weightx[-1] = weightx[-1]-1
    weightx = np.array(weightx)
    

    weightx = weightx.reshape(-1,1)


    weight = torch.tensor(weightx, device='cuda')
    weight = weight.flatten()
    hx = a/(nx-1)
    y = y.flatten()
    result = torch.sum(weight*y)*hx/3
    return result

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
        
        self.a1 = torch.nn.Parameter(torch.Tensor([0.1]))
        
        # 有可能用到加速收敛技术
        # self.a1 = torch.Tensor([0.1]).cuda()
        # self.a2 = torch.Tensor([0.1])
        # self.a3 = torch.Tensor([0.1])
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

    pred_fai = model(xy)
    return pred_fai

def evaluate(N_test):# calculate the prediction of the stress rr and theta
    # 生成theta数据点
    r_numpy = np.linspace(a, b, N_test)
    r = np.stack([r_numpy], 1) # 将mesh的点flatten
    r = torch.tensor(r,  requires_grad=True, device='cuda')

    f = pred(r).data.cpu().numpy()
    return  r_numpy, f

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
    pred_u_r = 1/E*(r*dfaidrr-nu*dfaidr)
    
    pred_sigma_rr = pred_sigma_rr.data.cpu().numpy().reshape(N_test, N_test)
    pred_sigma_theta = pred_sigma_theta.data.cpu().numpy().reshape(N_test, N_test)
    pred_u_r = pred_u_r.data.cpu().numpy().reshape(N_test, N_test)
    # sigma_r = a**2/(b**2-a**2)*(1-b**2/r**2)*Pi - b**2/(b**2-a**2)*(1-a**2/r**2)*Po
    # sigma_theta = a**2/(b**2-a**2)*(1+b**2/r**2)*Pi - b**2/(b**2-a**2)*(1+a**2/r**2)*Po

    return r_mesh, theta_mesh, pred_sigma_rr, pred_sigma_theta, pred_u_r

def evaluate_sigma_line(N_test):# output the prediction of the stress rr and theta along radius in direction of r without theta
    r_numpy = np.linspace(a, b, N_test) # generate the coordinate of r by numpy 
    r = torch.tensor(r_numpy,  requires_grad=True, device='cuda').unsqueeze(1) # change the type of array to tensor used to be AD
    fai = pred(r) # Input r to the pred function to get the necessary predition stress function
    dfaidr = grad(fai, r, torch.ones(r.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    dfaidrr = grad(dfaidr, r, torch.ones(r.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    pred_sigma_rr = (1/r)*dfaidr # rr方向的正应力
    pred_sigma_theta = dfaidrr # theta方向的正应力
    pred_u_r = 1/E*(r*dfaidrr-nu*dfaidr)
    pred_sigma_rr = pred_sigma_rr.data.cpu().numpy() # change the type of tensor to the array for the use of plotting
    pred_sigma_theta = pred_sigma_theta.data.cpu().numpy()
    pred_u_r = pred_u_r.data.cpu().numpy()
    return r_numpy, pred_sigma_rr, pred_sigma_theta, pred_u_r
# learning the homogenous network

model = FNN(1, 20, 1).cuda() # input: r; output: Airy stress function
optim = torch.optim.Adam(model.parameters(), lr= 0.001)
step_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1000, gamma=0.1)
loss_array = []
loss_dom_array = []
loss_ex_array = []
error_sigma_rr_array = []
error_sigma_theta_array = []
error_u_r_array = []
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
        
        J_dom = simpson_int(0.5/E*(1/r**2*dfaidr**2 + dfaidrr**2 - 2*nu/r*dfaidr*dfaidrr)*2*np.pi*r, r)
        #J_dom = torch.mean(0.5/E*(1/r**2*dfaidr**2 + dfaidrr**2 - 2*nu/r*dfaidr*dfaidrr)*2*np.pi*r) * (b-a) # 计算余能
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
        r_numpy, pred_sigma_rr, pred_sigma_theta, pred_u_r = evaluate_sigma_line(N_test)  
        exact_sigma_rr = a**2/(b**2-a**2)*(1-b**2/r_numpy**2)*Pi - b**2/(b**2-a**2)*(1-a**2/r_numpy**2)*Po
        exact_sigma_theta = a**2/(b**2-a**2)*(1+b**2/r_numpy**2)*Pi - b**2/(b**2-a**2)*(1+a**2/r_numpy**2)*Po        
        exact_u_r = 1/E*((1-nu)*(a**2*Pi-b**2*Po)/(b**2-a**2)*r_numpy + (1+nu)*(a**2*b**2)*(Pi-Po)/(b**2-a**2)/r_numpy)
        L2_rr_error = np.linalg.norm(pred_sigma_rr.flatten() - exact_sigma_rr.flatten())/np.linalg.norm(exact_sigma_rr.flatten())
        L2_theta_error = np.linalg.norm(pred_sigma_theta.flatten() - exact_sigma_theta.flatten())/np.linalg.norm(exact_sigma_theta.flatten())
        L2_u_error = np.linalg.norm(pred_u_r.flatten() - exact_u_r.flatten())/np.linalg.norm(exact_u_r.flatten())
        error_sigma_rr_array.append(L2_rr_error)
        error_sigma_theta_array.append(L2_theta_error)
        error_u_r_array.append(L2_u_error)
        if epoch%10==0:
            print(' epoch : %i, the loss : %f, loss dom : %f, loss ex : %f' % \
                  (epoch, loss.data, J_dom.data, J_ex.data))
        return loss
    optim.step(closure)
    step_scheduler.step()
    
    
    
r_mesh, theta_mesh, pred_sigma_rr, pred_sigma_theta, pred_u_r = evaluate_sigma(N_test) 
exact_sigma_rr = a**2/(b**2-a**2)*(1-b**2/r_mesh**2)*Pi - b**2/(b**2-a**2)*(1-a**2/r_mesh**2)*Po
exact_sigma_theta = a**2/(b**2-a**2)*(1+b**2/r_mesh**2)*Pi - b**2/(b**2-a**2)*(1+a**2/r_mesh**2)*Po  
exact_u_r = 1/E*((1-nu)*(a**2*Pi-b**2*Po)/(b**2-a**2)*r_mesh + (1+nu)*(a**2*b**2)*(Pi-Po)/(b**2-a**2)/r_mesh)
# =============================================================================
# plot the contouf of the sigma
# =============================================================================

# rr精确剪应力
x_mesh = r_mesh * np.cos(theta_mesh)
y_mesh = r_mesh * np.sin(theta_mesh)
h1 = plt.contourf(x_mesh, y_mesh, exact_sigma_rr,  cmap = 'jet', levels = 100)
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h1).ax.set_title(r'$\sigma_{rr}$', fontsize=13)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# zy精确剪应力
h2 = plt.contourf(x_mesh, y_mesh, exact_sigma_theta,  cmap = 'jet', levels = 100)
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h2).ax.set_title(r'$\sigma_{\theta\theta}$', fontsize=13)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# r精确位移
h2 = plt.contourf(x_mesh, y_mesh, exact_u_r,  cmap = 'jet', levels = 100)
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h2).ax.set_title(r'$u_{rr}$', fontsize=13)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# zx最小余能预测
h3 = plt.contourf(x_mesh, y_mesh, pred_sigma_rr,  cmap = 'jet', levels = 100)
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h3).ax.set_title(r'$\sigma_{rr}$', fontsize=13)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# zy最小余能预测
h4 = plt.contourf(x_mesh, y_mesh, pred_sigma_theta,  cmap = 'jet', levels = 100)
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h4).ax.set_title(r'$\sigma_{\theta\theta}$', fontsize=13)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# r最小余能预测
h4 = plt.contourf(x_mesh, y_mesh, pred_u_r,  cmap = 'jet', levels = 100)
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h4).ax.set_title(r'$u_{rr}$', fontsize=13)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# zx最小余能误差
h5 = plt.contourf(x_mesh, y_mesh, np.abs(pred_sigma_rr - exact_sigma_rr),  cmap = 'jet', levels = 100)
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h5).ax.set_title(r'$\sigma_{rr}$', fontsize=13)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# zy最小余能误差
h6 = plt.contourf(x_mesh, y_mesh, np.abs(pred_sigma_theta - exact_sigma_theta),  cmap = 'jet', levels = 100)
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h6).ax.set_title(r'$\sigma_{\theta\theta}$', fontsize=13)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# r最小余能预测
h4 = plt.contourf(x_mesh, y_mesh, np.abs(pred_u_r-exact_u_r),  cmap = 'jet', levels = 100)
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h4)#.ax.set_title(r'$u_{rr}$')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# =============================================================================
# plot line sigma_rr and theta
# =============================================================================
r_numpy, predline_sigma_rr, predline_sigma_theta, predline_u_r = evaluate_sigma_line(N_test)
exactline_sigma_rr = a**2/(b**2-a**2)*(1-b**2/r_numpy**2)*Pi - b**2/(b**2-a**2)*(1-a**2/r_numpy**2)*Po
exactline_sigma_theta = a**2/(b**2-a**2)*(1+b**2/r_numpy**2)*Pi - b**2/(b**2-a**2)*(1+a**2/r_numpy**2)*Po
exactline_u_r = 1/E*((1-nu)*(a**2*Pi-b**2*Po)/(b**2-a**2)*r_numpy + (1+nu)*(a**2*b**2)*(Pi-Po)/(b**2-a**2)/r_numpy)
plt.plot(r_numpy, exactline_sigma_rr, color = 'r')
plt.scatter(r_numpy[::2], predline_sigma_rr[::2], marker = '*', linestyle='-')
plt.xlabel('r', fontsize=13)
plt.ylabel(r'$\sigma_{rr}$', fontsize=13)
plt.legend(['Exact', 'DCEM'])
plt.show()

plt.plot(r_numpy, exactline_sigma_theta, color = 'r')
plt.scatter(r_numpy[::2], predline_sigma_theta[::2], marker = '*', linestyle='-')
plt.xlabel('r', fontsize=13)
plt.ylabel(r'$\sigma_{\theta\theta}$', fontsize=13)
plt.legend(['Exact', 'DCEM'])
plt.show()

plt.plot(r_numpy, exactline_u_r, color = 'r')
plt.scatter(r_numpy[::2], predline_u_r[::2], marker = '*', linestyle='-')
plt.xlabel('r', fontsize=13)
plt.ylabel(r'$u_{r}$', fontsize=13)
plt.legend(['Exact', 'DCEM'])
plt.show()


print('The total relative error of sigma rr is ' + str(error_sigma_rr_array[-1]))
print('The total relative error of sigma theta is ' + str(error_sigma_theta_array[-1]))
print('The total relative error of sigma theta is ' + str(error_u_r_array[-1]))

# plot loss error
#%%
def moving_average(interval, window_size):
    '''
    for smooth

    Parameters
    ----------
    interval : TYPE
        DESCRIPTION.
    window_size : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')  # numpy的卷积函数
error_sigma_rr_array_complementary_smo  = moving_average(interval = error_sigma_rr_array, window_size = 10)

error_sigma_theta_array_complementary_smo  = moving_average(interval = error_sigma_theta_array, window_size = 10)

error_dis_r_array_complementary_smo  = moving_average(interval = error_u_r_array, window_size = 10)




plt.plot(error_sigma_rr_array)
plt.yscale('log')
plt.xlabel('Iteration', fontsize=13)
plt.ylabel(r'$\sigma_{r}$ relative error', fontsize=13)
plt.show()


plt.plot(error_sigma_theta_array)
plt.yscale('log')
plt.xlabel('Iteration', fontsize=13)
plt.ylabel(r'$\sigma_{\theta}$ relative error', fontsize=13)
plt.show()

plt.plot(error_u_r_array)
plt.yscale('log')
plt.xlabel('Iteration', fontsize=13)
plt.ylabel(r'$\sigma_{\theta}$ relative error', fontsize=13)
plt.show()

r_numpy, fai_pred = evaluate(N_test)
r = np.linspace(a,b, N_test)
fai_exact = a**2/(b**2-a**2)*(r_numpy**2/2-b**2*np.log(r_numpy))*Pi-b**2/(b**2-a**2)*(r_numpy**2/2-a**2*np.log(r_numpy))*Po
plt.plot(r_numpy, fai_exact)
plt.plot(r_numpy, fai_pred)
plt.show()