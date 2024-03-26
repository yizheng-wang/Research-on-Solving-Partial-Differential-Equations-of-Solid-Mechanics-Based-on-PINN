# minimun potential energy mehtod

import numpy as np
import torch
import matplotlib.pyplot as plt
from  torch.autograd import grad
import time
import matplotlib as mpl
torch.set_default_tensor_type(torch.DoubleTensor) # 将tensor的类型变成默认的double
torch.set_default_tensor_type(torch.cuda.DoubleTensor) # 将cuda的tensor的类型变成默认的double
mpl.rcParams['figure.dpi'] = 1000
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

nepoch = 10000

dom_num = 101
N_test = 101
E = 1000
nu = 0.3
G = E/2/(1+nu)
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

    pred_dis = model(xy) * (xy - a) * (xy - b) + (1 - xy/a)*a/(a-b)*Urb + (1 - xy/b)*b/(b-a)*Ura # the admissiable displacement
    return pred_dis

def evaluate_sigma(N_test):# calculate the prediction of the stress rr and theta
# 分析sigma应力，输入坐标是极坐标r和theta
    r = np.linspace(a, b, N_test)
    theta = np.linspace(0, 2*np.pi, N_test) # y方向N_test个点
    r_mesh, theta_mesh = np.meshgrid(r, theta)
    xy = np.stack((r_mesh.flatten(), theta_mesh.flatten()), 1)
    X_test = torch.tensor(xy,  requires_grad=True, device='cuda')
    r = X_test[:, 0].unsqueeze(1)
    # 将r输入到pred中，输出应力函数
    Ur = pred(r) # Input r to the pred function to get the necessary predition stress function
    dUrdr = grad(Ur, r, torch.ones(r.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    pred_sigma_rr = 2*G/(1-nu)*(dUrdr+Ur*nu/r) # rr方向的正应力
    pred_sigma_theta = 2*G/(1-nu)*(Ur/r+nu*dUrdr) # theta方向的正应力
    
    pred_sigma_rr = pred_sigma_rr.data.cpu().numpy().reshape(N_test, N_test)
    pred_sigma_theta = pred_sigma_theta.data.cpu().numpy().reshape(N_test, N_test)
    # sigma_r = a**2/(b**2-a**2)*(1-b**2/r**2)*Pi - b**2/(b**2-a**2)*(1-a**2/r**2)*Po
    # sigma_theta = a**2/(b**2-a**2)*(1+b**2/r**2)*Pi - b**2/(b**2-a**2)*(1+a**2/r**2)*Po

    return r_mesh, theta_mesh, pred_sigma_rr, pred_sigma_theta

def evaluate_sigma_line(N_test):# output the prediction of the stress rr and theta along radius in direction of r without theta
    r_numpy = np.linspace(a, b, N_test) # generate the coordinate of r by numpy 
    r = torch.tensor(r_numpy,  requires_grad=True, device='cuda').unsqueeze(1) # change the type of array to tensor used to be AD
    Ur = pred(r) # Input r to the pred function to get the necessary predition stress function
    dUrdr = grad(Ur, r, torch.ones(r.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    pred_sigma_rr = 2*G/(1-nu)*(dUrdr+Ur*nu/r) # rr方向的正应力
    pred_sigma_theta = 2*G/(1-nu)*(Ur/r+nu*dUrdr) # theta方向的正应力
    pred_sigma_rr = pred_sigma_rr.data.cpu().numpy() # change the type of tensor to the array for the use of plotting
    pred_sigma_theta = pred_sigma_theta.data.cpu().numpy()

    return r_numpy, pred_sigma_rr, pred_sigma_theta

def evaluate_dis_line(N_test):# output the prediction of the displacement along the direction in r
    r_numpy = np.linspace(a, b, N_test) # generate the coordinate of r by numpy 
    r = torch.tensor(r_numpy,  requires_grad=True, device='cuda').unsqueeze(1) # change the type of array to tensor used to be AD
    Ur = pred(r) # Input r to the pred function to get the necessary predition stress function
    pred_dis_r = Ur.data.cpu().numpy()

    return r_numpy, pred_dis_r
# learning the homogenous network

model = FNN(1, 20, 1).cuda() # input: r; output: Airy stress function
optim = torch.optim.Adam(model.parameters(), lr= 0.001)
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
        Ur = pred(r)  
        dUrdr = grad(Ur, r, torch.ones(r.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        J_dom = simpson_int(0.5*(dUrdr*2*G/(1-nu)*(dUrdr+nu/r*Ur) + Ur/r*2*G/(1-nu)*(Ur/r+nu*dUrdr))*2*np.pi*r, r)
        # calculate the complementary external energy

        loss = J_dom 
        optim.zero_grad()
        loss.backward(retain_graph=True)
        loss_array.append(loss.data.cpu())
        loss_dom_array.append(J_dom.data.cpu())
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
            print(' epoch : %i, the loss : %f, loss dom : %f' % \
                  (epoch, loss.data, J_dom.data))
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
# theta精确剪应力
h2 = plt.contourf(x_mesh, y_mesh, exact_sigma_theta,  cmap = 'jet', levels = 100)
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h2).ax.set_title(r'$\sigma_{\theta\theta}$')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# rr最小势能预测
h3 = plt.contourf(x_mesh, y_mesh, pred_sigma_rr,  cmap = 'jet', levels = 100)
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h3).ax.set_title(r'$\sigma_{rr}$')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# theta最小势能预测
h4 = plt.contourf(x_mesh, y_mesh, pred_sigma_theta,  cmap = 'jet', levels = 100)
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h4).ax.set_title(r'$\sigma_{\theta\theta}$')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# rr最小势能误差
h5 = plt.contourf(x_mesh, y_mesh, np.abs(pred_sigma_rr - exact_sigma_rr),  cmap = 'jet', levels = 100)
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h5).ax.set_title(r'$\sigma_{rr}$')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# theta最小势能误差
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
plt.legend(['exact', 'DEM'])
plt.show()

plt.plot(r_numpy, exactline_sigma_theta, color = 'r')
plt.scatter(r_numpy[::2], predline_sigma_theta[::2], marker = '*', linestyle='-')
plt.xlabel('r')
plt.ylabel(r'$\sigma_{\theta\theta}$')
plt.legend(['exact', 'DEM'])
plt.show()

# =============================================================================
# plot the loss function
# =============================================================================

loss_array = np.array(loss_array)
loss_dom_array = np.array(loss_dom_array)

plt.plot(loss_array) 
plt.plot(loss_dom_array, linestyle='-.')
plt.legend(['Potential energy', "Strain energy"])
plt.xlabel('Training steps')
plt.ylabel('Loss')
plt.show()
# plt.title('loss evolution')

# plot the displacement of rr, compare the prediction and the exact solution


r_numpy, pred_dis_r = evaluate_dis_line(N_test)# output the prediction of the displacement along the direction in r
U_exact = 1/E*((1-nu)*(a**2*Pi-b**2*Po)/(b**2-a**2)*r_numpy + (1+nu)*(a**2*b**2)*(Pi-Po)/(b**2-a**2)/r_numpy) # the exact solution of the Lame 
plt.plot(r_numpy, U_exact, color = 'r')
plt.scatter(r_numpy[::2], pred_dis_r[::2], marker = '*', linestyle='-')
plt.xlabel('r')
plt.ylabel(r'$U_{r}$')
plt.legend(['exact', 'DEM'])
plt.show()

print('The total relative error of sigma rr is ' + str(error_sigma_rr_array[-1]))
print('The total relative error of sigma theta is ' + str(error_sigma_theta_array[-1]))
    
#%%
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
