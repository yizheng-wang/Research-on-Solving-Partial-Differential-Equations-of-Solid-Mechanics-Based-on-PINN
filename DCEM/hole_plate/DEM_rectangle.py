# DCM by gaussian integration

import numpy as np
import torch
import matplotlib.pyplot as plt
from  torch.autograd import grad
import time
import matplotlib as mpl
import numpy as np
from pyevtk.hl import pointsToVTK
from pyevtk.hl import gridToVTK
from torch.optim.lr_scheduler import MultiStepLR
from matplotlib import cm

torch.set_default_tensor_type(torch.DoubleTensor) # 将tensor的类型变成默认的double
torch.set_default_tensor_type(torch.cuda.DoubleTensor) # 将cuda的tensor的类型变成默认的double
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
setup_seed(0)

a = 1.
b = 1.
P = 100
nepoch = 20000

N_test = 101
N_bound = 101
E = 1000
nu = 0.3
G = E/2/(1+nu)

D11_mat = E/(1-nu**2)
D22_mat = E/(1-nu**2)
D12_mat = E*nu/(1-nu**2)
D21_mat = E*nu/(1-nu**2)
D33_mat = E/(2*(1+nu))

def boundary_data_force(Nf):
    '''
    generate the uniform points
    '''
    
    x = np.ones(Nf) * a
    y = np.linspace(0, b, Nf) # y方向N_test个点
    xy = np.stack((x.flatten(), y.flatten()), 1)
    xy_tensor = torch.tensor(xy,  requires_grad=True, device='cuda')
    return xy_tensor

def dom_data_uniform(Nf):
    '''
    generate the uniform points
    '''
    
    x = np.linspace(0, a, Nf)
    y = np.linspace(0, b, Nf) # y方向N_test个点
    x_mesh, y_mesh = np.meshgrid(x, y)
    xy = np.stack((x_mesh.flatten(), y_mesh.flatten()), 1)
    xy_tensor = torch.tensor(xy,  requires_grad=True, device='cuda')
    
    return xy_tensor

# NN architecture
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
        
        self.a = torch.nn.Parameter(torch.Tensor([0.1]))
        
        self.n = 1/self.a.data.cuda()


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

        y1 = torch.tanh(self.n*self.a*self.linear1(x))
        y2 = torch.tanh(self.n*self.a*self.linear2(y1))
        y3 = torch.tanh(self.n*self.a*self.linear3(y2))
        y = self.n*self.a*self.linear4(y3)
        return y
def simpson_int_1D(y, x,  nx = N_test):
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
    hx = torch.abs(x[0] - x[-1])/(nx-1) # 只有在右侧有外力势能
    y = y.flatten()
    result = torch.sum(weight*y)*hx/3
    return result  

def simpson_int_2D(y, x,  nx = N_test, ny = N_test):
    '''
    Simpson integration for 2D

    Parameters
    ----------
    y : tensor
        The value of the x.
    x : tensor
        Coordinate of the input.
    nx : int, optional
        The grid node number of x axis. The default is N_test.
    ny : int, optional
        The grid node number of y axis. The default is N_test.
        
    Returns
    -------
    result : tensor
        the result of the integration.

    '''
    weightx = [4, 2] * int((nx-1)/2)
    weightx = [1] + weightx
    weightx[-1] = weightx[-1]-1
    weightx = np.array(weightx)
    
    weighty = [4, 2] * int((ny-1)/2)
    weighty = [1] + weighty
    weighty[-1] = weighty[-1]-1
    weighty = np.array(weighty)

    weightx = weightx.reshape(-1,1)
    weighty = weighty.reshape(1,-1)
    weight = weightx*weighty
    weight = weight.flatten()
    weight = torch.tensor(weight, device='cuda')
    hx = a/(nx-1)
    hy = b/(ny-1)
    y = y.flatten()
    result = torch.sum(weight*y)*hx*hy/9
    return result    
    
# pred Airy stress function  
def pred(xy):
    '''
    

    Parameters
    ----------
    xy : tensor 
        the coordinate of the input tensor.

    Returns
    the prediction of dis

    '''
    x = xy[:, 0]
    y = xy[:, 1]
    dis = x.unsqueeze(1)
    pred_u = dis * model_g(xy) 
    return pred_u

def evaluate_sigma(N_test):# mesh spaced grid
    xy_dom_t = dom_data_uniform(N_test)
    
    u_pred = pred(xy_dom_t) # Input r and theta to the pred function to get the necessary predition stress function
    duxdxy = grad(u_pred[:, 0].unsqueeze(1), xy_dom_t, torch.ones(xy_dom_t.size()[0], 1, device="cuda"), create_graph=True, retain_graph=True)[0]
    duydxy = grad(u_pred[:, 1].unsqueeze(1), xy_dom_t, torch.ones(xy_dom_t.size()[0], 1, device="cuda"), create_graph=True, retain_graph=True)[0]
    duxdx = duxdxy[:, 0].unsqueeze(1)
    duxdy = duxdxy[:, 1].unsqueeze(1)
    
    duydx = duydxy[:, 0].unsqueeze(1)
    duydy = duydxy[:, 1].unsqueeze(1)
    
    exx_pred = duxdx
    eyy_pred = duydy
    e2xy_pred = duxdy + duydx 

    sxx_pred = D11_mat * exx_pred + D12_mat * eyy_pred
    syy_pred = D12_mat * exx_pred + D22_mat * eyy_pred
    sxy_pred = D33_mat * e2xy_pred

    dom = xy_dom_t.data.cpu().numpy()
    u_x = u_pred[:, 0].unsqueeze(1).data.cpu().numpy()
    u_y = u_pred[:, 1].unsqueeze(1).data.cpu().numpy()
    pred_sigma_x = sxx_pred.data.cpu().numpy() # change the type of tensor to the array for the use of plotting
    pred_sigma_y = syy_pred.data.cpu().numpy()
    pred_sigma_xy = sxy_pred.data.cpu().numpy()
    pred_u = np.sqrt(u_x**2 + u_y**2)
    pred_mise = np.sqrt(0.5*((pred_sigma_x-pred_sigma_y)**2+pred_sigma_x**2+pred_sigma_y**2+6*pred_sigma_xy**2))
    return dom, pred_sigma_x, pred_sigma_y, pred_sigma_xy, pred_mise, u_x, u_y, pred_u

model_g = FNN(2, 20, 2).cuda()
optim = torch.optim.Adam(params=model_g.parameters(), lr= 0.001)
scheduler = MultiStepLR(optim, milestones=[10000, 20000], gamma = 0.5)
loss_array = []
loss_dom_array = []
loss_ex_array = []
error_sigma_x_array = []
error_sigma_y_array = []
error_sigma_xy_array = []
nepoch = int(nepoch)
start = time.time()
xy_dom_t = dom_data_uniform(N_test)
boundary_t = boundary_data_force(N_bound)
for epoch in range(nepoch):
    if epoch == nepoch-1:
        end = time.time()
        consume_time = end-start
        print('time is %f' % consume_time)
    
    def closure():  
        u_pred = pred(xy_dom_t) # Input r and theta to the pred function to get the necessary predition stress function
        duxdxy = grad(u_pred[:, 0].unsqueeze(1), xy_dom_t, torch.ones(xy_dom_t.size()[0], 1, device="cuda"), create_graph=True, retain_graph=True)[0]
        duydxy = grad(u_pred[:, 1].unsqueeze(1), xy_dom_t, torch.ones(xy_dom_t.size()[0], 1, device="cuda"), create_graph=True, retain_graph=True)[0]
        duxdx = duxdxy[:, 0].unsqueeze(1)
        duxdy = duxdxy[:, 1].unsqueeze(1)
        
        duydx = duydxy[:, 0].unsqueeze(1)
        duydy = duydxy[:, 1].unsqueeze(1)
        
        exx_pred = duxdx
        eyy_pred = duydy
        e2xy_pred = duxdy + duydx 

        sxx_pred = D11_mat * exx_pred + D12_mat * eyy_pred
        syy_pred = D12_mat * exx_pred + D22_mat * eyy_pred
        sxy_pred = D33_mat * e2xy_pred

        J_dom_density = 0.5*(sxx_pred*exx_pred + syy_pred*eyy_pred + sxy_pred*e2xy_pred) # 计算应变能
        J_dom =  simpson_int_2D(J_dom_density, xy_dom_t,  nx = N_test, ny = N_test)

        # for external work
        u_pred_bound = pred(boundary_t)
        t = torch.zeros_like(u_pred_bound, device='cuda')
        t[:, 0] = P * torch.sin(boundary_t[:, 1]/b*torch.tensor(np.pi, device='cuda')) 
        ex_density = torch.sum(u_pred_bound * t, axis=1)
        J_ex = simpson_int_1D(ex_density, boundary_t[:,1],  nx = N_test)
        
        loss = J_dom - J_ex
        optim.zero_grad()
        loss.backward(retain_graph=True)
        loss_array.append(loss.data.cpu())
        loss_dom_array.append(J_dom.data.cpu())
        loss_ex_array.append(J_ex.data.cpu())
        if epoch%10==0:
            print(' epoch : %i, the loss : %f, loss dom : %f, loss ex : %f' % \
                  (epoch, loss.data, J_dom, J_ex))
        return loss
    optim.step(closure)
    scheduler.step()
    
    
#%%
def write_vtk_v2p(filename, dom, S11, S12, S22, von_mises,  u_x, u_y, pred_u): #已经将输出的感兴趣场进行了分类VTK导出
    xx = np.ascontiguousarray(dom[:, 0])
    yy = np.ascontiguousarray(dom[:, 1])
    zz = np.ascontiguousarray(dom[:, 1])*0 # 点的VTK
    S11 = S11.flatten()
    S12 = S12.flatten()
    S22 = S22.flatten()
    von_mises = von_mises.flatten()
    u_x = u_x.flatten()
    u_y = u_y.flatten()
    pred_u = pred_u.flatten()
    pointsToVTK(filename, xx, yy, zz, data={"S11": S11, "S12": S12, "S22": S22, "Von_mise": von_mises, "U_x": u_x, "U_y": u_y, "U_mag": pred_u})
        
def write_vtk_v2(filename, dom, S11, S12, S22, von_mises, u_x, u_y, pred_u):
    xx = np.ascontiguousarray(dom[:, 0]).reshape(N_test, N_test, 1)
    yy = np.ascontiguousarray(dom[:, 1]).reshape(N_test, N_test, 1)
    zz = 0*np.ascontiguousarray(dom[:, 1]).reshape(N_test, N_test, 1)
    gridToVTK(filename, xx, yy, zz, pointData={"S11": S11.reshape(N_test, N_test, 1), "S12": S12.reshape(N_test, N_test, 1), "S22": S22.reshape(N_test, N_test, 1), "Von_mise": von_mises.reshape(N_test, N_test, 1),\
                                               "U_x": u_x.reshape(N_test, N_test, 1), "U_y": u_y.reshape(N_test, N_test, 1), "U_mag": pred_u.reshape(N_test, N_test, 1) })
    # gridToVTK(filename, xx, yy, zz, pointData={"displacement": U})  
    
dom, pred_sigma_x, pred_sigma_y, pred_sigma_xy, pred_mise,  pred_u_x, pred_u_y, pred_u = evaluate_sigma(N_test)

write_vtk_v2("./output/DEM_rectangle",  dom, pred_sigma_x, pred_sigma_xy, pred_sigma_y, pred_mise, pred_u_x, pred_u_y, pred_u)


# =============================================================================
# SIGMA XX
# =============================================================================
plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(5.8, 4.8)) 
surf = ax.scatter(dom[:, 0], dom[:, 1], c = pred_sigma_x, cmap=cm.rainbow)# vmin=0, vmax=0.14, 
#cbar = fig.colorbar(ax)
cb = fig.colorbar(surf)
cb.ax.locator_params(nbins=7)
cb.ax.tick_params(labelsize=16)
#cb.set_label(label =r'$\sigma_xx (MPa)$', fontsize=16)
#cb.set_label(fontsize=16)
ax.axis('equal')
ax.set_xlabel('X Position (mm)', fontsize=18)
ax.set_ylabel('Y Position (mm)', fontsize=18)
for tick in ax.get_xticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
for tick in ax.get_yticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
#plt.savefig('Flat-U-Energy-10-10.png', dpi=600, transparent=True)
plt.show()
    
    
# =============================================================================
# SIGMA YY
# =============================================================================
plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(5.8, 4.8)) 
surf = ax.scatter(dom[:, 0], dom[:, 1], c = pred_sigma_y, cmap=cm.rainbow)# vmin=0, vmax=0.14, 
#cbar = fig.colorbar(ax)
cb = fig.colorbar(surf)
cb.ax.locator_params(nbins=7)
cb.ax.tick_params(labelsize=16)
#cb.set_label(label =r'$\sigma_xx (MPa)$', fontsize=16)
#cb.set_label(fontsize=16)
ax.axis('equal')
ax.set_xlabel('X Position (mm)', fontsize=18)
ax.set_ylabel('Y Position (mm)', fontsize=18)
for tick in ax.get_xticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
for tick in ax.get_yticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
#plt.savefig('Flat-U-Energy-10-10.png', dpi=600, transparent=True)
plt.show()

# =============================================================================
# SIGMA XY
# =============================================================================
plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(5.8, 4.8)) 
surf = ax.scatter(dom[:, 0], dom[:, 1], c = pred_sigma_xy, cmap=cm.rainbow)# vmin=0, vmax=0.14, 
#cbar = fig.colorbar(ax)
cb = fig.colorbar(surf)
cb.ax.locator_params(nbins=7)
cb.ax.tick_params(labelsize=16)
#cb.set_label(label =r'$\sigma_xx (MPa)$', fontsize=16)
#cb.set_label(fontsize=16)
ax.axis('equal')
ax.set_xlabel('X Position (mm)', fontsize=18)
ax.set_ylabel('Y Position (mm)', fontsize=18)
for tick in ax.get_xticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
for tick in ax.get_yticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
#plt.savefig('Flat-U-Energy-10-10.png', dpi=600, transparent=True)
plt.show()

# =============================================================================
# u_X
# =============================================================================
# Then, "ALWAYS use sans-serif fonts"
plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(5.8, 4.8)) 
surf = ax.scatter(dom[:, 0], dom[:, 1], c = pred_u_x, cmap=cm.rainbow)# vmin=0, vmax=0.14, 
#cbar = fig.colorbar(ax)
cb = fig.colorbar(surf)
cb.ax.locator_params(nbins=7)
cb.ax.tick_params(labelsize=16)
#cb.set_label(label =r'$\sigma_xx (MPa)$', fontsize=16)
#cb.set_label(fontsize=16)
ax.axis('equal')
ax.set_xlabel('X Position (mm)', fontsize=18)
ax.set_ylabel('Y Position (mm)', fontsize=18)
for tick in ax.get_xticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
for tick in ax.get_yticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
#plt.savefig('Flat-U-Energy-10-10.png', dpi=600, transparent=True)
plt.show()
    
    
# =============================================================================
# u_Y
# =============================================================================
# Then, "ALWAYS use sans-serif fonts"
plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(5.8, 4.8)) 
surf = ax.scatter(dom[:, 0], dom[:, 1], c = pred_u_y, cmap=cm.rainbow)# vmin=0, vmax=0.14, 
#cbar = fig.colorbar(ax)
cb = fig.colorbar(surf)
cb.ax.locator_params(nbins=7)
cb.ax.tick_params(labelsize=16)
#cb.set_label(label =r'$\sigma_xx (MPa)$', fontsize=16)
#cb.set_label(fontsize=16)
ax.axis('equal')
ax.set_xlabel('X Position (mm)', fontsize=18)
ax.set_ylabel('Y Position (mm)', fontsize=18)
for tick in ax.get_xticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
for tick in ax.get_yticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
#plt.savefig('Flat-U-Energy-10-10.png', dpi=600, transparent=True)
plt.show()    
    
    
    
