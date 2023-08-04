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
nepoch = 50000

N_test = 101
N_bound = 101
N_part = 1000
E = 1000
nu = 0.3
G = E/2/(1+nu)

tol_p = 0.0001
loss_p = 100
epoch_p = 0
cir = [[0.5, 0.5, 0.25, 0.25]]

m = 0
for i in cir:
    m = m + np.pi*i[2]*i[3]
S = a*b - m

D11_mat = E/(1-nu**2)
D22_mat = E/(1-nu**2)
D12_mat = E*nu/(1-nu**2)
D21_mat = E*nu/(1-nu**2)
D33_mat = E/(2*(1+nu))
training_part = 0
order = 2
factor = (4/a/b)**order
def boundary_data_force(Nf):
    '''
    generate the uniform points
    '''
    
    x = np.ones(Nf) * a
    y = np.linspace(0, b, Nf) # y方向N_test个点
    xy = np.stack((x.flatten(), y.flatten()), 1)
    xy_tensor = torch.tensor(xy,  requires_grad=True, device='cuda')
    return xy_tensor


def dom_data_uniform(Nf, cir):
    '''
    generate the uniform points
    '''
    
    x = np.linspace(0, a, Nf)
    y = np.linspace(0, b, Nf) # y方向N_test个点
    x_mesh, y_mesh = np.meshgrid(x, y)
    xy = np.stack((x_mesh.flatten(), y_mesh.flatten()), 1)
    xy_tensor = torch.tensor(xy,  requires_grad=True, device='cuda')

    index0 = []
    #Xf = np.copy(Xf0)
    for i, (x, y) in enumerate(xy_tensor): # determine whether the points is in the circle or not.

        #x, y = x*x0, y*y0
        good = 0
        for cir_info in cir:
            if (circle(cir_info[0], cir_info[1], cir_info[2], cir_info[3], x, y) > 0):
                good = good + 1
        if good == len(cir): 
            index0.append(i)
    Xf0 = xy_tensor[index0] # the domian points without the circle points    
    return Xf0
def circle(x0, y0, ra, rb, xt, yt):
    
    return (xt - x0)**2/ra**2 + (yt - y0)**2/rb**2 - 1
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


model_p = torch.load('./particular_DCEM_nn_%ihole_cir_fix' % len(cir))


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
    dis = factor * ((x-a)*y*(y-b))**order
    dis = dis.unsqueeze(1)
    pred_u = model_p(xy) + dis * model_g(xy) 
    return pred_u

def evaluate_sigma(N_test):# mesh spaced grid
    xy_dom_t = dom_data_uniform(N_test, cir)
    
    fai = pred(xy_dom_t) # Input r and theta to the pred function to get the necessary predition stress function
    
    dfaidxy = grad(fai, xy_dom_t, torch.ones(xy_dom_t.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    dfaidx = dfaidxy[:, 0].unsqueeze(1)
    dfaidy = dfaidxy[:, 1].unsqueeze(1)
    
    dfaidxdxy = grad(dfaidx,  xy_dom_t, torch.ones( xy_dom_t.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    dfaidxdx = dfaidxdxy[:, 0].unsqueeze(1)
    dfaidxdy = dfaidxdxy[:, 1].unsqueeze(1)

    dfaidydxy = grad(dfaidy,  xy_dom_t, torch.ones( xy_dom_t.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    dfaidydx = dfaidydxy[:, 0].unsqueeze(1)
    dfaidydy = dfaidydxy[:, 1].unsqueeze(1)

    # 通过应力函数计算应力
    sigma_x = dfaidydy
    sigma_y = dfaidxdx
    sigma_xy = -dfaidxdy
    dom = xy_dom_t.data.cpu().numpy()
    pred_sigma_x = sigma_x.data.cpu().numpy() # change the type of tensor to the array for the use of plotting
    pred_sigma_y = sigma_y.data.cpu().numpy()
    pred_sigma_xy = sigma_xy.data.cpu().numpy()

    pred_mise = np.sqrt(0.5*((pred_sigma_x-pred_sigma_y)**2+pred_sigma_x**2+pred_sigma_y**2+6*pred_sigma_xy**2))

    return dom, pred_sigma_x, pred_sigma_y, pred_sigma_xy, pred_mise

def test_particular_with_pred(N_test):
    equal_line_x = np.linspace(0, a, N_part)
    equal_line_y = np.linspace(0, b, N_part)
    
    up_boundary = np.stack([equal_line_x, b*np.ones(len(equal_line_x))], 1) 
    right_boundary = np.stack([a*np.ones(len(equal_line_y)), equal_line_y], 1)
    down_boundary = np.stack([equal_line_x, 0*np.ones(len(equal_line_x))], 1)
    
    
    tx_right = P*np.sin(np.pi/b*right_boundary[:,1])
    tx_right = tx_right[:,np.newaxis]
    ty_right = 0*np.ones(len(right_boundary))
    ty_right = ty_right[:,np.newaxis]   
    
    # tx_right = P*np.ones(len(right_boundary[:,1]))
    # tx_right = tx_right[:,np.newaxis]
    # ty_right = 0*np.ones(len(right_boundary))
    # ty_right = ty_right[:,np.newaxis]   
    
    tx_up = 0*np.ones(len(up_boundary))
    tx_up = tx_up[:,np.newaxis] 
    ty_up = 0*np.ones(len(up_boundary))
    ty_up = ty_up[:,np.newaxis]     
    
    tx_down = 0*np.ones(len(down_boundary))
    tx_down = tx_down[:,np.newaxis] 
    ty_down = 0*np.ones(len(down_boundary))
    ty_down = ty_down[:,np.newaxis]         
    
    Xb_x_right = torch.tensor(right_boundary,  requires_grad=True, device='cuda') 
    
    Xb_y_right = torch.tensor(right_boundary,  requires_grad=True, device='cuda') 
    
    Xb_x_up = torch.tensor(up_boundary,  requires_grad=True, device='cuda')
    
    Xb_y_up = torch.tensor(up_boundary,  requires_grad=True, device='cuda')    

    Xb_x_down = torch.tensor(down_boundary,  requires_grad=True, device='cuda')
    
    Xb_y_down = torch.tensor(down_boundary,  requires_grad=True, device='cuda')  
    
    fai_y_right = pred(Xb_y_right) # predict the boundary condition
    dfaidxy_y_right = grad(fai_y_right, Xb_y_right, torch.ones(Xb_y_right.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]            
    dfaidx_y_right = dfaidxy_y_right[:, 0].unsqueeze(1)
    #dfaidy_y_right = dfaidxy_y_right[:, 1].unsqueeze(1)             
    dfaidxdxy_y_right = grad(dfaidx_y_right, Xb_y_right, torch.ones(Xb_y_right.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    dfaidxdy_y_right = dfaidxdxy_y_right[:, 1].unsqueeze(1)


    fai_x_up = pred(Xb_x_up) # predict the boundary condition
    dfaidxy_x_up = grad(fai_x_up, Xb_x_up, torch.ones(Xb_x_up.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]            
    dfaidx_x_up = dfaidxy_x_up[:, 0].unsqueeze(1)
    #dfaidy_x_up = dfaidxy_x_up[:, 1].unsqueeze(1)             
    dfaidxdxy_x_up = grad(dfaidx_x_up, Xb_x_up, torch.ones(Xb_x_up.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    dfaidxdy_x_up = dfaidxdxy_x_up[:, 1].unsqueeze(1)

    fai_y_up = pred(Xb_y_up) # predict the boundary condition
    dfaidxy_y_up = grad(fai_y_up, Xb_y_up, torch.ones(Xb_y_up.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]            
    dfaidx_y_up = dfaidxy_y_up[:, 0].unsqueeze(1)
    #dfaidy_y_up = dfaidxy_y_up[:, 1].unsqueeze(1)             
    dfaidxdxy_y_up = grad(dfaidx_y_up, Xb_y_up, torch.ones(Xb_y_up.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    dfaidxdx_y_up = dfaidxdxy_y_up[:, 0].unsqueeze(1)


    fai_x_right = pred(Xb_x_right) # predict the boundary condition
    dfaidxy_x_right = grad(fai_x_right, Xb_x_right, torch.ones(Xb_x_right.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]            
    #dfaidx_x_right = dfaidxy_x_right[:, 0].unsqueeze(1)
    dfaidy_x_right = dfaidxy_x_right[:, 1].unsqueeze(1)             
    dfaidydxy_x_right = grad(dfaidy_x_right, Xb_x_right, torch.ones(Xb_x_right.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    dfaidydy_x_right = dfaidydxy_x_right[:, 1].unsqueeze(1)

    fai_x_down = pred(Xb_x_down) # predict the boundary condition
    dfaidxy_x_down = grad(fai_x_down, Xb_x_down, torch.ones(Xb_x_down.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]            
    dfaidx_x_down = dfaidxy_x_down[:, 0].unsqueeze(1)
    #dfaidy_x_down = dfaidxy_x_down[:, 1].unsqueeze(1)             
    dfaidxdxy_x_down = grad(dfaidx_x_down, Xb_x_down, torch.ones(Xb_x_down.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    dfaidxdy_x_down = dfaidxdxy_x_down[:, 1].unsqueeze(1)

    fai_y_down = pred(Xb_y_down) # predict the boundary condition
    dfaidxy_y_down = grad(fai_y_down, Xb_y_down, torch.ones(Xb_y_down.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]            
    dfaidx_y_down = dfaidxy_y_down[:, 0].unsqueeze(1)
    #dfaidy_y_down = dfaidxy_y_down[:, 1].unsqueeze(1)             
    dfaidxdxy_y_down = grad(dfaidx_y_down, Xb_y_down, torch.ones(Xb_y_down.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    dfaidxdx_y_down = dfaidxdxy_y_down[:, 0].unsqueeze(1)

    
    pred_sigma_yy_up = dfaidxdx_y_up.data.cpu().numpy()
    pred_sigma_xy_up = dfaidxdy_x_up.data.cpu().numpy()
    
    pred_sigma_xx_right = dfaidydy_x_right.data.cpu().numpy()
    pred_sigma_xy_right = dfaidxdy_y_right.data.cpu().numpy()
    
    pred_sigma_yy_down = dfaidxdx_y_down.data.cpu().numpy()
    pred_sigma_xy_down = dfaidxdy_x_down.data.cpu().numpy()
    return equal_line_x, pred_sigma_xy_up, tx_up,  pred_sigma_yy_up, ty_up, pred_sigma_yy_down, ty_down, pred_sigma_xy_down, tx_down,  equal_line_y,  pred_sigma_xx_right, tx_right, pred_sigma_xy_right, ty_right

criterion = torch.nn.MSELoss()
model_g = FNN(2, 20, 1).cuda()
optim = torch.optim.Adam(params=model_g.parameters(), lr= 0.01)
scheduler = MultiStepLR(optim, milestones=[10000, 20000], gamma = 0.5)
loss_array = []
loss_dom_array = []
loss_ex_array = []
error_sigma_x_array = []
error_sigma_y_array = []
error_sigma_xy_array = []
nepoch = int(nepoch)
start = time.time()
xy_dom_t = dom_data_uniform(N_test, cir)
boundary_t = boundary_data_force(N_bound)
for epoch in range(nepoch):
    if epoch == nepoch-1:
        end = time.time()
        consume_time = end-start
        print('time is %f' % consume_time)
    
    def closure():  
        fai = pred(xy_dom_t) # Input r and theta to the pred function to get the necessary predition stress function
        
        dfaidxy = grad(fai, xy_dom_t, torch.ones(xy_dom_t.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        dfaidx = dfaidxy[:, 0].unsqueeze(1)
        dfaidy = dfaidxy[:, 1].unsqueeze(1)
        
        dfaidxdxy = grad(dfaidx,  xy_dom_t, torch.ones( xy_dom_t.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        dfaidxdx = dfaidxdxy[:, 0].unsqueeze(1)
        dfaidxdy = dfaidxdxy[:, 1].unsqueeze(1)
    
        dfaidydxy = grad(dfaidy,  xy_dom_t, torch.ones( xy_dom_t.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        dfaidydx = dfaidydxy[:, 0].unsqueeze(1)
        dfaidydy = dfaidydxy[:, 1].unsqueeze(1)
    
        # 通过应力函数计算应力
        sigma_x = dfaidydy
        sigma_y = dfaidxdx
        sigma_xy = -dfaidxdy

        # 计算应变
        epsilon_x = 1/E*(sigma_x - nu*sigma_y)
        epsilon_y = 1/E*(sigma_y - nu*sigma_x)
        epsilon_xy = 1/G*sigma_xy

        J_dom_density = 0.5*(sigma_x*epsilon_x + sigma_y*epsilon_y + sigma_xy*epsilon_xy) # 计算余能
        J_dom =  torch.mean(J_dom_density) * (S)
      
        loss =  J_dom 
        optim.zero_grad()
        loss.backward(retain_graph=True)
        loss_array.append(loss.data.cpu())
        loss_dom_array.append(J_dom.data.cpu())
        if epoch%10==0:
            print(' epoch : %i, the loss : %f, loss dom : %f' % \
                  (epoch, loss.data, J_dom))
        return loss
    optim.step(closure)
    scheduler.step()
    
    
#%%
def write_vtk_v2p(filename, dom, S11, S12, S22, mise): #已经将输出的感兴趣场进行了分类VTK导出
    xx = np.ascontiguousarray(dom[:, 0])
    yy = np.ascontiguousarray(dom[:, 1])
    zz = np.ascontiguousarray(dom[:, 1])*0 # 点的VTK
    S11 = S11.flatten()
    S12 = S12.flatten()
    S22 = S22.flatten()
    mise = mise.flatten()
    pointsToVTK(filename, xx, yy, zz, data={"S11": S11, "S12": S12, "S22": S22, "Mises": mise})
        

dom, pred_sigma_x, pred_sigma_y, pred_sigma_xy, pred_mise = evaluate_sigma(N_test)

write_vtk_v2p("./output/DCEM_%ihole_NN_cir_fixed" % len(cir),  dom, pred_sigma_x, pred_sigma_xy, pred_sigma_y, pred_mise)


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

  
    
equal_line_x, pred_sigma_xy_up, tx_up,  pred_sigma_yy_up, ty_up, pred_sigma_yy_down, ty_down,\
    pred_sigma_xy_down, tx_down,  equal_line_y,  pred_sigma_xx_right, tx_right, pred_sigma_xy_right, ty_right = test_particular_with_pred(N_test)

internal = 4
plt.plot(equal_line_x, tx_up)
plt.plot(equal_line_x, ty_up)
plt.scatter(equal_line_x[::internal], pred_sigma_xy_up[::internal], marker = '*', linestyle='-')
plt.scatter(equal_line_x[::internal], pred_sigma_yy_up[::internal], marker = '+', linestyle='-')
plt.xlabel('x')
plt.ylabel('sigma_up')
plt.legend(['exact_tx_up', 'exact_ty_up', 'pred_xy_up',  'pred_yy_up'])
plt.title('up')
plt.show()

plt.plot(equal_line_y, tx_right)
plt.plot(equal_line_y, ty_right)
plt.scatter(equal_line_y[::internal], pred_sigma_xx_right[::internal], marker = '*', linestyle='-')
plt.scatter(equal_line_y[::internal], pred_sigma_xy_right[::internal], marker = '+', linestyle='-')
plt.xlabel('x')
plt.ylabel('sigma_right')
plt.legend(['exact_tx_right', 'exact_ty_right', 'pred_xx_right', 'pred_xy_right'])
plt.title('right')
plt.show()    
    
plt.plot(equal_line_x, tx_down)
plt.plot(equal_line_x, ty_down)
plt.scatter(equal_line_x[::internal], pred_sigma_xy_down[::internal], marker = '*', linestyle='-')
plt.scatter(equal_line_x[::internal], pred_sigma_yy_down[::internal], marker = '+', linestyle='-')
plt.xlabel('x')
plt.ylabel('sigma_down')
plt.legend(['exact_tx_down', 'exact_ty_down', 'pred_xy_down',  'pred_yy_down'])
plt.title('down')
plt.show()

# =============================================================================
# SIGMA mise
# =============================================================================
plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(5.8, 4.8)) 
surf = ax.scatter(dom[:, 0], dom[:, 1], c = pred_mise, cmap=cm.rainbow)# vmin=0, vmax=0.14, 
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