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
import cmath
FEM_node = np.load("node_coordinate_abaqus_1hole_all_pressure.npy")
node_stress_abaqus_rectangle = np.load("node_stress_abaqus_1hole_all_pressure.npy")

FEM_mise = node_stress_abaqus_rectangle[:,0]
FEM_stress11 = node_stress_abaqus_rectangle[:,1]
FEM_stress22 = node_stress_abaqus_rectangle[:,2]
FEM_stress12 = node_stress_abaqus_rectangle[:,3]

bound_len = 0.0
index_without_boundary = np.where((FEM_node[:, 1] > bound_len)&(FEM_node[:, 0] > bound_len)&(FEM_node[:, 1] < 1-bound_len)&(FEM_node[:, 0] < 1-bound_len)\
                                  &((FEM_node[:, 0]-0.5)**2+(FEM_node[:, 1]-0.5)**2 > 0.25**2*(1+bound_len)))
FEM_node = FEM_node[index_without_boundary, :][0] # 找到左边界条件的坐标点
FEM_mise = FEM_mise[index_without_boundary] # 找到左边界条件的坐标点
FEM_stress11 = FEM_stress11[index_without_boundary] # 找到左边界条件的坐标点
FEM_stress22 = FEM_stress22[index_without_boundary] # 找到左边界条件的坐标点
FEM_stress12 = FEM_stress12[index_without_boundary] # 找到左边界条件的坐标点


torch.set_default_tensor_type(torch.DoubleTensor) # 将tensor的类型变成默认的double
torch.set_default_tensor_type(torch.cuda.DoubleTensor) # 将cuda的tensor的类型变成默认的double
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
setup_seed(2023)
scaler = 100
dom_dmsh = np.load('1hole_dmsh.npy')
a = 1.
b = 1.
P = 100
P_left = 110
nepoch = 6000

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

Nf = 10000
Nf_aug = 30000
ratio = 0.2625

m = 0
n = 0
for i in cir:
    m = m + np.pi*i[2]*i[3]*(1+ratio)
    n = n + np.pi*i[2]*i[3]*(ratio)
S_n = torch.tensor(a*b-m, device='cuda')
S_a = torch.tensor(n, device='cuda')

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

def train_data(Nf, Nf_aug, cir, ratio = 0.5625):
    """
    

    Parameters
    ----------
    Nb : int
        the number of the boundary points
    Nf : int
        the number of the domain points
    Nf_aug : int
        the number of the domain points (augmented)
    cir : list
        [[x, y, ra, rb], [x, y, ra, rb], ...]
    ratio: int
        the augmented number points on the position
    Returns
    -------
    Xl : tensor [:,2]
        the boundary points on the left
    Xd : tensor [:,2]
        the boundary points on the bottom
    Xr : tensor [:,2]
        the boundary points on the right
    Xu : tensor [:,2]
        the boundary points on the up
    Xf0 : tensor [:, 2]
        the domain points
    Xf1 : tensor [:, 2]
        the domain points (more dense)


    """
      
    Xf_x = np.random.rand(Nf, 1)*a # domain points
    Xf_y = np.random.rand(Nf, 1)*b  # domain points
    Xf = np.hstack([Xf_x, Xf_y])
    index0 = []
    #Xf = np.copy(Xf0)
    for i, (x, y) in enumerate(Xf): # determine whether the points is in the circle or not.

        #x, y = x*x0, y*y0
        good = 0
        for cir_info in cir:
            if (circle(cir_info[0], cir_info[1], cir_info[2], cir_info[3], x, y) > ratio):
                good = good + 1
        if good == len(cir): 
            index0.append(i)
    Xf0 = Xf[index0] # the domian points without the circle points
    
    Xf_x = np.random.rand(Nf_aug, 1)*a  # domain points
    Xf_y = np.random.rand(Nf_aug, 1)*b  # domain points
    Xf = np.hstack([Xf_x, Xf_y])
    index_a = []

    
    for i, (x, y) in enumerate(Xf):
        for cir_info in cir:
            if (circle(cir_info[0], cir_info[1], cir_info[2], cir_info[3], x, y) > 0 and circle(cir_info[0], cir_info[1], cir_info[2], cir_info[3], x, y) < ratio):
                index_a.append(i)
    
    Xf1 = Xf[index_a]
    
    Xf0 = torch.tensor(Xf0, dtype=torch.float64, requires_grad=True)
    Xf1 = torch.tensor(Xf1, dtype=torch.float64, requires_grad=True)
    
    return  Xf0, Xf1
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
            if (circle(cir_info[0], cir_info[1], cir_info[2], cir_info[3], x, y) >= 0):
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

class RBF(torch.nn.Module):
    def __init__(self, N_c):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(RBF, self).__init__()
        x = np.linspace(0-a*0.1, a*1.1, N_c)
        y = np.linspace(0-b*0.1, b*1.1, N_c) # y方向N_test个点
        x_mesh, y_mesh = np.meshgrid(x, y)
        Xc = np.stack((x_mesh.flatten(), y_mesh.flatten()), 1)
        self.Xc = torch.tensor(Xc, dtype=torch.float64)
        
        self.a = torch.nn.Parameter(torch.ones(1, N_c**2)*0)
        self.b = torch.nn.Parameter(torch.ones(1, N_c**2)*3)
        
    def forward(self, xy):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = xy[:, 0].unsqueeze(1)
        y = xy[:, 1].unsqueeze(1)
        Xc_x = self.Xc[:, 0].unsqueeze(0)
        Xc_y = self.Xc[:, 1].unsqueeze(0)
        gama = (x-Xc_x)**2 + (y-Xc_y)**2
        y1 = (self.b)**2*gama
        y2 = torch.exp(-y1) 
        y3 = 1/len(self.Xc)**0.5 * y2* self.a
        y = torch.sum(y3, axis=1).unsqueeze(1)
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


model_p = torch.load('./particular_DCEM_nn_%ihole_cir_fix_all_pressure' % len(cir))


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
    dis = factor * (x*(x-a)*y*(y-b))**order
    dis = dis.unsqueeze(1)
    pred_u = model_p(xy) + dis * model_g(xy) 
    return pred_u

def evaluate_sigma(N_test):# mesh spaced grid
    xy_dom_t = dom_data_uniform(N_test, cir)
# add cir boundary points
    Nb = N_test *10
    boundary_point = np.ones([len(cir)*Nb, 2])
    theta = np.linspace(0, 2*np.pi, Nb)
    for index, eve_cir in enumerate(cir):
      x = eve_cir[0] + np.cos(theta) * (eve_cir[2]**2)**0.5   # the circle is default.
      y = eve_cir[1] + np.sin(theta) * (eve_cir[2]**2)**0.5
      boundary_point[Nb*index: Nb*(index+1), 0] = x
      boundary_point[Nb*index: Nb*(index+1), 1] = y
    boundary_point = torch.tensor(boundary_point, dtype=torch.float64, requires_grad=True)
    
    xy_dom_t = torch.cat([boundary_point, xy_dom_t], 0)
    
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
    sigma_x = scaler*dfaidydy
    sigma_y = scaler*dfaidxdx
    sigma_xy = -scaler*dfaidxdy
    dom = xy_dom_t.data.cpu().numpy()
    pred_sigma_x = sigma_x.data.cpu().numpy() # change the type of tensor to the array for the use of plotting
    pred_sigma_y = sigma_y.data.cpu().numpy()
    pred_sigma_xy = sigma_xy.data.cpu().numpy()

    pred_mise = np.sqrt(0.5*((pred_sigma_x-pred_sigma_y)**2+pred_sigma_x**2+pred_sigma_y**2+6*pred_sigma_xy**2))

    return dom, pred_sigma_x, pred_sigma_y, pred_sigma_xy, pred_mise
def plot_pos(Xf0, Xf1):

    Xf0 = Xf0.cpu().detach().numpy()
    Xf1 = Xf1.cpu().detach().numpy()
    
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(figsize=(5.8, 4.8)) 
    plt.scatter(Xf0[:,0], Xf0[:,1], s=1, c='b', marker='x')
    plt.scatter(Xf1[:,0], Xf1[:,1], s=1, c='r', marker='x')  

    ax.axis('equal')
    ax.set_xlabel('X Position (mm)', fontsize=18)
    ax.set_ylabel('Y Position (mm)', fontsize=18)
    for tick in ax.get_xticklabels():
        #tick.set_fontname('Times New Roman')
        tick.set_fontsize(16)
    for tick in ax.get_yticklabels():
        #tick.set_fontname('Times New Roman')
        tick.set_fontsize(16)
    #plt.savefig('CH-2hole-Sample.png', dpi=600, transparent=True)
    plt.show()
    
    
def test_particular_with_pred(N_test):
    equal_line_x = np.linspace(0, a, N_part)
    equal_line_y = np.linspace(0, b, N_part)
    
    up_boundary = np.stack([equal_line_x, b*np.ones(len(equal_line_x))], 1) 
    right_boundary = np.stack([a*np.ones(len(equal_line_y)), equal_line_y], 1)
    left_boundary = np.stack([0*np.ones(len(equal_line_y)), equal_line_y], 1)
    down_boundary = np.stack([equal_line_x, 0*np.ones(len(equal_line_x))], 1)
    
    
    tx_right = P*np.sin(np.pi/b*right_boundary[:,1])
    tx_right = tx_right[:,np.newaxis]
    ty_right = 0*np.ones(len(right_boundary))
    ty_right = ty_right[:,np.newaxis]   
    
    tx_left = -P_left*np.sin(np.pi/b*right_boundary[:,1])
    tx_left = tx_left[:,np.newaxis]
    ty_left = 0*np.ones(len(left_boundary))
    ty_left = ty_left[:,np.newaxis]  
    
    tx_up = 0*np.ones(len(up_boundary))
    tx_up = tx_up[:,np.newaxis] 
    ty_up = P*np.sin(np.pi/a*up_boundary[:,0])
    ty_up = ty_up[:,np.newaxis]     
    
    tx_down = 0*np.ones(len(down_boundary))
    tx_down = tx_down[:,np.newaxis] 
    ty_down = -P*np.sin(np.pi/a*down_boundary[:,0])
    ty_down = ty_down[:,np.newaxis]         
    
    Xb_x_right = torch.tensor(right_boundary,  requires_grad=True, device='cuda') 
    
    Xb_y_right = torch.tensor(right_boundary,  requires_grad=True, device='cuda') 
    
    Xb_x_up = torch.tensor(up_boundary,  requires_grad=True, device='cuda')
    
    Xb_y_up = torch.tensor(up_boundary,  requires_grad=True, device='cuda')    

    Xb_x_down = torch.tensor(down_boundary,  requires_grad=True, device='cuda')
    
    Xb_y_down = torch.tensor(down_boundary,  requires_grad=True, device='cuda')  
    
    Xb_x_left = torch.tensor(left_boundary,  requires_grad=True, device='cuda')
    target_x_left = torch.tensor(tx_left, device='cuda')   
    
    Xb_y_left = torch.tensor(left_boundary,  requires_grad=True, device='cuda')
    target_y_left = torch.tensor(ty_left, device='cuda')       
    
    fai_y_right = pred(Xb_y_right) # predict the boundary condition
    dfaidxy_y_right = grad(fai_y_right, Xb_y_right, torch.ones(Xb_y_right.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]            
    dfaidx_y_right = dfaidxy_y_right[:, 0].unsqueeze(1)
    #dfaidy_y_right = dfaidxy_y_right[:, 1].unsqueeze(1)             
    dfaidxdxy_y_right = grad(dfaidx_y_right, Xb_y_right, torch.ones(Xb_y_right.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    dfaidxdy_y_right = scaler*dfaidxdxy_y_right[:, 1].unsqueeze(1)


    fai_x_up = pred(Xb_x_up) # predict the boundary condition
    dfaidxy_x_up = grad(fai_x_up, Xb_x_up, torch.ones(Xb_x_up.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]            
    dfaidx_x_up = dfaidxy_x_up[:, 0].unsqueeze(1)
    #dfaidy_x_up = dfaidxy_x_up[:, 1].unsqueeze(1)             
    dfaidxdxy_x_up = grad(dfaidx_x_up, Xb_x_up, torch.ones(Xb_x_up.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    dfaidxdy_x_up = scaler*dfaidxdxy_x_up[:, 1].unsqueeze(1)

    fai_y_up = pred(Xb_y_up) # predict the boundary condition
    dfaidxy_y_up = grad(fai_y_up, Xb_y_up, torch.ones(Xb_y_up.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]            
    dfaidx_y_up = dfaidxy_y_up[:, 0].unsqueeze(1)
    #dfaidy_y_up = dfaidxy_y_up[:, 1].unsqueeze(1)             
    dfaidxdxy_y_up = grad(dfaidx_y_up, Xb_y_up, torch.ones(Xb_y_up.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    dfaidxdx_y_up = scaler*dfaidxdxy_y_up[:, 0].unsqueeze(1)


    fai_x_right = pred(Xb_x_right) # predict the boundary condition
    dfaidxy_x_right = grad(fai_x_right, Xb_x_right, torch.ones(Xb_x_right.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]            
    #dfaidx_x_right = dfaidxy_x_right[:, 0].unsqueeze(1)
    dfaidy_x_right = dfaidxy_x_right[:, 1].unsqueeze(1)             
    dfaidydxy_x_right = grad(dfaidy_x_right, Xb_x_right, torch.ones(Xb_x_right.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    dfaidydy_x_right = scaler*dfaidydxy_x_right[:, 1].unsqueeze(1)

    fai_x_down = pred(Xb_x_down) # predict the boundary condition
    dfaidxy_x_down = grad(fai_x_down, Xb_x_down, torch.ones(Xb_x_down.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]            
    dfaidx_x_down = dfaidxy_x_down[:, 0].unsqueeze(1)
    #dfaidy_x_down = dfaidxy_x_down[:, 1].unsqueeze(1)             
    dfaidxdxy_x_down = grad(dfaidx_x_down, Xb_x_down, torch.ones(Xb_x_down.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    dfaidxdy_x_down = scaler*dfaidxdxy_x_down[:, 1].unsqueeze(1)

    fai_y_down = pred(Xb_y_down) # predict the boundary condition
    dfaidxy_y_down = grad(fai_y_down, Xb_y_down, torch.ones(Xb_y_down.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]            
    dfaidx_y_down = dfaidxy_y_down[:, 0].unsqueeze(1)
    #dfaidy_y_down = dfaidxy_y_down[:, 1].unsqueeze(1)             
    dfaidxdxy_y_down = grad(dfaidx_y_down, Xb_y_down, torch.ones(Xb_y_down.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    dfaidxdx_y_down = scaler*dfaidxdxy_y_down[:, 0].unsqueeze(1)

    fai_y_left = pred(Xb_y_left) # predict the boundary condition
    dfaidxy_y_left = grad(fai_y_left, Xb_y_left, torch.ones(Xb_y_left.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]            
    dfaidx_y_left = dfaidxy_y_left[:, 0].unsqueeze(1)
    #dfaidy_y_left = dfaidxy_y_left[:, 1].unsqueeze(1)             
    dfaidxdxy_y_left = grad(dfaidx_y_left, Xb_y_left, torch.ones(Xb_y_left.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    dfaidxdy_y_left = scaler*dfaidxdxy_y_left[:, 1].unsqueeze(1)

    fai_x_left = pred(Xb_x_left) # predict the boundary condition
    dfaidxy_x_left = grad(fai_x_left, Xb_x_left, torch.ones(Xb_x_left.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]            
    #dfaidx_x_left = dfaidxy_x_left[:, 0].unsqueeze(1)
    dfaidy_x_left = dfaidxy_x_left[:, 1].unsqueeze(1)             
    dfaidydxy_x_left = grad(dfaidy_x_left, Xb_x_left, torch.ones(Xb_x_left.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    dfaidydy_x_left = scaler*dfaidydxy_x_left[:, 1].unsqueeze(1)

    
    pred_sigma_yy_up = dfaidxdx_y_up.data.cpu().numpy()
    pred_sigma_xy_up = dfaidxdy_x_up.data.cpu().numpy()
    
    pred_sigma_xx_right = dfaidydy_x_right.data.cpu().numpy()
    pred_sigma_xy_right = dfaidxdy_y_right.data.cpu().numpy()
    
    pred_sigma_yy_down = dfaidxdx_y_down.data.cpu().numpy()
    pred_sigma_xy_down = dfaidxdy_x_down.data.cpu().numpy()
    
    pred_sigma_xx_left = dfaidydy_x_left.data.cpu().numpy()
    pred_sigma_xy_left = dfaidxdy_y_left.data.cpu().numpy()    
    return equal_line_x, pred_sigma_xy_up, tx_up,  pred_sigma_yy_up, ty_up, pred_sigma_yy_down, ty_down, pred_sigma_xy_down, tx_down,  \
        equal_line_y,  pred_sigma_xx_right, tx_right, pred_sigma_xy_right, ty_right,  pred_sigma_xx_left, tx_left, pred_sigma_xy_left, ty_left



criterion = torch.nn.MSELoss()
model_g = FNN(2, 20, 1).cuda()
# model_g = RBF(20)
optim = torch.optim.Adam(params=model_g.parameters(), lr= 0.01)
scheduler = MultiStepLR(optim, milestones=[10000, 20000], gamma = 0.5)
loss_array = []
loss_dom_array = []
loss_ex_array = []
error_sigma_x_array = []
error_sigma_y_array = []
error_sigma_xy_array = []
error_sigma_mise_array = []
nepoch = int(nepoch)
start = time.time()
#xy_dom_t = torch.tensor(dom_dmsh[:,:2],   requires_grad=True, device='cuda')
xy_dom_t = dom_data_uniform(N_test, cir)
# weight = torch.tensor(dom_dmsh[:,2],   requires_grad=True, device='cuda').unsqueeze(1)
# xy_dom_t, xy_dom_t_aug = train_data(Nf, Nf_aug, cir, ratio)
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
        #dfaidydx = dfaidydxy[:, 0].unsqueeze(1)
        dfaidydy = dfaidydxy[:, 1].unsqueeze(1)
    
        # 通过应力函数计算应力
        sigma_x = dfaidydy
        sigma_y = dfaidxdx
        sigma_xy = -dfaidxdy

        # 计算应变
        epsilon_x = 1/E*(sigma_x - nu*sigma_y)*E
        epsilon_y = 1/E*(sigma_y - nu*sigma_x)*E
        epsilon_xy = 1/G*sigma_xy*E

        J_dom_density = 0.5*(sigma_x*epsilon_x + sigma_y*epsilon_y + sigma_xy*epsilon_xy) # 计算余能
        #J_dom =  torch.sum(J_dom_density * weight)
        J_dom =  torch.mean(J_dom_density) * S
        loss =  J_dom 
        optim.zero_grad()
        loss.backward(retain_graph=True)
        loss_array.append(loss.data.cpu())
        loss_dom_array.append(J_dom.data.cpu())
        if epoch%50==0:
            print(' epoch : %i, the loss : %f, loss dom : %f' % \
                  (epoch, loss.data, J_dom))
        if epoch%10 ==0:
            # von_mise stress L2 error: test the error in the FEM coordinate
            fem_dom_t = torch.tensor(FEM_node,  requires_grad=True, device='cuda')
            fai = pred(fem_dom_t) # Input r and theta to the pred function to get the necessary predition stress function
            
            dfaidxy = grad(fai, fem_dom_t, torch.ones(fem_dom_t.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
            dfaidx = dfaidxy[:, 0].unsqueeze(1)
            dfaidy = dfaidxy[:, 1].unsqueeze(1)
            
            dfaidxdxy = grad(dfaidx,  fem_dom_t, torch.ones( fem_dom_t.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
            dfaidxdx = dfaidxdxy[:, 0].unsqueeze(1)
            dfaidxdy = dfaidxdxy[:, 1].unsqueeze(1)
        
            dfaidydxy = grad(dfaidy,  fem_dom_t, torch.ones( fem_dom_t.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
            dfaidydx = dfaidydxy[:, 0].unsqueeze(1)
            dfaidydy = dfaidydxy[:, 1].unsqueeze(1)
        
            # 通过应力函数计算应力
            sxx_pred = scaler*dfaidydy.data.cpu().numpy().flatten()
            syy_pred = scaler*dfaidxdx.data.cpu().numpy().flatten()
            sxy_pred = -scaler*dfaidxdy.data.cpu().numpy().flatten()
            mise_pred = np.sqrt(0.5*((sxx_pred-syy_pred)**2+sxx_pred**2+syy_pred**2+6*sxy_pred**2)).flatten()
            
            L2_mise = np.linalg.norm(mise_pred - FEM_mise)/np.linalg.norm(FEM_mise)
            L2_sigma11 = np.linalg.norm(sxx_pred - FEM_stress11)/np.linalg.norm(FEM_stress11)
            L2_sigma22 = np.linalg.norm(syy_pred - FEM_stress22)/np.linalg.norm(FEM_stress22)
            L2_sigma12 = np.linalg.norm(sxy_pred - FEM_stress12)/np.linalg.norm(FEM_stress12)
            
            error_sigma_mise_array.append(L2_mise)
            error_sigma_x_array.append(L2_sigma11)
            error_sigma_y_array.append(L2_sigma22)
            error_sigma_xy_array.append(L2_sigma12)
            print(' epoch : %i, the loss : %f, loss dom : %f, L2_mise: %f, L2_xx: %f, L2_yy: %f, L2_xy: %f' % \
                  (epoch, loss.data, J_dom, L2_mise, L2_sigma11, L2_sigma22, L2_sigma12)) 
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

write_vtk_v2p("../../../output/DCEM_%ihole_NN_cir_fixed_all_pressure" % len(cir),  dom, pred_sigma_x, pred_sigma_xy, pred_sigma_y, pred_mise)


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


# =============================================================================
# # theta=2*pi and x = 1.0
# =============================================================================
FEM_node = np.load("node_coordinate_abaqus_1hole_all_pressure.npy")
node_stress_abaqus_rectangle = np.load("node_stress_abaqus_1hole_all_pressure.npy")

FEM_mise = node_stress_abaqus_rectangle[:,0]
FEM_stress11 = node_stress_abaqus_rectangle[:,1]
FEM_stress22 = node_stress_abaqus_rectangle[:,2]
FEM_stress12 = node_stress_abaqus_rectangle[:,3]
    
# # theta = 2*pi
FEM_theta = np.load("boundary_coordinate_abaqus_1hole_all_pressure.npy")
FEM_mise_theta = node_stress_abaqus_rectangle[list(int(i) for i in FEM_theta[:, 0]), 0]

xy_dom_t_theta = torch.tensor(FEM_theta[:,1:],  requires_grad=True, device='cuda')
fai_theta = pred(xy_dom_t_theta) # Input r and theta to the pred function to get the necessary predition stress function

dfaidxy_theta = grad(fai_theta, xy_dom_t_theta, torch.ones(xy_dom_t_theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
dfaidx_theta = dfaidxy_theta[:, 0].unsqueeze(1)
dfaidy_theta = dfaidxy_theta[:, 1].unsqueeze(1)

dfaidxdxy_theta = grad(dfaidx_theta,  xy_dom_t_theta, torch.ones( xy_dom_t_theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
dfaidxdx_theta = dfaidxdxy_theta[:, 0].unsqueeze(1)
dfaidxdy_theta = dfaidxdxy_theta[:, 1].unsqueeze(1)

dfaidydxy_theta = grad(dfaidy_theta,  xy_dom_t_theta, torch.ones( xy_dom_t_theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
dfaidydx_theta = dfaidydxy_theta[:, 0].unsqueeze(1)
dfaidydy_theta = dfaidydxy_theta[:, 1].unsqueeze(1)

# 通过应力函数计算应力
sxx_pred_theta = scaler*dfaidydy_theta.data.cpu().numpy().flatten()
syy_pred_theta = scaler*dfaidxdx_theta.data.cpu().numpy().flatten()
sxy_pred_theta = -scaler*dfaidxdy_theta.data.cpu().numpy().flatten()
mise_pred_theta = np.sqrt(0.5*((sxx_pred_theta-syy_pred_theta)**2+sxx_pred_theta**2+syy_pred_theta**2+6*sxy_pred_theta**2)).flatten()

# # x = 1.0
x10 = np.where((FEM_node[:, 0] == 1.0 ))
FEM_node_x10 = FEM_node[x10, :][0] # 
FEM_mise_x10 = FEM_mise[x10] 
FEM_stress11_x10 = FEM_stress11[x10] 
FEM_stress22_x10 = FEM_stress22[x10] 
FEM_stress12_x10 = FEM_stress12[x10] 


xy_dom_t_x10 = torch.tensor(FEM_node_x10,  requires_grad=True, device='cuda')
fai_x10 = pred(xy_dom_t_x10) # Input r and theta to the pred function to get the necessary predition stress function

dfaidxy_x10 = grad(fai_x10, xy_dom_t_x10, torch.ones(xy_dom_t_x10.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
dfaidx_x10 = dfaidxy_x10[:, 0].unsqueeze(1)
dfaidy_x10 = dfaidxy_x10[:, 1].unsqueeze(1)

dfaidxdxy_x10 = grad(dfaidx_x10,  xy_dom_t_x10, torch.ones( xy_dom_t_x10.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
dfaidxdx_x10 = dfaidxdxy_x10[:, 0].unsqueeze(1)
dfaidxdy_x10 = dfaidxdxy_x10[:, 1].unsqueeze(1)

dfaidydxy_x10 = grad(dfaidy_x10,  xy_dom_t_x10, torch.ones( xy_dom_t_x10.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
dfaidydx_x10 = dfaidydxy_x10[:, 0].unsqueeze(1)
dfaidydy_x10 = dfaidydxy_x10[:, 1].unsqueeze(1)

# 通过应力函数计算应力
sxx_pred_x10 = scaler*dfaidydy_x10.data.cpu().numpy().flatten()
syy_pred_x10 = scaler*dfaidxdx_x10.data.cpu().numpy().flatten()
sxy_pred_x10 = -scaler*dfaidxdy_x10.data.cpu().numpy().flatten()
mise_pred_x10 = np.sqrt(0.5*((sxx_pred_x10-syy_pred_x10)**2+sxx_pred_x10**2+syy_pred_x10**2+6*sxy_pred_x10**2)).flatten()

# # x = 0.0
x00 = np.where((FEM_node[:, 0] == 0.0 ))
FEM_node_x00 = FEM_node[x00, :][0] # 
FEM_mise_x00 = FEM_mise[x00] 
FEM_stress11_x00 = FEM_stress11[x00] 
FEM_stress22_x00 = FEM_stress22[x00] 
FEM_stress12_x00 = FEM_stress12[x00] 


xy_dom_t_x00 = torch.tensor(FEM_node_x00,  requires_grad=True, device='cuda')
fai_x00 = pred(xy_dom_t_x00) # Input r and theta to the pred function to get the necessary predition stress function

dfaidxy_x00 = grad(fai_x00, xy_dom_t_x00, torch.ones(xy_dom_t_x00.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
dfaidx_x00 = dfaidxy_x00[:, 0].unsqueeze(1)
dfaidy_x00 = dfaidxy_x00[:, 1].unsqueeze(1)

dfaidxdxy_x00 = grad(dfaidx_x00,  xy_dom_t_x00, torch.ones( xy_dom_t_x00.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
dfaidxdx_x00 = dfaidxdxy_x00[:, 0].unsqueeze(1)
dfaidxdy_x00 = dfaidxdxy_x00[:, 1].unsqueeze(1)

dfaidydxy_x00 = grad(dfaidy_x00,  xy_dom_t_x00, torch.ones( xy_dom_t_x00.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
dfaidydx_x00 = dfaidydxy_x00[:, 0].unsqueeze(1)
dfaidydy_x00 = dfaidydxy_x00[:, 1].unsqueeze(1)

# 通过应力函数计算应力
sxx_pred_x00 = scaler*dfaidydy_x00.data.cpu().numpy().flatten()
syy_pred_x00 = scaler*dfaidxdx_x00.data.cpu().numpy().flatten()
sxy_pred_x00 = -scaler*dfaidxdy_x00.data.cpu().numpy().flatten()
mise_pred_x00 = np.sqrt(0.5*((sxx_pred_x00-syy_pred_x00)**2+sxx_pred_x00**2+syy_pred_x00**2+6*sxy_pred_x00**2)).flatten()

# # y = 0.0
y00 = np.where((FEM_node[:, 1] == 0.0 ))
FEM_node_y00 = FEM_node[y00, :][0] # 
FEM_mise_y00 = FEM_mise[y00] 
FEM_stress11_y00 = FEM_stress11[y00] 
FEM_stress22_y00 = FEM_stress22[y00] 
FEM_stress12_y00 = FEM_stress12[y00] 


xy_dom_t_y00 = torch.tensor(FEM_node_y00,  requires_grad=True, device='cuda')
fai_y00 = pred(xy_dom_t_y00) # Input r and theta to the pred function to get the necessary predition stress function

dfaidxy_y00 = grad(fai_y00, xy_dom_t_y00, torch.ones(xy_dom_t_y00.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
dfaidx_y00 = dfaidxy_y00[:, 0].unsqueeze(1)
dfaidy_y00 = dfaidxy_y00[:, 1].unsqueeze(1)

dfaidxdxy_y00 = grad(dfaidx_y00,  xy_dom_t_y00, torch.ones( xy_dom_t_y00.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
dfaidxdx_y00 = dfaidxdxy_y00[:, 0].unsqueeze(1)
dfaidxdy_y00 = dfaidxdxy_y00[:, 1].unsqueeze(1)

dfaidydxy_y00 = grad(dfaidy_y00,  xy_dom_t_y00, torch.ones( xy_dom_t_y00.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
dfaidydx_y00 = dfaidydxy_y00[:, 0].unsqueeze(1)
dfaidydy_y00 = dfaidydxy_y00[:, 1].unsqueeze(1)

# 通过应力函数计算应力
sxx_pred_y00 = scaler*dfaidydy_y00.data.cpu().numpy().flatten()
syy_pred_y00 = scaler*dfaidxdx_y00.data.cpu().numpy().flatten()
sxy_pred_y00 = -scaler*dfaidxdy_y00.data.cpu().numpy().flatten()
mise_pred_y00 = np.sqrt(0.5*((sxx_pred_y00-syy_pred_y00)**2+sxx_pred_y00**2+syy_pred_y00**2+6*sxy_pred_y00**2)).flatten()


# # y = 1.0
y10 = np.where((FEM_node[:, 1] == 1.0 ))
FEM_node_y10 = FEM_node[y10, :][0] # 
FEM_mise_y10 = FEM_mise[y10] 
FEM_stress11_y10 = FEM_stress11[y10] 
FEM_stress22_y10 = FEM_stress22[y10] 
FEM_stress12_y10 = FEM_stress12[y10] 


xy_dom_t_y10 = torch.tensor(FEM_node_y10,  requires_grad=True, device='cuda')
fai_y10 = pred(xy_dom_t_y10) # Input r and theta to the pred function to get the necessary predition stress function

dfaidxy_y10 = grad(fai_y10, xy_dom_t_y10, torch.ones(xy_dom_t_y10.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
dfaidx_y10 = dfaidxy_y10[:, 0].unsqueeze(1)
dfaidy_y10 = dfaidxy_y10[:, 1].unsqueeze(1)

dfaidxdxy_y10 = grad(dfaidx_y10,  xy_dom_t_y10, torch.ones( xy_dom_t_y10.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
dfaidxdx_y10 = dfaidxdxy_y10[:, 0].unsqueeze(1)
dfaidxdy_y10 = dfaidxdxy_y10[:, 1].unsqueeze(1)

dfaidydxy_y10 = grad(dfaidy_y10,  xy_dom_t_y10, torch.ones( xy_dom_t_y10.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
dfaidydx_y10 = dfaidydxy_y10[:, 0].unsqueeze(1)
dfaidydy_y10 = dfaidydxy_y10[:, 1].unsqueeze(1)

# 通过应力函数计算应力
sxx_pred_y10 = scaler*dfaidydy_y10.data.cpu().numpy().flatten()
syy_pred_y10 = scaler*dfaidxdx_y10.data.cpu().numpy().flatten()
sxy_pred_y10 = -scaler*dfaidxdy_y10.data.cpu().numpy().flatten()
mise_pred_y10 = np.sqrt(0.5*((sxx_pred_y10-syy_pred_y10)**2+sxx_pred_y10**2+syy_pred_y10**2+6*sxy_pred_y10**2)).flatten()
# =============================================================================
# line
# =============================================================================
equal_line_x, pred_sigma_xy_up, tx_up,  pred_sigma_yy_up, ty_up, pred_sigma_yy_down, ty_down,\
    pred_sigma_xy_down, tx_down,  equal_line_y,  pred_sigma_xx_right, tx_right, pred_sigma_xy_right, ty_right,  pred_sigma_xx_left, tx_left, pred_sigma_xy_left, ty_left = test_particular_with_pred(N_test)

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
plt.scatter(equal_line_x[::internal], -pred_sigma_yy_down[::internal], marker = '+', linestyle='-')
plt.xlabel('x')
plt.ylabel('sigma_down')
plt.legend(['exact_tx_down', 'exact_ty_down', 'pred_xy_down',  'pred_yy_down'])
plt.title('down')
plt.show()

plt.plot(equal_line_y, tx_left)
plt.plot(equal_line_y, ty_left)
plt.scatter(equal_line_y[::internal], -pred_sigma_xx_left[::internal], marker = '*', linestyle='-')
plt.scatter(equal_line_y[::internal], pred_sigma_xy_left[::internal], marker = '+', linestyle='-')
plt.xlabel('x')
plt.ylabel('sigma_left')
plt.legend(['exact_tx_left', 'exact_ty_left', 'pred_xx_left', 'pred_xy_left'])
plt.title('left')
plt.show()   