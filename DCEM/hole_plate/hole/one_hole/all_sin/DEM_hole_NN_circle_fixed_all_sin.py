import numpy as np
import torch
import matplotlib.pyplot as plt 
from matplotlib import cm
import numpy as np
import torch
from  torch.autograd import grad
import time
import matplotlib as mpl
from pyevtk.hl import pointsToVTK
from pyevtk.hl import gridToVTK
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
import cmath
mpl.rcParams['figure.dpi'] = 1000


FEM_node = np.load("node_coordinate_abaqus_1hole_all_pressure.npy")
node_stress_abaqus_rectangle = np.load("node_stress_abaqus_1hole_all_pressure.npy")

FEM_mise = node_stress_abaqus_rectangle[:,0]
FEM_stress11 = node_stress_abaqus_rectangle[:,1]
FEM_stress22 = node_stress_abaqus_rectangle[:,2]
FEM_stress12 = node_stress_abaqus_rectangle[:,3]

bound_len = 0.05
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
    
setup_seed(1)
dom_dmsh = np.load('1hole_dmsh.npy')
a = 1.
b = 1.
P = 100
P_left = 110
beta = 0.001
ratio = 0
N_part = 1000
Nb = 1000
Nf = 10000
Nf_aug = 20000
nepoch = 30000

# cir = [[0.2, 0.2, 0.2, 0.2], [-0.2, 0.4, 0.1, 0.1], [-0.2, -0.3, 0.3, 0.3]]
cir = [[0.5, 0.5, 0.25, 0.25]]
m = 0
for i in cir:
    m = m + np.pi*i[2]*i[3]
S = a*b - m

N_test_plot = 100
E = 1000
nu = 0.3
G = E/2/(1+nu)

tol_p = 0.0001
loss_p = 100
epoch_p = 0
N_test = 1000
N_uni = 101

D11_mat = E/(1-nu**2)
D22_mat = E/(1-nu**2)
D12_mat = E*nu/(1-nu**2)
D21_mat = E*nu/(1-nu**2)
D33_mat = E/(2*(1+nu))
training_part = 0
order = 2
factor = (4/a/b)**order



# Prepare training data




def train_data(Nb, Nf, Nf_aug, cir, ratio = 0.5625):
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



    
    Xr = np.hstack([ a*np.ones([Nb,1]), np.random.rand(Nb,1)*b*2 -b]) # the boundary points on the right side.
    Xr = torch.tensor(Xr, dtype=torch.float64, requires_grad=True)
    
    
    Xf_x = np.random.rand(Nf, 1)*2*a - a # domain points
    Xf_y = np.random.rand(Nf, 1)*2*b - b # domain points
    Xf = np.hstack([Xf_x, Xf_y])
    index0 = []
    #Xf = np.copy(Xf0)
    for i, (x, y) in enumerate(Xf): # determine whether the points is in the circle or not.

        #x, y = x*x0, y*y0
        good = 0
        for cir_info in cir:
            if (circle(cir_info[0], cir_info[1], cir_info[2], cir_info[3], x, y) >= ratio):
                good = good + 1
        if good == len(cir): 
            index0.append(i)
    Xf0 = Xf[index0] # the domian points without the circle points
    
    Xf_x = np.random.rand(Nf_aug, 1)*2*a - a # domain points
    Xf_y = np.random.rand(Nf_aug, 1)*2*b - b # domain points
    Xf = np.hstack([Xf_x, Xf_y])
    index_a = []

    
    for i, (x, y) in enumerate(Xf):
        for cir_info in cir:
            if (circle(cir_info[0], cir_info[1], cir_info[2], cir_info[3], x, y) > 0 and circle(cir_info[0], cir_info[1], cir_info[2], cir_info[3], x, y) < ratio):
                index_a.append(i)
    
    Xf1 = Xf[index_a]
    
    Xf0 = torch.tensor(Xf0, dtype=torch.float64, requires_grad=True)
    Xf1 = torch.tensor(Xf1, dtype=torch.float64, requires_grad=True)
    
    return  Xr, Xf0, Xf1

def train_data_uniform(Nb, Nf, cir):
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



    
    Xr = np.hstack([ a*np.ones([Nb,1]), np.linspace(0, b, Nb)[:, np.newaxis]]) # the boundary points on the right side.
    Xr = torch.tensor(Xr, dtype=torch.float64, requires_grad=True)

    Xu = np.hstack([ np.linspace(0, a, Nb)[:, np.newaxis],  b*np.ones([Nb,1])]) # the boundary points on the right side.
    Xu = torch.tensor(Xu, dtype=torch.float64, requires_grad=True)
    
    Xl = np.hstack([ 0*np.ones([Nb,1]), np.linspace(0, b, Nb)[:, np.newaxis]]) # the boundary points on the right side.
    Xl = torch.tensor(Xl, dtype=torch.float64, requires_grad=True)

    Xd = np.hstack([ np.linspace(0, a, Nb)[:, np.newaxis],  0*np.ones([Nb,1])]) # the boundary points on the right side.
    Xd = torch.tensor(Xd, dtype=torch.float64, requires_grad=True)

    
    x = np.linspace(0, a, Nf)
    y = np.linspace(0, b, Nf) # y方向N_test个点
    x_mesh, y_mesh = np.meshgrid(x, y)
    Xf = np.stack((x_mesh.flatten(), y_mesh.flatten()), 1)
    index0 = []
    #Xf = np.copy(Xf0)
    for i, (x, y) in enumerate(Xf): # determine whether the points is in the circle or not.

        #x, y = x*x0, y*y0
        good = 0
        for cir_info in cir:
            if (circle(cir_info[0], cir_info[1], cir_info[2], cir_info[3], x, y) >= 0):
                good = good + 1
        if good == len(cir): 
            index0.append(i)
    Xf0 = Xf[index0] # the domian points without the circle points
    

    
    Xf0 = torch.tensor(Xf0, dtype=torch.float64, requires_grad=True)
    
    return  Xr, Xu, Xl, Xd, Xf0
def circle(x0, y0, ra, rb, xt, yt):
    
    return (xt - x0)**2/ra**2 + (yt - y0)**2/rb**2 - 1


class FNN(torch.nn.Module):
    def __init__(self, n_input, n_output, n_layer, n_nodes):
        super(FNN, self).__init__()
        self.n_layer = n_layer
        
        self.Input = nn.Linear(n_input, n_nodes)   # linear layer
        nn.init.xavier_uniform_(self.Input.weight) # wigths and bias initiation
        nn.init.normal_(self.Input.bias)

        self.Output = nn.Linear(n_nodes, n_output)
        nn.init.xavier_uniform_(self.Output.weight)
        nn.init.normal_(self.Output.bias)
        
        self.Hidden = nn.ModuleList() # hidden layer list
        for i in range(n_layer):
            self.Hidden.append(nn.Linear(n_nodes, n_nodes))
        for layer in self.Hidden:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.normal_(layer.bias)
        

    def forward(self, x):
        y = torch.tanh(self.Input(x)) # tanh activation function
        for layer in self.Hidden:
            y = torch.tanh(layer(y))
        y = self.Output(y)
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

class distance_net(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(distance_net, self).__init__()
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
    # dis = model_dis(xy)
    dis =  ((x-0.5)**2/0.25**2 + (y-0.5)**2/0.25**2 -1).unsqueeze(1)
    general = torch.cat([Net_u(xy), Net_v(xy)],1)
    pred_u = dis * general 
    return pred_u

    
def evaluate_sigma(N_test):# mesh spaced grid
    _, _, _, _, xy_dom_t = train_data_uniform(Nb, N_test, cir)

# add cir boundary points
    N_cir = N_test *10
    boundary_point = np.ones([len(cir)*N_cir, 2])
    theta = np.linspace(0, 2*np.pi, N_cir)
    for index, eve_cir in enumerate(cir):
      x = eve_cir[0] + np.cos(theta) * (eve_cir[2]**2)**0.5   # the circle is default.
      y = eve_cir[1] + np.sin(theta) * (eve_cir[2]**2)**0.5
      boundary_point[N_cir*index: N_cir*(index+1), 0] = x
      boundary_point[N_cir*index: N_cir*(index+1), 1] = y
    boundary_point = torch.tensor(boundary_point, dtype=torch.float64, requires_grad=True)
    
    xy_dom_t = torch.cat([boundary_point, xy_dom_t], 0)
    
    u_pred =  pred(xy_dom_t) # Input r and theta to the pred function to get the necessary predition stress function
    duxdxy = grad(u_pred[:, 0].unsqueeze(1), xy_dom_t, torch.ones(xy_dom_t.size()[0], 1, device="cuda"), create_graph=True, retain_graph=True)[0]
    duydxy = grad(u_pred[:, 1].unsqueeze(1), xy_dom_t, torch.ones(xy_dom_t.size()[0], 1, device="cuda"), create_graph=True, retain_graph=True)[0]
    duxdx = duxdxy[:, 0].unsqueeze(1)
    duxdy = duxdxy[:, 1].unsqueeze(1)
    
    duydx = duydxy[:, 0].unsqueeze(1)
    duydy = duydxy[:, 1].unsqueeze(1)
    
    exx_pred = duxdx*P/E
    eyy_pred = duydy*P/E
    e2xy_pred = (duxdy + duydx)*P/E 

    sxx_pred = D11_mat * exx_pred + D12_mat * eyy_pred
    syy_pred = D12_mat * exx_pred + D22_mat * eyy_pred
    sxy_pred = D33_mat * e2xy_pred

    dom = xy_dom_t.data.cpu().numpy()
    u_x = u_pred[:, 0].unsqueeze(1).data.cpu().numpy()*P/E
    u_y = u_pred[:, 1].unsqueeze(1).data.cpu().numpy()*P/E
    u_mag = np.sqrt(u_x**2 + u_y**2)
    pred_sigma_x = sxx_pred.data.cpu().numpy() # change the type of tensor to the array for the use of plotting
    pred_sigma_y = syy_pred.data.cpu().numpy()
    pred_sigma_xy = sxy_pred.data.cpu().numpy()

    pred_mise = np.sqrt(0.5*((pred_sigma_x-pred_sigma_y)**2+pred_sigma_x**2+pred_sigma_y**2+6*pred_sigma_xy**2))

    return dom, pred_sigma_x, pred_sigma_y, pred_sigma_xy, pred_mise, u_mag


# =============================================================================
#%%
# =============================================================================
Net_u = FNN(2, 1, 5, 5)
Net_v = FNN(2, 1, 5, 5)
# Net_u = RBF(20)
# Net_v = RBF(20)
optim = torch.optim.Adam(list(Net_u.parameters())+list(Net_v.parameters()), lr=0.01)
scheduler = MultiStepLR(optim, milestones=[10000, 20000], gamma = 0.1)
loss_array = []
loss_dom_array = []
loss_ex_array = []
error_sigma_x_array = []
error_sigma_y_array = []
error_sigma_xy_array = []
error_sigma_mise_array = []
nepoch = int(nepoch)
start = time.time()

criterion = torch.nn.MSELoss()
Xr, Xu, Xl, Xd, xy_dom_t = train_data_uniform(Nb, N_uni, cir)
# xy_dom_t = torch.tensor(dom_dmsh[:,:2],   requires_grad=True, device='cuda')
weight = torch.tensor(dom_dmsh[:,2],   requires_grad=True, device='cuda').unsqueeze(1)
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

        J_dom_density = 0.5*(sxx_pred*exx_pred + syy_pred*eyy_pred + sxy_pred*e2xy_pred)/E # 计算应变能
        # J_dom =  torch.sum(J_dom_density * weight)
        J_dom =  torch.mean(J_dom_density) * S
    
        u_pred_bound_r = pred(Xr)
        t_r = torch.zeros_like(u_pred_bound_r, device='cuda')
        t_r[:, 0] = P * torch.sin(Xr[:, 1]/b*torch.tensor(np.pi, device='cuda')) 
        ex_density_r = torch.sum(u_pred_bound_r * t_r, axis=1)
        J_ex_r = torch.mean(ex_density_r)*b/P
  
        u_pred_bound_u = pred(Xu)
        t_u = torch.zeros_like(u_pred_bound_u, device='cuda')
        t_u[:, 1] = P * torch.sin(Xu[:, 0]/a*torch.tensor(np.pi, device='cuda')) 
        ex_density_u = torch.sum(u_pred_bound_u * t_u, axis=1)
        J_ex_u = torch.mean(ex_density_u)*a/P

        u_pred_bound_l = pred(Xl)
        t_l = torch.zeros_like(u_pred_bound_l, device='cuda')
        t_l[:, 0] = -P_left * torch.sin(Xl[:, 1]/b*torch.tensor(np.pi, device='cuda')) 
        ex_density_l = torch.sum(u_pred_bound_l * t_l, axis=1)
        J_ex_l = torch.mean(ex_density_l)*b/P
 
        u_pred_bound_d = pred(Xd)
        t_d = torch.zeros_like(u_pred_bound_d, device='cuda')
        t_d[:, 1] = -P * torch.sin(Xd[:, 0]/a*torch.tensor(np.pi, device='cuda')) 
        ex_density_d = torch.sum(u_pred_bound_d * t_d, axis=1)
        J_ex_d = torch.mean(ex_density_d)*a/P
        
        J_ex = J_ex_r + J_ex_u + J_ex_l + J_ex_d
        
        loss = J_dom - J_ex
        optim.zero_grad()
        loss.backward(retain_graph=True)
        loss_array.append(loss.data.cpu())
        loss_dom_array.append(J_dom.data.cpu())
        loss_ex_array.append(J_ex.data.cpu())
        if epoch%100==0:
            print(' epoch : %i, the loss : %f, loss dom : %f, loss ex : %f' % \
                  (epoch, loss.data, J_dom, J_ex))
        if epoch%10 == 0 :
                        # von_mise stress L2 error: test the error in the FEM coordinate
            fem_dom_t = torch.tensor(FEM_node,  requires_grad=True, device='cuda')
            u_pred = pred(fem_dom_t) # Input r and theta to the pred function to get the necessary predition stress function
            duxdxy = grad(u_pred[:, 0].unsqueeze(1), fem_dom_t, torch.ones(fem_dom_t.size()[0], 1, device="cuda"), create_graph=True, retain_graph=True)[0]
            duydxy = grad(u_pred[:, 1].unsqueeze(1), fem_dom_t, torch.ones(fem_dom_t.size()[0], 1, device="cuda"), create_graph=True, retain_graph=True)[0]
            duxdx = duxdxy[:, 0].unsqueeze(1)
            duxdy = duxdxy[:, 1].unsqueeze(1)
            
            duydx = duydxy[:, 0].unsqueeze(1)
            duydy = duydxy[:, 1].unsqueeze(1)
            
            exx_pred = duxdx.data.cpu().numpy()*P/E
            eyy_pred = duydy.data.cpu().numpy()*P/E
            e2xy_pred = (duxdy + duydx).data.cpu().numpy()*P/E
    
            sxx_pred = (D11_mat * exx_pred + D12_mat * eyy_pred).flatten()
            syy_pred = (D12_mat * exx_pred + D22_mat * eyy_pred).flatten()
            sxy_pred = (D33_mat * e2xy_pred).flatten()
            mise_pred = np.sqrt(0.5*((sxx_pred-syy_pred)**2+sxx_pred**2+syy_pred**2+6*sxy_pred**2))
            
            L2_mise = np.linalg.norm(mise_pred - FEM_mise)/np.linalg.norm(FEM_mise)
            L2_sigma11 = np.linalg.norm(sxx_pred - FEM_stress11)/np.linalg.norm(FEM_stress11)
            L2_sigma22 = np.linalg.norm(syy_pred - FEM_stress22)/np.linalg.norm(FEM_stress22)
            L2_sigma12 = np.linalg.norm(sxy_pred - FEM_stress12)/np.linalg.norm(FEM_stress12)
            
            error_sigma_mise_array.append(L2_mise)
            error_sigma_x_array.append(L2_sigma11)
            error_sigma_y_array.append(L2_sigma22)
            error_sigma_xy_array.append(L2_sigma12)
            print(' epoch : %i, the loss : %f, loss dom : %f, loss ex : %f, L2_mise: %f, L2_xx: %f, L2_yy: %f, L2_xy: %f' % \
                  (epoch, loss.data, J_dom, J_ex, L2_mise, L2_sigma11, L2_sigma22, L2_sigma12))
        return loss
    optim.step(closure)
    scheduler.step()
    
#%%
def write_vtk_v2p(filename, dom, S11, S12, S22, mise, u_mag): #已经将输出的感兴趣场进行了分类VTK导出
    xx = np.ascontiguousarray(dom[:, 0])
    yy = np.ascontiguousarray(dom[:, 1])
    zz = np.ascontiguousarray(dom[:, 1])*0 # 点的VTK
    S11 = S11.flatten()
    S12 = S12.flatten()
    S22 = S22.flatten()
    mise = mise.flatten()
    u_mag = u_mag.flatten()
    pointsToVTK(filename, xx, yy, zz, data={"S11": S11, "S12": S12, "S22": S22, "Mises": mise, "U_mag": u_mag})
        
def write_vtk_v2(filename, dom, S11, S12, S22, mise, u_mag):
    xx = np.ascontiguousarray(dom[:, 0]).reshape(N_test_plot, N_test_plot, 1)
    yy = np.ascontiguousarray(dom[:, 1]).reshape(N_test_plot, N_test_plot, 1)
    zz = 0*np.ascontiguousarray(dom[:, 1]).reshape(N_test_plot, N_test_plot, 1)
    gridToVTK(filename, xx, yy, zz, pointData={"S11": S11.reshape(N_test_plot, N_test_plot, 1), \
                                               "S12": S12.reshape(N_test_plot, N_test_plot, 1), \
                                               "S22": S22.reshape(N_test_plot, N_test_plot, 1), \
                                               "Mise": mise.reshape(N_test_plot, N_test_plot, 1), \
                                               "U_mag": u_mag.reshape(N_test_plot, N_test_plot, 1)})
    # gridToVTK(filename, xx, yy, zz, pointData={"displacement": U})  
    
dom, pred_sigma_x, pred_sigma_y, pred_sigma_xy, pred_mise, pred_u_mag = evaluate_sigma(N_uni)

write_vtk_v2p("../../../output/DEM_%ihole_NN_cir_fixed_all_pressure" % len(cir),  dom, pred_sigma_x, pred_sigma_xy, pred_sigma_y, pred_mise, pred_u_mag)


# =============================================================================
# SIGMA XX
# =============================================================================
plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(5.8, 4.8)) 
surf = ax.scatter(dom[:, 0], dom[:, 1], c = pred_sigma_x, cmap=cm.rainbow)
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
# U mag
# =============================================================================
plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(5.8, 4.8)) 
surf = ax.scatter(dom[:, 0], dom[:, 1], c = pred_u_mag, cmap=cm.rainbow)# vmin=0, vmax=0.14, 
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
u_pred_theta =  pred(xy_dom_t_theta) * P/E # Input r and theta to the pred function to get the necessary predition stress function
duxdxy_theta = grad(u_pred_theta[:, 0].unsqueeze(1), xy_dom_t_theta, torch.ones(xy_dom_t_theta.size()[0], 1, device="cuda"), create_graph=True, retain_graph=True)[0]
duydxy_theta = grad(u_pred_theta[:, 1].unsqueeze(1), xy_dom_t_theta, torch.ones(xy_dom_t_theta.size()[0], 1, device="cuda"), create_graph=True, retain_graph=True)[0]
duxdx_theta = duxdxy_theta[:, 0].unsqueeze(1)
duxdy_theta = duxdxy_theta[:, 1].unsqueeze(1)

duydx_theta = duydxy_theta[:, 0].unsqueeze(1)
duydy_theta = duydxy_theta[:, 1].unsqueeze(1)

exx_pred_theta = duxdx_theta
eyy_pred_theta = duydy_theta
e2xy_pred_theta = duxdy_theta + duydx_theta 

sxx_pred_theta = (D11_mat * exx_pred_theta + D12_mat * eyy_pred_theta).data.cpu().numpy()
syy_pred_theta = (D12_mat * exx_pred_theta + D22_mat * eyy_pred_theta).data.cpu().numpy()
sxy_pred_theta = (D33_mat * e2xy_pred_theta).data.cpu().numpy()
mise_pred_theta = np.sqrt(0.5*((sxx_pred_theta-syy_pred_theta)**2+sxx_pred_theta**2+syy_pred_theta**2+6*sxy_pred_theta**2))



# # x = 1.0
x10 = np.where((FEM_node[:, 0] == 1.0 ))
FEM_node_x10 = FEM_node[x10, :][0] # 
FEM_mise_x10 = FEM_mise[x10] 
FEM_stress11_x10 = FEM_stress11[x10] 
FEM_stress22_x10 = FEM_stress22[x10] 
FEM_stress12_x10 = FEM_stress12[x10] 

xy_dom_t_x10 = torch.tensor(FEM_node_x10,  requires_grad=True, device='cuda')
u_pred_x10 = pred(xy_dom_t_x10)*P/E # Input r and theta to the pred function to get the necessary predition stress function
duxdxy_x10 = grad(u_pred_x10[:, 0].unsqueeze(1), xy_dom_t_x10, torch.ones(xy_dom_t_x10.size()[0], 1, device="cuda"), create_graph=True, retain_graph=True)[0]
duydxy_x10 = grad(u_pred_x10[:, 1].unsqueeze(1), xy_dom_t_x10, torch.ones(xy_dom_t_x10.size()[0], 1, device="cuda"), create_graph=True, retain_graph=True)[0]
duxdx_x10 = duxdxy_x10[:, 0].unsqueeze(1)
duxdy_x10 = duxdxy_x10[:, 1].unsqueeze(1)

duydx_x10 = duydxy_x10[:, 0].unsqueeze(1)
duydy_x10 = duydxy_x10[:, 1].unsqueeze(1)

exx_pred_x10 = duxdx_x10
eyy_pred_x10 = duydy_x10
e2xy_pred_x10 = duxdy_x10 + duydx_x10 

sxx_pred_x10 = (D11_mat * exx_pred_x10 + D12_mat * eyy_pred_x10).data.cpu().numpy()
syy_pred_x10 = (D12_mat * exx_pred_x10 + D22_mat * eyy_pred_x10).data.cpu().numpy()
sxy_pred_x10 = (D33_mat * e2xy_pred_x10).data.cpu().numpy()
mise_pred_x10 = np.sqrt(0.5*((sxx_pred_x10-syy_pred_x10)**2+sxx_pred_x10**2+syy_pred_x10**2+6*sxy_pred_x10**2))

# # x = 0.0
x00 = np.where((FEM_node[:, 0] == 0.0 ))
FEM_node_x00 = FEM_node[x00, :][0] # 
FEM_mise_x00 = FEM_mise[x00] 
FEM_stress11_x00 = FEM_stress11[x00] 
FEM_stress22_x00 = FEM_stress22[x00] 
FEM_stress12_x00 = FEM_stress12[x00] 

xy_dom_t_x00 = torch.tensor(FEM_node_x00,  requires_grad=True, device='cuda')
u_pred_x00 = pred(xy_dom_t_x00)*P/E # Input r and theta to the pred function to get the necessary predition stress function
duxdxy_x00 = grad(u_pred_x00[:, 0].unsqueeze(1), xy_dom_t_x00, torch.ones(xy_dom_t_x00.size()[0], 1, device="cuda"), create_graph=True, retain_graph=True)[0]
duydxy_x00 = grad(u_pred_x00[:, 1].unsqueeze(1), xy_dom_t_x00, torch.ones(xy_dom_t_x00.size()[0], 1, device="cuda"), create_graph=True, retain_graph=True)[0]
duxdx_x00 = duxdxy_x00[:, 0].unsqueeze(1)
duxdy_x00 = duxdxy_x00[:, 1].unsqueeze(1)

duydx_x00 = duydxy_x00[:, 0].unsqueeze(1)
duydy_x00 = duydxy_x00[:, 1].unsqueeze(1)

exx_pred_x00 = duxdx_x00
eyy_pred_x00 = duydy_x00
e2xy_pred_x00 = duxdy_x00 + duydx_x00 

sxx_pred_x00 = (D11_mat * exx_pred_x00 + D12_mat * eyy_pred_x00).data.cpu().numpy()
syy_pred_x00 = (D12_mat * exx_pred_x00 + D22_mat * eyy_pred_x00).data.cpu().numpy()
sxy_pred_x00 = (D33_mat * e2xy_pred_x00).data.cpu().numpy()
mise_pred_x00 = np.sqrt(0.5*((sxx_pred_x00-syy_pred_x00)**2+sxx_pred_x00**2+syy_pred_x00**2+6*sxy_pred_x00**2))

# # y = 1.0
y10 = np.where((FEM_node[:, 1] == 1.0 ))
FEM_node_y10 = FEM_node[y10, :][0] # 
FEM_mise_y10 = FEM_mise[y10] 
FEM_stress11_y10 = FEM_stress11[y10] 
FEM_stress22_y10 = FEM_stress22[y10] 
FEM_stress12_y10 = FEM_stress12[y10] 

xy_dom_t_y10 = torch.tensor(FEM_node_y10,  requires_grad=True, device='cuda')
u_pred_y10 = pred(xy_dom_t_y10)*P/E # Input r and theta to the pred function to get the necessary predition stress function
duxdxy_y10 = grad(u_pred_y10[:, 0].unsqueeze(1), xy_dom_t_y10, torch.ones(xy_dom_t_y10.size()[0], 1, device="cuda"), create_graph=True, retain_graph=True)[0]
duydxy_y10 = grad(u_pred_y10[:, 1].unsqueeze(1), xy_dom_t_y10, torch.ones(xy_dom_t_y10.size()[0], 1, device="cuda"), create_graph=True, retain_graph=True)[0]
duxdx_y10 = duxdxy_y10[:, 0].unsqueeze(1)
duxdy_y10 = duxdxy_y10[:, 1].unsqueeze(1)

duydx_y10 = duydxy_y10[:, 0].unsqueeze(1)
duydy_y10 = duydxy_y10[:, 1].unsqueeze(1)

exx_pred_y10 = duxdx_y10
eyy_pred_y10 = duydy_y10
e2xy_pred_y10 = duxdy_y10 + duydx_y10 

sxx_pred_y10 = (D11_mat * exx_pred_y10 + D12_mat * eyy_pred_y10).data.cpu().numpy()
syy_pred_y10 = (D12_mat * exx_pred_y10 + D22_mat * eyy_pred_y10).data.cpu().numpy()
sxy_pred_y10 = (D33_mat * e2xy_pred_y10).data.cpu().numpy()
mise_pred_y10 = np.sqrt(0.5*((sxx_pred_y10-syy_pred_y10)**2+sxx_pred_y10**2+syy_pred_y10**2+6*sxy_pred_y10**2))

# # y = 0.0
y00 = np.where((FEM_node[:, 1] == 0.0 ))
FEM_node_y00 = FEM_node[y00, :][0] # 
FEM_mise_y00 = FEM_mise[y00] 
FEM_stress11_y00 = FEM_stress11[y00] 
FEM_stress22_y00 = FEM_stress22[y00] 
FEM_stress12_y00 = FEM_stress12[y00] 

xy_dom_t_y00 = torch.tensor(FEM_node_y00,  requires_grad=True, device='cuda')
u_pred_y00 = pred(xy_dom_t_y00)*P/E # Input r and theta to the pred function to get the necessary predition stress function
duxdxy_y00 = grad(u_pred_y00[:, 0].unsqueeze(1), xy_dom_t_y00, torch.ones(xy_dom_t_y00.size()[0], 1, device="cuda"), create_graph=True, retain_graph=True)[0]
duydxy_y00 = grad(u_pred_y00[:, 1].unsqueeze(1), xy_dom_t_y00, torch.ones(xy_dom_t_y00.size()[0], 1, device="cuda"), create_graph=True, retain_graph=True)[0]
duxdx_y00 = duxdxy_y00[:, 0].unsqueeze(1)
duxdy_y00 = duxdxy_y00[:, 1].unsqueeze(1)

duydx_y00 = duydxy_y00[:, 0].unsqueeze(1)
duydy_y00 = duydxy_y00[:, 1].unsqueeze(1)

exx_pred_y00 = duxdx_y00
eyy_pred_y00 = duydy_y00
e2xy_pred_y00 = duxdy_y00 + duydx_y00 

sxx_pred_y00 = (D11_mat * exx_pred_y00 + D12_mat * eyy_pred_y00).data.cpu().numpy()
syy_pred_y00 = (D12_mat * exx_pred_y00 + D22_mat * eyy_pred_y00).data.cpu().numpy()
sxy_pred_y00 = (D33_mat * e2xy_pred_y00).data.cpu().numpy()
mise_pred_y00 = np.sqrt(0.5*((sxx_pred_y00-syy_pred_y00)**2+sxx_pred_y00**2+syy_pred_y00**2+6*sxy_pred_y00**2))