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
from sklearn.neighbors import KDTree
import torch.nn as nn
mpl.rcParams['figure.dpi'] = 1000
torch.set_default_tensor_type(torch.DoubleTensor) # 将tensor的类型变成默认的double
torch.set_default_tensor_type(torch.cuda.DoubleTensor) # 将cuda的tensor的类型变成默认的double
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
setup_seed(1)
training_dis = 1
a = 1.
b = 1.
tol_dis = 0.00001
Nb = 3000
add_corn = 100
Nf = 100
Nf_aug = 20000
cir = [[0.5, 0.5, 0.25, 0.25]]


def train_data_uniform(Nb, Nf, cir,num_corner):
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
    

    
    Xl = np.hstack([0*np.linspace(0, a, Nb)[:, np.newaxis],  np.linspace(0, b, Nb)[:, np.newaxis]]) # the boundary points on the right side.
    Xl = torch.tensor(Xl, dtype=torch.float64, requires_grad=True)


    
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
            if (circle(cir_info[0], cir_info[1], cir_info[2], cir_info[3], x, y) > 0):
                good = good + 1
        if good == len(cir): 
            index0.append(i)
    Xf0 = Xf[index0] # the domian points without the circle points
    Xf0 = torch.tensor(Xf0, dtype=torch.float64, requires_grad=True)
    
    # for interface point
    boundary_point = np.ones([len(cir)*Nb, 2])
    theta = np.linspace(0, 2*np.pi, Nb)
    for index, eve_cir in enumerate(cir):
      x = eve_cir[0] + np.cos(theta) * (eve_cir[2]**2)**0.5   # the circle is default.
      y = eve_cir[1] + np.sin(theta) * (eve_cir[2]**2)**0.5
      boundary_point[Nb*index: Nb*(index+1), 0] = x
      boundary_point[Nb*index: Nb*(index+1), 1] = y
    boundary_point = torch.tensor(boundary_point, dtype=torch.float64, requires_grad=True)
    boundary_point = torch.cat([boundary_point, Xl], 0)
    return  Xf0, boundary_point

def circle(x0, y0, ra, rb, xt, yt):
    
    return (xt - x0)**2/ra**2 + (yt - y0)**2/rb**2 - 1


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
def plot_pos(Xb, Xf):
    Xb = Xb.cpu().detach().numpy()
    Xf = Xf.cpu().detach().numpy()

    
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(figsize=(5.8, 4.8)) 
    plt.scatter(Xf[:,0], Xf[:,1], s=1, c='b', marker='x')
    
    plt.scatter(Xb[:,0], Xb[:,1], s=0.1, c='k', marker='x')
 

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

def plot_dis(N_test_dis):
    x = np.linspace(0, a, N_test_dis)
    y = np.linspace(0, b, N_test_dis) # y方向N_test个点
    x_mesh, y_mesh = np.meshgrid(x, y)
    Xf = np.stack((x_mesh.flatten(), y_mesh.flatten()), 1)
    index0 = []
    #Xf = np.copy(Xf0)
    for i, (x, y) in enumerate(Xf): # determine whether the points is in the circle or not.

        #x, y = x*x0, y*y0
        good = 0
        for cir_info in cir:
            if (circle(cir_info[0], cir_info[1], cir_info[2], cir_info[3], x, y) > 0):
                good = good + 1
        if good == len(cir): 
            index0.append(i)
    Xf = Xf[index0] # the domian points without the circle points
    Xf_tensor = torch.tensor(Xf, dtype=torch.float64, requires_grad=True)
    dis_tensor = model_dis(Xf_tensor)
    dis = dis_tensor.cpu().detach().numpy()
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(figsize=(5.8, 4.8)) 
    surf = ax.scatter(Xf[:, 0], Xf[:, 1], c = dis, cmap=cm.rainbow)# vmin=0, vmax=0.14, 
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
    
Xf_tensor, boundary_point_tensor = train_data_uniform(Nb, Nf, cir, add_corn)
plot_pos(boundary_point_tensor, Xf_tensor)

boundary_point = boundary_point_tensor.cpu().detach().numpy()
Xf = Xf_tensor.cpu().detach().numpy()

kdt = KDTree(boundary_point, metric='euclidean') # 将本质边界条件封装成一个对象

d_dir, _ = kdt.query(boundary_point, k=1, return_distance = True) # obtain the target distance of the boundary points
d_dom, _ = kdt.query(Xf, k=1, return_distance = True) # obtain the target distance of the domain points

input_d = np.concatenate((boundary_point, Xf)) # concatenate the input and the output
output_d = np.concatenate((d_dir, d_dom))

# transform the numpy to tensor for training
input_d_tensor =  torch.tensor(input_d, device='cuda')
output_d_tensor =  torch.tensor(output_d, device='cuda')

loss_dis = 100
epoch_b = 0
criterion = torch.nn.MSELoss()

loss_dis_array = []
if training_dis ==1:
    model_dis = distance_net(2, 20, 1).cuda() # 定义两个特解，这是因为在裂纹处是间断值
    optim = torch.optim.Adam(params=model_dis.parameters(), lr= 0.001)
    scheduler = MultiStepLR(optim, milestones=[5000, 10000, 30000, 40000, 50000], gamma = 0.5)
    while loss_dis>tol_dis:
        epoch_b = epoch_b + 1
        def closure():  
            pred_d = model_dis(input_d_tensor) # predict the boundary condition
            loss_dis = criterion(pred_d, output_d_tensor)
            optim.zero_grad()
            loss_dis.backward()
            loss_dis_array.append(loss_dis.data.cpu())
            if epoch_b%10==0:
                print('trianing particular network : the number of epoch is %i, the loss is %f' % (epoch_b, loss_dis.data))
            return loss_dis
        optim.step(closure)
        scheduler.step()
        loss_dis = loss_dis_array[-1]
    torch.save(model_dis, './distance_nn_DEM_%ihole_fixed' % len(cir))
model_dis = torch.load( './distance_nn_DEM_%ihole_fixed' % len(cir))
plot_dis(100)
#%% dis_n
def evaluate_dis(N_test):
    equal_line_x = np.linspace(0, a, N_test)
    equal_line_y = np.linspace(0, b, N_test)
# circle points
    circle_boundary = np.ones([len(cir)*N_test, 2])
    theta = np.linspace(0, 2*np.pi, N_test)
    for index, eve_cir in enumerate(cir):
      x = eve_cir[0] + np.cos(theta) * (eve_cir[2]**2)**0.5   # the circle is default.
      y = eve_cir[1] + np.sin(theta) * (eve_cir[2]**2)**0.5
      circle_boundary[Nb*index: Nb*(index+1), 0] = x
      circle_boundary[Nb*index: Nb*(index+1), 1] = y  
    
    left_boundary = np.stack([0*np.ones(len(equal_line_y)), equal_line_y], 1)
    
      
    Xb_left = torch.tensor(left_boundary,  requires_grad=True, device='cuda')

    
    Xb_circle = torch.tensor(circle_boundary,  requires_grad=True, device='cuda')
  
    pred_dis_left = model_dis(Xb_left)        
   
    pred_t_n_circle_tensor = torch.ones([len(cir)*N_test, 1], device='cuda')
    for index, eve_cir in enumerate(cir):
        Xb_eve_cir = Xb_circle[N_test*index: N_test*(index+1)]
        pred_dis_circle = model_dis(Xb_eve_cir)**2
        disdxy_circle = grad(pred_dis_circle, Xb_eve_cir, torch.ones(Xb_eve_cir.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]            
        disdx_circle = disdxy_circle[:, 0].unsqueeze(1)   
        disdy_circle = disdxy_circle[:, 1].unsqueeze(1)   
        pred_t_n_circle_tensor[N_test*index: N_test*(index+1)] = (Xb_eve_cir[:, 0]-eve_cir[0]).unsqueeze(1) *disdx_circle +  (Xb_eve_cir[:, 1]-eve_cir[1]).unsqueeze(1) *disdy_circle
   
    pred_dis_left_array = pred_dis_left.data.cpu().numpy()
    circle_n_array = pred_t_n_circle_tensor.data.cpu().numpy()
    return pred_dis_left_array, circle_n_array

equal_line_x = np.linspace(0, a, Nb)
equal_line_y = np.linspace(0, b, Nb)
# circle points
theta = np.linspace(0, 2*np.pi, Nb)

pred_dis_left_array, circle_n_array = evaluate_dis(Nb)

internal = 1
plt.plot(equal_line_y, pred_dis_left_array)
plt.plot(theta, circle_n_array)
plt.legend(['disdy_down', 'disdy_right', 'disdy_up', 'circle'])
plt.xlabel('coordinate')
plt.ylabel('dis_n')
plt.show()
