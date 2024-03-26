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

FEM_node = np.load("node_coordinate_abaqus_rectangle.npy")
node_stress_abaqus_rectangle = np.load("node_stress_abaqus_rectangle.npy")

FEM_mise = node_stress_abaqus_rectangle[:,0]
FEM_stress11 = node_stress_abaqus_rectangle[:,1]
FEM_stress22 = node_stress_abaqus_rectangle[:,2]
FEM_stress12 = node_stress_abaqus_rectangle[:,3]

bound_len = 0.05
index_without_boundary = np.where((FEM_node[:, 1] > bound_len)&(FEM_node[:, 0] > bound_len)&(FEM_node[:, 1] < 1-bound_len)&(FEM_node[:, 0] < 1-bound_len))
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
    
setup_seed(0)

a = 1.
b = 1.
P = 100
nepoch = 10000

N_test = 101
N_bound = 101
N_part = 1000
E = 1000
nu = 0.3
G = E/2/(1+nu)

tol_p = 0.0001
loss_p = 100
epoch_p = 0



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

# =============================================================================
# training the particular network
# =============================================================================
equal_line_x = np.linspace(0, a, N_part)
equal_line_y = np.linspace(0, b, N_part)

up_boundary = np.stack([equal_line_x, b*np.ones(len(equal_line_x))], 1) 
right_boundary = np.stack([a*np.ones(len(equal_line_y)), equal_line_y], 1)
down_boundary = np.stack([equal_line_x, 0*np.ones(len(equal_line_x))], 1)


tx_right = P*np.sin(np.pi/b*right_boundary[:,1])
tx_right = tx_right[:,np.newaxis]
ty_right = 0*np.ones(len(right_boundary))
ty_right = ty_right[:,np.newaxis]   

tx_up = 0*np.ones(len(up_boundary))
tx_up = tx_up[:,np.newaxis] 
ty_up = 0*np.ones(len(up_boundary))
ty_up = ty_up[:,np.newaxis]     

tx_down = 0*np.ones(len(down_boundary))
tx_down = tx_down[:,np.newaxis] 
ty_down = 0*np.ones(len(down_boundary))
ty_down = ty_down[:,np.newaxis]     

   

Xb_x_down = torch.tensor(down_boundary,  requires_grad=True, device='cuda')
target_x_down = torch.tensor(tx_down, device='cuda')   

Xb_y_down = torch.tensor(down_boundary,  requires_grad=True, device='cuda')
target_y_down = torch.tensor(ty_down, device='cuda')  

Xb_x_right = torch.tensor(right_boundary,  requires_grad=True, device='cuda')
target_x_right = torch.tensor(tx_right, device='cuda')   

Xb_y_right = torch.tensor(right_boundary,  requires_grad=True, device='cuda')
target_y_right = torch.tensor(ty_right, device='cuda')   

Xb_x_up = torch.tensor(up_boundary,  requires_grad=True, device='cuda')
target_x_up = torch.tensor(tx_up, device='cuda')   

Xb_y_up = torch.tensor(up_boundary,  requires_grad=True, device='cuda')
target_y_up = torch.tensor(ty_up, device='cuda')   
    

if training_part == 1:    
    loss_p_array = []
    criterion = torch.nn.MSELoss()
    start = time.time()
    model_p = particular(2, 20, 1).cuda() # 定义两个特解，这是因为在裂纹处是间断值
    optimp = torch.optim.Adam(params=model_p.parameters(), lr= 0.0005)
    scheduler = MultiStepLR(optimp, milestones=[10000, 20000, 30000, 40000, 50000], gamma = 0.5)

    
    # training the particular network for particular solution on the bonudaray
    while loss_p>tol_p:
        epoch_p = epoch_p + 1
        def closure():  

###    The mix boundary are satisfied by the penalty method
            fai_y_right = model_p(Xb_y_right) # predict the boundary condition
            dfaidxy_y_right = grad(fai_y_right, Xb_y_right, torch.ones(Xb_y_right.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]            
            dfaidx_y_right = dfaidxy_y_right[:, 0].unsqueeze(1)
            #dfaidy_y_right = dfaidxy_y_right[:, 1].unsqueeze(1)             
            dfaidxdxy_y_right = grad(dfaidx_y_right, Xb_y_right, torch.ones(Xb_y_right.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
            dfaidxdy_y_right = dfaidxdxy_y_right[:, 1].unsqueeze(1)
            loss_y_right = criterion(dfaidxdy_y_right, target_y_right)

            fai_x_up = model_p(Xb_x_up) # predict the boundary condition
            dfaidxy_x_up = grad(fai_x_up, Xb_x_up, torch.ones(Xb_x_up.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]            
            dfaidx_x_up = dfaidxy_x_up[:, 0].unsqueeze(1)
            #dfaidy_x_up = dfaidxy_x_up[:, 1].unsqueeze(1)             
            dfaidxdxy_x_up = grad(dfaidx_x_up, Xb_x_up, torch.ones(Xb_x_up.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
            dfaidxdy_x_up = dfaidxdxy_x_up[:, 1].unsqueeze(1)
            loss_x_up = criterion(dfaidxdy_x_up, target_x_up)

            fai_y_up = model_p(Xb_y_up) # predict the boundary condition
            dfaidxy_y_up = grad(fai_y_up, Xb_y_up, torch.ones(Xb_y_up.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]            
            dfaidx_y_up = dfaidxy_y_up[:, 0].unsqueeze(1)
            #dfaidy_y_up = dfaidxy_y_up[:, 1].unsqueeze(1)             
            dfaidxdxy_y_up = grad(dfaidx_y_up, Xb_y_up, torch.ones(Xb_y_up.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
            dfaidxdx_y_up = dfaidxdxy_y_up[:, 0].unsqueeze(1)
            loss_y_up = criterion(dfaidxdx_y_up, target_y_up)

            fai_x_right = model_p(Xb_x_right) # predict the boundary condition
            dfaidxy_x_right = grad(fai_x_right, Xb_x_right, torch.ones(Xb_x_right.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]            
            #dfaidx_x_right = dfaidxy_x_right[:, 0].unsqueeze(1)
            dfaidy_x_right = dfaidxy_x_right[:, 1].unsqueeze(1)             
            dfaidydxy_x_right = grad(dfaidy_x_right, Xb_x_right, torch.ones(Xb_x_right.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
            dfaidydy_x_right = dfaidydxy_x_right[:, 1].unsqueeze(1)
            loss_x_right = criterion(dfaidydy_x_right, target_x_right)

            fai_x_down = model_p(Xb_x_down) # predict the boundary condition
            dfaidxy_x_down = grad(fai_x_down, Xb_x_down, torch.ones(Xb_x_down.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]            
            dfaidx_x_down = dfaidxy_x_down[:, 0].unsqueeze(1)
            #dfaidy_x_down = dfaidxy_x_down[:, 1].unsqueeze(1)             
            dfaidxdxy_x_down = grad(dfaidx_x_down, Xb_x_down, torch.ones(Xb_x_down.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
            dfaidxdy_x_down = dfaidxdxy_x_down[:, 1].unsqueeze(1)
            loss_x_down = criterion(dfaidxdy_x_down, target_x_down)

            fai_y_down = model_p(Xb_y_down) # predict the boundary condition
            dfaidxy_y_down = grad(fai_y_down, Xb_y_down, torch.ones(Xb_y_down.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]            
            dfaidx_y_down = dfaidxy_y_down[:, 0].unsqueeze(1)
            #dfaidy_y_down = dfaidxy_y_down[:, 1].unsqueeze(1)             
            dfaidxdxy_y_down = grad(dfaidx_y_down, Xb_y_down, torch.ones(Xb_y_down.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
            dfaidxdx_y_down = dfaidxdxy_y_down[:, 0].unsqueeze(1)
            loss_y_down = criterion(dfaidxdx_y_down, target_y_down)
            
            loss_p =  loss_y_right + loss_x_up + loss_y_up + loss_x_right + loss_y_down + loss_x_down
            
            optimp.zero_grad()
            loss_p.backward()
            loss_p_array.append(loss_p.data.cpu())
            if epoch_p%10==0:
                print('epoch  %i, loss: %f, x_right: %f, y_right: %f, x_up: %f, y_up: %f, x_down: %f, y_down: %f' %\
                      (epoch_p, loss_p.data, loss_x_right.data, loss_y_right.data, loss_x_up.data, loss_y_up.data\
                       , loss_x_down.data, loss_y_down.data))
            return loss_p
        optimp.step(closure)
        scheduler.step()
        loss_p = loss_p_array[-1]
    torch.save(model_p, './particular_nn_rectangle')
model_p = torch.load('./particular_nn_rectangle')


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
    dis = factor * ((x-a)*y*(y-b))**order
    dis = dis.unsqueeze(1)
    pred_u = model_p(xy) + dis * model_g(xy) 
    return pred_u

def evaluate_sigma(N_test):# mesh spaced grid
    xy_dom_t = dom_data_uniform(N_test)
    
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
optim = torch.optim.Adam(params=model_g.parameters(), lr= 0.001)
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
xy_dom_t = dom_data_uniform(N_test)
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
        J_dom =  simpson_int_2D(J_dom_density, xy_dom_t,  nx = N_test, ny = N_test)
      
        loss =  J_dom 
        optim.zero_grad()
        loss.backward(retain_graph=True)
        loss_array.append(loss.data.cpu())
        loss_dom_array.append(J_dom.data.cpu())
        if epoch%100==0:
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
            sxx_pred = dfaidydy.data.cpu().numpy().flatten()
            syy_pred = dfaidxdx.data.cpu().numpy().flatten()
            sxy_pred = -dfaidxdy.data.cpu().numpy().flatten()
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
def write_vtk_v2p(filename, dom, S11, S12, S22): #已经将输出的感兴趣场进行了分类VTK导出
    xx = np.ascontiguousarray(dom[:, 0])
    yy = np.ascontiguousarray(dom[:, 1])
    zz = np.ascontiguousarray(dom[:, 1])*0 # 点的VTK
    S11 = S11.flatten()
    S12 = S12.flatten()
    S22 = S22.flatten()
    pointsToVTK(filename, xx, yy, zz, data={"S11": S11, "S12": S12, "S22": S22})
        
def write_vtk_v2(filename, dom, S11, S12, S22, von_mises):
    xx = np.ascontiguousarray(dom[:, 0]).reshape(N_test, N_test, 1)
    yy = np.ascontiguousarray(dom[:, 1]).reshape(N_test, N_test, 1)
    zz = 0*np.ascontiguousarray(dom[:, 1]).reshape(N_test, N_test, 1)
    gridToVTK(filename, xx, yy, zz, pointData={"S11": S11.reshape(N_test, N_test, 1), "S12": S12.reshape(N_test, N_test, 1), "S22": S22.reshape(N_test, N_test, 1), "Von_mise": von_mises.reshape(N_test, N_test, 1)})
    # gridToVTK(filename, xx, yy, zz, pointData={"displacement": U})  
    
dom, pred_sigma_x, pred_sigma_y, pred_sigma_xy, pred_von = evaluate_sigma(N_test)

write_vtk_v2("../output/DCEM_rectangle",  dom, pred_sigma_x, pred_sigma_xy, pred_sigma_y, pred_von)


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
# # x = 0.5, x = 1.0, y = 0.5
# =============================================================================
FEM_node = np.load("node_coordinate_abaqus_rectangle.npy")
node_stress_abaqus_rectangle = np.load("node_stress_abaqus_rectangle.npy")

FEM_mise = node_stress_abaqus_rectangle[:,0]
FEM_stress11 = node_stress_abaqus_rectangle[:,1]
FEM_stress22 = node_stress_abaqus_rectangle[:,2]
FEM_stress12 = node_stress_abaqus_rectangle[:,3]
    
# # x = 0.5
x05 = np.where((FEM_node[:, 0] == 0.5 ))
FEM_node_x05 = FEM_node[x05, :][0] # 
FEM_mise_x05 = FEM_mise[x05] 
FEM_stress11_x05 = FEM_stress11[x05] 
FEM_stress22_x05 = FEM_stress22[x05] 
FEM_stress12_x05 = FEM_stress12[x05] 


xy_dom_t_x05 = torch.tensor(FEM_node_x05,  requires_grad=True, device='cuda')
fai_x05 = pred(xy_dom_t_x05) # Input r and theta to the pred function to get the necessary predition stress function

dfaidxy_x05 = grad(fai_x05, xy_dom_t_x05, torch.ones(xy_dom_t_x05.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
dfaidx_x05 = dfaidxy_x05[:, 0].unsqueeze(1)
dfaidy_x05 = dfaidxy_x05[:, 1].unsqueeze(1)

dfaidxdxy_x05 = grad(dfaidx_x05,  xy_dom_t_x05, torch.ones( xy_dom_t_x05.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
dfaidxdx_x05 = dfaidxdxy_x05[:, 0].unsqueeze(1)
dfaidxdy_x05 = dfaidxdxy_x05[:, 1].unsqueeze(1)

dfaidydxy_x05 = grad(dfaidy_x05,  xy_dom_t_x05, torch.ones( xy_dom_t_x05.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
dfaidydx_x05 = dfaidydxy_x05[:, 0].unsqueeze(1)
dfaidydy_x05 = dfaidydxy_x05[:, 1].unsqueeze(1)

# 通过应力函数计算应力
sxx_pred_x05 = dfaidydy_x05.data.cpu().numpy().flatten()
syy_pred_x05 = dfaidxdx_x05.data.cpu().numpy().flatten()
sxy_pred_x05 = -dfaidxdy_x05.data.cpu().numpy().flatten()
mise_pred_x05 = np.sqrt(0.5*((sxx_pred_x05-syy_pred_x05)**2+sxx_pred_x05**2+syy_pred_x05**2+6*sxy_pred_x05**2)).flatten()

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
sxx_pred_x10 = dfaidydy_x10.data.cpu().numpy().flatten()
syy_pred_x10 = dfaidxdx_x10.data.cpu().numpy().flatten()
sxy_pred_x10 = -dfaidxdy_x10.data.cpu().numpy().flatten()
mise_pred_x10 = np.sqrt(0.5*((sxx_pred_x10-syy_pred_x10)**2+sxx_pred_x10**2+syy_pred_x10**2+6*sxy_pred_x10**2)).flatten()

# # y = 0.5
y05 = np.where((FEM_node[:, 1] == 0.5 ))
FEM_node_y05 = FEM_node[y05, :][0] # 
FEM_mise_y05 = FEM_mise[y05] 
FEM_stress11_y05 = FEM_stress11[y05] 
FEM_stress22_y05 = FEM_stress22[y05] 
FEM_stress12_y05 = FEM_stress12[y05] 


xy_dom_t_y05 = torch.tensor(FEM_node_y05,  requires_grad=True, device='cuda')
fai_y05 = pred(xy_dom_t_y05) # Input r and theta to the pred function to get the necessary predition stress function

dfaidxy_y05 = grad(fai_y05, xy_dom_t_y05, torch.ones(xy_dom_t_y05.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
dfaidx_y05 = dfaidxy_y05[:, 0].unsqueeze(1)
dfaidy_y05 = dfaidxy_y05[:, 1].unsqueeze(1)

dfaidxdxy_y05 = grad(dfaidx_y05,  xy_dom_t_y05, torch.ones( xy_dom_t_y05.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
dfaidxdx_y05 = dfaidxdxy_y05[:, 0].unsqueeze(1)
dfaidxdy_y05 = dfaidxdxy_y05[:, 1].unsqueeze(1)

dfaidydxy_y05 = grad(dfaidy_y05,  xy_dom_t_y05, torch.ones( xy_dom_t_y05.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
dfaidydx_y05 = dfaidydxy_y05[:, 0].unsqueeze(1)
dfaidydy_y05 = dfaidydxy_y05[:, 1].unsqueeze(1)

# 通过应力函数计算应力
sxx_pred_y05 = dfaidydy_y05.data.cpu().numpy().flatten()
syy_pred_y05 = dfaidxdx_y05.data.cpu().numpy().flatten()
sxy_pred_y05 = -dfaidxdy_y05.data.cpu().numpy().flatten()
mise_pred_y05 = np.sqrt(0.5*((sxx_pred_y05-syy_pred_y05)**2+sxx_pred_y05**2+syy_pred_y05**2+6*sxy_pred_y05**2)).flatten()