# 这是构造可能位移场的程序，不再用距离神经网络，而是用RBF来解析的构造，虽然不是精确满足，但是精度已经非常的好了
import numpy as np
import torch
import matplotlib.pyplot as plt
from  torch.autograd import grad
from sklearn.neighbors import KDTree
import time
from itertools import chain
from pyevtk.hl import gridToVTK
from pyevtk.hl import pointsToVTK
import meshio

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # 设置随机数种子
torch.set_default_tensor_type(torch.DoubleTensor) # 将tensor的类型变成默认的double
torch.set_default_tensor_type(torch.cuda.DoubleTensor) # 将cuda的tensor的类型变成默认的double
femdis = meshio.read('./output/fem2d/elasticity/displacement000000.vtu') # 读入有限元的位移解
femvon = meshio.read('./output/fem2d/elasticity/von_mises000000.vtu') # 读入有限元的位移解
femdisx = sorted(femdis.point_data.values())[0][:,0] # 得到有限元相应的y方向的位移解，这是一个一维度的ARRAY，用来评估error
femdisy = sorted(femdis.point_data.values())[0][:,1] # 得到有限元相应的y方向的位移解，这是一个一维度的ARRAY，用来评估error
femdisz =sorted(femdis.point_data.values())[0][:,2] # 得到有限元相应的y方向的位移解，这是一个一维度的ARRAY，用来评估error
femvon = sorted(femvon.point_data.values())[0] # 得到有限元的vonmise应力，是一个一维度的array
train_p = 0
nepoch_u0 = 10000 # the float type
penalty = 700
gama = 0.2
E0 = 1000
nu0 = 0.3
E1 = 10000
nu1 = 0.3
nx = 400
ny = 100
param_c1 = 630
param_c2 = -1.2
param_c = 100
ty = -5
dd = 0
setup_seed(dd)
nepoch_u0 = 100*dd + 10000
nepoch_u0 = 10000
def write_vtk_vts(filename, x_space, y_space, z_space, Ux, Uy, Uz, S11, S12, S22, E11, E12, E22, SVonMises):
    #已经将输出的感兴趣场进行了分类VTK导出,用VTs格式方便数据可视化
    xx, yy, zz = np.meshgrid(x_space, y_space, z_space)
    gridToVTK(filename, xx, yy, zz, pointData={ "displacement": (Ux, Uy, Uz), \
                                               "S-VonMises": SVonMises, \
                                               "S11": S11, "S12": S12, "S22": S22, \
                                               "E11": E11, "E12": E12, "E22": E22
                                               })


def write_vtk_vtu(filename, x_space, y_space, z_space, Ux, Uy, Uz, S11, S12, S22, E11, E12, E22, SVonMises):
    #已经将输出的感兴趣场进行了分类VTK导出,用VTU格式方便数据转化为array，所以用了flatten，并且pointsToVTK
    xx, yy, zz = np.meshgrid(x_space, y_space, z_space)
    pointsToVTK(filename, xx.flatten(), yy.flatten(), zz.flatten(), data={ "displacementx": Ux.flatten(), "displacementy": Uy.flatten(), "displacementz": Uz.flatten(), \
                                               "S-VonMises": SVonMises.flatten(), \
                                               "S11": S11.flatten(), "S12": S12.flatten(),  "S22": S22.flatten(),\
                                               "E11": E11.flatten(), "E12": E12.flatten(), "E22": E22.flatten()
                                               })

# 定义一个能量class
class Material:
    # ---------------------------------------------------------------------------------------------------------------------------
    def __init__(self, energy, E=None, nu=None, param_c1=None, param_c2=None, param_c=None, rou = 1000):
        """
        

        Parameters
        ----------
        energy : TYPE
            DESCRIPTION.
        dim : TYPE
            DESCRIPTION.
        E : TYPE, optional
            DESCRIPTION. The default is None.
        nu : TYPE, optional
            DESCRIPTION. The default is None.
        param_c1 : TYPE, optional
            DESCRIPTION. The default is None.
        param_c2 : TYPE, optional
            DESCRIPTION. The default is None.
        param_c : TYPE, optional
            DESCRIPTION. The default is None.
        rou : float, optional
            The density of the material. The default is 1000.

        Returns
        -------
        None.

        """
        self.type = energy
        if self.type == 'neohookean':
            self.mu = E / (2 * (1 + nu))
            self.lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
        if self.type == 'mooneyrivlin':
            self.param_c1 = param_c1
            self.param_c2 = param_c2
            self.param_c = param_c
            self.param_d = 2 * (self.param_c1 + 2 * self.param_c2)
    def getStoredEnergy(self, u, x):
        if self.type == 'neohookean':
            return self.NeoHookean2D(u, x) # 将每一个点位移以及位置输入到这个函数中，获得每一个点的应变能密度
        if self.type == 'mooneyrivlin':
            return self.MooneyRivlin2D(u, x)
    def MooneyRivlin2D(self, u, x):
        duxdxy = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device='cuda'), create_graph=True, retain_graph=True)[0]
        duydxy = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device='cuda'), create_graph=True, retain_graph=True)[0]
        Fxx = duxdxy[:, 0].unsqueeze(1) + 1
        Fxy = duxdxy[:, 1].unsqueeze(1) + 0
        Fyx = duydxy[:, 0].unsqueeze(1) + 0
        Fyy = duydxy[:, 1].unsqueeze(1) + 1
        detF = Fxx * Fyy - Fxy * Fyx
        C11 = Fxx * Fxx + Fyx * Fyx
        C12 = Fxx * Fxy + Fyx * Fyy
        C21 = Fxy * Fxx + Fyy * Fyx
        C22 = Fxy * Fxy + Fyy * Fyy
        J = detF
        traceC = C11 + C22
        I1 = traceC
        trace_C2 = C11 * C11 + C12 * C21 + C21 * C12 + C22 * C22
        I2 = 0.5 * (traceC ** 2 - trace_C2)
        strainEnergy = self.param_c * (J - 1) ** 2 - self.param_d * torch.log(J) + self.param_c1 * (I1 - 2) + self.param_c2 * (I2 - 1)
        return strainEnergy
    def NeoHookean2D(self, u, x):
        duxdxy = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device='cuda'), create_graph=True, retain_graph=True)[0]
        duydxy = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device='cuda'), create_graph=True, retain_graph=True)[0]
        Fxx = duxdxy[:, 0].unsqueeze(1) + 1
        Fxy = duxdxy[:, 1].unsqueeze(1) + 0
        Fyx = duydxy[:, 0].unsqueeze(1) + 0
        Fyy = duydxy[:, 1].unsqueeze(1) + 1
        detF = Fxx * Fyy - Fxy * Fyx
        trC = Fxx ** 2 + Fxy ** 2 + Fyx ** 2 + Fyy ** 2
        strainEnergy = 0.5 * self.lam * (torch.log(detF) * torch.log(detF)) - self.mu * torch.log(detF) + 0.5 * self.mu * (trC - 2)
        return strainEnergy

            
            

def interface(Ni): # 交界面点
    '''
     生成裂纹尖端上半圆的点，为了多分配点
    '''
    x = 4*np.random.rand(Ni)
    y = 0.5*np.ones(Ni)
    xi = np.stack([x, y], 1)
    xi = torch.tensor(xi, requires_grad=True, device='cuda')
    return xi

def essential_bound(Ni): # 本质边界点
    '''
     生成裂纹尖端上半圆的点，为了多分配点
    '''
    x = np.zeros(Ni)
    y = np.random.rand(Ni)
    xeb = np.stack([x, y], 1)
    xeb = torch.tensor(xeb, requires_grad=True, device='cuda')
    return xeb

def train_data(Nb, Nf): # 上下两个域以及力边界条件点
    '''
    生成强制边界点，四周以及裂纹处
    生成上下的内部点
    '''
    

    x = 4*np.ones(Nb)
    y = np.random.rand(Nb)
    xnb = np.stack([x, y], 1)
    xnb = torch.tensor(xnb, requires_grad=True, device='cuda') # 生成力边界条件
    
    
    x = 4*np.random.rand(Nf)
    y = np.random.rand(Nf)
    xf = np.stack([x, y], 1)
    xf = torch.tensor(xf, requires_grad=True, device='cuda') # 生成力边界条件
    

    xf1 = xf[(xf[:, 1]>0.5)] # 上区域点，去除内部多配的点
    xf2 = xf[(xf[:, 1]<0.5)]  # 下区域点，去除内部多配的点
    
    xf1 = torch.tensor(xf1, requires_grad=True, device='cuda')
    xf2 = torch.tensor(xf2, requires_grad=True, device='cuda')
    
    return xnb, xf1, xf2

# for the u0 solution satisfying the homogenous boundary condition
class homo(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(homo, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, D_out)
        
        # self.a1 = torch.nn.Parameter(torch.Tensor([0.2]))
        

        self.a1 = torch.Tensor([0.1]).cuda()
        self.a2 = torch.Tensor([0.1])
        self.a3 = torch.Tensor([0.1])
        self.n = 1/self.a1.data
        torch.nn.init.normal_(self.linear1.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear2.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear3.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear4.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear5.bias, mean=0, std=1)

        torch.nn.init.normal_(self.linear1.weight, mean=0, std=np.sqrt(2/(D_in+H)))
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear3.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear4.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear5.weight, mean=0, std=np.sqrt(2/(H+D_out)))

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
    
def pred(xy): # 从坐标的位置判断用哪个神经网络
    '''
    

    Parameters
    ----------
    Nb : int
        the  number of boundary point.
    Nf : int
        the  number of internal point.

    Returns
    -------
    Xb : tensor
        The boundary points coordinates.
    Xf1 : tensor
        interior hole.
    Xf2 : tensor
        exterior region.
    '''

    pred = xy[:,0].unsqueeze(1) * model_h(xy)
    return pred    
    
def evaluate_model(material, x, y, z):
    energy_type = material.type
    if energy_type == 'neohookean':
        mu = material.mu
        lmbda = material.lam
        Nx = len(x)
        Ny = len(y)
        xGrid, yGrid = np.meshgrid(x, y)
        x1D = xGrid.flatten()
        y1D = yGrid.flatten()
        xy = np.concatenate((np.array([x1D]).T, np.array([y1D]).T), axis=-1)
        xy_tensor = torch.from_numpy(xy)
        xy_tensor = xy_tensor.cuda()
        xy_tensor.requires_grad_(True)
        # u_pred_torch = self.model(xy_tensor)
        u_pred_torch = pred(xy_tensor)
        duxdxy = grad(u_pred_torch[:, 0].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device='cuda'),
                       create_graph=True, retain_graph=True)[0]
        duydxy = grad(u_pred_torch[:, 1].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device='cuda'),
                       create_graph=True, retain_graph=True)[0]
        F11 = duxdxy[:, 0].unsqueeze(1) + 1
        F12 = duxdxy[:, 1].unsqueeze(1) + 0
        F21 = duydxy[:, 0].unsqueeze(1) + 0
        F22 = duydxy[:, 1].unsqueeze(1) + 1
        detF = F11 * F22 - F12 * F21
        invF11 = F22 / detF
        invF22 = F11 / detF
        invF12 = -F12 / detF
        invF21 = -F21 / detF
        C11 = F11**2 + F21**2
        C12 = F11*F12 + F21*F22
        C21 = F12*F11 + F22*F21
        C22 = F12**2 + F22**2
        E11 = 0.5 * (C11 - 1)
        E12 = 0.5 * C12
        E21 = 0.5 * C21
        E22 = 0.5 * (C22 - 1)

        P11 = mu * F11 + (lmbda * torch.log(detF) - mu) * invF11
        P12 = mu * F12 + (lmbda * torch.log(detF) - mu) * invF21
        P21 = mu * F21 + (lmbda * torch.log(detF) - mu) * invF12
        P22 = mu * F22 + (lmbda * torch.log(detF) - mu) * invF22
        S11 = invF11 * P11 + invF12 * P21
        S12 = invF11 * P12 + invF12 * P22
        S21 = invF21 * P11 + invF22 * P21
        S22 = invF21 * P12 + invF22 * P22
        u_pred = u_pred_torch.detach().cpu().numpy()
        F11_pred = F11.detach().cpu().numpy()
        F12_pred = F12.detach().cpu().numpy()
        F21_pred = F21.detach().cpu().numpy()
        F22_pred = F22.detach().cpu().numpy()
        E11_pred = E11.detach().cpu().numpy()
        E12_pred = E12.detach().cpu().numpy()
        E21_pred = E21.detach().cpu().numpy()
        E22_pred = E22.detach().cpu().numpy()
        S11_pred = S11.detach().cpu().numpy()
        S12_pred = S12.detach().cpu().numpy()
        S21_pred = S21.detach().cpu().numpy()
        S22_pred = S22.detach().cpu().numpy()
        surUx = u_pred[:, 0].reshape(Ny, Nx, 1)
        surUy = u_pred[:, 1].reshape(Ny, Nx, 1)
        surUz = np.zeros([Ny, Nx, 1])
        surE11 = E11_pred.reshape(Ny, Nx, 1)
        surE12 = E12_pred.reshape(Ny, Nx, 1)
        surE13 = np.zeros([Nx, Ny, 1])
        surE21 = E21_pred.reshape(Ny, Nx, 1)
        surE22 = E22_pred.reshape(Ny, Nx, 1)
        surE23 = np.zeros([Nx, Ny, 1])
        surE33 = np.zeros([Nx, Ny, 1])
        surS11 = S11_pred.reshape(Ny, Nx, 1)
        surS12 = S12_pred.reshape(Ny, Nx, 1)
        surS13 = np.zeros([Nx, Ny, 1])
        surS21 = S21_pred.reshape(Ny, Nx, 1)
        surS22 = S22_pred.reshape(Ny, Nx, 1)
        surS23 = np.zeros([Nx, Ny, 1])
        surS33 = np.zeros([Nx, Ny, 1])
        SVonMises = np.float64(np.sqrt(0.5 * ((surS11 - surS22) ** 2 + (surS22) ** 2 + (-surS11) ** 2 + 6 * (surS12 ** 2))))
        U = [np.float64(surUx), np.float64(surUy), np.float64(surUz)]
        return U, np.float64(surS11), np.float64(surS12), np.float64(surS22), \
            np.float64(surE11), np.float64(surE12), np.float64(surE22), \
                np.float64(SVonMises), np.float64(F11_pred), np.float64(F12_pred), np.float64(F21_pred), np.float64(F22_pred)
    if energy_type == 'mooneyrivlin':
        c1 = material.param_c1
        c2 = material.param_c2
        c = material.param_c
        d = material.param_d
        Nx = len(x)
        Ny = len(y)
        xGrid, yGrid = np.meshgrid(x, y)
        x1D = xGrid.flatten()
        y1D = yGrid.flatten()
        xy = np.concatenate((np.array([x1D]).T, np.array([y1D]).T), axis=-1)
        xy_tensor = torch.from_numpy(xy)
        xy_tensor = xy_tensor.cuda()
        xy_tensor.requires_grad_(True)
        # u_pred_torch = self.model(xy_tensor)
        u_pred_torch = pred(xy_tensor)
        duxdxy = \
        grad(u_pred_torch[:, 0].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device='cuda'),
             create_graph=True, retain_graph=True)[0]
        duydxy = \
        grad(u_pred_torch[:, 1].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device='cuda'),
             create_graph=True, retain_graph=True)[0]
        F11 = duxdxy[:, 0].unsqueeze(1) + 1
        F12 = duxdxy[:, 1].unsqueeze(1) + 0
        F21 = duydxy[:, 0].unsqueeze(1) + 0
        F22 = duydxy[:, 1].unsqueeze(1) + 1
        detF = F11 * F22 - F12 * F21
        C11 = F11 ** 2 + F21 ** 2
        C12 = F11 * F12 + F21 * F22
        C21 = F12 * F11 + F22 * F21
        C22 = F12 ** 2 + F22 ** 2
        detC = C11 * C22 - C12 * C21
        invC11 = C22 / detC
        invC22 = C11 / detC
        invC12 = -C12 / detC
        invC21 = -C21 / detC
        J = detC ** 0.5
        I1= C11 + C22
        trC2 = C11*C11 + C12*C21 + C21*C12 + C22*C22
        I2 = 0.5*(I1**2 - trC2)
        S11 = (2 * c1 + 2 * c2 * I1) - 2 * c2 * C11 + (2 * c * (J - 1) * J - d) * invC11
        S12 = - 2 * c2 * C21 + (2 * c * (J - 1) * J - d) * invC21
        S21 = - 2 * c2 * C12 + (2 * c * (J - 1) * J - d) * invC12
        S22 = (2 * c1 + 2 * c2 * I1) - 2 * c2 * C22 + (2 * c * (J - 1) * J - d) * invC22
        E11 = 0.5 * (C11 - 1)
        E12 = 0.5 * C12
        E21 = 0.5 * C21
        E22 = 0.5 * (C22 - 1)
        u_pred = u_pred_torch.detach().cpu().numpy()
        F11_pred = F11.detach().cpu().numpy()
        F12_pred = F12.detach().cpu().numpy()
        F21_pred = F21.detach().cpu().numpy()
        F22_pred = F22.detach().cpu().numpy()
        E11_pred = E11.detach().cpu().numpy()
        E12_pred = E12.detach().cpu().numpy()
        E21_pred = E21.detach().cpu().numpy()
        E22_pred = E22.detach().cpu().numpy()
        S11_pred = S11.detach().cpu().numpy()
        S12_pred = S12.detach().cpu().numpy()
        S21_pred = S21.detach().cpu().numpy()
        S22_pred = S22.detach().cpu().numpy()
        surUx = u_pred[:, 0].reshape(Ny, Nx, 1)
        surUy = u_pred[:, 1].reshape(Ny, Nx, 1)
        surUz = np.zeros([Nx, Ny, 1])
        surE11 = E11_pred.reshape(Ny, Nx, 1)
        surE12 = E12_pred.reshape(Ny, Nx, 1)
        surE13 = np.zeros([Nx, Ny, 1])
        surE21 = E21_pred.reshape(Ny, Nx, 1)
        surE22 = E22_pred.reshape(Ny, Nx, 1)
        surE23 = np.zeros([Nx, Ny, 1])
        surE33 = np.zeros([Nx, Ny, 1])
        surS11 = S11_pred.reshape(Ny, Nx, 1)
        surS12 = S12_pred.reshape(Ny, Nx, 1)
        surS13 = np.zeros([Nx, Ny, 1])
        surS21 = S21_pred.reshape(Ny, Nx, 1)
        surS22 = S22_pred.reshape(Ny, Nx, 1)
        surS23 = np.zeros([Nx, Ny, 1])
        surS33 = np.zeros([Nx, Ny, 1])
        SVonMises = np.float64(
            np.sqrt(0.5 * ((surS11 - surS22) ** 2 + (surS22) ** 2 + (-surS11) ** 2 + 6 * (surS12 ** 2))))
        U = [np.float64(surUx), np.float64(surUy), np.float64(surUz)]
        return U, np.float64(surS11), np.float64(surS12), np.float64(surS22), \
            np.float64(surE11), np.float64(surE12), np.float64(surE22), \
                np.float64(SVonMises), np.float64(F11_pred), np.float64(F12_pred), np.float64(F21_pred), np.float64(F22_pred)




# get f_p from model_p 

# learning the homogenous network



model_h = homo(2, 15, 2).cuda()
criterion = torch.nn.MSELoss()
optim_h = torch.optim.Adam(params=model_h.parameters(), lr= 0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optim_h, milestones=[5000, 10000, 30000], gamma = 0.1)
loss_array = []
errorL2_array = []
errorvon_array = []
loss1_array = []
loss2_array = []   
lossi_array = []

nepoch_u0 = int(nepoch_u0)
start = time.time()

neo0 = Material('neohookean', E0, nu0)
neo1 = Material('neohookean', E1, nu1)
moon = Material('mooneyrivlin', param_c1=param_c1, param_c2=param_c2, param_c=param_c)
for epoch in range(nepoch_u0):
    
    if epoch ==1000:
        end = time.time()
        consume_time = end-start
        print('time is %f' % consume_time)
    if epoch%100 == 0:
        Xn, Xf1, Xf2 = train_data(256, 4096)

    def closure():  
        u_pred1 = pred(Xf1)
        u_pred2 = pred(Xf2)
        storedEnergy1 = neo0.getStoredEnergy(u_pred1, Xf1)
        storedEnergy2 = neo1.getStoredEnergy(u_pred2, Xf2)

        J1 = torch.sum(storedEnergy1) * 2/len(Xf1) 
        
        J2 = torch.sum(storedEnergy2) * 2/len(Xf2) 
        #  构造外力功
        u_predn = pred(Xn)
        external_density = u_predn[:, 1] * (ty) # 外力功密度
        external = torch.sum(external_density) * 1 / len(external_density)
        
        loss = J1 + J2 - external  # the functional
        optim_h.zero_grad()
        loss.backward()
        loss_array.append(loss.data.cpu())
        loss1_array.append(J1.data.cpu())
        loss2_array.append(J2.data.cpu())
        
        if epoch%10==0:
            x = np.linspace(0, 4, nx+1)
            y = np.linspace(0, 1, ny+1)
            z = np.array([0])
            xyz = np.meshgrid(x, y, z)
            # 分别得到上下两个区域全部的输出场
            Uu, S11u, S12u, S22u, E11u, E12u,  E22u, SVonMisesu, F11u, F12u, F21u, F22u = evaluate_model(neo0, x, y, z)
            Ud, S11d, S12d, S22d, E11d, E12d,  E22d, SVonMisesd, F11d, F12d, F21d, F22d = evaluate_model(neo1, x, y, z)
            # 将上面的neo场的下半部分用moon的下半部分rewrite，通过xyz[1] 的y坐标进行识别
            Uu[0][xyz[1]<0.5] = Ud[0][xyz[1]<0.5] # Uu是一个list进行改动，原先的程序时一个tuple不可以更改,将UD作为一个基准
            Uu[1][xyz[1]<0.5] = Ud[1][xyz[1]<0.5]
            Uux = Uu[0].copy() # flags出了问题
            Uuy = Uu[1].copy()
            Uuz = Uu[2].copy()
            S11u[xyz[1]<0.5] = S11d[xyz[1]<0.5]
            S12u[xyz[1]<0.5] = S12d[xyz[1]<0.5]
            S22u[xyz[1]<0.5] = S22d[xyz[1]<0.5]

            E11u[xyz[1]<0.5] = E11d[xyz[1]<0.5]
            E12u[xyz[1]<0.5] = E12d[xyz[1]<0.5]
            E22u[xyz[1]<0.5] = E22d[xyz[1]<0.5]

            SVonMisesu[xyz[1]<0.5] = SVonMisesd[xyz[1]<0.5]
            # F11u[xyz[1]<0.5] = F11d[xyz[1]<0.5]
            # F12u[xyz[1]<0.5] = F12d[xyz[1]<0.5]
            # F21u[xyz[1]<0.5] = F21d[xyz[1]<0.5]
            # F22u[xyz[1]<0.5] = F22d[xyz[1]<0.5] # 上述输出的全部是array，但是是具有grid结构的
            Upredmag = np.sqrt(Uux**2+Uuy**2+Uuz**2).transpose(2,0,1).flatten() # 得到位移的输出大小，这里是包含每一个点的预测,由于meshgrid前面两个维度是相反的，所以我们将前面两个维度进行变换
            Ufemmag = np.sqrt(femdisx**2+femdisy**2+femdisz**2) # 得到位移的有限元的大小，这里是包含每一个点的预测
            predvon = SVonMisesu.transpose(2,0,1).flatten()
            
            
            # 因为fenics在中心面插值不准确，所以我们去处中性面附近的点，即是去除y等于0.48-0.52（包含）的点，在
            # 一维的array是去除[401*47:401*53]
            delete_idx = range(401*47,401*53)
            Upredmag_d = np.delete(Upredmag, delete_idx)
            Ufemmag_d = np.delete(Ufemmag, delete_idx)
            predvon_d = np.delete(predvon, delete_idx)
            femvon_d = np.delete(femvon, delete_idx)
            
            errorL2 = np.linalg.norm(Upredmag_d-Ufemmag_d)/np.linalg.norm(Ufemmag_d)
            errorvon = np.abs((np.linalg.norm(predvon_d) - np.linalg.norm(femvon_d))/np.linalg.norm(femvon_d))

            errorL2_array.append(errorL2)
            errorvon_array.append(errorvon)
            print(' epoch : %i, the loss : %f ,  J1 : %f , J2 : %f, ErrorL2 : %f, Errorvon : %f' % (epoch, loss.data, J1.data, J2.data, errorL2, errorvon))
        return loss
    optim_h.step(closure)
    scheduler.step()
    if np.isnan(np.array(loss_array[-1])): # 防止nan出现浪费时间
        break
    
# if np.isnan(np.array(loss_array[-1])):
#     continue
# %%
# for plotting 
# 将位移图，应力图进行输出
x = np.linspace(0, 4, 401)
y = np.linspace(0, 1, 101)
z = np.zeros(1)
xyz = np.meshgrid(x, y, z)
# 分别得到上下两个区域全部的输出场
Uu, S11u, S12u, S22u, E11u, E12u, E22u, SVonMisesu, F11u, F12u, F21u, F22u = evaluate_model(neo0, x, y, z)
Ud, S11d, S12d, S22d, E11d, E12d, E22d, SVonMisesd, F11d, F12d, F21d, F22d = evaluate_model(neo1, x, y, z)
# 将上面的neo场的下半部分用moon的下半部分rewrite，通过xyz[1] 的y坐标进行识别
Uu[0][xyz[1]<0.5] = Ud[0][xyz[1]<0.5] # Uu是一个list进行改动，原先的程序时一个tuple不可以更改
Uu[1][xyz[1]<0.5] = Ud[1][xyz[1]<0.5]
Uux = Uu[0].copy() # flags出了问题
Uuy = Uu[1].copy()
Uuz = Uu[2].copy()
S11u[xyz[1]<0.5] = S11d[xyz[1]<0.5]
S12u[xyz[1]<0.5] = S12d[xyz[1]<0.5]
S22u[xyz[1]<0.5] = S22d[xyz[1]<0.5]
E11u[xyz[1]<0.5] = E11d[xyz[1]<0.5]
E12u[xyz[1]<0.5] = E12d[xyz[1]<0.5]
E22u[xyz[1]<0.5] = E22d[xyz[1]<0.5]
SVonMisesu[xyz[1]<0.5] = SVonMisesd[xyz[1]<0.5]
# F11u[xyz[1]<0.5] = F11d[xyz[1]<0.5]
# F12u[xyz[1]<0.5] = F12d[xyz[1]<0.5]
# F21u[xyz[1]<0.5] = F21d[xyz[1]<0.5]
# F22u[xyz[1]<0.5] = F22d[xyz[1]<0.5]


filename_out = "./output/dem2d/neo+moon_energy_epoch%i_s%i" % (nepoch_u0, dd) # 输出的VTS文件
write_vtk_vtu(filename_out, x, y, z, Uux, Uuy, Uuz, S11u, S12u, S22u, E11u, E12u, E22u, SVonMisesu)
write_vtk_vts(filename_out, x, y, z, Uux, Uuy, Uuz, S11u, S12u, S22u, E11u, E12u, E22u, SVonMisesu)
# %%

loss_array = np.array(loss_array) # make the list to array
np.save('./loss_error/hyper_energy_loss_epoch%i_s%i' % (nepoch_u0, dd), loss_array)
errorL2_array = np.array(errorL2_array) # make the list to array
np.save('./loss_error/hyper_energy_errorL2_epoch%i_s%i' % (nepoch_u0, dd), errorL2_array)
errorvon_array = np.array(errorvon_array) # make the list to array
np.save('./loss_error/hyper_energy_errorvon_epoch%i_s%i' % (nepoch_u0, dd), errorvon_array)

femdismag = meshio.read('./output/fem2d/elasticity/magnitude000000.vtu') # 读入有限元的位移的大小
femdis = meshio.read('./output/fem2d/elasticity/displacement000000.vtu') # 读入有限元的y位移
femvon = meshio.read('./output/fem2d/elasticity/von_mises000000.vtu') # 读入有限元的位移解
femdismag = sorted(femdismag.point_data.values())[0] # 得到有限元相应的y方向的位移解，这是一个一维度的ARRAY，用来评估error
femdisy = sorted(femdis.point_data.values())[0][:, 1] # 得到有限元相应的y方向的位移解，这是一个一维度的ARRAY，用来评估error
femvon = sorted(femvon.point_data.values())[0] # 得到有限元的vonmise应力，是一个一维度的array
femcoor = femdis.points
# 画x=2的中线，# 有限元中线没有，就自己差值了
femcoorx2= femcoor[(femcoor[:, 0]==2)] # 只要用到y，所以其他两个方向的坐标无关紧要

femdismagx2 = femdismag[(femcoor[:, 0]==2)]
femvonx2 = femvon[(femcoor[:, 0]==2)]
# 接下来我们要中性面上的位移
femcoormid = femcoor[(femcoor[:, 1]==0.5)] # 只要用到x的坐标就行了,fem没有过材料中线
femdisyymid = femdisy[(femcoor[:, 1]==0.5)] # 中性面的disy
femdisyyx2 = femdisy[(femcoor[:, 0]==2)]        # x=2的disy
energyvtu = meshio.read("./output/dem2d/neo+moon_energy_epoch%i_s%i.vtu" % (nepoch_u0, dd))
######################################################################################################
energycoor = energyvtu.points
energydismag = np.sqrt(energyvtu.point_data['displacementx']**2 + energyvtu.point_data['displacementy']**2 + energyvtu.point_data['displacementz']**2).flatten()
energyvon =  energyvtu.point_data['S-VonMises'].flatten()
energydisy =  energyvtu.point_data['displacementy'].flatten()
 
# 获得中线的位移大小以及mises应力
energycoorx2 = energycoor[(energycoor[:, 0]==2)] 
energydismagx2 = energydismag[(energycoor[:, 0]==2)] # x=2的位移大小
energyvonx2 =  energyvon[(energycoor[:, 0]==2)]
energydisyx2 = energydisy[(energycoor[:, 0]==2)] 
# 中性面的位移
energycoormid = energycoor[(energycoor[:, 1]==0.5)]    
energydisyymid = energydisy[(energycoor[:, 1]==0.5)]
 
 
fig = plt.figure(figsize=(20, 12), dpi=1000)

plt.subplot(2, 3, 1)
iteration = np.array(range(int(len(loss_array)/2), len(loss_array))) # 从中间循环开始分析损失函数,因为无法使用log，由于损失是负数
plt.plot(iteration, loss_array[iteration], ls = '--')
plt.legend(loc = 'upper right')
plt.xlabel('the iteration')
plt.ylabel('loss')
plt.title('loss', fontsize = 20) 

plt.subplot(2, 3, 2) # 画L2损失
iteration = np.array(range(int(len(errorL2_array)/2), len(errorL2_array))) # 从中间循环开始分析损失函数,因为无法使用log，由于损失是负数
plt.plot(iteration*10, errorL2_array[iteration], ls = '--') # 每10次储存，所以这里循环数目乘以10
plt.legend(loc = 'upper right')
plt.xlabel('the iteration')
plt.ylabel('error L2')
plt.title('L2', fontsize = 20) 

plt.subplot(2, 3, 3) # 画L2损失
iteration = np.array(range(int(len(errorvon_array)/2), len(errorvon_array))) # 从中间循环开始分析损失函数,因为无法使用log，由于损失是负数
plt.plot(iteration*10, errorvon_array[iteration], ls = '--') # 每10次储存，所以这里循环数目乘以10
plt.legend(loc = 'upper right')
plt.xlabel('the iteration')
plt.ylabel('error von')
plt.title('VON', fontsize = 20) 
plt.suptitle('ENERGY hyper',   size = 25)

plt.subplot(2, 3, 4) # 画中线的位移大小
plt.scatter(femcoorx2[:, 1], femdismagx2, ls = '--', label = 'fem mag') # 先画参照解
plt.plot(energycoorx2[:, 1], energydismagx2, ls = ':', label = 'energy mag') # 先画参照解
plt.scatter(femcoorx2[:, 1], femdisyyx2, ls = '--', label = 'fem disy') # 先画参照解
plt.plot(energycoorx2[:, 1], energydisyx2, ls = ':', label = 'energy disy') # 先画参照解
plt.legend(loc = 'upper right')
plt.xlabel('y')
plt.ylabel('dis')
plt.title('dismag x=2', fontsize = 20) 

plt.subplot(2, 3, 5)  # 画中线的von大小
plt.scatter(femcoorx2[:, 1], femvonx2, ls = '--', label = 'fem') # 先画参照解
plt.plot(energycoorx2[:, 1], energyvonx2, ls = ':', label = 'energy') # 先画参照解
plt.legend( loc = 'upper right')
plt.xlabel('y')
plt.ylabel('von')
plt.title('von x=2', fontsize = 20) 

plt.subplot(2, 3, 6) # 画中性面的y方向位移
plt.scatter(femcoormid[:, 0], femdisyymid, ls = '--', label = 'fem') # 先画参照解
plt.plot(energycoormid[:, 0], energydisyymid, ls = ':', label = 'energy') # 先画参照解
plt.legend( loc = 'upper right')
plt.xlabel('x')
plt.ylabel('dis')
plt.title('disy y=0.5', fontsize = 20) 

plt.suptitle('ENERGY hyper  %i' % nepoch_u0,   size = 25)     


plt.savefig('./loss_error/hyper_energy_errorvon_epoch%i_s%i.pdf' % (nepoch_u0, dd))
  #  plt.show()