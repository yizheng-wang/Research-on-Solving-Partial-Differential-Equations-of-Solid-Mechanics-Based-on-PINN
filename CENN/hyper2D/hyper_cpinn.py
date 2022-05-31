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

torch.set_default_tensor_type(torch.DoubleTensor) # 将tensor的类型变成默认的double
torch.set_default_tensor_type(torch.cuda.DoubleTensor) # 将cuda的tensor的类型变成默认的double
femdis = meshio.read('./output/fem/elasticity/displacement000000.vtu') # 读入有限元的位移解
femvon = meshio.read('./output/fem/elasticity/von_mises000000.vtu') # 读入有限元的位移解
femdisx = femdis.point_data['f_9'][:, 0] # 得到有限元相应的y方向的位移解，这是一个一维度的ARRAY，用来评估error
femdisy = femdis.point_data['f_9'][:, 1] # 得到有限元相应的y方向的位移解，这是一个一维度的ARRAY，用来评估error
femdisz = femdis.point_data['f_9'][:, 2] # 得到有限元相应的y方向的位移解，这是一个一维度的ARRAY，用来评估error
femvon = femvon.point_data['f_60'] # 得到有限元的vonmise应力，是一个一维度的array
train_p = 0

penalty = 2000
gama = 0.2
E = 1000
nu = 0.3
param_c1 = 63000
param_c2 = -1.2
param_c = 100
tx = 0
ty = -5
tz = 0
b1 = 1
b2 = 1
b3 = 1 
b4 = 1 
b5 = 1
for dd in range(1):
    nepoch_u0 = 100*dd+100 # the float type
    def write_vtk_vts(filename, x_space, y_space, z_space, Ux, Uy, Uz, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises):
        #已经将输出的感兴趣场进行了分类VTK导出,用VTs格式方便数据可视化
        xx, yy, zz = np.meshgrid(x_space, y_space, z_space)
        gridToVTK(filename, xx, yy, zz, pointData={ "displacement": (Ux, Uy, Uz),\
                                                   "S-VonMises": SVonMises, \
                                                   "S11": S11, "S12": S12, "S13": S13, \
                                                   "S22": S22, "S23": S23, "S33": S33, \
                                                   "E11": E11, "E12": E12, "E13": E13, \
                                                   "E22": E22, "E23": E23, "E33": E33\
                                                   })
    
    
    def write_vtk_vtu(filename, x_space, y_space, z_space, Ux, Uy, Uz, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises):
        #已经将输出的感兴趣场进行了分类VTK导出,用VTU格式方便数据转化为array，所以用了flatten，并且pointsToVTK
        xx, yy, zz = np.meshgrid(x_space, y_space, z_space)
        pointsToVTK(filename, xx.flatten(), yy.flatten(), zz.flatten(), data={ "displacementx": Ux.flatten(), "displacementy": Uy.flatten(), "displacementz": Uz.flatten(), \
                                                   "S-VonMises": SVonMises.flatten(), \
                                                   "S11": S11.flatten(), "S12": S12.flatten(), "S13": S13.flatten(), \
                                                   "S22": S22.flatten(), "S23": S23.flatten(), "S33": S33.flatten(), \
                                                   "E11": E11.flatten(), "E12": E12.flatten(), "E13": E13.flatten(), \
                                                   "E22": E22.flatten(), "E23": E23.flatten(), "E33": E33.flatten()\
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
                return self.NeoHookean3D(u, x) # 将每一个点位移以及位置输入到这个函数中，获得每一个点的应变能密度
            if self.type == 'mooneyrivlin':
                return self.MooneyRivlin3D(u, x)
        def MooneyRivlin3D(self, u, x):
            duxdxyz = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device='cuda'), create_graph=True, retain_graph=True)[0]
            duydxyz = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device='cuda'), create_graph=True, retain_graph=True)[0]
            duzdxyz = grad(u[:, 2].unsqueeze(1), x, torch.ones(x.size()[0], 1, device='cuda'), create_graph=True, retain_graph=True)[0]
            Fxx = duxdxyz[:, 0].unsqueeze(1) + 1
            Fxy = duxdxyz[:, 1].unsqueeze(1) + 0
            Fxz = duxdxyz[:, 2].unsqueeze(1) + 0
            Fyx = duydxyz[:, 0].unsqueeze(1) + 0
            Fyy = duydxyz[:, 1].unsqueeze(1) + 1
            Fyz = duydxyz[:, 2].unsqueeze(1) + 0
            Fzx = duzdxyz[:, 0].unsqueeze(1) + 0
            Fzy = duzdxyz[:, 1].unsqueeze(1) + 0
            Fzz = duzdxyz[:, 2].unsqueeze(1) + 1
            detF = Fxx * (Fyy * Fzz - Fyz * Fzy) - Fxy * (Fyx * Fzz - Fyz * Fzx) + Fxz * (Fyx * Fzy - Fyy * Fzx)
            C11 = Fxx ** 2 + Fyx ** 2 + Fzx ** 2
            C12 = Fxx * Fxy + Fyx * Fyy + Fzx * Fzy
            C13 = Fxx * Fxz + Fyx * Fyz + Fzx * Fzz
            C21 = Fxy * Fxx + Fyy * Fyx + Fzy * Fzx
            C22 = Fxy ** 2 + Fyy ** 2 + Fzy ** 2
            C23 = Fxy * Fxz + Fyy * Fyz + Fzy * Fzz
            C31 = Fxz * Fxx + Fyz * Fyx + Fzz * Fzx
            C32 = Fxz * Fxy + Fyz * Fyy + Fzz * Fzy
            C33 = Fxz ** 2 + Fyz ** 2 + Fzz ** 2
            trC = C11 + C22 + C33
            trC2 = C11*C11 + C12*C21 + C13*C31 + C21*C12 + C22*C22 + C23*C32 + C31*C13 + C32*C23 + C33*C33
            I1 = trC
            I2 = 0.5 * (trC*trC - trC2)
            J = detF
            strainEnergy = self.param_c * (J - 1) ** 2 - self.param_d * torch.log(J) + self.param_c1 * (
                    I1 - 3) + self.param_c2 * (I2 - 3)
            return strainEnergy
        def NeoHookean3D(self, u, x):
            duxdxyz = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device='cuda'), create_graph=True, retain_graph=True)[0]
            duydxyz = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device='cuda'), create_graph=True, retain_graph=True)[0]
            duzdxyz = grad(u[:, 2].unsqueeze(1), x, torch.ones(x.size()[0], 1, device='cuda'), create_graph=True, retain_graph=True)[0]
            Fxx = duxdxyz[:, 0].unsqueeze(1) + 1 # 将位移梯度变成F
            Fxy = duxdxyz[:, 1].unsqueeze(1) + 0
            Fxz = duxdxyz[:, 2].unsqueeze(1) + 0
            Fyx = duydxyz[:, 0].unsqueeze(1) + 0
            Fyy = duydxyz[:, 1].unsqueeze(1) + 1
            Fyz = duydxyz[:, 2].unsqueeze(1) + 0
            Fzx = duzdxyz[:, 0].unsqueeze(1) + 0
            Fzy = duzdxyz[:, 1].unsqueeze(1) + 0
            Fzz = duzdxyz[:, 2].unsqueeze(1) + 1
            detF = Fxx * (Fyy * Fzz - Fyz * Fzy) - Fxy * (Fyx * Fzz - Fyz * Fzx) + Fxz * (Fyx * Fzy - Fyy * Fzx) # 自己手写行列式
            trC = Fxx ** 2 + Fxy ** 2 + Fxz ** 2 + Fyx ** 2 + Fyy ** 2 + Fyz ** 2 + Fzx ** 2 + Fzy ** 2 + Fzz ** 2 # 格林张量的迹
            strainEnergy = 0.5 * self.lam * (torch.log(detF) * torch.log(detF)) - self.mu * torch.log(detF) + 0.5 * self.mu * (trC - 3) # 定义neohookean的能量密度
            return strainEnergy # 返回的是一个二维的列向量，包含了所有点的应变能密度
    
                
                
    
    def interface(Ni): # 交界面点
        '''
         生成裂纹尖端上半圆的点，为了多分配点
        '''
        x = 4*np.random.rand(Ni)
        y = 0.5*np.ones(Ni)
        z = np.random.rand(Ni)
        xi = np.stack([x, y, z], 1)
        xi = torch.tensor(xi, requires_grad=True, device='cuda')
        return xi
    
    def essential_bound(Ni): # 本质边界点
        '''
         生成裂纹尖端上半圆的点，为了多分配点
        '''
        x = np.zeros(Ni)
        y = np.random.rand(Ni)
        z = np.random.rand(Ni)
        xeb = np.stack([x, y, z], 1)
        xeb = torch.tensor(xeb, requires_grad=True, device='cuda')
        return xeb
        
    def train_data(Nb, Nf): # 上下两个域以及力边界条件点
        '''
        生成强制边界点，四周以及裂纹处
        生成上下的内部点
        '''
        
    
        x = 4*np.ones(Nb)
        y = np.random.rand(Nb)
        z = np.random.rand(Nb)
        xnb = np.stack([x, y, z], 1)
        xnb = torch.tensor(xnb, requires_grad=True, device='cuda') # 生成力边界条件
        xnb1 = xnb[(xnb[:, 1]>0.5)]
        xnb2 = xnb[(xnb[:, 1]<0.5)]
        
        x = 4*np.random.rand(Nf)
        y = np.random.rand(Nf)
        z = np.random.rand(Nf)
        xf = np.stack([x, y, z], 1)
        xf = torch.tensor(xf, requires_grad=True, device='cuda') # 生成力边界条件
        
    
        xf1 = xf[(xf[:, 1]>0.5)] # 上区域点，去除内部多配的点
        xf2 = xf[(xf[:, 1]<0.5)]  # 下区域点，去除内部多配的点
        
        xf1 = torch.tensor(xf1, requires_grad=True, device='cuda')
        xf2 = torch.tensor(xf2, requires_grad=True, device='cuda')
        
        return xnb1, xnb2, xf1, xf2
    
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
        pred = torch.zeros((len(xy), 3), device = 'cuda')
        # # 上区域的预测，由于要用到不同的特解网络
        # pred[xy[:, 1]>0.5] = RBF(xy[xy[:, 1]>0.5]) * model_h1(xy[xy[:, 1]>0.5])
        # # 下区域的预测，由于要用到不同的特解网络
        # pred[xy[:, 1]<0.5] = RBF(xy[xy[:, 1]<0.5]) * model_h2(xy[xy[:, 1]<0.5])
        # # 平局值，即是分片的交界面
        # pred[xy[:, 1]==0.5] = RBF(xy[xy[:, 1]==0.5]) * (model_h1(xy[xy[:, 1]==0.5]) + model_h2(xy[xy[:, 1]==0.5]))/2
        # 试一试乘以坐标的方式
        # 上区域的预测，由于要用到不同的特解网络
        pred[xy[:, 1]>0.5] =  model_h1(xy[xy[:, 1]>0.5])
        # 下区域的预测，由于要用到不同的特解网络
        pred[xy[:, 1]<0.5] = model_h2(xy[xy[:, 1]<0.5])
        # 平局值，即是分片的交界面
        pred[xy[:, 1]==0.5] = (model_h1(xy[xy[:, 1]==0.5]) + model_h2(xy[xy[:, 1]==0.5]))/2
        return pred    
        
    def evaluate_model( material, x, y, z): # 输入的是array
        energy_type = material.type
    
        Nx = len(x)
        Ny = len(y)
        Nz = len(z)
        xGrid, yGrid, zGrid = np.meshgrid(x, y, z)
        x1D = xGrid.flatten()
        y1D = yGrid.flatten()
        z1D = zGrid.flatten()
        xyz = np.concatenate((np.array([x1D]).T, np.array([y1D]).T, np.array([z1D]).T), axis=-1)
        xyz_tensor = torch.from_numpy(xyz)
        xyz_tensor = xyz_tensor.cuda()
        xyz_tensor.requires_grad_(True)
        # u_pred_torch = self.model(xyz_tensor)
        u_pred_torch = pred(xyz_tensor)
        duxdxyz = grad(u_pred_torch[:, 0].unsqueeze(1), xyz_tensor, torch.ones(xyz_tensor.size()[0], 1, device='cuda'),
                       create_graph=True, retain_graph=True)[0]
        duydxyz = grad(u_pred_torch[:, 1].unsqueeze(1), xyz_tensor, torch.ones(xyz_tensor.size()[0], 1, device='cuda'),
                       create_graph=True, retain_graph=True)[0]
        duzdxyz = grad(u_pred_torch[:, 2].unsqueeze(1), xyz_tensor, torch.ones(xyz_tensor.size()[0], 1, device='cuda'),
                       create_graph=True, retain_graph=True)[0]
        F11 = duxdxyz[:, 0].unsqueeze(1) + 1
        F12 = duxdxyz[:, 1].unsqueeze(1) + 0
        F13 = duxdxyz[:, 2].unsqueeze(1) + 0
        F21 = duydxyz[:, 0].unsqueeze(1) + 0
        F22 = duydxyz[:, 1].unsqueeze(1) + 1
        F23 = duydxyz[:, 2].unsqueeze(1) + 0
        F31 = duzdxyz[:, 0].unsqueeze(1) + 0
        F32 = duzdxyz[:, 1].unsqueeze(1) + 0
        F33 = duzdxyz[:, 2].unsqueeze(1) + 1
        detF = F11 * (F22 * F33 - F23 * F32) - F12 * (F21 * F33 - F23 * F31) + F13 * (F21 * F32 - F22 * F31)
        invF11 = (F22 * F33 - F23 * F32) / detF
        invF12 = -(F12 * F33 - F13 * F32) / detF
        invF13 = (F12 * F23 - F13 * F22) / detF
        invF21 = -(F21 * F33 - F23 * F31) / detF
        invF22 = (F11 * F33 - F13 * F31) / detF
        invF23 = -(F11 * F23 - F13 * F21) / detF
        invF31 = (F21 * F32 - F22 * F31) / detF
        invF32 = -(F11 * F32 - F12 * F31) / detF
        invF33 = (F11 * F22 - F12 * F21) / detF
        C11 = F11 ** 2 + F21 ** 2 + F31 ** 2
        C12 = F11 * F12 + F21 * F22 + F31 * F32
        C13 = F11 * F13 + F21 * F23 + F31 * F33
        C21 = F12 * F11 + F22 * F21 + F32 * F31
        C22 = F12 ** 2 + F22 ** 2 + F32 ** 2
        C23 = F12 * F13 + F22 * F23 + F32 * F33
        C31 = F13 * F11 + F23 * F21 + F33 * F31
        C32 = F13 * F12 + F23 * F22 + F33 * F32
        C33 = F13 ** 2 + F23 ** 2 + F33 ** 2
        E11 = 0.5 * (C11 - 1)
        E12 = 0.5 * C12
        E13 = 0.5 * C13
        E21 = 0.5 * C21
        E22 = 0.5 * (C22 - 1)
        E23 = 0.5 * C23
        E31 = 0.5 * C31
        E32 = 0.5 * C32
        E33 = 0.5 * (C33 - 1)
        if energy_type == 'neohookean':
            mu = material.mu
            lmbda = material.lam
            P11 = mu * F11 + (lmbda * torch.log(detF) - mu) * invF11
            P12 = mu * F12 + (lmbda * torch.log(detF) - mu) * invF21
            P13 = mu * F13 + (lmbda * torch.log(detF) - mu) * invF31
            P21 = mu * F21 + (lmbda * torch.log(detF) - mu) * invF12
            P22 = mu * F22 + (lmbda * torch.log(detF) - mu) * invF22
            P23 = mu * F23 + (lmbda * torch.log(detF) - mu) * invF32
            P31 = mu * F31 + (lmbda * torch.log(detF) - mu) * invF13
            P32 = mu * F32 + (lmbda * torch.log(detF) - mu) * invF23
            P33 = mu * F33 + (lmbda * torch.log(detF) - mu) * invF33
            S11 = invF11 * P11 + invF12 * P21 + invF13 * P31
            S12 = invF11 * P12 + invF12 * P22 + invF13 * P32
            S13 = invF11 * P13 + invF12 * P23 + invF13 * P33
            S21 = invF21 * P11 + invF22 * P21 + invF23 * P31
            S22 = invF21 * P12 + invF22 * P22 + invF23 * P32
            S23 = invF21 * P13 + invF22 * P23 + invF23 * P33
            S31 = invF31 * P11 + invF32 * P21 + invF33 * P31
            S32 = invF31 * P12 + invF32 * P22 + invF33 * P32
            S33 = invF31 * P13 + invF32 * P23 + invF33 * P33
            u_pred = u_pred_torch.detach().cpu().numpy()
            F11_pred = F11.detach().cpu().numpy()
            F12_pred = F12.detach().cpu().numpy()
            F13_pred = F13.detach().cpu().numpy()
            F21_pred = F21.detach().cpu().numpy()
            F22_pred = F22.detach().cpu().numpy()
            F23_pred = F23.detach().cpu().numpy()
            F31_pred = F31.detach().cpu().numpy()
            F32_pred = F32.detach().cpu().numpy()
            F33_pred = F33.detach().cpu().numpy()
            E11_pred = E11.detach().cpu().numpy()
            E12_pred = E12.detach().cpu().numpy()
            E13_pred = E13.detach().cpu().numpy()
            E21_pred = E21.detach().cpu().numpy()
            E22_pred = E22.detach().cpu().numpy()
            E23_pred = E23.detach().cpu().numpy()
            E31_pred = E31.detach().cpu().numpy()
            E32_pred = E32.detach().cpu().numpy()
            E33_pred = E33.detach().cpu().numpy()
            S11_pred = S11.detach().cpu().numpy()
            S12_pred = S12.detach().cpu().numpy()
            S13_pred = S13.detach().cpu().numpy()
            S21_pred = S21.detach().cpu().numpy()
            S22_pred = S22.detach().cpu().numpy()
            S23_pred = S23.detach().cpu().numpy()
            S31_pred = S31.detach().cpu().numpy()
            S32_pred = S32.detach().cpu().numpy()
            S33_pred = S33.detach().cpu().numpy()
            surUx = u_pred[:, 0].reshape(Ny, Nx, Nz)
            surUy = u_pred[:, 1].reshape(Ny, Nx, Nz)
            surUz = u_pred[:, 2].reshape(Ny, Nx, Nz)
            surE11 = E11_pred.reshape(Ny, Nx, Nz)
            surE12 = E12_pred.reshape(Ny, Nx, Nz)
            surE13 = E13_pred.reshape(Ny, Nx, Nz)
            surE21 = E21_pred.reshape(Ny, Nx, Nz)
            surE22 = E22_pred.reshape(Ny, Nx, Nz)
            surE23 = E23_pred.reshape(Ny, Nx, Nz)
            surE31 = E31_pred.reshape(Ny, Nx, Nz)
            surE32 = E32_pred.reshape(Ny, Nx, Nz)
            surE33 = E33_pred.reshape(Ny, Nx, Nz)
            surS11 = S11_pred.reshape(Ny, Nx, Nz)
            surS12 = S12_pred.reshape(Ny, Nx, Nz)
            surS13 = S13_pred.reshape(Ny, Nx, Nz)
            surS21 = S21_pred.reshape(Ny, Nx, Nz)
            surS22 = S22_pred.reshape(Ny, Nx, Nz)
            surS23 = S23_pred.reshape(Ny, Nx, Nz)
            surS31 = S31_pred.reshape(Ny, Nx, Nz)
            surS32 = S32_pred.reshape(Ny, Nx, Nz)
            surS33 = S33_pred.reshape(Ny, Nx, Nz)
            
            F11_pred = F11_pred.reshape(Ny, Nx, Nz)
            F12_pred = F12_pred.reshape(Ny, Nx, Nz)
            F13_pred = F13_pred.reshape(Ny, Nx, Nz)
            F21_pred = F21_pred.reshape(Ny, Nx, Nz)
            F22_pred = F22_pred.reshape(Ny, Nx, Nz)
            F23_pred = F23_pred.reshape(Ny, Nx, Nz)
            F31_pred = F31_pred.reshape(Ny, Nx, Nz)
            F32_pred = F32_pred.reshape(Ny, Nx, Nz)
            F33_pred = F33_pred.reshape(Ny, Nx, Nz)
            SVonMises = np.float64(
                np.sqrt(0.5 * ((surS11 - surS22) ** 2 + (surS22 - surS33) ** 2 + (surS33 - surS11) ** 2 + 6 * (
                        surS12 ** 2 + surS23 ** 2 + surS31 ** 2))))
            U = [np.float64(surUx), np.float64(surUy), np.float64(surUz)]
            S1 = (np.float64(surS11), np.float64(surS12), np.float64(surS13))
            S2 = (np.float64(surS21), np.float64(surS22), np.float64(surS23))
            S3 = (np.float64(surS31), np.float64(surS32), np.float64(surS33))
            E1 = (np.float64(surE11), np.float64(surE12), np.float64(surE13))
            E2 = (np.float64(surE21), np.float64(surE22), np.float64(surE23))
            E3 = (np.float64(surE31), np.float64(surE32), np.float64(surE33))
            return U, np.float64(surS11), np.float64(surS12), np.float64(surS13), np.float64(surS22), np.float64(surS23), \
                   np.float64(surS33), np.float64(surE11), np.float64(surE12), \
                   np.float64(surE13), np.float64(surE22), np.float64(surE23), np.float64(surE33), np.float64(SVonMises), \
                   np.float64(F11_pred), np.float64(F12_pred), np.float64(F13_pred), \
                   np.float64(F21_pred), np.float64(F22_pred), np.float64(F23_pred), \
                   np.float64(F31_pred), np.float64(F32_pred), np.float64(F33_pred)
        if energy_type == 'mooneyrivlin':
            c1 = material.param_c1
            c2 = material.param_c2
            c = material.param_c
            d = material.param_d
            Nx = len(x)
            Ny = len(y)
            Nz = len(z)
            xGrid, yGrid, zGrid = np.meshgrid(x, y, z)
            x1D = xGrid.flatten()
            y1D = yGrid.flatten()
            z1D = zGrid.flatten()
            xyz = np.concatenate((np.array([x1D]).T, np.array([y1D]).T, np.array([z1D]).T), axis=-1)
            xyz_tensor = torch.from_numpy(xyz)
            xyz_tensor = xyz_tensor.cuda()
            xyz_tensor.requires_grad_(True)
            # u_pred_torch = self.model(xy_tensor)
            u_pred_torch = pred(xyz_tensor)
            duxdxyz = \
            grad(u_pred_torch[:, 0].unsqueeze(1), xyz_tensor, torch.ones(xyz_tensor.size()[0], 1, device='cuda'),
                 create_graph=True, retain_graph=True)[0]
            duydxyz = \
            grad(u_pred_torch[:, 1].unsqueeze(1), xyz_tensor, torch.ones(xyz_tensor.size()[0], 1, device='cuda'),
                 create_graph=True, retain_graph=True)[0]
            duzdxyz = \
            grad(u_pred_torch[:, 2].unsqueeze(1), xyz_tensor, torch.ones(xyz_tensor.size()[0], 1, device='cuda'),
                 create_graph=True, retain_graph=True)[0]
            
            F11 = duxdxyz[:, 0].unsqueeze(1) + 1
            F12 = duxdxyz[:, 1].unsqueeze(1) + 0
            F13 = duxdxyz[:, 2].unsqueeze(1) + 0
            F21 = duydxyz[:, 0].unsqueeze(1) + 0
            F22 = duydxyz[:, 1].unsqueeze(1) + 1
            F23 = duydxyz[:, 2].unsqueeze(1) + 0
            F31 = duzdxyz[:, 0].unsqueeze(1) + 0
            F32 = duzdxyz[:, 1].unsqueeze(1) + 0
            F33 = duzdxyz[:, 2].unsqueeze(1) + 1
            detF = F11 * (F22 * F33 - F23 * F32) - F12 * (F21 * F33 - F23 * F31) + F13 * (F21 * F32 - F22 * F31)
    
            C11 = F11 ** 2 + F21 ** 2 + F31 ** 2
            C12 = F11 * F12 + F21 * F22 + F31 * F32
            C13 = F11 * F13 + F21 * F23 + F31 * F33
            C21 = F12 * F11 + F22 * F21 + F32 * F31
            C22 = F12 ** 2 + F22 ** 2 + F32 ** 2
            C23 = F12 * F13 + F22 * F23 + F32 * F33
            C31 = F13 * F11 + F23 * F21 + F33 * F31
            C32 = F13 * F12 + F23 * F22 + F33 * F32
            C33 = F13 ** 2 + F23 ** 2 + F33 ** 2
            invC11 = (C22 * C33 - C23 * C32) / detF**2
            invC12 = -(C12 * C33 - C13 * C32) / detF**2
            invC13 = (C12 * C23 - C13 * C22) / detF**2
            invC21 = -(C21 * C33 - C23 * C31) / detF**2
            invC22 = (C11 * C33 - C13 * C31) / detF**2
            invC23 = -(C11 * C23 - C13 * C21) / detF**2
            invC31 = (C21 * C32 - C22 * C31) / detF**2
            invC32 = -(C11 * C32 - C12 * C31) / detF**2
            invC33 = (C11 * C22 - C12 * C21) / detF**2
            E11 = 0.5 * (C11 - 1)
            E12 = 0.5 * C12
            E13 = 0.5 * C13
            E21 = 0.5 * C21
            E22 = 0.5 * (C22 - 1)
            E23 = 0.5 * C23
            E31 = 0.5 * C31
            E32 = 0.5 * C32
            E33 = 0.5 * (C33 - 1)
            
            J = detF
            I1= C11 + C22 + C33
            trC2 = C11*C11 + C12*C21 + C13*C31 + C21*C12 + C22*C22 + C23*C32 + C31*C13 + C32*C23 + C33*C33
            I2 = 0.5*(I1**2 - trC2)
            S11 = (2 * c1 + 2 * c2 * I1) - 2 * c2 * C11 + (2 * c * (J - 1) * J - d) * invC11
            S12 = - 2 * c2 * C21 + (2 * c * (J - 1) * J - d) * invC21
            S13 = - 2 * c2 * C31 + (2 * c * (J - 1) * J - d) * invC31
            S21 = - 2 * c2 * C12 + (2 * c * (J - 1) * J - d) * invC12
            S22 = (2 * c1 + 2 * c2 * I1) - 2 * c2 * C22 + (2 * c * (J - 1) * J - d) * invC22
            S23 = - 2 * c2 * C32 + (2 * c * (J - 1) * J - d) * invC32
            S31 = - 2 * c2 * C13 + (2 * c * (J - 1) * J - d) * invC13
            S32 = - 2 * c2 * C23 + (2 * c * (J - 1) * J - d) * invC23
            S33 = (2 * c1 + 2 * c2 * I1) - 2 * c2 * C33 + (2 * c * (J - 1) * J - d) * invC33
            
            E11 = 0.5 * (C11 - 1)
            E12 = 0.5 * C12
            E13 = 0.5 * C13
            E21 = 0.5 * C21
            E22 = 0.5 * (C22 - 1)
            E23 = 0.5 * C23
            E31 = 0.5 * C31
            E32 = 0.5 * C32
            E33 = 0.5 * (C33 - 1)
            u_pred = u_pred_torch.detach().cpu().numpy()
            F11_pred = F11.detach().cpu().numpy()
            F12_pred = F12.detach().cpu().numpy()
            F13_pred = F13.detach().cpu().numpy()
            F21_pred = F21.detach().cpu().numpy()
            F22_pred = F22.detach().cpu().numpy()
            F23_pred = F23.detach().cpu().numpy()
            F31_pred = F31.detach().cpu().numpy()
            F32_pred = F32.detach().cpu().numpy()
            F33_pred = F33.detach().cpu().numpy()
            E11_pred = E11.detach().cpu().numpy()
            E12_pred = E12.detach().cpu().numpy()
            E13_pred = E13.detach().cpu().numpy()
            E21_pred = E21.detach().cpu().numpy()
            E22_pred = E22.detach().cpu().numpy()
            E23_pred = E23.detach().cpu().numpy()
            E31_pred = E31.detach().cpu().numpy()
            E32_pred = E32.detach().cpu().numpy()
            E33_pred = E33.detach().cpu().numpy()
            S11_pred = S11.detach().cpu().numpy()
            S12_pred = S12.detach().cpu().numpy()
            S13_pred = S13.detach().cpu().numpy()
            S21_pred = S21.detach().cpu().numpy()
            S22_pred = S22.detach().cpu().numpy()
            S23_pred = S23.detach().cpu().numpy()
            S31_pred = S31.detach().cpu().numpy()
            S32_pred = S32.detach().cpu().numpy()
            S33_pred = S33.detach().cpu().numpy()
            surUx = u_pred[:, 0].reshape(Ny, Nx, Nz)
            surUy = u_pred[:, 1].reshape(Ny, Nx, Nz)
            surUz = u_pred[:, 2].reshape(Ny, Nx, Nz)
            surE11 = E11_pred.reshape(Ny, Nx, Nz)
            surE12 = E12_pred.reshape(Ny, Nx, Nz)
            surE13 = E13_pred.reshape(Ny, Nx, Nz)
            surE21 = E21_pred.reshape(Ny, Nx, Nz)
            surE22 = E22_pred.reshape(Ny, Nx, Nz)
            surE23 = E23_pred.reshape(Ny, Nx, Nz)
            surE31 = E31_pred.reshape(Ny, Nx, Nz)
            surE32 = E32_pred.reshape(Ny, Nx, Nz)
            surE33 = E33_pred.reshape(Ny, Nx, Nz)
            surS11 = S11_pred.reshape(Ny, Nx, Nz)
            surS12 = S12_pred.reshape(Ny, Nx, Nz)
            surS13 = S13_pred.reshape(Ny, Nx, Nz)
            surS21 = S21_pred.reshape(Ny, Nx, Nz)
            surS22 = S22_pred.reshape(Ny, Nx, Nz)
            surS23 = S23_pred.reshape(Ny, Nx, Nz)
            surS31 = S31_pred.reshape(Ny, Nx, Nz)
            surS32 = S32_pred.reshape(Ny, Nx, Nz)
            surS33 = S33_pred.reshape(Ny, Nx, Nz)
            
            F11_pred = F11_pred.reshape(Ny, Nx, Nz)
            F12_pred = F12_pred.reshape(Ny, Nx, Nz)
            F13_pred = F13_pred.reshape(Ny, Nx, Nz)
            F21_pred = F21_pred.reshape(Ny, Nx, Nz)
            F22_pred = F22_pred.reshape(Ny, Nx, Nz)
            F23_pred = F23_pred.reshape(Ny, Nx, Nz)
            F31_pred = F31_pred.reshape(Ny, Nx, Nz)
            F32_pred = F32_pred.reshape(Ny, Nx, Nz)
            F33_pred = F33_pred.reshape(Ny, Nx, Nz)
            SVonMises = np.float64(
                np.sqrt(0.5 * ((surS11 - surS22) ** 2 + (surS22 - surS33) ** 2 + (surS33 - surS11) ** 2 + 6 * (
                        surS12 ** 2 + surS23 ** 2 + surS31 ** 2))))
            U = [np.float64(surUx), np.float64(surUy), np.float64(surUz)]
            return U, np.float64(surS11), np.float64(surS12), np.float64(surS13), np.float64(surS22), np.float64(surS23), \
                   np.float64(surS33), np.float64(surE11), np.float64(surE12), \
                   np.float64(surE13), np.float64(surE22), np.float64(surE23), np.float64(surE33), np.float64(SVonMises), \
                   np.float64(F11_pred), np.float64(F12_pred), np.float64(F13_pred), \
                   np.float64(F21_pred), np.float64(F22_pred), np.float64(F23_pred), \
                   np.float64(F31_pred), np.float64(F32_pred), np.float64(F33_pred)

    
    
    # get f_p from model_p 
    
    # learning the homogenous network
    
    
    
    model_h1 = homo(3, 15, 3).cuda()
    model_h2 = homo(3, 15, 3).cuda()
    criterion = torch.nn.MSELoss()
    optim_h = torch.optim.LBFGS(params=chain(model_h1.parameters(), model_h2.parameters()), lr= 0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim_h, milestones=[500, 3000, 5000], gamma = 0.1)
    errorL2_array = []
    errorvon_array = []
    loss_array = []
    loss1_array = []
    loss2_array = []
    lossi_array = []
    lossid_array = []           
    lossn1_array = []
    lossn2_array = []   
    lossb_array = []
    
    nepoch_u0 = int(nepoch_u0)
    start = time.time()
    
    neo = Material('neohookean', E, nu)
    moon = Material('mooneyrivlin', param_c1=param_c1, param_c2=param_c2, param_c=param_c)
    for epoch in range(nepoch_u0):
        if epoch ==1000:
            end = time.time()
            consume_time = end-start
            print('time is %f' % consume_time)
        if epoch%100 == 0:
            Xn1, Xn2, Xf1, Xf2 = train_data(256, 4096)
            Xi = interface(3000)
            Xb = essential_bound(3000)
        def closure():  
    
            u_pred1 = pred(Xf1)
            u_pred2 = pred(Xf2)
            
            # 先处理上面的NEO
            mu = neo.mu
            lmbda = neo.lam
            duxdxyz = grad(u_pred1[:, 0].unsqueeze(1), Xf1, torch.ones(Xf1.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]
            duydxyz = grad(u_pred1[:, 1].unsqueeze(1), Xf1, torch.ones(Xf1.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]
            duzdxyz = grad(u_pred1[:, 2].unsqueeze(1), Xf1, torch.ones(Xf1.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]
            F11 = duxdxyz[:, 0].unsqueeze(1) + 1
            F12 = duxdxyz[:, 1].unsqueeze(1) + 0
            F13 = duxdxyz[:, 2].unsqueeze(1) + 0
            F21 = duydxyz[:, 0].unsqueeze(1) + 0
            F22 = duydxyz[:, 1].unsqueeze(1) + 1
            F23 = duydxyz[:, 2].unsqueeze(1) + 0
            F31 = duzdxyz[:, 0].unsqueeze(1) + 0
            F32 = duzdxyz[:, 1].unsqueeze(1) + 0
            F33 = duzdxyz[:, 2].unsqueeze(1) + 1
            detF = F11 * (F22 * F33 - F23 * F32) - F12 * (F21 * F33 - F23 * F31) + F13 * (F21 * F32 - F22 * F31)
            invF11 = (F22 * F33 - F23 * F32) / detF
            invF12 = -(F12 * F33 - F13 * F32) / detF
            invF13 = (F12 * F23 - F13 * F22) / detF
            invF21 = -(F21 * F33 - F23 * F31) / detF
            invF22 = (F11 * F33 - F13 * F31) / detF
            invF23 = -(F11 * F23 - F13 * F21) / detF
            invF31 = (F21 * F32 - F22 * F31) / detF
            invF32 = -(F11 * F32 - F12 * F31) / detF
            invF33 = (F11 * F22 - F12 * F21) / detF

            P11 = mu * F11 + (lmbda * torch.log(detF) - mu) * invF11
            P12 = mu * F12 + (lmbda * torch.log(detF) - mu) * invF21
            P13 = mu * F13 + (lmbda * torch.log(detF) - mu) * invF31
            P21 = mu * F21 + (lmbda * torch.log(detF) - mu) * invF12
            P22 = mu * F22 + (lmbda * torch.log(detF) - mu) * invF22
            P23 = mu * F23 + (lmbda * torch.log(detF) - mu) * invF32
            P31 = mu * F31 + (lmbda * torch.log(detF) - mu) * invF13
            P32 = mu * F32 + (lmbda * torch.log(detF) - mu) * invF23
            P33 = mu * F33 + (lmbda * torch.log(detF) - mu) * invF33
            # 计算上面neo区域内部的损失
            #x方向的平衡方程
            dP11dxyz = grad(P11, Xf1, torch.ones(Xf1.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]
            dP12dxyz = grad(P12, Xf1, torch.ones(Xf1.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]
            dP13dxyz = grad(P13, Xf1, torch.ones(Xf1.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]            
            P11dx = dP11dxyz[:, 0].unsqueeze(1)
            P12dy = dP12dxyz[:, 1].unsqueeze(1)
            P13dz = dP13dxyz[:, 2].unsqueeze(1)
            # y方向的平衡方程
            dP21dxyz = grad(P21, Xf1, torch.ones(Xf1.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]
            dP22dxyz = grad(P22, Xf1, torch.ones(Xf1.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]
            dP23dxyz = grad(P23, Xf1, torch.ones(Xf1.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]            
            P21dx = dP21dxyz[:, 0].unsqueeze(1)
            P22dy = dP22dxyz[:, 1].unsqueeze(1)
            P23dz = dP23dxyz[:, 2].unsqueeze(1)
            # z方向的平衡方程
            dP31dxyz = grad(P31, Xf1, torch.ones(Xf1.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]
            dP32dxyz = grad(P32, Xf1, torch.ones(Xf1.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]
            dP33dxyz = grad(P33, Xf1, torch.ones(Xf1.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]            
            P31dx = dP31dxyz[:, 0].unsqueeze(1)
            P32dy = dP32dxyz[:, 1].unsqueeze(1)
            P33dz = dP33dxyz[:, 2].unsqueeze(1)
            
            f0 = torch.zeros((len(Xf1), 1)) # 做一个平衡方程的标签，这里的体力项相当于是0，正好可以用来做MSE的标签
            J1 = criterion(P11dx+P12dy+P13dz, f0)+criterion(P21dx+P22dy+P23dz, f0)+criterion(P31dx+P32dy+P33dz, f0)
            # 计算下面moon区域的内部损失
            c1 = moon.param_c1
            c2 = moon.param_c2
            c = moon.param_c
            d = moon.param_d
            duxdxyz = grad(u_pred2[:, 0].unsqueeze(1), Xf2, torch.ones(Xf2.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]
            duydxyz = grad(u_pred2[:, 1].unsqueeze(1), Xf2, torch.ones(Xf2.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]
            duzdxyz = grad(u_pred2[:, 2].unsqueeze(1), Xf2, torch.ones(Xf2.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]
            F11 = duxdxyz[:, 0].unsqueeze(1) + 1
            F12 = duxdxyz[:, 1].unsqueeze(1) + 0
            F13 = duxdxyz[:, 2].unsqueeze(1) + 0
            F21 = duydxyz[:, 0].unsqueeze(1) + 0
            F22 = duydxyz[:, 1].unsqueeze(1) + 1
            F23 = duydxyz[:, 2].unsqueeze(1) + 0
            F31 = duzdxyz[:, 0].unsqueeze(1) + 0
            F32 = duzdxyz[:, 1].unsqueeze(1) + 0
            F33 = duzdxyz[:, 2].unsqueeze(1) + 1
            detF = F11 * (F22 * F33 - F23 * F32) - F12 * (F21 * F33 - F23 * F31) + F13 * (F21 * F32 - F22 * F31)
            C11 = F11 ** 2 + F21 ** 2 + F31 ** 2
            C12 = F11 * F12 + F21 * F22 + F31 * F32
            C13 = F11 * F13 + F21 * F23 + F31 * F33
            C21 = F12 * F11 + F22 * F21 + F32 * F31
            C22 = F12 ** 2 + F22 ** 2 + F32 ** 2
            C23 = F12 * F13 + F22 * F23 + F32 * F33
            C31 = F13 * F11 + F23 * F21 + F33 * F31
            C32 = F13 * F12 + F23 * F22 + F33 * F32
            C33 = F13 ** 2 + F23 ** 2 + F33 ** 2
            invC11 = (C22 * C33 - C23 * C32) / detF**2
            invC12 = -(C12 * C33 - C13 * C32) / detF**2
            invC13 = (C12 * C23 - C13 * C22) / detF**2
            invC21 = -(C21 * C33 - C23 * C31) / detF**2
            invC22 = (C11 * C33 - C13 * C31) / detF**2
            invC23 = -(C11 * C23 - C13 * C21) / detF**2
            invC31 = (C21 * C32 - C22 * C31) / detF**2
            invC32 = -(C11 * C32 - C12 * C31) / detF**2
            invC33 = (C11 * C22 - C12 * C21) / detF**2
            J = detF
            I1= C11 + C22 + C33
            trC2 = C11*C11 + C12*C21 + C13*C31 + C21*C12 + C22*C22 + C23*C32 + C31*C13 + C32*C23 + C33*C33
            I2 = 0.5*(I1**2 - trC2)
            S11 = (2 * c1 + 2 * c2 * I1) - 2 * c2 * C11 + (2 * c * (J - 1) * J - d) * invC11
            S12 = - 2 * c2 * C21 + (2 * c * (J - 1) * J - d) * invC21
            S13 = - 2 * c2 * C31 + (2 * c * (J - 1) * J - d) * invC31
            S21 = - 2 * c2 * C12 + (2 * c * (J - 1) * J - d) * invC12
            S22 = (2 * c1 + 2 * c2 * I1) - 2 * c2 * C22 + (2 * c * (J - 1) * J - d) * invC22
            S23 = - 2 * c2 * C32 + (2 * c * (J - 1) * J - d) * invC32
            S31 = - 2 * c2 * C13 + (2 * c * (J - 1) * J - d) * invC13
            S32 = - 2 * c2 * C23 + (2 * c * (J - 1) * J - d) * invC23
            S33 = (2 * c1 + 2 * c2 * I1) - 2 * c2 * C33 + (2 * c * (J - 1) * J - d) * invC33
            P11 =  F11*S11 + F12*S21 + F13*S31
            P12 =  F11*S12 + F12*S22 + F13*S32
            P13 =  F11*S13 + F12*S23 + F13*S33
            P21 =  F21*S11 + F22*S21 + F23*S31
            P22 =  F21*S12 + F22*S22 + F23*S32
            P23 =  F21*S13 + F22*S23 + F23*S33
            P31 =  F31*S11 + F32*S21 + F33*S31
            P32 =  F31*S12 + F32*S22 + F33*S32
            P33 =  F31*S13 + F32*S23 + F33*S33
            #x方向的平衡方程
            dP11dxyz = grad(P11, Xf2, torch.ones(Xf2.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]
            dP12dxyz = grad(P12, Xf2, torch.ones(Xf2.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]
            dP13dxyz = grad(P13, Xf2, torch.ones(Xf2.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]            
            P11dx = dP11dxyz[:, 0].unsqueeze(1)
            P12dy = dP12dxyz[:, 1].unsqueeze(1)
            P13dz = dP13dxyz[:, 2].unsqueeze(1)
            # y方向的平衡方程
            dP21dxyz = grad(P21, Xf2, torch.ones(Xf2.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]
            dP22dxyz = grad(P22, Xf2, torch.ones(Xf2.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]
            dP23dxyz = grad(P23, Xf2, torch.ones(Xf2.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]            
            P21dx = dP21dxyz[:, 0].unsqueeze(1)
            P22dy = dP22dxyz[:, 1].unsqueeze(1)
            P23dz = dP23dxyz[:, 2].unsqueeze(1)
            # z方向的平衡方程
            dP31dxyz = grad(P31, Xf2, torch.ones(Xf2.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]
            dP32dxyz = grad(P32, Xf2, torch.ones(Xf2.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]
            dP33dxyz = grad(P33, Xf2, torch.ones(Xf2.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]            
            P31dx = dP31dxyz[:, 0].unsqueeze(1)
            P32dy = dP32dxyz[:, 1].unsqueeze(1)
            P33dz = dP33dxyz[:, 2].unsqueeze(1)
            
            f0 = torch.zeros((len(Xf2), 1)) # 做一个平衡方程的标签，这里的体力项相当于是0，正好可以用来做MSE的标签
            J2 = criterion(P11dx+P12dy+P13dz, f0)+criterion(P21dx+P22dy+P23dz, f0)+criterion(P31dx+P32dy+P33dz, f0)

            #  构造外力损失，又要将上面的步骤重新算一遍
            u_pred1 = pred(Xn1)
            mu = neo.mu
            lmbda = neo.lam
            duxdxyz = grad(u_pred1[:, 0].unsqueeze(1), Xn1, torch.ones(Xn1.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]
            duydxyz = grad(u_pred1[:, 1].unsqueeze(1), Xn1, torch.ones(Xn1.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]
            duzdxyz = grad(u_pred1[:, 2].unsqueeze(1), Xn1, torch.ones(Xn1.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]
            F11 = duxdxyz[:, 0].unsqueeze(1) + 1
            F12 = duxdxyz[:, 1].unsqueeze(1) + 0
            F13 = duxdxyz[:, 2].unsqueeze(1) + 0
            F21 = duydxyz[:, 0].unsqueeze(1) + 0
            F22 = duydxyz[:, 1].unsqueeze(1) + 1
            F23 = duydxyz[:, 2].unsqueeze(1) + 0
            F31 = duzdxyz[:, 0].unsqueeze(1) + 0
            F32 = duzdxyz[:, 1].unsqueeze(1) + 0
            F33 = duzdxyz[:, 2].unsqueeze(1) + 1
            detF = F11 * (F22 * F33 - F23 * F32) - F12 * (F21 * F33 - F23 * F31) + F13 * (F21 * F32 - F22 * F31)
            invF11 = (F22 * F33 - F23 * F32) / detF
            invF12 = -(F12 * F33 - F13 * F32) / detF
            invF13 = (F12 * F23 - F13 * F22) / detF
            invF21 = -(F21 * F33 - F23 * F31) / detF
            invF22 = (F11 * F33 - F13 * F31) / detF
            invF23 = -(F11 * F23 - F13 * F21) / detF
            invF31 = (F21 * F32 - F22 * F31) / detF
            invF32 = -(F11 * F32 - F12 * F31) / detF
            invF33 = (F11 * F22 - F12 * F21) / detF

            P11 = mu * F11 + (lmbda * torch.log(detF) - mu) * invF11
            P12 = mu * F12 + (lmbda * torch.log(detF) - mu) * invF21
            P13 = mu * F13 + (lmbda * torch.log(detF) - mu) * invF31
            P21 = mu * F21 + (lmbda * torch.log(detF) - mu) * invF12
            P22 = mu * F22 + (lmbda * torch.log(detF) - mu) * invF22
            P23 = mu * F23 + (lmbda * torch.log(detF) - mu) * invF32
            P31 = mu * F31 + (lmbda * torch.log(detF) - mu) * invF13
            P32 = mu * F32 + (lmbda * torch.log(detF) - mu) * invF23
            P33 = mu * F33 + (lmbda * torch.log(detF) - mu) * invF33      
            nx = tx * torch.ones((len(Xn1), 1))
            ny = ty * torch.ones((len(Xn1), 1))
            nz = tz * torch.ones((len(Xn1), 1))
            
            Jn1 = criterion(P11, nx)+criterion(P21, ny)+criterion(P31, nz) # 上面区域的力边界条件
            #求下面区域的力边界条件
            u_pred2 = pred(Xn2)
            c1 = moon.param_c1
            c2 = moon.param_c2
            c = moon.param_c
            d = moon.param_d
            duxdxyz = grad(u_pred2[:, 0].unsqueeze(1), Xn2, torch.ones(Xn2.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]
            duydxyz = grad(u_pred2[:, 1].unsqueeze(1), Xn2, torch.ones(Xn2.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]
            duzdxyz = grad(u_pred2[:, 2].unsqueeze(1), Xn2, torch.ones(Xn2.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]
            F11 = duxdxyz[:, 0].unsqueeze(1) + 1
            F12 = duxdxyz[:, 1].unsqueeze(1) + 0
            F13 = duxdxyz[:, 2].unsqueeze(1) + 0
            F21 = duydxyz[:, 0].unsqueeze(1) + 0
            F22 = duydxyz[:, 1].unsqueeze(1) + 1
            F23 = duydxyz[:, 2].unsqueeze(1) + 0
            F31 = duzdxyz[:, 0].unsqueeze(1) + 0
            F32 = duzdxyz[:, 1].unsqueeze(1) + 0
            F33 = duzdxyz[:, 2].unsqueeze(1) + 1
            detF = F11 * (F22 * F33 - F23 * F32) - F12 * (F21 * F33 - F23 * F31) + F13 * (F21 * F32 - F22 * F31)
            C11 = F11 ** 2 + F21 ** 2 + F31 ** 2
            C12 = F11 * F12 + F21 * F22 + F31 * F32
            C13 = F11 * F13 + F21 * F23 + F31 * F33
            C21 = F12 * F11 + F22 * F21 + F32 * F31
            C22 = F12 ** 2 + F22 ** 2 + F32 ** 2
            C23 = F12 * F13 + F22 * F23 + F32 * F33
            C31 = F13 * F11 + F23 * F21 + F33 * F31
            C32 = F13 * F12 + F23 * F22 + F33 * F32
            C33 = F13 ** 2 + F23 ** 2 + F33 ** 2
            invC11 = (C22 * C33 - C23 * C32) / detF**2
            invC12 = -(C12 * C33 - C13 * C32) / detF**2
            invC13 = (C12 * C23 - C13 * C22) / detF**2
            invC21 = -(C21 * C33 - C23 * C31) / detF**2
            invC22 = (C11 * C33 - C13 * C31) / detF**2
            invC23 = -(C11 * C23 - C13 * C21) / detF**2
            invC31 = (C21 * C32 - C22 * C31) / detF**2
            invC32 = -(C11 * C32 - C12 * C31) / detF**2
            invC33 = (C11 * C22 - C12 * C21) / detF**2
            J = detF
            I1= C11 + C22 + C33
            trC2 = C11*C11 + C12*C21 + C13*C31 + C21*C12 + C22*C22 + C23*C32 + C31*C13 + C32*C23 + C33*C33
            I2 = 0.5*(I1**2 - trC2)
            S11 = (2 * c1 + 2 * c2 * I1) - 2 * c2 * C11 + (2 * c * (J - 1) * J - d) * invC11
            S12 = - 2 * c2 * C21 + (2 * c * (J - 1) * J - d) * invC21
            S13 = - 2 * c2 * C31 + (2 * c * (J - 1) * J - d) * invC31
            S21 = - 2 * c2 * C12 + (2 * c * (J - 1) * J - d) * invC12
            S22 = (2 * c1 + 2 * c2 * I1) - 2 * c2 * C22 + (2 * c * (J - 1) * J - d) * invC22
            S23 = - 2 * c2 * C32 + (2 * c * (J - 1) * J - d) * invC32
            S31 = - 2 * c2 * C13 + (2 * c * (J - 1) * J - d) * invC13
            S32 = - 2 * c2 * C23 + (2 * c * (J - 1) * J - d) * invC23
            S33 = (2 * c1 + 2 * c2 * I1) - 2 * c2 * C33 + (2 * c * (J - 1) * J - d) * invC33
            P11 =  F11*S11 + F12*S21 + F13*S31
            P12 =  F11*S12 + F12*S22 + F13*S32
            P13 =  F11*S13 + F12*S23 + F13*S33
            P21 =  F21*S11 + F22*S21 + F23*S31
            P22 =  F21*S12 + F22*S22 + F23*S32
            P23 =  F21*S13 + F22*S23 + F23*S33
            P31 =  F31*S11 + F32*S21 + F33*S31
            P32 =  F31*S12 + F32*S22 + F33*S32
            P33 =  F31*S13 + F32*S23 + F33*S33

            nx = tx * torch.ones((len(Xn2), 1))
            ny = ty * torch.ones((len(Xn2), 1))
            nz = tz * torch.ones((len(Xn2), 1))
            
            Jn2 = criterion(P11, nx)+criterion(P21, ny)+criterion(P31, nz) # 上面区域的力边界条件            
            
            # 交界面条件，有两个
            # 添加交界面的原函数损失
            u_ii1 = model_h1(Xi)  # 分别得到两个网络的交界面预测
            u_ii2 = model_h2(Xi)  
            Ji = criterion(u_ii1, u_ii2)
            # 交界面的力平衡，又要重新算一遍
            # 算上面区域的交界面的平衡方程
            u_pred1 = model_h1(Xi)
            mu = neo.mu
            lmbda = neo.lam
            duxdxyz = grad(u_pred1[:, 0].unsqueeze(1), Xi, torch.ones(Xi.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]
            duydxyz = grad(u_pred1[:, 1].unsqueeze(1), Xi, torch.ones(Xi.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]
            duzdxyz = grad(u_pred1[:, 2].unsqueeze(1), Xi, torch.ones(Xi.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]
            F11 = duxdxyz[:, 0].unsqueeze(1) + 1
            F12 = duxdxyz[:, 1].unsqueeze(1) + 0
            F13 = duxdxyz[:, 2].unsqueeze(1) + 0
            F21 = duydxyz[:, 0].unsqueeze(1) + 0
            F22 = duydxyz[:, 1].unsqueeze(1) + 1
            F23 = duydxyz[:, 2].unsqueeze(1) + 0
            F31 = duzdxyz[:, 0].unsqueeze(1) + 0
            F32 = duzdxyz[:, 1].unsqueeze(1) + 0
            F33 = duzdxyz[:, 2].unsqueeze(1) + 1
            detF = F11 * (F22 * F33 - F23 * F32) - F12 * (F21 * F33 - F23 * F31) + F13 * (F21 * F32 - F22 * F31)
            invF11 = (F22 * F33 - F23 * F32) / detF
            invF12 = -(F12 * F33 - F13 * F32) / detF
            invF13 = (F12 * F23 - F13 * F22) / detF
            invF21 = -(F21 * F33 - F23 * F31) / detF
            invF22 = (F11 * F33 - F13 * F31) / detF
            invF23 = -(F11 * F23 - F13 * F21) / detF
            invF31 = (F21 * F32 - F22 * F31) / detF
            invF32 = -(F11 * F32 - F12 * F31) / detF
            invF33 = (F11 * F22 - F12 * F21) / detF

            P11n = mu * F11 + (lmbda * torch.log(detF) - mu) * invF11
            P12n = mu * F12 + (lmbda * torch.log(detF) - mu) * invF21
            P13n = mu * F13 + (lmbda * torch.log(detF) - mu) * invF31
            P21n = mu * F21 + (lmbda * torch.log(detF) - mu) * invF12
            P22n = mu * F22 + (lmbda * torch.log(detF) - mu) * invF22
            P23n = mu * F23 + (lmbda * torch.log(detF) - mu) * invF32
            P31n = mu * F31 + (lmbda * torch.log(detF) - mu) * invF13
            P32n = mu * F32 + (lmbda * torch.log(detF) - mu) * invF23
            P33n = mu * F33 + (lmbda * torch.log(detF) - mu) * invF33      

            #求下面区域的交界面的平衡方程
            u_pred2 = model_h2(Xi)
            c1 = moon.param_c1
            c2 = moon.param_c2
            c = moon.param_c
            d = moon.param_d
            duxdxyz = grad(u_pred2[:, 0].unsqueeze(1), Xi, torch.ones(Xi.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]
            duydxyz = grad(u_pred2[:, 1].unsqueeze(1), Xi, torch.ones(Xi.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]
            duzdxyz = grad(u_pred2[:, 2].unsqueeze(1), Xi, torch.ones(Xi.size()[0], 1, device='cuda'),
                           create_graph=True, retain_graph=True)[0]
            F11 = duxdxyz[:, 0].unsqueeze(1) + 1
            F12 = duxdxyz[:, 1].unsqueeze(1) + 0
            F13 = duxdxyz[:, 2].unsqueeze(1) + 0
            F21 = duydxyz[:, 0].unsqueeze(1) + 0
            F22 = duydxyz[:, 1].unsqueeze(1) + 1
            F23 = duydxyz[:, 2].unsqueeze(1) + 0
            F31 = duzdxyz[:, 0].unsqueeze(1) + 0
            F32 = duzdxyz[:, 1].unsqueeze(1) + 0
            F33 = duzdxyz[:, 2].unsqueeze(1) + 1
            detF = F11 * (F22 * F33 - F23 * F32) - F12 * (F21 * F33 - F23 * F31) + F13 * (F21 * F32 - F22 * F31)
            C11 = F11 ** 2 + F21 ** 2 + F31 ** 2
            C12 = F11 * F12 + F21 * F22 + F31 * F32
            C13 = F11 * F13 + F21 * F23 + F31 * F33
            C21 = F12 * F11 + F22 * F21 + F32 * F31
            C22 = F12 ** 2 + F22 ** 2 + F32 ** 2
            C23 = F12 * F13 + F22 * F23 + F32 * F33
            C31 = F13 * F11 + F23 * F21 + F33 * F31
            C32 = F13 * F12 + F23 * F22 + F33 * F32
            C33 = F13 ** 2 + F23 ** 2 + F33 ** 2
            invC11 = (C22 * C33 - C23 * C32) / detF**2
            invC12 = -(C12 * C33 - C13 * C32) / detF**2
            invC13 = (C12 * C23 - C13 * C22) / detF**2
            invC21 = -(C21 * C33 - C23 * C31) / detF**2
            invC22 = (C11 * C33 - C13 * C31) / detF**2
            invC23 = -(C11 * C23 - C13 * C21) / detF**2
            invC31 = (C21 * C32 - C22 * C31) / detF**2
            invC32 = -(C11 * C32 - C12 * C31) / detF**2
            invC33 = (C11 * C22 - C12 * C21) / detF**2
            J = detF
            I1= C11 + C22 + C33
            trC2 = C11*C11 + C12*C21 + C13*C31 + C21*C12 + C22*C22 + C23*C32 + C31*C13 + C32*C23 + C33*C33
            I2 = 0.5*(I1**2 - trC2)
            S11 = (2 * c1 + 2 * c2 * I1) - 2 * c2 * C11 + (2 * c * (J - 1) * J - d) * invC11
            S12 = - 2 * c2 * C21 + (2 * c * (J - 1) * J - d) * invC21
            S13 = - 2 * c2 * C31 + (2 * c * (J - 1) * J - d) * invC31
            S21 = - 2 * c2 * C12 + (2 * c * (J - 1) * J - d) * invC12
            S22 = (2 * c1 + 2 * c2 * I1) - 2 * c2 * C22 + (2 * c * (J - 1) * J - d) * invC22
            S23 = - 2 * c2 * C32 + (2 * c * (J - 1) * J - d) * invC32
            S31 = - 2 * c2 * C13 + (2 * c * (J - 1) * J - d) * invC13
            S32 = - 2 * c2 * C23 + (2 * c * (J - 1) * J - d) * invC23
            S33 = (2 * c1 + 2 * c2 * I1) - 2 * c2 * C33 + (2 * c * (J - 1) * J - d) * invC33
            P11m =  F11*S11 + F12*S21 + F13*S31
            P12m =  F11*S12 + F12*S22 + F13*S32
            P13m =  F11*S13 + F12*S23 + F13*S33
            P21m =  F21*S11 + F22*S21 + F23*S31
            P22m =  F21*S12 + F22*S22 + F23*S32
            P23m =  F21*S13 + F22*S23 + F23*S33
            P31m =  F31*S11 + F32*S21 + F33*S31
            P32m =  F31*S12 + F32*S22 + F33*S32
            P33m =  F31*S13 + F32*S23 + F33*S33      
            
            Jid = criterion(P12n, P12m)+criterion(P22n, P22m)+criterion(P32n, P32m)
            
            # 接下来算本质边界条件的损失
            u_predeb = pred(Xb)
            ueb = torch.zeros((len(Xb), 1))
            Jb = criterion(u_predeb, ueb)
            
            
            loss = b1*J1 + b2*J2 + b3*(Ji + Jid) + b4*(Jn1+Jn2) + b5*Jb # the functional
            optim_h.zero_grad()
            loss.backward(retain_graph=True)
            loss_array.append(loss.data.cpu())
            loss1_array.append(J1.data.cpu())
            loss2_array.append(J2.data.cpu())
            lossi_array.append(Ji.data.cpu())
            lossid_array.append(Jid.data.cpu())            
            lossn1_array.append(Jn1.data.cpu())
            lossn2_array.append(Jn2.data.cpu())    
            lossb_array.append(Jb.data.cpu())
            if epoch%10==0:
                x = np.linspace(0, 4, 69)
                y = np.linspace(0, 1, 18)
                z = np.linspace(0, 1, 18)
                xyz = np.meshgrid(x, y, z)
                # 分别得到上下两个区域全部的输出场
                Uu, S11u, S12u, S13u, S22u, S23u, S33u, E11u, E12u, E13u, E22u, E23u, E33u, SVonMisesu, F11u, F12u, F13u, F21u, F22u, F23u, F31u, F32u, F33u = evaluate_model(neo, x, y, z)
                Ud, S11d, S12d, S13d, S22d, S23d, S33d, E11d, E12d, E13d, E22d, E23d, E33d, SVonMisesd, F11d, F12d, F13d, F21d, F22d, F23d, F31d, F32d, F33d = evaluate_model(moon, x, y, z)
                # 将上面的neo场的下半部分用moon的下半部分rewrite，通过xyz[1] 的y坐标进行识别
                Uu[0][xyz[1]<0.5] = Ud[0][xyz[1]<0.5] # Uu是一个list进行改动，原先的程序时一个tuple不可以更改
                Uu[1][xyz[1]<0.5] = Ud[1][xyz[1]<0.5]
                Uu[2][xyz[1]<0.5] = Ud[2][xyz[1]<0.5]
                Uux = Uu[0].copy() # flags出了问题
                Uuy = Uu[1].copy()
                Uuz = Uu[2].copy()
                S11u[xyz[1]<0.5] = S11d[xyz[1]<0.5]
                S12u[xyz[1]<0.5] = S12d[xyz[1]<0.5]
                S13u[xyz[1]<0.5] = S13d[xyz[1]<0.5]
                S22u[xyz[1]<0.5] = S22d[xyz[1]<0.5]
                S23u[xyz[1]<0.5] = S23d[xyz[1]<0.5]
                S33u[xyz[1]<0.5] = S33d[xyz[1]<0.5]
                E11u[xyz[1]<0.5] = E11d[xyz[1]<0.5]
                E12u[xyz[1]<0.5] = E12d[xyz[1]<0.5]
                E13u[xyz[1]<0.5] = E13d[xyz[1]<0.5]
                E22u[xyz[1]<0.5] = E22d[xyz[1]<0.5]
                E23u[xyz[1]<0.5] = E23d[xyz[1]<0.5]
                E33u[xyz[1]<0.5] = E33d[xyz[1]<0.5]
                SVonMisesu[xyz[1]<0.5] = SVonMisesd[xyz[1]<0.5]
                F11u[xyz[1]<0.5] = F11d[xyz[1]<0.5]
                F12u[xyz[1]<0.5] = F12d[xyz[1]<0.5]
                F13u[xyz[1]<0.5] = F13d[xyz[1]<0.5]
                F21u[xyz[1]<0.5] = F21d[xyz[1]<0.5]
                F22u[xyz[1]<0.5] = F22d[xyz[1]<0.5]
                F23u[xyz[1]<0.5] = F23d[xyz[1]<0.5]
                F31u[xyz[1]<0.5] = F31d[xyz[1]<0.5]
                F32u[xyz[1]<0.5] = F32d[xyz[1]<0.5]
                F33u[xyz[1]<0.5] = F33d[xyz[1]<0.5] # 上述输出的全部是array，但是是具有grid结构的
                Upredmag = np.sqrt(Uux**2+Uuy**2+Uuz**2).transpose(2,0,1).flatten() # 得到位移的输出大小，这里是包含每一个点的预测,由于meshgrid前面两个维度是相反的，所以我们将前面两个维度进行变换
                Ufemmag = np.sqrt(femdisx**2+femdisy**2+femdisz**2) # 得到位移的有限元的大小，这里是包含每一个点的预测
                predvon = SVonMisesu.transpose(2,0,1).flatten()
                if epoch == 200:
                    print()
                errorL2 = np.linalg.norm(Upredmag-Ufemmag)/np.linalg.norm(Ufemmag)
                errorvon = np.abs((np.linalg.norm(predvon) - np.linalg.norm(femvon))/np.linalg.norm(femvon))
    
                errorL2_array.append(errorL2)
                errorvon_array.append(errorvon)
                print(' epoch : %i, the loss : %f ,  J1 : %f , J2 : %f, Ji : %f, Jid : %f, Jn1 : %f, Jn2 : %f, Jb : %f, ErrorL2 : %f, Errorvon : %f' % (epoch, loss.data, J1.data, J2.data, Ji.data, Jid.data, Jn1.data, Jn2.data, Jb.data, errorL2, errorvon))
            return loss
        optim_h.step(closure)
        scheduler.step()
    # %%
    # for plotting 
    # 将位移图，应力图进行输出
    x = np.linspace(0, 4, 73) # 60
    y = np.linspace(0, 1, 19)
    z = np.linspace(0, 1, 19)
    xyz = np.meshgrid(x, y, z)
    # 分别得到上下两个区域全部的输出场
    Uu, S11u, S12u, S13u, S22u, S23u, S33u, E11u, E12u, E13u, E22u, E23u, E33u, SVonMisesu, F11u, F12u, F13u, F21u, F22u, F23u, F31u, F32u, F33u = evaluate_model(neo, x, y, z)
    Ud, S11d, S12d, S13d, S22d, S23d, S33d, E11d, E12d, E13d, E22d, E23d, E33d, SVonMisesd, F11d, F12d, F13d, F21d, F22d, F23d, F31d, F32d, F33d = evaluate_model(moon, x, y, z)
    # 将上面的neo场的下半部分用moon的下半部分rewrite，通过xyz[1] 的y坐标进行识别
    Uu[0][xyz[1]<0.5] = Ud[0][xyz[1]<0.5] # Uu是一个list进行改动，原先的程序时一个tuple不可以更改
    Uu[1][xyz[1]<0.5] = Ud[1][xyz[1]<0.5]
    Uu[2][xyz[1]<0.5] = Ud[2][xyz[1]<0.5]
    Uux = Uu[0].copy() # flags出了问题
    Uuy = Uu[1].copy()
    Uuz = Uu[2].copy()
    S11u[xyz[1]<0.5] = S11d[xyz[1]<0.5]
    S12u[xyz[1]<0.5] = S12d[xyz[1]<0.5]
    S13u[xyz[1]<0.5] = S13d[xyz[1]<0.5]
    S22u[xyz[1]<0.5] = S22d[xyz[1]<0.5]
    S23u[xyz[1]<0.5] = S23d[xyz[1]<0.5]
    S33u[xyz[1]<0.5] = S33d[xyz[1]<0.5]
    E11u[xyz[1]<0.5] = E11d[xyz[1]<0.5]
    E12u[xyz[1]<0.5] = E12d[xyz[1]<0.5]
    E13u[xyz[1]<0.5] = E13d[xyz[1]<0.5]
    E22u[xyz[1]<0.5] = E22d[xyz[1]<0.5]
    E23u[xyz[1]<0.5] = E23d[xyz[1]<0.5]
    E33u[xyz[1]<0.5] = E33d[xyz[1]<0.5]
    SVonMisesu[xyz[1]<0.5] = SVonMisesd[xyz[1]<0.5]
    F11u[xyz[1]<0.5] = F11d[xyz[1]<0.5]
    F12u[xyz[1]<0.5] = F12d[xyz[1]<0.5]
    F13u[xyz[1]<0.5] = F13d[xyz[1]<0.5]
    F21u[xyz[1]<0.5] = F21d[xyz[1]<0.5]
    F22u[xyz[1]<0.5] = F22d[xyz[1]<0.5]
    F23u[xyz[1]<0.5] = F23d[xyz[1]<0.5]
    F31u[xyz[1]<0.5] = F31d[xyz[1]<0.5]
    F32u[xyz[1]<0.5] = F32d[xyz[1]<0.5]
    F33u[xyz[1]<0.5] = F33d[xyz[1]<0.5]
    
    filename_out = "./output/dem/neo+moon_cpinn_epoch%i" % nepoch_u0 # 输出的VTS文件
    write_vtk_vtu(filename_out, x, y, z, Uux, Uuy, Uuz, S11u, S12u, S13u, S22u, S23u, S33u, E11u, E12u, E13u, E22u, E23u, E33u, SVonMisesu)
    write_vtk_vts(filename_out, x, y, z, Uux, Uuy, Uuz, S11u, S12u, S13u, S22u, S23u, S33u, E11u, E12u, E13u, E22u, E23u, E33u, SVonMisesu)
    
    # %%
    
    loss_array = np.array(loss_array) # make the list to array
    np.save('./loss_error/hyper_cpinn_loss_epoch%i' % nepoch_u0, loss_array)
    errorL2_array = np.array(errorL2_array) # make the list to array
    np.save('./loss_error/hyper_cpinn_errorL2_epoch%i' % nepoch_u0, errorL2_array)
    errorvon_array = np.array(errorvon_array) # make the list to array
    np.save('./loss_error/hyper_cpinn_errorvon_epoch%i' % nepoch_u0, errorvon_array)
    
    fig = plt.figure(figsize=(20, 6), dpi=1000)
    
    plt.subplot(1, 3, 1)
    loss_array = loss_array[::20]
    iteration = np.array(range(int(len(loss_array)/2), len(loss_array))) # 从中间循环开始分析损失函数,因为无法使用log，由于损失是负数
    plt.plot(iteration, loss_array[iteration], ls = '--')
    plt.legend(loc = 'upper right')
    plt.xlabel('the iteration')
    plt.ylabel('loss')
    plt.title('LOSS', fontsize = 20) 
    
    plt.subplot(1, 3, 2) # 画L2损失
    errorL2_array = errorL2_array[::20]
    iteration = np.array(range(int(len(errorL2_array)/2), len(errorL2_array))) # 从中间循环开始分析损失函数,因为无法使用log，由于损失是负数
    plt.plot(iteration*10, errorL2_array[iteration], ls = '--') # 每10次储存，所以这里循环数目乘以10
    plt.legend(loc = 'upper right')
    plt.xlabel('the iteration')
    plt.ylabel('error L2')
    plt.title('L2', fontsize = 20) 
    
    plt.subplot(1, 3, 3) # 画L2损失
    errorvon_array = errorvon_array[::20]
    iteration = np.array(range(int(len(errorvon_array)/2), len(errorvon_array))) # 从中间循环开始分析损失函数,因为无法使用log，由于损失是负数
    plt.plot(iteration*10, errorvon_array[iteration], ls = '--')
    plt.legend(loc = 'upper right')
    plt.xlabel('the iteration')
    plt.ylabel('error von')
    plt.title('VON', fontsize = 20) 
    plt.suptitle('CPINN hyper',   size = 25)
    plt.savefig('./loss_error/hyper_cpinn_errorvon_epoch%i.pdf' % nepoch_u0)
    plt.show()