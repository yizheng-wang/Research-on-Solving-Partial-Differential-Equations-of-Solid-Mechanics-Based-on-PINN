"""
@author: 王一铮, 447650327@qq.com
"""
import sys
sys.path.insert(0, '/home/sg/SeaDrive/My Libraries/硕士学位论文/王一铮硕士论文/代码') # add路径
from DEM.plate_hole import Plate_hole_define_structure as des
from DEM.MultiLayerNet import *
from DEM import EnergyModel as md
from DEM import Utility as util
from DEM.plate_hole import config as cf
from DEM.IntegrationLoss import *
from DEM.EnergyModel import *
import numpy as np
import time
import torch
import xlrd

import matplotlib as mpl
mpl.rcParams['font.sans-serif']=['SimHei'] # 为了显示中文matplotlib

mpl.rcParams['figure.dpi'] = 1000
# fix random seeds
axes = {'labelsize' : 'large'}
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 5}
legend = {'fontsize': 'medium'}
lines = {'linewidth': 0.5,
         'markersize' : 10}
mpl.rc('font', **font)
mpl.rc('axes', **axes)
mpl.rc('legend', **legend)
mpl.rc('lines', **lines)

class DeepEnergyMethod_ritz:
    # Instance attributes
    def __init__(self, model, numIntType, energy, dim, defect=False, r=0):
        # self.data = data
        self.model = MultiLayerNet(model[0], model[1], model[2])
        self.model = self.model.to(dev)
        self.intLoss = IntegrationLoss(numIntType, dim, defect = defect)
        self.energy = energy
        # self.post = PostProcessing(energy, dim)
        self.dim = dim
        self.lossArray = []
        self.r = r

    def train_model(self, shape, dxdydz, data, neumannBC, dirichletBC, iteration, learning_rate, penalty_boundary_loss=5000):
        x = torch.from_numpy(data).float()
        x = x.to(dev)
        x.requires_grad_(True)
        # get tensor inputs and outputs for boundary conditions
        # -------------------------------------------------------------------------------
        #                             Dirichlet BC
        # -------------------------------------------------------------------------------
        dirBC_coordinates = {}  # declare a dictionary
        dirBC_values = {}  # declare a dictionary
        dirBC_penalty = {}
        dirBC_normal = {}
        for i, keyi in enumerate(dirichletBC):
            dirBC_coordinates[i] = torch.from_numpy(dirichletBC[keyi]['coord']).float().to(dev)
            dirBC_values[i] = torch.from_numpy(dirichletBC[keyi]['known_value']).float().to(dev)
            dirBC_penalty[i] = torch.tensor(dirichletBC[keyi]['penalty']).float().to(dev)
            dirBC_normal[i] = torch.tensor(dirichletBC[keyi]['dir_normal2d']).float().to(dev)
        neuBC_coordinates = {}  # declare a dictionary
        neuBC_values = {}  # declare a dictionary
        neuBC_penalty = {}
        for i, keyi in enumerate(neumannBC):
            neuBC_coordinates[i] = torch.from_numpy(neumannBC[keyi]['coord']).float().to(dev)
            neuBC_coordinates[i].requires_grad_(True) # 这里感觉不用设置梯度为True
            neuBC_values[i] = torch.from_numpy(neumannBC[keyi]['known_value']).float().to(dev) # 这里只有一个边界，所以这个dict目前只有1个，这里应该是根据边界条件的个数确定dict有多少组的
            neuBC_penalty[i] = torch.tensor(neumannBC[keyi]['penalty']).float().to(dev)
        # ----------------------------------------------------------------------------------
        # Minimizing loss function (energy and boundary conditions)
        # ----------------------------------------------------------------------------------
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)# 每一个循环需要的循环次数，这里是20，也就是每一个iter需要有20个循环
        start_time = time.time()
        energy_loss_array = []
        lambda_loss_array = []
        for t in range(iteration):
            # Zero gradients, perform a backward pass, and update the weights.
            def closure():
                it_time = time.time()
                # ----------------------------------------------------------------------------------
                # Internal Energy
                # ----------------------------------------------------------------------------------
                ust_pred = self.getUST(x)
                ust_pred.double()
                storedEnergy = self.energy.getStoredEnergy(ust_pred, x)
                internal2 = self.intLoss.lossInternalEnergy(storedEnergy, dx=dxdydz[0], dy=dxdydz[1], shape=shape)
                external2 = torch.zeros(len(neuBC_coordinates)) # 外力功,到时候损失要加负号
                for i, vali in enumerate(neuBC_coordinates):
                    neu_u_pred = self.getUST(neuBC_coordinates[i])
                    fext = torch.bmm((neu_u_pred + neuBC_coordinates[i]).unsqueeze(1), neuBC_values[i].unsqueeze(2))
                    external2[i] = self.intLoss.lossExternalEnergy(fext, dx=dxdydz[1])
                bc_u_crit = torch.zeros((len(dirBC_coordinates))) # 维度是1
                for i, vali in enumerate(dirBC_coordinates):
                    dir_u_pred = self.getUST(dirBC_coordinates[i])
                    bc_u_crit[i] = self.loss_squared_sum(dir_u_pred[:, i].unsqueeze(1), dirBC_values[i][:, i].unsqueeze(1)) # 输入维度是50，2的张量
                energy_loss = internal2 - torch.sum(external2)
                boundary_loss = torch.sum(bc_u_crit)
                loss = energy_loss +  penalty_boundary_loss * boundary_loss
                optimizer.zero_grad()
                loss.backward()
                print('Iter: %d Loss: %.9e Energy: %.9e Boundary: %.9e Internal Energy: %.9e  External Energy: %.9e Time: %.3e'
                      % (t + 1, loss.item(), energy_loss.item(), boundary_loss.item(), internal2.item(), torch.sum(external2).item(), time.time() - it_time))
                energy_loss_array.append(energy_loss.data)
                #lambda_loss_array.append(lambda_loss.data)
                self.lossArray.append(loss.data)
                return loss
            optimizer.step(closure)
        elapsed = time.time() - start_time
        print('Training time: %.4f' % elapsed)




    def getUST(self, x):
        '''
        

        Parameters
        ----------
        x : tensor
            coordinate of 2 dimensionality.

        Returns
        -------
        ust_pred : tensor
            get the displacement, strain and stress in 2 dimensionality.

        '''
        ust_pred = x/20*self.model(x) # 尝试归一化

        return ust_pred

    def evaluate_model(self, datatest):
        energy_type = self.energy.type
        dim = self.dim
        if dim == 2:
            Nx = len(datatest)
            x = datatest[:, 0].reshape(Nx, 1)
            y = datatest[:, 1].reshape(Nx, 1)
#            z = datatest[:, 2].reshape(Nx, 1)
            
            xy = np.concatenate((x, y), axis=1)
            xy_tensor = torch.from_numpy(xy).float()
            xy_tensor = xy_tensor.to(dev)
            xy_tensor.requires_grad_(True)
            ust_pred_torch = self.getUST(xy_tensor)
            duxdxy = grad(ust_pred_torch[:, 0].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev),
                           create_graph=True, retain_graph=True)[0]
            duydxy = grad(ust_pred_torch[:, 1].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev),
                           create_graph=True, retain_graph=True)[0]
            dudx = duxdxy[:, 0]
            dudy = duxdxy[:, 1]
            dvdx = duydxy[:, 0]
            dvdy = duydxy[:, 1]
            exx_pred = dudx
            eyy_pred = dvdy
            e2xy_pred = dudy + dvdx     
            sxx_pred = self.energy.D11_mat * exx_pred + self.energy.D12_mat * eyy_pred
            syy_pred = self.energy.D12_mat * exx_pred + self.energy.D22_mat * eyy_pred
            sxy_pred = self.energy.D33_mat * e2xy_pred
            
            ust_pred = ust_pred_torch.detach().cpu().numpy()
            exx_pred = exx_pred.detach().cpu().numpy()
            eyy_pred = eyy_pred.detach().cpu().numpy()
            e2xy_pred = e2xy_pred.detach().cpu().numpy()
            sxx_pred = sxx_pred.detach().cpu().numpy()
            syy_pred = syy_pred.detach().cpu().numpy()
            sxy_pred = sxy_pred.detach().cpu().numpy()
            ust_pred = ust_pred_torch.detach().cpu().numpy()
            F11_pred = np.zeros(Nx) # 因为是小变形，所以我不关心这个量，先全部设为0
            F12_pred = np.zeros(Nx)
            F21_pred = np.zeros(Nx)
            F22_pred = np.zeros(Nx)
            surUx = ust_pred[:, 0]
            surUy = ust_pred[:, 1]
            surUz = np.zeros(Nx)
            surE11 = exx_pred
            surE12 = 0.5*e2xy_pred
            surE13 = np.zeros(Nx)
            surE21 = 0.5*e2xy_pred
            surE22 = eyy_pred
            surE23 = np.zeros(Nx)
            surE33 = np.zeros(Nx)
           
            surS11 = sxx_pred
            surS12 = sxy_pred
            surS13 = np.zeros(Nx)
            surS21 = sxy_pred
            surS22 = syy_pred
            surS23 = np.zeros(Nx)
            surS33 = np.zeros(Nx)

            
            SVonMises = np.float64(np.sqrt(0.5 * ((surS11 - surS22) ** 2 + (surS22) ** 2 + (-surS11) ** 2 + 6 * (surS12 ** 2))))
            U = (np.float64(surUx), np.float64(surUy), np.float64(surUz))
            return U, np.float64(surS11), np.float64(surS12), np.float64(surS13), np.float64(surS22), np.float64(
                surS23), \
                   np.float64(surS33), np.float64(surE11), np.float64(surE12), \
                   np.float64(surE13), np.float64(surE22), np.float64(surE23), np.float64(surE33), np.float64(
                SVonMises), \
                   np.float64(F11_pred), np.float64(F12_pred), np.float64(F21_pred), np.float64(F22_pred)
        else: # 之后要改动，目前二维不用太大变化
            Nx = len(x)
            Ny = len(y)
            Nz = len(z)
            xGrid, yGrid, zGrid = np.meshgrid(x, y, z)
            x1D = xGrid.flatten()
            y1D = yGrid.flatten()
            z1D = zGrid.flatten()
            xyz = np.concatenate((np.array([x1D]).T, np.array([y1D]).T, np.array([z1D]).T), axis=-1)
            xyz_tensor = torch.from_numpy(xyz).float()
            xyz_tensor = xyz_tensor.to(dev)
            xyz_tensor.requires_grad_(True)
            # u_pred_torch = self.model(xyz_tensor)
            u_pred_torch = self.getU(xyz_tensor)
            duxdxyz = grad(u_pred_torch[:, 0].unsqueeze(1), xyz_tensor, torch.ones(xyz_tensor.size()[0], 1, device=dev),
                           create_graph=True, retain_graph=True)[0]
            duydxyz = grad(u_pred_torch[:, 1].unsqueeze(1), xyz_tensor, torch.ones(xyz_tensor.size()[0], 1, device=dev),
                           create_graph=True, retain_graph=True)[0]
            duzdxyz = grad(u_pred_torch[:, 2].unsqueeze(1), xyz_tensor, torch.ones(xyz_tensor.size()[0], 1, device=dev),
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
            if energy_type == 'neohookean' and dim == 3:
                P11 = mu * F11 + (lmbda * torch.log(detF) - mu) * invF11
                P12 = mu * F12 + (lmbda * torch.log(detF) - mu) * invF21
                P13 = mu * F13 + (lmbda * torch.log(detF) - mu) * invF31
                P21 = mu * F21 + (lmbda * torch.log(detF) - mu) * invF12
                P22 = mu * F22 + (lmbda * torch.log(detF) - mu) * invF22
                P23 = mu * F23 + (lmbda * torch.log(detF) - mu) * invF32
                P31 = mu * F31 + (lmbda * torch.log(detF) - mu) * invF13
                P32 = mu * F32 + (lmbda * torch.log(detF) - mu) * invF23
                P33 = mu * F33 + (lmbda * torch.log(detF) - mu) * invF33
            else:
                print("This energy model will be implemented later !!!")
                exit()
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
            SVonMises = np.float64(
                np.sqrt(0.5 * ((surS11 - surS22) ** 2 + (surS22 - surS33) ** 2 + (surS33 - surS11) ** 2 + 6 * (
                        surS12 ** 2 + surS23 ** 2 + surS31 ** 2))))
            U = (np.float64(surUx), np.float64(surUy), np.float64(surUz))
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
    # --------------------------------------------------------------------------------
    # method: loss sum for the energy part
    # --------------------------------------------------------------------------------
    @staticmethod
    def loss_sum(tinput):
        return torch.sum(tinput) / tinput.data.nelement()

    # --------------------------------------------------------------------------------
    # purpose: loss square sum for the boundary part
    # --------------------------------------------------------------------------------
    @staticmethod
    def loss_squared_sum(tinput, target):
        row, column = tinput.shape
        loss = 0
        for j in range(column):
            loss += torch.sum((tinput[:, j] - target[:, j]) ** 2) / tinput[:, j].data.nelement()
        return loss

    def printLoss(self):
        self.los


class DeepEnergyMethod:
    # Instance attributes
    def __init__(self, model, numIntType, energy, dim, defect=False, r=0):
        # self.data = data
        self.model = MultiLayerNet(model[0], model[1], model[2])
        self.model = self.model.to(dev)
        self.intLoss = IntegrationLoss(numIntType, dim, defect = defect)
        self.energy = energy
        # self.post = PostProcessing(energy, dim)
        self.dim = dim
        self.lossArray = []
        self.r = r

    def train_model(self, shape, dxdydz, data, neumannBC, dirichletBC, iteration, learning_rate, penalty_boundary_loss=5000):
        x = torch.from_numpy(data).float()
        x = x.to(dev)
        x.requires_grad_(True)
        # get tensor inputs and outputs for boundary conditions
        # -------------------------------------------------------------------------------
        #                             Dirichlet BC
        # -------------------------------------------------------------------------------
        dirBC_coordinates = {}  # declare a dictionary
        dirBC_values = {}  # declare a dictionary
        dirBC_penalty = {}
        dirBC_normal = {}
        for i, keyi in enumerate(dirichletBC):
            dirBC_coordinates[i] = torch.from_numpy(dirichletBC[keyi]['coord']).float().to(dev)
            dirBC_values[i] = torch.from_numpy(dirichletBC[keyi]['known_value']).float().to(dev)
            dirBC_penalty[i] = torch.tensor(dirichletBC[keyi]['penalty']).float().to(dev)
            dirBC_normal[i] = torch.tensor(dirichletBC[keyi]['dir_normal2d']).float().to(dev)
        # -------------------------------------------------------------------------------
        #                           Neumann BC
        # -------------------------------------------------------------------------------
        neuBC_coordinates = {}  # declare a dictionary
        neuBC_values = {}  # declare a dictionary
        neuBC_penalty = {}
        for i, keyi in enumerate(neumannBC):
            neuBC_coordinates[i] = torch.from_numpy(neumannBC[keyi]['coord']).float().to(dev)
            neuBC_coordinates[i].requires_grad_(True) # 这里感觉不用设置梯度为True
            neuBC_values[i] = torch.from_numpy(neumannBC[keyi]['known_value']).float().to(dev) # 这里只有一个边界，所以这个dict目前只有1个，这里应该是根据边界条件的个数确定dict有多少组的
            neuBC_penalty[i] = torch.tensor(neumannBC[keyi]['penalty']).float().to(dev)
        # ----------------------------------------------------------------------------------
        # Minimizing loss function (energy and boundary conditions)
        # ----------------------------------------------------------------------------------
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)# 每一个循环需要的循环次数，这里是20，也就是每一个iter需要有20个循环
        start_time = time.time()
        energy_loss_array = []
        lambda_loss_array = []
        for t in range(iteration):
            # Zero gradients, perform a backward pass, and update the weights.
            def closure():
                it_time = time.time()
                # ----------------------------------------------------------------------------------
                # Internal Energy
                # ----------------------------------------------------------------------------------
                ust_pred = self.getUST(x)
                ust_pred.double()
                storedEnergy = self.energy.getStoredEnergy(ust_pred, x)
                internal2 = self.intLoss.lossInternalEnergy(storedEnergy, dx=dxdydz[0], dy=dxdydz[1], shape=shape)
                external2 = torch.zeros(len(neuBC_coordinates)) # 外力功,到时候损失要加负号
                for i, vali in enumerate(neuBC_coordinates):
                    neu_u_pred = self.getUST(neuBC_coordinates[i])
                    fext = torch.bmm((neu_u_pred + neuBC_coordinates[i]).unsqueeze(1), neuBC_values[i].unsqueeze(2))
                    external2[i] = self.intLoss.lossExternalEnergy(fext, dx=dxdydz[1])
                bc_u_crit = torch.zeros((len(dirBC_coordinates))) # 维度是1
                for i, vali in enumerate(dirBC_coordinates):
                    dir_u_pred = self.getUST(dirBC_coordinates[i])
                    bc_u_crit[i] = self.loss_squared_sum(dir_u_pred[:, i].unsqueeze(1), dirBC_values[i][:, i].unsqueeze(1)) # 输入维度是50，2的张量
                energy_loss = internal2 - torch.sum(external2)
                boundary_loss = torch.sum(bc_u_crit)
                loss = energy_loss +  penalty_boundary_loss * boundary_loss
                optimizer.zero_grad()
                loss.backward()
                print('Iter: %d Loss: %.9e Energy: %.9e Boundary: %.9e Internal Energy: %.9e  External Energy: %.9e Time: %.3e'
                      % (t + 1, loss.item(), energy_loss.item(), boundary_loss.item(), internal2.item(), torch.sum(external2).item(), time.time() - it_time))
                energy_loss_array.append(energy_loss.data)
                #lambda_loss_array.append(lambda_loss.data)
                self.lossArray.append(loss.data)
                return loss
            optimizer.step(closure)
        elapsed = time.time() - start_time
        print('Training time: %.4f' % elapsed)

    def getUST(self, x):
        '''
        

        Parameters
        ----------
        x : tensor
            coordinate of 2 dimensionality.

        Returns
        -------
        ust_pred : tensor
            get the displacement, strain and stress in 2 dimensionality.

        '''
        ust = self.model(x)
        Ux = ust[:, 0] # 如果x坐标是0的话，x方向位移也是0
        Uy = ust[:, 1] # 如果y坐标是0的话，y方向位移也是0

        Ux = Ux.reshape(Ux.shape[0], 1)
        Uy = Uy.reshape(Uy.shape[0], 1)

        ust_pred = torch.cat((Ux, Uy), -1)
        return ust_pred

    # --------------------------------------------------------------------------------
    # Evaluate model to obtain:
    # 1. U - Displacement
    # 2. E - Green Lagrange Strain
    # 3. S - 2nd Piola Kirchhoff Stress
    # 4. F - Deformation Gradient
    # Date implement: 20.06.2019
    # --------------------------------------------------------------------------------
    def evaluate_model(self, datatest):
        energy_type = self.energy.type
        dim = self.dim
        if dim == 2:
            Nx = len(datatest)
            x = datatest[:, 0].reshape(Nx, 1)
            y = datatest[:, 1].reshape(Nx, 1)
#            z = datatest[:, 2].reshape(Nx, 1)
            
            xy = np.concatenate((x, y), axis=1)
            xy_tensor = torch.from_numpy(xy).float()
            xy_tensor = xy_tensor.to(dev)
            xy_tensor.requires_grad_(True)
            ust_pred_torch = self.getUST(xy_tensor)
            duxdxy = grad(ust_pred_torch[:, 0].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev),
                           create_graph=True, retain_graph=True)[0]
            duydxy = grad(ust_pred_torch[:, 1].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev),
                           create_graph=True, retain_graph=True)[0]
            dudx = duxdxy[:, 0]
            dudy = duxdxy[:, 1]
            dvdx = duydxy[:, 0]
            dvdy = duydxy[:, 1]
            exx_pred = dudx
            eyy_pred = dvdy
            e2xy_pred = dudy + dvdx     
            sxx_pred = self.energy.D11_mat * exx_pred + self.energy.D12_mat * eyy_pred
            syy_pred = self.energy.D12_mat * exx_pred + self.energy.D22_mat * eyy_pred
            sxy_pred = self.energy.D33_mat * e2xy_pred
            
            ust_pred = ust_pred_torch.detach().cpu().numpy()
            exx_pred = exx_pred.detach().cpu().numpy()
            eyy_pred = eyy_pred.detach().cpu().numpy()
            e2xy_pred = e2xy_pred.detach().cpu().numpy()
            sxx_pred = sxx_pred.detach().cpu().numpy()
            syy_pred = syy_pred.detach().cpu().numpy()
            sxy_pred = sxy_pred.detach().cpu().numpy()
            ust_pred = ust_pred_torch.detach().cpu().numpy()
            F11_pred = np.zeros(Nx) # 因为是小变形，所以我不关心这个量，先全部设为0
            F12_pred = np.zeros(Nx)
            F21_pred = np.zeros(Nx)
            F22_pred = np.zeros(Nx)
            surUx = ust_pred[:, 0]
            surUy = ust_pred[:, 1]
            surUz = np.zeros(Nx)
            surE11 = exx_pred
            surE12 = 0.5*e2xy_pred
            surE13 = np.zeros(Nx)
            surE21 = 0.5*e2xy_pred
            surE22 = eyy_pred
            surE23 = np.zeros(Nx)
            surE33 = np.zeros(Nx)
           
            surS11 = sxx_pred
            surS12 = sxy_pred
            surS13 = np.zeros(Nx)
            surS21 = sxy_pred
            surS22 = syy_pred
            surS23 = np.zeros(Nx)
            surS33 = np.zeros(Nx)

            
            SVonMises = np.float64(np.sqrt(0.5 * ((surS11 - surS22) ** 2 + (surS22) ** 2 + (-surS11) ** 2 + 6 * (surS12 ** 2))))
            U = (np.float64(surUx), np.float64(surUy), np.float64(surUz))
            return U, np.float64(surS11), np.float64(surS12), np.float64(surS13), np.float64(surS22), np.float64(
                surS23), \
                   np.float64(surS33), np.float64(surE11), np.float64(surE12), \
                   np.float64(surE13), np.float64(surE22), np.float64(surE23), np.float64(surE33), np.float64(
                SVonMises), \
                   np.float64(F11_pred), np.float64(F12_pred), np.float64(F21_pred), np.float64(F22_pred)
        else: # 之后要改动，目前二维不用太大变化
            Nx = len(x)
            Ny = len(y)
            Nz = len(z)
            xGrid, yGrid, zGrid = np.meshgrid(x, y, z)
            x1D = xGrid.flatten()
            y1D = yGrid.flatten()
            z1D = zGrid.flatten()
            xyz = np.concatenate((np.array([x1D]).T, np.array([y1D]).T, np.array([z1D]).T), axis=-1)
            xyz_tensor = torch.from_numpy(xyz).float()
            xyz_tensor = xyz_tensor.to(dev)
            xyz_tensor.requires_grad_(True)
            # u_pred_torch = self.model(xyz_tensor)
            u_pred_torch = self.getU(xyz_tensor)
            duxdxyz = grad(u_pred_torch[:, 0].unsqueeze(1), xyz_tensor, torch.ones(xyz_tensor.size()[0], 1, device=dev),
                           create_graph=True, retain_graph=True)[0]
            duydxyz = grad(u_pred_torch[:, 1].unsqueeze(1), xyz_tensor, torch.ones(xyz_tensor.size()[0], 1, device=dev),
                           create_graph=True, retain_graph=True)[0]
            duzdxyz = grad(u_pred_torch[:, 2].unsqueeze(1), xyz_tensor, torch.ones(xyz_tensor.size()[0], 1, device=dev),
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
            if energy_type == 'neohookean' and dim == 3:
                P11 = mu * F11 + (lmbda * torch.log(detF) - mu) * invF11
                P12 = mu * F12 + (lmbda * torch.log(detF) - mu) * invF21
                P13 = mu * F13 + (lmbda * torch.log(detF) - mu) * invF31
                P21 = mu * F21 + (lmbda * torch.log(detF) - mu) * invF12
                P22 = mu * F22 + (lmbda * torch.log(detF) - mu) * invF22
                P23 = mu * F23 + (lmbda * torch.log(detF) - mu) * invF32
                P31 = mu * F31 + (lmbda * torch.log(detF) - mu) * invF13
                P32 = mu * F32 + (lmbda * torch.log(detF) - mu) * invF23
                P33 = mu * F33 + (lmbda * torch.log(detF) - mu) * invF33
            else:
                print("This energy model will be implemented later !!!")
                exit()
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
            SVonMises = np.float64(
                np.sqrt(0.5 * ((surS11 - surS22) ** 2 + (surS22 - surS33) ** 2 + (surS33 - surS11) ** 2 + 6 * (
                        surS12 ** 2 + surS23 ** 2 + surS31 ** 2))))
            U = (np.float64(surUx), np.float64(surUy), np.float64(surUz))
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
    # --------------------------------------------------------------------------------
    # method: loss sum for the energy part
    # --------------------------------------------------------------------------------
    @staticmethod
    def loss_sum(tinput):
        return torch.sum(tinput) / tinput.data.nelement()

    # --------------------------------------------------------------------------------
    # purpose: loss square sum for the boundary part
    # --------------------------------------------------------------------------------
    @staticmethod
    def loss_squared_sum(tinput, target):
        row, column = tinput.shape
        loss = 0
        for j in range(column):
            loss += torch.sum((tinput[:, j] - target[:, j]) ** 2) / tinput[:, j].data.nelement()
        return loss

    def printLoss(self):
        self.los


if __name__ == '__main__':
    datax0 = xlrd.open_workbook("hole_abaqus_x=0.xlsx") # 获得x=0的数据，需要提取y方向位移以及mise应力
    tablex0 = datax0.sheet_by_index(0)
    x0 = tablex0.col_values(0)
    x0 = np.array(x0)
    x0_2d = np.zeros((len(x0), 2))
    x0_2d[:, 1] = x0    
    disy_abqusx0 = tablex0.col_values(1)
    mise_abaqusx0 = tablex0.col_values(2)    

    datay0 = xlrd.open_workbook("hole_abaqus_y=0.xlsx") # 获得x=0的数据，需要提取y方向位移以及mise应力
    tabley0 = datay0.sheet_by_index(0)
    y0 = tabley0.col_values(0)
    y0 = np.array(y0)
    y0_2d = np.zeros((len(y0), 2))
    y0_2d[:, 0] = y0
    disx_abqusy0 = tabley0.col_values(1)   
    
    pred_disy_array = [] # 估计 x=0的y方向位移
    pred_disx_array = [] # 估计 y=0的x方向位移
    pred_mise_array = [] # 估计 x=0的mise应力
    penalty_boundary_loss_array = []
    for i in range(1):
        penalty_boundary_loss = 1000 * (i+1) 
        # ----------------------------------------------------------------------
        #                   STEP 1: SETUP DOMAIN - COLLECT CLEAN DATABASE
        # ----------------------------------------------------------------------
        dom, boundary_neumann, boundary_dirichlet, bound_id = des.setup_domain() # 返回的boundary value是一个字典，里面不仅仅有坐标还有力大小，力大小没有标准化
        datatest = des.get_datatest()
        # ----------------------------------------------------------------------
        #                   STEP 2: SETUP MODEL
        # ----------------------------------------------------------------------
        mat = md.EnergyModel('elasticityMP', 2, cf.E, cf.nu)
        dem = DeepEnergyMethod([cf.D_in, cf.H, cf.D_out], 'simpson', mat, 2, defect=True, r = cf.r)
        # ----------------------------------------------------------------------
        #                   STEP 3: TRAINING MODEL
        # ----------------------------------------------------------------------
        start_time = time.time()
        shape = [cf.Nx, cf.Ny]
        dxdy = [cf.hx, cf.hy]
        cf.iteration = 10000
        cf.filename_out = "./output/dem/P%iPlate_hole_Elasticity_minpotential_2Layer_mesh200x200_iter%i_simpsonADAM" % (penalty_boundary_loss, cf.iteration)
        dem.train_model(shape, dxdy, dom, boundary_neumann, boundary_dirichlet, cf.iteration, cf.lr, penalty_boundary_loss=penalty_boundary_loss)
        end_time = time.time() - start_time
        print("End time: %.5f" % end_time)
        z = np.zeros((datatest.shape[0], 1))
        datatest = np.concatenate((datatest, z), 1)
        U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises, F11, F12, F21, F22 = dem.evaluate_model(datatest)
#        util.write_vtk_v2p(cf.filename_out, datatest, U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises)
        # 接下来比较两条线x=0和y=0，首先比较x=0的y方向位移以及mise应力
        U_2dx0, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMisesx0, F11, F12, F21, F22 = dem.evaluate_model(x0_2d)
        pred_disy_array.append(U_2dx0[1]) # 将y方向位移放入到array中
        pred_mise_array.append(SVonMisesx0) # 将mise应力放入到array中
        # 接下来放y=0的x方向位移
        U_2dy0, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises, F11, F12, F21, F22 = dem.evaluate_model(y0_2d)
        pred_disx_array.append(U_2dy0[0]) # 将y方向位移放入到array中
        # 储存罚函数        
        penalty_boundary_loss_array.append(penalty_boundary_loss)
        print('Loss convergence')
        
    # 接下来分析deep nitsche ritz方法
    cf.iteration = 10000
    cf.filename_out = "./output/ritz_P%iPlate_hole_2Layer_mesh200x200_iter%i_simpsonLBFGS" % (penalty_boundary_loss, cf.iteration)
    dem_r = DeepEnergyMethod_ritz([cf.D_in, cf.H, cf.D_out], 'simpson', mat, 2, defect=True, r = cf.r)
    dem_r.train_model(shape, dxdy, dom, boundary_neumann, boundary_dirichlet, cf.iteration, cf.lr, penalty_boundary_loss=0) # 因为是用解析的方法构造的，所以边界损失有没有都没有影响
    end_time = time.time() - start_time
    print("End time: %.5f" % end_time)
    # 分析x=0这条线
    U_2dx0_r, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMisesx0_r, F11, F12, F21, F22 = dem_r.evaluate_model(x0_2d)
    pred_disy_r = U_2dx0_r[1]
    pred_mise_r = SVonMisesx0_r
    # 分析y=0这条线
    U_2dy0_r, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises, F11, F12, F21, F22 = dem_r.evaluate_model(y0_2d)
    pred_disx_r = U_2dy0_r[0]
    # 画曲线评估，需要画3幅图，第一幅是x0的y方向位移
    plt.show()
    plt.plot(x0, disy_abqusx0, color = 'red', label = 'Abaqus参照解', lw = 2)
    plt.plot(x0, pred_disy_r, color = 'blue', label = 'DEM', lw = 2)
    for idx, bp in enumerate(penalty_boundary_loss_array):
        plt.plot(x0, pred_disy_array[idx], label = 'p%i 罚函数解' % bp )
    plt.xlabel('y坐标') 
    plt.ylabel('y方向位移')
    plt.title('x=0处y方向位移，撒点数：2383')
    plt.legend()
    plt.show()
    # 第二幅是x0的mise应力
    plt.plot(x0, mise_abaqusx0, color = 'red', label = 'Abaqus参照解', lw = 2)
    plt.plot(x0, pred_mise_r, color = 'blue', label = 'DEM', lw = 2)
    for idx, bp in enumerate(penalty_boundary_loss_array):
        plt.plot(x0, pred_mise_array[idx], label = 'p%i 罚函数解' % bp )
    plt.xlabel('y坐标') 
    plt.ylabel('Mises应力')
    plt.title('x=0处Mises应力，撒点数：2383')
    plt.legend()
    plt.show()
    # 第三幅是y0的x方向位移
    plt.plot(y0, disx_abqusy0, color = 'red', label = 'Abaqus参照解', lw = 2)
    plt.plot(y0, pred_disx_r, color = 'blue', label = 'DEM', lw = 2)
    for idx, bp in enumerate(penalty_boundary_loss_array):
        plt.plot(y0, pred_disx_array[idx], label = 'p%i 罚函数解' % bp )
    plt.xlabel('x坐标') 
    plt.ylabel('x方向位移')
    plt.title('y=0处x方向位移，撒点数：2383')
    plt.legend()
    plt.show()












