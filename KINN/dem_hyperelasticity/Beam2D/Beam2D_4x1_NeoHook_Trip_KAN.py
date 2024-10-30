"""
@author: sfmt4368 (Simon), texa5140 (Cosmin), minh.nguyen@ikm.uni-hannover.de

Implements the 2D Hyperelastic beam models (Neo-Hookean)
"""
import sys
sys.path.append("../..") 
from dem_hyperelasticity.Beam2D import define_structure as des
from dem_hyperelasticity.MultiLayerNet import *
from dem_hyperelasticity import EnergyModel as md
from dem_hyperelasticity import Utility as util
from dem_hyperelasticity.Beam2D import config as cf
from dem_hyperelasticity.IntegrationLoss import *
from dem_hyperelasticity.EnergyModel import *
import numpy as np
import time
import torch
import pyvista as pv

mpl.rcParams['figure.dpi'] = 100
# fix random seeds
axes = {'labelsize' : 'large'}
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 17}
legend = {'fontsize': 'medium'}
lines = {'linewidth': 3,
         'markersize' : 7}
mpl.rc('font', **font)
mpl.rc('axes', **axes)
mpl.rc('legend', **legend)
mpl.rc('lines', **lines)


class DeepEnergyMethod:
    # Instance attributes
    def __init__(self, model, numIntType, energy, dim):
        # self.data = data
        self.model = KAN(model, base_activation=torch.nn.SiLU, grid_size=15, grid_range=[0, 1], spline_order=3)
        self.model = self.model.to(dev)
        self.intLoss = IntegrationLoss(numIntType, dim)
        self.energy = energy
        # self.post = PostProcessing(energy, dim)
        self.dim = dim
        self.lossArray = []

    def train_model(self, shape, dxdydz, data, neumannBC, dirichletBC, iteration, learning_rate):
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
        for i, keyi in enumerate(dirichletBC):
            dirBC_coordinates[i] = torch.from_numpy(dirichletBC[keyi]['coord']).float().to(dev)
            dirBC_values[i] = torch.from_numpy(dirichletBC[keyi]['known_value']).float().to(dev)
            dirBC_penalty[i] = torch.tensor(dirichletBC[keyi]['penalty']).float().to(dev)
        # -------------------------------------------------------------------------------
        #                           Neumann BC
        # -------------------------------------------------------------------------------
        neuBC_coordinates = {}  # declare a dictionary
        neuBC_values = {}  # declare a dictionary
        neuBC_penalty = {}
        for i, keyi in enumerate(neumannBC):
            neuBC_coordinates[i] = torch.from_numpy(neumannBC[keyi]['coord']).float().to(dev)
            neuBC_coordinates[i].requires_grad_(True)
            neuBC_values[i] = torch.from_numpy(neumannBC[keyi]['known_value']).float().to(dev)
            neuBC_penalty[i] = torch.tensor(neumannBC[keyi]['penalty']).float().to(dev)
        # ----------------------------------------------------------------------------------
        # Minimizing loss function (energy and boundary conditions)
        # ----------------------------------------------------------------------------------
        optimizer = torch.optim.LBFGS(self.model.parameters(), lr=learning_rate, max_iter=20)
        start_time = time.time()
        energy_loss_array = []
        boundary_loss_array = []
        self.L2_array = []
        self.H1_array = []
        # loss_array = []
        for t in range(iteration):
            # Zero gradients, perform a backward pass, and update the weights.
            def closure():
                it_time = time.time()
                # ----------------------------------------------------------------------------------
                # Internal Energy
                # ----------------------------------------------------------------------------------
                u_pred = self.getU(x)
                #  u_pred.double()
                storedEnergy = self.energy.getStoredEnergy(u_pred, x)
                internal2 = self.intLoss.lossInternalEnergy(storedEnergy, dx=dxdydz[0], dy=dxdydz[1], shape=shape)
                external2 = torch.zeros(len(neuBC_coordinates))
                for i, vali in enumerate(neuBC_coordinates):
                    neu_u_pred = self.getU(neuBC_coordinates[i])
                    fext = torch.bmm((neu_u_pred + neuBC_coordinates[i]).unsqueeze(1), neuBC_values[i].unsqueeze(2))
                    external2[i] = self.intLoss.lossExternalEnergy(fext, dx=dxdydz[1])
                bc_u_crit = torch.zeros((len(dirBC_coordinates)))
                for i, vali in enumerate(dirBC_coordinates):
                    dir_u_pred = self.getU(dirBC_coordinates[i])
                    bc_u_crit[i] = self.loss_squared_sum(dir_u_pred, dirBC_values[i])
                energy_loss = internal2 - torch.sum(external2)
                boundary_loss = torch.sum(bc_u_crit)
                loss = energy_loss + boundary_loss
                optimizer.zero_grad()
                loss.backward()
                # 算一下error
                surUx_pred, surUy_pred = self.evaluate_u(points)
                Von_pred = self.evaluate_von(points)
                dis_x_pred = surUx_pred.flatten()
                dis_y_pred = surUy_pred.flatten()
                Von_pred = Von_pred.flatten()
                
                L2norm = np.linalg.norm(dis_y_pred - dis_y_fem)/np.linalg.norm( dis_y_fem)
                H1norm = np.linalg.norm(Von_pred - von_fem)/np.linalg.norm(von_fem)
                

                print('Iter: %d Loss: %.9e Energy: %.9e Boundary: %.9e , L2 error: %.9e, H1 error: %.9e,  Time: %.3e'
                      % (t + 1, loss.item(), energy_loss.item(), boundary_loss.item(), L2norm,  H1norm, time.time() - it_time))
                energy_loss_array.append(energy_loss.data)
                boundary_loss_array.append(boundary_loss.data)
                self.lossArray.append(loss.data)
                return loss
            optimizer.step(closure)
            # 算一下error
            surUx_pred, surUy_pred = self.evaluate_u(points)
            Von_pred = self.evaluate_von(points)
            dis_x_pred = surUx_pred.flatten()
            dis_y_pred = surUy_pred.flatten()
            Von_pred = Von_pred.flatten()
            
            L2norm = np.linalg.norm(dis_y_pred - dis_y_fem)/np.linalg.norm( dis_y_fem)
            H1norm = np.linalg.norm(Von_pred - von_fem)/np.linalg.norm(von_fem)
            self.L2_array.append(L2norm)
            self.H1_array.append(H1norm)            
        elapsed = time.time() - start_time
        print('Training time: %.4f' % elapsed)

    def getU(self, x):
        x_scale = x/cf.Length
        u = self.model(x_scale)
        Ux = x[:, 0] * u[:, 0]
        Uy = x[:, 0] * u[:, 1]
        Ux = Ux.reshape(Ux.shape[0], 1)
        Uy = Uy.reshape(Uy.shape[0], 1)
        u_pred = torch.cat((Ux, Uy), -1)
        return u_pred

    # --------------------------------------------------------------------------------
    # Evaluate model to obtain:
    # 1. U - Displacement
    # 2. E - Green Lagrange Strain
    # 3. S - 2nd Piola Kirchhoff Stress
    # 4. F - Deformation Gradient
    # Date implement: 20.06.2019
    # --------------------------------------------------------------------------------
    def evaluate_u(self, points): # 仅仅输入位移来降低运算量
        energy_type = self.energy.type
        mu = self.energy.mu
        lmbda = self.energy.lam
        xy_tensor = torch.from_numpy(points[:,:-1]).float() # 输入的points是三维度的，最后一个维度先不做模型的输入
        xy_tensor = xy_tensor.to(dev)
        xy_tensor.requires_grad_(True)
        u_pred = self.getU(xy_tensor)
        surUx = u_pred[:, 0].cpu().detach().numpy()
        surUy = u_pred[:, 1].cpu().detach().numpy()

        return surUx, surUy
    
    def evaluate_von(self, points): # 仅仅输入位移来降低运算量
        energy_type = self.energy.type
        mu = self.energy.mu
        lmbda = self.energy.lam
        xy_tensor = torch.from_numpy(points[:,:-1]).float() # 输入的points是三维度的，最后一个维度先不做模型的输入
        xy_tensor = xy_tensor.to(dev)
        xy_tensor.requires_grad_(True)
        u_pred_torch = self.getU(xy_tensor)
        duxdxy = grad(u_pred_torch[:, 0].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev),
                       create_graph=True, retain_graph=True)[0]
        duydxy = grad(u_pred_torch[:, 1].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev),
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

        if energy_type == 'neohookean':
            P11 = mu * F11 + (lmbda * torch.log(detF) - mu) * invF11
            P12 = mu * F12 + (lmbda * torch.log(detF) - mu) * invF21
            P21 = mu * F21 + (lmbda * torch.log(detF) - mu) * invF12
            P22 = mu * F22 + (lmbda * torch.log(detF) - mu) * invF22
        S11 = invF11 * P11 + invF12 * P21
        S12 = invF11 * P12 + invF12 * P22
        S21 = invF21 * P11 + invF22 * P21
        S22 = invF21 * P12 + invF22 * P22
        S11_pred = S11.detach().cpu().numpy()
        S12_pred = S12.detach().cpu().numpy()
        S21_pred = S21.detach().cpu().numpy()
        S22_pred = S22.detach().cpu().numpy()
        SVonMises = np.float64(np.sqrt(0.5 * ((S11_pred - S22_pred) ** 2 + (S22_pred) ** 2 + (-S11_pred) ** 2 + 6 * (S12_pred ** 2))))
            
        return SVonMises
    
    
    def evaluate_model(self, x, y, z):
        energy_type = self.energy.type
        mu = self.energy.mu
        lmbda = self.energy.lam
        dim = self.dim
        if dim == 2:
            Nx = len(x)
            Ny = len(y)
            xGrid, yGrid = np.meshgrid(x, y)
            x1D = xGrid.flatten()
            y1D = yGrid.flatten()
            xy = np.concatenate((np.array([x1D]).T, np.array([y1D]).T), axis=-1)
            xy_tensor = torch.from_numpy(xy).float()
            xy_tensor = xy_tensor.to(dev)
            xy_tensor.requires_grad_(True)
            u_pred_torch = self.getU(xy_tensor)
            duxdxy = grad(u_pred_torch[:, 0].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev),
                           create_graph=True, retain_graph=True)[0]
            duydxy = grad(u_pred_torch[:, 1].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev),
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
            if energy_type == 'neohookean' and dim == 2:
                P11 = mu * F11 + (lmbda * torch.log(detF) - mu) * invF11
                P12 = mu * F12 + (lmbda * torch.log(detF) - mu) * invF21
                P21 = mu * F21 + (lmbda * torch.log(detF) - mu) * invF12
                P22 = mu * F22 + (lmbda * torch.log(detF) - mu) * invF22
            else:
                print("This energy model will be implemented later !!!")
                exit()
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
            surE13 = np.zeros([Ny, Nx, 1])
            surE21 = E21_pred.reshape(Ny, Nx, 1)
            surE22 = E22_pred.reshape(Ny, Nx, 1)
            surE23 = np.zeros([Ny, Nx, 1])
            surE33 = np.zeros([Ny, Nx, 1])
            surS11 = S11_pred.reshape(Ny, Nx, 1)
            surS12 = S12_pred.reshape(Ny, Nx, 1)
            surS13 = np.zeros([Ny, Nx, 1])
            surS21 = S21_pred.reshape(Ny, Nx, 1)
            surS22 = S22_pred.reshape(Ny, Nx, 1)
            surS23 = np.zeros([Ny, Nx, 1])
            surS33 = np.zeros([Ny, Nx, 1])
            SVonMises = np.float64(np.sqrt(0.5 * ((surS11 - surS22) ** 2 + (surS22) ** 2 + (-surS11) ** 2 + 6 * (surS12 ** 2))))
            U = (np.float64(surUx), np.float64(surUy), np.float64(surUz))
            return U, np.float64(surS11), np.float64(surS12), np.float64(surS13), np.float64(surS22), np.float64(
                surS23), \
                   np.float64(surS33), np.float64(surE11), np.float64(surE12), \
                   np.float64(surE13), np.float64(surE22), np.float64(surE23), np.float64(surE33), np.float64(
                SVonMises), \
                   np.float64(F11_pred), np.float64(F12_pred), np.float64(F21_pred), np.float64(F22_pred)
        else:
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


if __name__ == '__main__':
    # 读取 VTU 文件(FEM的参照解) 首先读取位移
    vtu_file_dis = "./output/fem/E=1000/beam2d_4x1_fem_v1000000.vtu"
    mesh_dis = pv.read(vtu_file_dis)
    
    # 获取坐标点
    points = mesh_dis.points
    
    # 获取所有数组数据的名称
    array_names_dis = mesh_dis.array_names
    
    
    # 打印所有数组数据
    for name in array_names_dis:
        dis_exact = mesh_dis.point_data[name]
    
    dis_x_fem = dis_exact[:,0]
    dis_y_fem = dis_exact[:,1]
    dis_z_fem = dis_exact[:,2]
    
    # 读取 VTU 文件(FEM的参照解) 然后读取vonmise应力
    vtu_file_von = "./output/fem/E=1000/elasticity/von_mises000000.vtu"
    mesh_von = pv.read(vtu_file_von)
    
    # 获取坐标点
    points = mesh_von.points
    
    # 获取所有数组数据的名称
    array_names_von = mesh_von.array_names
    
    
    # 打印所有数组数据
    for name in array_names_von:
        mises_exact = mesh_von.point_data[name]
    
    von_fem = mises_exact.flatten()
   
    
    
    # ----------------------------------------------------------------------
    #                   STEP 1: SETUP DOMAIN - COLLECT CLEAN DATABASE
    # ----------------------------------------------------------------------
    dom, boundary_neumann, boundary_dirichlet = des.setup_domain()
    x, y, datatest = des.get_datatest()
    # ----------------------------------------------------------------------
    #                   STEP 2: SETUP MODEL
    # ----------------------------------------------------------------------
    mat = md.EnergyModel('neohookean', 2, cf.E, cf.nu)
    #dem = DeepEnergyMethod([cf.D_in, cf.H, cf.D_out], 'simpson', mat, 2)
    dem = DeepEnergyMethod([cf.D_in, 5,5,5, cf.D_out], 'trapezoidal', mat, 2)
    # ----------------------------------------------------------------------
    #                   STEP 3: TRAINING MODEL
    # ----------------------------------------------------------------------
    start_time = time.time()
    shape = [cf.Nx, cf.Ny]
    dxdy = [cf.hx, cf.hy]
    cf.iteration = 1000
    cf.filename_out = "./output/dem/NeoHook_KAN_trap"
    cf.lr = 0.001
    dem.train_model(shape, dxdy, dom, boundary_neumann, boundary_dirichlet, cf.iteration, cf.lr)
    end_time = time.time() - start_time
    print("End time: %.5f" % end_time)
    z = np.array([0.])
    disx_pred, disy_pred = dem.evaluate_u(points)

    dis_y_abs_error = np.abs(disy_pred.flatten() - dis_y_fem.flatten())
    util.write_vtk_v2p("./output/dem/NeoHook_KAN_trap_disy_error", points, dis_y_abs_error)
    
    # error_storage
    np.save("./output/dem/NeoHook_KAN_trap_L2_norm.npy", dem.L2_array)
    np.save("./output/dem/NeoHook_KAN_trap_H1_norm.npy", dem.H1_array)
    

    # y=0.5 storage
    points_y05 = points[points[:,1]==0.5]
    disx_pred_y05, disy_pred_y05 = dem.evaluate_u(points_y05) # prediction
    von_pred_y05 = dem.evaluate_von(points_y05)
    dis_x_fem_y05 = dis_x_fem[points[:,1]==0.5]
    dis_y_fem_y05 = dis_y_fem[points[:,1]==0.5]
    von_fem_y05 = von_fem[points[:,1]==0.5]
    
    dis_pred_y05 =  (disx_pred_y05**2 + disy_pred_y05**2)**0.5
    dis_fem_y05 =  (dis_x_fem_y05**2 + dis_y_fem_y05**2)**0.5
    
    dict_y05 = {'X': points_y05, 'Dis_pred': dis_pred_y05, 'Dis_fem': dis_fem_y05, \
                'Von_pred': von_pred_y05, 'Von_exact': von_fem_y05}

    np.save("./output/dem/NeoHook_KAN_trap_y05.npy", dict_y05)


    # x=2.0 storage
    points_x2 = points[points[:,0]==2.0]
    disx_pred_x2, disy_pred_x2 = dem.evaluate_u(points_x2) # prediction
    von_pred_x2 = dem.evaluate_von(points_x2)
    dis_x_fem_x2 = dis_x_fem[points[:,0]==2.0]
    dis_y_fem_x2 = dis_y_fem[points[:,0]==2.0]
    von_fem_x2 = von_fem[points[:,0]==2.0]
    
    dis_pred_x2 =  (disx_pred_x2**2 + disy_pred_x2**2)**0.5
    dis_fem_x2 =  (dis_x_fem_x2**2 + dis_y_fem_x2**2)**0.5
    
    dict_x2 = {'X': points_x2, 'Dis_pred': dis_pred_x2, 'Dis_fem': dis_fem_x2, \
                'Von_pred': von_pred_x2, 'Von_exact': von_fem_x2}

    np.save("./output/dem/NeoHook_KAN_trap_x2.npy", dict_x2)
