"""
@author: 王一铮, 447650327@qq.com
改编自：https://github.com/MinhNguyenIKM/dem_hyperelasticity
"""
import sys
sys.path.insert(0, '/home/sg/SeaDrive/My Libraries/硕士学位论文/PINN最小能量原理/dem_hyperelasticity-master') # add路径
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
        self.model = MultiLayerNet(model[0], model[1], model[2])
        self.model = self.model.to(dev)
        self.intLoss = IntegrationLoss(numIntType, dim)
        self.energy = energy
        # self.post = PostProcessing(energy, dim)
        self.dim = dim
        self.lossArray = []

    def train_model(self, data, neumannBC, dirichletBC, LHD, iteration, learning_rate):
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
#            dirBC_normal[i] = torch.tensor(dirichletBC[keyi]['dir_normal2d']).float().to(dev)
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
        optimizer = torch.optim.LBFGS(self.model.parameters(), lr=learning_rate, max_iter=20)# 每一个循环需要的循环次数，这里是20，也就是每一个iter需要有20个循环
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
                strainEnergy = self.energy.getStoredEnergy(ust_pred, x) # 最小势能仅仅是应变能**************************
                internal2 = self.intLoss.montecarlo2D(strainEnergy, LHD[0], LHD[1])
               
                external2 = torch.zeros(len(neuBC_coordinates)) # 外力功,到时候损失要加负号
                for i, vali in enumerate(neuBC_coordinates):
                    neu_ust_pred = self.getUST(neuBC_coordinates[i])
                    neu_u_pred = neu_ust_pred[:,(0, 1)]
                    fext = torch.bmm((neu_u_pred).unsqueeze(1), neuBC_values[i].unsqueeze(2)) # 每一个节点所作的功
                    external2[i] = self.intLoss.montecarlo1D(fext, LHD[1])
                bc_u_crit = torch.zeros((len(dirBC_coordinates))) # 维度是1
                for i, vali in enumerate(dirBC_coordinates):
                    dir_u_pred = self.getUST(dirBC_coordinates[i])
                    bc_u_crit[i] = self.loss_squared_sum(dir_u_pred, dirBC_values[i]) # 输入维度是50，2的张量
                energy_loss = internal2 - torch.sum(external2)
                #lambda_loss = -torch.sum(bc_u_crit) - internal2g
                loss = energy_loss 
                optimizer.zero_grad()
                loss.backward()
                print('Iter: %d Loss: %.9e Energy: %.9e  Time: %.3e'
                      % (t + 1, loss.item(), energy_loss.item(), time.time() - it_time))
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
        Ux = x[:, 0] * ust[:, 0]
        Uy = x[:, 0] * ust[:, 1]

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
    def evaluate_model(self, x, y, z):
        energy_type = self.energy.type
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
            ust_pred_torch = self.getUST(xy_tensor)
            duxdxy = grad(ust_pred_torch[:, 0].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev),
                           create_graph=True, retain_graph=True)[0]
            duydxy = grad(ust_pred_torch[:, 1].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev),
                           create_graph=True, retain_graph=True)[0]
            dudx = duxdxy[:, 0].unsqueeze(1)
            dudy = duxdxy[:, 1].unsqueeze(1)
            dvdx = duydxy[:, 0].unsqueeze(1)
            dvdy = duydxy[:, 1].unsqueeze(1)
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
            '''duxdxy = grad(ust_pred_torch[:, 0].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev),
                           create_graph=True, retain_graph=True)[0]
            duydxy = grad(ust_pred_torch[:, 1].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev),
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
            S22 = invF21 * P12 + invF22 * P22'''
            ust_pred = ust_pred_torch.detach().cpu().numpy()
            F11_pred = np.zeros([Nx, Ny, 1]) # 因为是小变形，所以我不关心这个量，先全部设为0
            F12_pred = np.zeros([Nx, Ny, 1])
            F21_pred = np.zeros([Nx, Ny, 1])
            F22_pred = np.zeros([Nx, Ny, 1])
            surUx = ust_pred[:, 0].reshape(Ny, Nx, 1)
            surUy = ust_pred[:, 1].reshape(Ny, Nx, 1)
            surUz = np.zeros([Nx, Ny, 1])
            surE11 = exx_pred.reshape(Ny, Nx, 1)
            surE12 = 0.5*e2xy_pred.reshape(Ny, Nx, 1)
            surE13 = np.zeros([Nx, Ny, 1])
            surE21 = 0.5*e2xy_pred.reshape(Ny, Nx, 1)
            surE22 = eyy_pred.reshape(Ny, Nx, 1)
            surE23 = np.zeros([Nx, Ny, 1])
            surE33 = np.zeros([Nx, Ny, 1])
           
            surS11 = sxx_pred.reshape(Ny, Nx, 1)
            surS12 = sxy_pred.reshape(Ny, Nx, 1)
            surS13 = np.zeros([Nx, Ny, 1])
            surS21 = sxy_pred.reshape(Ny, Nx, 1)
            surS22 = syy_pred.reshape(Ny, Nx, 1)
            surS23 = np.zeros([Nx, Ny, 1])
            surS33 = np.zeros([Nx, Ny, 1])

            
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
    # ----------------------------------------------------------------------
    #                   STEP 1: SETUP DOMAIN - COLLECT CLEAN DATABASE
    # ----------------------------------------------------------------------
    dom, boundary_neumann, boundary_dirichlet = des.setup_domain() # 返回的boundary value是一个字典，里面不仅仅有坐标还有力大小，力大小没有标准化
    x, y, datatest = des.get_datatest()
    # ----------------------------------------------------------------------
    #                   STEP 2: SETUP MODEL
    # ----------------------------------------------------------------------
    mat = md.EnergyModel('elasticityMP', 2, cf.E, cf.nu)
    dem = DeepEnergyMethod([cf.D_in, cf.H, cf.D_out], 'montecarlo', mat, 2)
    # ----------------------------------------------------------------------
    #                   STEP 3: TRAINING MODEL
    # ----------------------------------------------------------------------
    start_time = time.time()
    shape = [cf.Nx, cf.Ny]
    dxdy = [cf.hx, cf.hy]
    cf.iteration = 500
    cf.filename_out = "./output/dem/Elasticity_minpotential_2Layer_mesh200x50_iter500_mont"
    dem.train_model(dom, boundary_neumann, boundary_dirichlet, [cf.Length, cf.Height, cf.Depth], cf.iteration, cf.lr)
    end_time = time.time() - start_time
    print("End time: %.5f" % end_time)
    z = np.array([0])
    U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises, F11, F12, F21, F22 = dem.evaluate_model(x, y, z)
    util.write_vtk_v2(cf.filename_out, x, y, z, U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises)
    surUx, surUy, surUz = U
    L2norm = util.getL2norm2D(surUx, surUy, cf.Nx, cf.Ny, cf.hx, cf.hy)
    H10norm = util.getH10norm2D(F11, F12, F21, F22, cf.Nx, cf.Ny, cf.hx, cf.hy)
    print("L2 norm = %.10f" % L2norm)
    print("H10 norm = %.10f" % H10norm)
    print('Loss convergence')
