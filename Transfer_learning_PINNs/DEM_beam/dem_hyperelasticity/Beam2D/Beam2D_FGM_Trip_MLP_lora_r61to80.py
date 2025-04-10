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
import os

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


class DeepEnergyMethod1to2:
    # Instance attributes
    def __init__(self, model, numIntType, energy, dim, r):
        # self.data = data
        self.model = MultiLayerNetLoRA(model[0], model[1], model[2], r=r)
        self.model = self.model.to(dev)
        self.model.load_state_dict(torch.load('./model/PINN/model_FGM1.pth'), strict = False)
        self.intLoss = IntegrationLoss(numIntType, dim)
        self.energy = energy
        # self.post = PostProcessing(energy, dim)
        self.dim = dim
        self.lossArray = []

    def freeze_lora_layers(self, model: nn.Module, train_layer_keywords: list, lr=1e-3):
        """
        冻结 model 中除含有指定关键字的 LoRA 参数以外的所有参数，
        并返回一个只更新指定层 LoRA 参数的优化器。
        
        参数:
            model (nn.Module): 你的 PyTorch 模型
            train_layer_keywords (list): 需要更新(微调)的层名称关键字的列表。
            lr (float): 优化器学习率
            
        返回:
            optimizer (torch.optim.Optimizer): 只更新指定层 LoRA 参数的优化器
        """
        # 遍历所有参数
        for name, param in model.named_parameters():
            # 如果关键词命中，就保持 requires_grad=True；否则 requires_grad=False
            if any(keyword in name for keyword in train_layer_keywords):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # 收集需要训练的参数
        params_to_optimize = [p for p in model.parameters() if p.requires_grad]
        # 构建优化器
        optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
        
        return optimizer

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
        optimizer = self.freeze_lora_layers(self.model, train_layer_keywords=["linear2.lora", "linear2.bias","linear3.lora", "linear3.bias","linear4.lora", "linear4.bias"], lr=0.001)
        start_time = time.time()
        energy_loss_array = []
        boundary_loss_array = []
        self.L2_array = []
        self.H1_array = []
        # loss_array = []
        start_time = time.time()
        for t in range(iteration):
            # Zero gradients, perform a backward pass, and update the weights.
            if t % 1000 == 0:
                end_time = time.time()
                print("1000:", end_time - start_time)
                start_time = time.time()
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
                    fext = torch.bmm(neu_u_pred.unsqueeze(1), neuBC_values[i].unsqueeze(2))
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
                
                L2norm = np.linalg.norm(dis_y_pred - dis_y_iga)/np.linalg.norm( dis_y_iga)
                H1norm = np.linalg.norm(Von_pred - von_iga)/np.linalg.norm(von_iga)
                

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
            
            L2norm = np.linalg.norm(dis_y_pred - dis_y_iga)/np.linalg.norm( dis_y_iga)
            H1norm = np.linalg.norm(Von_pred - von_iga)/np.linalg.norm(von_iga)
            self.L2_array.append(L2norm)
            self.H1_array.append(H1norm)            
        elapsed = time.time() - start_time
        print('Training time: %.4f' % elapsed)
        torch.save(self.model.state_dict(), f'model/PINN/model_FGM1to2_lora_r1.pth')
    def getU(self, x):
        x_scale = x/cf.Length
        u = self.model(x_scale)
        Ux = x[:, 0] * u[:, 0] * (cf.Length-x[:, 0])
        Uy = x[:, 0] * u[:, 1] * (cf.Length-x[:, 0])
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
        xy_tensor = torch.from_numpy(points[:,:-1]).float() # 输入的points是三维度的，最后一个维度先不做模型的输入
        xy_tensor = xy_tensor.to(dev)
        xy_tensor.requires_grad_(True)
        u_pred = self.getU(xy_tensor)
        surUx = u_pred[:, 0].cpu().detach().numpy()
        surUy = u_pred[:, 1].cpu().detach().numpy()

        return surUx, surUy
    
    def evaluate_von(self, points): # 仅仅输入位移来降低运算量
        xy_tensor = torch.from_numpy(points[:,:-1]).float() # 输入的points是三维度的，最后一个维度先不做模型的输入
        xy_tensor = xy_tensor.to(dev)
        xy_tensor.requires_grad_(True)
        u_pred_torch = self.getU(xy_tensor)
        duxdxy = grad(u_pred_torch[:, 0].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev),
                       create_graph=True, retain_graph=True)[0]
        duydxy = grad(u_pred_torch[:, 1].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev),
                       create_graph=True, retain_graph=True)[0]
        
        dudx = duxdxy[:, 0].unsqueeze(1)
        dudy = duxdxy[:, 1].unsqueeze(1)
        dvdx = duydxy[:, 0].unsqueeze(1)
        dvdy = duydxy[:, 1].unsqueeze(1)
        exx_pred = dudx
        eyy_pred = dvdy
        e2xy_pred = dudy + dvdx     
        
        y = points[:,1]
        ratio = 1 - 0.5*np.cos(torch.pi*(y/2)).reshape(-1,1)
        
        sxx_pred = (self.energy.D11_mat * exx_pred + self.energy.D12_mat * eyy_pred) 
        syy_pred = (self.energy.D12_mat * exx_pred + self.energy.D22_mat * eyy_pred) 
        sxy_pred = (self.energy.D33_mat * e2xy_pred)
        
        exx_pred = exx_pred.detach().cpu().numpy()
        eyy_pred = eyy_pred.detach().cpu().numpy()
        e2xy_pred = e2xy_pred.detach().cpu().numpy()
        sxx_pred = sxx_pred.detach().cpu().numpy()* ratio
        syy_pred = syy_pred.detach().cpu().numpy()* ratio
        sxy_pred = sxy_pred.detach().cpu().numpy()* ratio


        
        SVonMises = np.float64(np.sqrt(0.5 * ((sxx_pred - syy_pred) ** 2 + (syy_pred) ** 2 + (-sxx_pred) ** 2 + 6 * (sxy_pred ** 2))))
            
        return SVonMises
    

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

class DeepEnergyMethod2to1:
    # Instance attributes
    def __init__(self, model, numIntType, energy, dim, r):
        # self.data = data
        self.model = MultiLayerNetLoRA(model[0], model[1], model[2], r=r)
        self.model = self.model.to(dev)
        self.model.load_state_dict(torch.load('./model/PINN/model_FGM2.pth'), strict = False)
        self.intLoss = IntegrationLoss(numIntType, dim)
        self.energy = energy
        # self.post = PostProcessing(energy, dim)
        self.dim = dim
        self.lossArray = []

    def freeze_lora_layers(self, model: nn.Module, train_layer_keywords: list, lr=1e-3):
        """
        冻结 model 中除含有指定关键字的 LoRA 参数以外的所有参数，
        并返回一个只更新指定层 LoRA 参数的优化器。
        
        参数:
            model (nn.Module): 你的 PyTorch 模型
            train_layer_keywords (list): 需要更新(微调)的层名称关键字的列表。
            lr (float): 优化器学习率
            
        返回:
            optimizer (torch.optim.Optimizer): 只更新指定层 LoRA 参数的优化器
        """
        # 遍历所有参数
        for name, param in model.named_parameters():
            # 如果关键词命中，就保持 requires_grad=True；否则 requires_grad=False
            if any(keyword in name for keyword in train_layer_keywords):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # 收集需要训练的参数
        params_to_optimize = [p for p in model.parameters() if p.requires_grad]
        # 构建优化器
        optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
        
        return optimizer


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
        optimizer = self.freeze_lora_layers(self.model, train_layer_keywords=["linear2.lora", "linear2.bias","linear3.lora", "linear3.bias","linear4.lora", "linear4.bias"], lr=0.001)

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
                    fext = torch.bmm(neu_u_pred.unsqueeze(1), neuBC_values[i].unsqueeze(2))
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
                
                L2norm = np.linalg.norm(dis_y_pred - dis_y_iga)/np.linalg.norm( dis_y_iga)
                H1norm = np.linalg.norm(Von_pred - von_iga)/np.linalg.norm(von_iga)
                

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
            
            L2norm = np.linalg.norm(dis_y_pred - dis_y_iga)/np.linalg.norm( dis_y_iga)
            H1norm = np.linalg.norm(Von_pred - von_iga)/np.linalg.norm(von_iga)
            self.L2_array.append(L2norm)
            self.H1_array.append(H1norm)            
        elapsed = time.time() - start_time
        print('Training time: %.4f' % elapsed)
        torch.save(self.model.state_dict(), f'model/PINN/model_FGM2to1_lora_r1.pth')
    def getU(self, x):
        x_scale = x/cf.Length
        u = self.model(x_scale)
        Ux = x[:, 0] * u[:, 0] * (cf.Length-x[:, 0])
        Uy = x[:, 0] * u[:, 1] * (cf.Length-x[:, 0])
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
        xy_tensor = torch.from_numpy(points[:,:-1]).float() # 输入的points是三维度的，最后一个维度先不做模型的输入
        xy_tensor = xy_tensor.to(dev)
        xy_tensor.requires_grad_(True)
        u_pred = self.getU(xy_tensor)
        surUx = u_pred[:, 0].cpu().detach().numpy()
        surUy = u_pred[:, 1].cpu().detach().numpy()

        return surUx, surUy
    
    def evaluate_von(self, points): # 仅仅输入位移来降低运算量
        xy_tensor = torch.from_numpy(points[:,:-1]).float() # 输入的points是三维度的，最后一个维度先不做模型的输入
        xy_tensor = xy_tensor.to(dev)
        xy_tensor.requires_grad_(True)
        u_pred_torch = self.getU(xy_tensor)
        duxdxy = grad(u_pred_torch[:, 0].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev),
                       create_graph=True, retain_graph=True)[0]
        duydxy = grad(u_pred_torch[:, 1].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev),
                       create_graph=True, retain_graph=True)[0]
        
        dudx = duxdxy[:, 0].unsqueeze(1)
        dudy = duxdxy[:, 1].unsqueeze(1)
        dvdx = duydxy[:, 0].unsqueeze(1)
        dvdy = duydxy[:, 1].unsqueeze(1)
        exx_pred = dudx
        eyy_pred = dvdy
        e2xy_pred = dudy + dvdx     
        
        y = points[:,1]
        ratio = 1 - 0.5*np.cos(torch.pi*(y/1-0.5)).reshape(-1,1)
        
        sxx_pred = (self.energy.D11_mat * exx_pred + self.energy.D12_mat * eyy_pred) 
        syy_pred = (self.energy.D12_mat * exx_pred + self.energy.D22_mat * eyy_pred) 
        sxy_pred = (self.energy.D33_mat * e2xy_pred)
        
        exx_pred = exx_pred.detach().cpu().numpy()
        eyy_pred = eyy_pred.detach().cpu().numpy()
        e2xy_pred = e2xy_pred.detach().cpu().numpy()
        sxx_pred = sxx_pred.detach().cpu().numpy()* ratio
        syy_pred = syy_pred.detach().cpu().numpy()* ratio
        sxy_pred = sxy_pred.detach().cpu().numpy()* ratio


        
        SVonMises = np.float64(np.sqrt(0.5 * ((sxx_pred - syy_pred) ** 2 + (syy_pred) ** 2 + (-sxx_pred) ** 2 + 6 * (sxy_pred ** 2))))
            
        return SVonMises
    

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
    r_list = list(range(61, 81))
    r_list = [x for x in r_list if x!= 4 and x %4 ==0]
    for r in r_list:
        path = f'output/dem_lora/r={r}'

        # 检查路径是否存在，如果不存在则创建
        if not os.path.exists(path):
            os.makedirs(path)
        #  以2为目标，1是被迁移
        x_iga = np.load('./output/IGA/const-dist2/x_plot_grid.npy').flatten()
        y_iga = np.load('./output/IGA/const-dist2/y_plot_grid.npy').flatten()
        points = np.stack([x_iga, y_iga, np.zeros(len(x_iga))], axis=1)
        
        # 获取位移场y和vonmise应力
        dis_y_iga = np.load('./output/IGA/const-dist2/v_exact.npy').flatten()
        von_iga = np.load('./output/IGA/const-dist2/von_mise.npy').flatten()
        
        # ----------------------------------------------------------------------
        #                   STEP 1: SETUP DOMAIN - COLLECT CLEAN DATABASE
        # ----------------------------------------------------------------------
        dom, boundary_neumann, boundary_dirichlet = des.setup_domain()
        x, y, datatest = des.get_datatest()
        # ----------------------------------------------------------------------
        #                   STEP 2: SETUP MODEL
        # ----------------------------------------------------------------------
        mat = md.EnergyModel('FGM2', 2, cf.E, cf.nu)
        #dem = DeepEnergyMethod([cf.D_in, cf.H, cf.D_out], 'simpson', mat, 2)
        dem = DeepEnergyMethod1to2([cf.D_in, cf.H, cf.D_out], 'trapezoidal', mat, 2, r)
        # ----------------------------------------------------------------------
        #                   STEP 3: TRAINING MODEL
        # ----------------------------------------------------------------------
        start_time = time.time()
        shape = [cf.Nx, cf.Ny]
        dxdy = [cf.hx, cf.hy]
        cf.iteration = 100000
    
        cf.lr = 0.001
        dem.train_model(shape, dxdy, dom, boundary_neumann, boundary_dirichlet, cf.iteration, cf.lr)
        end_time = time.time() - start_time
        print("End time: %.5f" % end_time)
        z = np.array([0.])
    
        # error_storage
        np.save(f"./output/dem_lora/r={r}/FGM1to2_MLP_trap_L2_norm.npy", dem.L2_array)
        np.save(f"./output/dem_lora/r={r}/FGM1to2_MLP_trap_H1_norm.npy", dem.H1_array)
        
        #  以1为目标，2是被迁移
        x_iga = np.load('./output/IGA/const-dist1/x_plot_grid.npy').flatten()
        y_iga = np.load('./output/IGA/const-dist1/y_plot_grid.npy').flatten()
        points = np.stack([x_iga, y_iga, np.zeros(len(x_iga))], axis=1)
        
        # 获取位移场y和vonmise应力
        dis_y_iga = np.load('./output/IGA/const-dist1/v_exact.npy').flatten()
        von_iga = np.load('./output/IGA/const-dist1/von_mise.npy').flatten()
        
        # ----------------------------------------------------------------------
        #                   STEP 1: SETUP DOMAIN - COLLECT CLEAN DATABASE
        # ----------------------------------------------------------------------
        dom, boundary_neumann, boundary_dirichlet = des.setup_domain()
        x, y, datatest = des.get_datatest()
        # ----------------------------------------------------------------------
        #                   STEP 2: SETUP MODEL
        # ----------------------------------------------------------------------
        mat = md.EnergyModel('FGM1', 2, cf.E, cf.nu)
        #dem = DeepEnergyMethod([cf.D_in, cf.H, cf.D_out], 'simpson', mat, 2)
        dem = DeepEnergyMethod2to1([cf.D_in, cf.H, cf.D_out], 'trapezoidal', mat, 2, r)
        # ----------------------------------------------------------------------
        #                   STEP 3: TRAINING MODEL
        # ----------------------------------------------------------------------
        start_time = time.time()
        shape = [cf.Nx, cf.Ny]
        dxdy = [cf.hx, cf.hy]
    
        cf.lr = 0.001
        dem.train_model(shape, dxdy, dom, boundary_neumann, boundary_dirichlet, cf.iteration, cf.lr)
        end_time = time.time() - start_time
        print("End time: %.5f" % end_time)
        z = np.array([0.])
    
        # error_storage
        np.save(f"./output/dem_lora/r={r}/FGM2to1_MLP_trap_L2_norm.npy", dem.L2_array)
        np.save(f"./output/dem_lora/r={r}/FGM2to1_MLP_trap_H1_norm.npy", dem.H1_array)
