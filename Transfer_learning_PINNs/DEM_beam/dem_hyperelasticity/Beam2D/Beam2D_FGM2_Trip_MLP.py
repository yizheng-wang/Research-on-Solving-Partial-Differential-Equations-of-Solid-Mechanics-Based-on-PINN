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
        self.model = MultiLayerNet(model[0], model[1], model[2])
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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
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
        torch.save(self.model.state_dict(), f'model/PINN/model_FGM2.pth')
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
        ratio = 1 - 0.5*np.cos(torch.pi*y/2).reshape(-1,1)
        
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
    # 获取坐标点
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
    dem = DeepEnergyMethod([cf.D_in, cf.H, cf.D_out], 'trapezoidal', mat, 2)
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
    disx_pred, disy_pred = dem.evaluate_u(points)

    dis_y_abs_error = np.abs(disy_pred.flatten() - dis_y_iga.flatten())
    util.write_vtk_v2p("./output/dem/FGM2_IGA_disy", points, dis_y_iga)
    util.write_vtk_v2p("./output/dem/FGM2_MLP_trap_disy", points, disy_pred)
    util.write_vtk_v2p("./output/dem/FGM2_MLP_trap_disy_error", points, dis_y_abs_error)
    
    von_pred = dem.evaluate_von(points).flatten()
    
    von_abs_error = np.abs(von_pred.flatten() - von_iga.flatten())
    util.write_vtk_v2p("./output/dem/FGM2_IGA_von", points, von_iga)
    util.write_vtk_v2p("./output/dem/FGM2_MLP_trap_von", points, von_pred)
    util.write_vtk_v2p("./output/dem/FGM2_MLP_trap_von_error", points, von_abs_error)
    # error_storage
    np.save("./output/dem/FGM2_MLP_trap_L2_norm.npy", dem.L2_array)
    np.save("./output/dem/FGM2_MLP_trap_H1_norm.npy", dem.H1_array)
    

    # # y=0.5 storage
    # points_y05 = points[points[:,1]==0.5]
    # disx_pred_y05, disy_pred_y05 = dem.evaluate_u(points_y05) # prediction
    # von_pred_y05 = dem.evaluate_von(points_y05)
    # # dis_x_fem_y05 = dis_x_fem[points[:,1]==0.5]
    # dis_y_iga_y05 = dis_y_iga[points[:,1]==0.5]
    # von_iga_y05 = von_iga[points[:,1]==0.5]
    
    
    # dict_y05 = {'X': points_y05, 'Dis_pred': disy_pred_y05, 'Dis_fem': dis_y_iga_y05, \
    #             'Von_pred': von_pred_y05, 'Von_exact': von_iga_y05}

    # np.save("./output/dem/FGM_MLP_trap_dis_y05.npy", dict_y05)


    # # x=2.0 storage
    # points_x2 = points[points[:,0]==2.0]
    # disx_pred_x2, disy_pred_x2 = dem.evaluate_u(points_x2) # prediction
    # von_pred_x2 = dem.evaluate_von(points_x2)
    # dis_x_fem_x2 = dis_x_fem[points[:,0]==2.0]
    # dis_y_fem_x2 = dis_y_fem[points[:,0]==2.0]
    # von_fem_x2 = von_fem[points[:,0]==2.0]
    
    # dis_pred_x2 =  (disx_pred_x2**2 + disy_pred_x2**2)**0.5
    # dis_fem_x2 =  (dis_x_fem_x2**2 + dis_y_fem_x2**2)**0.5
    
    # dict_x2 = {'X': points_x2, 'Dis_pred': dis_pred_x2, 'Dis_fem': dis_fem_x2, \
    #             'Von_pred': von_pred_x2, 'Von_exact': von_fem_x2}

    # np.save("./output/dem/NeoHook_MLP_trap_x2.npy", dict_x2)
