"""
@author: 王一铮, 447650327@qq.com
"""
import sys
sys.path.insert(0, '../..') # add路径
from DEM_plate_hole.plate_hole import define_structure as des
from DEM_plate_hole.MultiLayerNet import *
from DEM_plate_hole import EnergyModel as md
from DEM_plate_hole import Utility as util
from DEM_plate_hole.plate_hole import config as cf
from DEM_plate_hole.IntegrationLoss import *
from DEM_plate_hole.EnergyModel import *
import numpy as np
import time
import torch
import xlrd

def setup_seed(seed):
# random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(2024)

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
    def __init__(self, model, grid_size, spline_order, numIntType, energy, dim, defect=False, r=0):
        # self.data = data
        #self.model = MultiLayerNet(model[0], model[1], model[2])
        self.model = KAN(model, base_activation=torch.nn.SiLU, grid_size=grid_size, grid_range=[0, 1], spline_order=spline_order)
        self.model = self.model.to(dev)
        self.intLoss = IntegrationLoss(numIntType, dim, defect = defect)
        self.energy = energy
        # self.post = PostProcessing(energy, dim)
        self.dim = dim
        self.lossArray = []
        self.r = r

    def train_model(self, data, neumannBC, dirichletBC, boundary_id, LHD, iteration, learning_rate):
        x = torch.from_numpy(data[:,0:2]).float() # 将data中的前两个维度放入X作为输入
        J = torch.from_numpy(data[:,2, np.newaxis]).float()
        x = x.to(dev)
        J = J.to(dev)
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
        #optimizer = torch.optim.LBFGS(self.model.parameters(), lr=learning_rate, max_iter=20)# 每一个循环需要的循环次数，这里是20，也就是每一个iter需要有20个循环
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)# 每一个循环需要的循环次数，这里是20，也就是每一个iter需要有20个循环
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5000, 6000, 7000], gamma = 0.1)
        start_time = time.time()
        energy_loss_array = []
        lambda_loss_array = []
        self.U_mag_loss_array = []
        self.Mise_loss_array = []
        U_mag_exact = np.load('../../abaqus_reference/U_mag.npy')[:,1]
        MISES_exact = np.load('../../abaqus_reference/MISES.npy')[:,1]
        for t in range(iteration):
            # Zero gradients, perform a backward pass, and update the weights.
            def closure():
                it_time = time.time()
                # ----------------------------------------------------------------------------------
                # Internal Energy
                # ----------------------------------------------------------------------------------
                ust_pred = self.getUST(x)
                storedEnergy = self.energy.getStoredEnergy(ust_pred, x) # 最小势能仅仅是应变能**************************
                internal2 = self.intLoss.ele2d(storedEnergy, J)
        
                external2 = torch.zeros(len(neuBC_coordinates)) # 外力功,到时候损失要加负号
                for i, vali in enumerate(neuBC_coordinates):
                    neu_ust_pred = self.getUST(neuBC_coordinates[i])
                    neu_u_pred = neu_ust_pred[:,(0, 1)]
                    fext = torch.bmm((neu_u_pred).unsqueeze(1), neuBC_values[i].unsqueeze(2))
                    external2[i] = self.intLoss.montecarlo1D(fext, LHD[1])
                bc_u_crit = torch.zeros((len(dirBC_coordinates))) # 维度是1
                for i, vali in enumerate(dirBC_coordinates):
                    dir_u_pred = self.getUST(dirBC_coordinates[i])
                    bc_u_crit[i] = self.loss_squared_sum(dir_u_pred, dirBC_values[i]) # 输入维度是50，2的张量
                energy_loss = internal2 - torch.sum(external2)
                # record_l2 error(u_mag), and H1_error(mise)
                U, U_mag, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises, F11, F12, F21, F22 = self.evaluate_model(x.detach().cpu().numpy())
                L2 = np.linalg.norm(U_mag - U_mag_exact) / np.linalg.norm(U_mag_exact)
                H1 = np.linalg.norm(SVonMises - MISES_exact) / np.linalg.norm(MISES_exact)
                self.U_mag_loss_array.append(L2)
                self.Mise_loss_array.append(H1)           
                # try to evaluate dis_mag and mise error in every epoch
                loss = energy_loss 
                optimizer.zero_grad()
                loss.backward()
                print('Iter: %d Loss: %.9e Internal Energy: %.9e  External Energy: %.9e  L2: %.9e  H1: %.9e Time: %.3e'
                      % (t + 1, loss.item(), internal2.item(), torch.sum(external2).item(), L2, H1, time.time() - it_time))
                energy_loss_array.append(energy_loss.data)
                #lambda_loss_array.append(lambda_loss.data)
                self.lossArray.append(loss.data)
                return loss
            optimizer.step(closure)
            scheduler.step()
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
        x_scale = x/cf.Length
        ust = self.model(x_scale)
        Ux = x[:, 0] * ust[:, 0] # 如果x坐标是0的话，x方向位移也是0
        Uy = x[:, 1] * ust[:, 1] # 如果y坐标是0的话，y方向位移也是0

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
            U_mag = (np.float64(surUx)**2 + np.float64(surUy)**2 + np.float64(surUz)**2)**(0.5)
            return U, np.float64(U_mag), np.float64(surS11), np.float64(surS12), np.float64(surS13), np.float64(surS22), np.float64(
                surS23), \
                   np.float64(surS33), np.float64(surE11), np.float64(surE12), \
                   np.float64(surE13), np.float64(surE22), np.float64(surE23), np.float64(surE33), np.float64(
                SVonMises), \
                   np.float64(F11_pred), np.float64(F12_pred), np.float64(F21_pred), np.float64(F22_pred)

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
        self.loss


if __name__ == '__main__':
    # ----------------------------------------------------------------------
    #                   STEP 1: SETUP DOMAIN - COLLECT CLEAN DATABASE
    # ----------------------------------------------------------------------
    dom, boundary_neumann, boundary_dirichlet, bound_id = des.setup_domain_tri() # 返回的boundary value是一个字典，里面不仅仅有坐标还有力大小，力大小没有标准化
    datatest = des.get_datatest()
    
    mat = md.EnergyModel('elasticityMP', 2, cf.E, cf.nu)
    # ----------------------------------------------------------------------
    #                   STEP 2: SETUP MODEL
    # ----------------------------------------------------------------------
    para_list = []
    error_L2_list = []
    error_H1_list = []
    layers_list = [[2,5,2],[2,5,5,2],[2,5,5,5,2]]
    grid_size_list = [3, 5, 8, 10, 12, 15, 18, 20, 22, 25, 27, 30]
    dict_scale_law = {}
    for layer in layers_list:
        for grid_size in grid_size_list:
    #dem = DeepEnergyMethod([cf.D_in, cf.H, cf.D_out], 'montecarlo', mat, 2, defect=True, r = cf.r)
            dem = DeepEnergyMethod(layer, grid_size, 3, 'montecarlo', mat, 2, defect=True, r = cf.r)
            # ----------------------------------------------------------------------
            #                   STEP 3: TRAINING MODEL
            # ----------------------------------------------------------------------
            start_time = time.time()
            shape = [cf.Nx, cf.Ny]
            dxdy = [cf.hx, cf.hy]
            cf.iteration = 8000
            cf.filename_out = '../../results/KINN_DEM_rbf'
            # 将坐标点先存起来
            N = dom.shape[0] # abaqus最后一个维度必须是0，就是必须要输入三位坐标才行
            dom_aba = np.hstack((dom[:,:-1], np.zeros((N, 1))))
            dem.train_model(dom, boundary_neumann, boundary_dirichlet, bound_id, [cf.Length, cf.Height, cf.Depth], cf.iteration, cf.lr)
            end_time = time.time() - start_time
            print("End time: %.5f" % end_time)
            z = np.zeros((datatest.shape[0], 1))
            datatest = np.concatenate((datatest, z), 1)
            U, U_mag, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises, F11, F12, F21, F22 = dem.evaluate_model(dom_aba)
            # 储存一下error contourf
            U_mag_exact = np.load('../../abaqus_reference/U_mag.npy')[:,1]
            MISES_exact = np.load('../../abaqus_reference/MISES.npy')[:,1]
            L2 = np.linalg.norm(U_mag - U_mag_exact) / np.linalg.norm(U_mag_exact)
            H1 = np.linalg.norm(SVonMises - MISES_exact) / np.linalg.norm(MISES_exact)
        
            #L2norm = util.getL2norm2D(surUx, surUy, cf.Nx, cf.Ny, cf.hx, cf.hy) # 因为这里也没有和精确解比较，而且是一个缺口的方板，没有必要计算此范数
            #H10norm = util.getH10norm2D(F11, F12, F21, F22, cf.Nx, cf.Ny, cf.hx, cf.hy)
            #print("L2 norm = %.10f" % L2norm)
            #print("H10 norm = %.10f" % H10norm)
            print('Loss convergence')
            num_pa
            para_list.append(num_para)
            error_L2_list.append(L2)
            error_H1_list.append(H1)
        dict_scale_law = {
            'layers_list': layers_list,
            'grid_size_list': grid_size_list,
            'parameters': para_list,
            'errors_L2': error_L2_list,
            'errors_H1': error_H1_list
        }
        np.save( '../../results/plate_hole_grid_size.npy', dict_scale_law)
        loaded_data = np.load('../../results/plate_hole_grid_size.npy', allow_pickle=True).item()
        print(loaded_data)    
    
    
    
