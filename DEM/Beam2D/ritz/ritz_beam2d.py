"""
@author: 王一铮, 447650327@qq.com
改编自：https://github.com/MinhNguyenIKM/dem_hyperelasticity
"""
import sys
sys.path.insert(0, '/home/sg/SeaDrive/My Libraries/硕士学位论文/王一铮硕士论文/代码') # add路径
from DEM.Beam2D.ritz import define_structure as des
from DEM.MultiLayerNet import *
from DEM import EnergyModel as md
from DEM import Utility as util
from DEM.Beam2D.ritz import config as cf
from DEM.IntegrationLoss import *
from DEM.EnergyModel import *
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import xlrd

import matplotlib as mpl
mpl.rcParams['font.sans-serif']=['SimHei'] # 为了显示中文matplotlib

mpl.rcParams['figure.dpi'] = 1000
# fix random seeds
axes = {'labelsize' : 'large'}
# font = {'family' : 'serif',
#         'weight' : 'normal',
#         'size'   : 17}
legend = {'fontsize': 'medium'}
lines = {'linewidth': 3,
         'markersize' : 7}
# mpl.rc('font', **font)
mpl.rc('axes', **axes)
mpl.rc('legend', **legend)
mpl.rc('lines', **lines)


class DeepEnergyMethod:
    # Instance attributes
    def __init__(self, model, numIntType, energy, dim, defect = False, r=0):
        self.model = MultiLayerNet(model[0], model[1], model[2]) # 输出的数据结构并没有发生改变
        self.model = self.model.to(dev) # 将神经网络放入相应的设备中，这里可以是GPU
        self.intLoss = IntegrationLoss(numIntType, dim, defect = defect)
        self.energy = energy
        # self.post = PostProcessing(energy, dim)
        self.dim = dim
        self.lossArray = []
        self.r = r
        self.particular = MultiLayerNet_p(model[0], model[1], model[2]).to(dev)  # 定义一个特解的网络结构
        self.distance = MultiLayerNet_d_b(model[0], model[1], model[2]).to(dev)  # 定义一个距离的网络结构

    def train_particular_soluton(self, dirichletBC, tol = cf.particular_tol):
        dirBC_coordinates = {}  # declare a dictionary
        dirBC_values = {}  # declare a dictionary
        dirBC_penalty = {}
        dirBC_normal = {} # 得到一个储存相应tensor的字典{0：tensor}这种形式
        n = 0 # 获得dir边界点的数目
        for i, keyi in enumerate(dirichletBC):
            dirBC_coordinates[i] = torch.from_numpy(dirichletBC[keyi]['coord']).float().to(dev)
            dirBC_values[i] = torch.from_numpy(dirichletBC[keyi]['known_value']).float().to(dev)
            dirBC_penalty[i] = torch.tensor(dirichletBC[keyi]['penalty']).float().to(dev)
            n += len(dirichletBC[keyi]['coord'])
        self.boundary_dir_n = n # 将边界dir的数目储存到类中的属性attribute中
        start_time = time.time()
        particular_loss_array = []
        # 构建输入和输出，将所有边界上面的坐标和位移值都变成一个统一的输入和输出
        x_train = dirBC_coordinates[0] # 由于悬臂梁问题只有一个dir条件
        y_train = dirBC_values[0]
        criterion = torch.nn.L1Loss(reduction = 'sum')
        optimizer = torch.optim.Adam(self.particular.parameters(), lr = 1e-4)
        # 接下来进行特解网络的训练
        t = 0
        while True: # 将最近的两个损失进行对比，如果差不多就可以停止迭代了
            
            it_time = time.time()
            t += 1
            y_pred = self.particular(x_train) # 这里输出的是二维
            loss = criterion(y_pred, y_train)
            particular_loss_array.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if t % 10 == 0:
                print('Train particular solution phrase: Iter: %d Loss: %.9e Time: %.3e'
                      % (t, loss.item(), time.time() - it_time))
            if loss<tol: break        

    # def train_distance(self, dirichletBC, dom, proportion=0.5, tol=0.1):
    #     # 这里距离网络也是一个二维的输出
    #     '''
        

    #     Parameters
    #     ----------
    #     dirichletBC : diction
    #         The dictionary include the numpy belong to different Dirlet boundaries.
    #     dom : numpy
    #         The domain point, whose shape is n*2 due to the problem is 2 dimensionality.
    #     proportion : float, optional
    #         DESCRIPTION. The default is 0.3. Because the domain has too many point, it is not necessary to use all the domain
    #         point to train the distance neural network.
    #     tol : float, optional
    #         DESCRIPTION. The default is 0.3. The MSE loss stop criterion.

    #     Returns
    #     -------
    #     None.

    #     '''
    #     dirBC_coordinates = {}  # declare a dictionary
    #     dirBC_values = {}  # declare a dictionary
    #     dirBC_penalty = {}
    #     dirBC_normal = {} # 得到一个储存相应tensor的字典{0：tensor}这种形式
    #     for i, keyi in enumerate(dirichletBC):
    #         dirBC_coordinates[i] = torch.from_numpy(dirichletBC[keyi]['coord']).float().to(dev)
    #         dirBC_values[i] = torch.from_numpy(dirichletBC[keyi]['known_value']).float().to(dev)
    #         dirBC_penalty[i] = torch.tensor(dirichletBC[keyi]['penalty']).float().to(dev)

    #     start_time = time.time()
    #     distance_loss_array = []
    #     # 构建输入和输出，将所有边界上面的坐标和位移值都变成一个统一的输入和输出
    #     B_train = dirBC_coordinates[0]
    #     B_train = B_train.cpu().numpy() #  将放入边界训练的张量转化为array
    #     criterion = torch.nn.L1Loss(reduction = 'sum')
    #     optimizer = torch.optim.Adam(self.distance.parameters(), lr = 1e-4)
    #     # 利用KNN的最邻近KDtree的方法进行logn的搜索，这是一种快速搜索的技术，balltree也可以
    #     kdt = KDTree(B_train, metric='euclidean')
    #     train_d_idx = np.random.choice(len(dom), int(len(dom)*proportion)) # 从全域中均匀取得一部分用来训练位移网络的点,这一步是用来获得相应的训练点的idx的
    #     D_train = dom[train_d_idx, :] # 获得用来训练距离神经网络的坐标点
    #     d, _ = kdt.query(D_train, k=1, return_distance = True) # d是域内训练点离边界点的最近距离，用来给距离神经网络做标签用
    #     # 将边界点放入数据集中
    #     D_train = np.concatenate((D_train, B_train)) #  这是一个二维的数据
    #     d = np.concatenate((d, np.zeros((len(B_train), 1)))) # 这是一个一维度的距离
    #     d = np.concatenate((d, d), 1) #  将d变成一个两列相同的二维数据，这是由于悬臂梁问题的特殊性，才这么取假设的
    #     # 将numpy转化为tensor进行训练
    #     y_train= torch.from_numpy(d).float().to(dev)
    #     # 由于两个维度的距离数据应该相同，所以这里应该y_train 扩展到两列
        
    #     D_train = torch.from_numpy(D_train).float().to(dev)
    #     # 接下来进行特解网络的训练
    #     t = 0
    #     while True: # 将最近的两个损失进行对比，如果差不多就可以停止迭代了
            
    #         it_time = time.time()
    #         t += 1
    #         y_pred = self.distance(D_train) # 获得两位数据
    #         loss = criterion(y_pred, y_train)
    #         distance_loss_array.append(loss)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         if t % 10 == 0:
    #             print('Train distance phrase: Iter: %d Loss: %.9e Time: %.3e'
    #                   % (t, loss.item(), time.time() - it_time))
    #         if loss<tol: break


    def train_model(self, shape, dxdydz, data, neumannBC, dirichletBC, iteration, learning_rate, penalty_boundary_loss):
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
                storedEnergy = self.energy.getStoredEnergy(ust_pred, x) # 最小势能仅仅是应变能**************************
                internal2 = self.intLoss.lossInternalEnergy(storedEnergy, dx=dxdydz[0], dy=dxdydz[1], shape=shape)
               
                external2 = torch.zeros(len(neuBC_coordinates)) # 外力功,到时候损失要加负号
                for i, vali in enumerate(neuBC_coordinates):
                    neu_u_pred = self.getUST(neuBC_coordinates[i])
                    fext = torch.bmm((neu_u_pred + neuBC_coordinates[i]).unsqueeze(1), neuBC_values[i].unsqueeze(2))
                    external2[i] = self.intLoss.lossExternalEnergy(fext, dx=dxdydz[1])
                bc_u_crit = torch.zeros(len(dirBC_coordinates)) # 维度是1
                for i, vali in enumerate(dirBC_coordinates):
                    dir_u_pred = self.getUST(dirBC_coordinates[i])
                    bc_u_crit[i] = self.loss_squared_sum(dir_u_pred, dirBC_values[i]) # 输入维度是50，2的张量
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
            get the z displacement

        '''
        ust = self.particular(x) + x[:, 0].unsqueeze(1) * self.model(x)

        return ust
    def evaluate_model_y0(self, line):
        
        # 评估y=0的位移扰度曲线
        line2d = np.zeros((len(line), 2))
        line2d[:, 0] = line
        line2d[:, 1] = 0.5
        line2d_tensor = torch.from_numpy(line2d).float().to(dev)
        ust_pred_y = self.getUST(line2d_tensor)
        v = ust_pred_y[:, 1] # 获得扰度曲线
        v = v.cpu().detach().numpy()
        return v
    # --------------------------------------------------------------------------------
    # Evaluate model to obtain:
    # 1. U - Displacement
    # 2. E - Green Lagrange Strain
    # 3. S - 2nd Piola Kirchhoff Stress
    # 4. F - Deformation Gradient
    # Date implement: 20.06.2019
    # --------------------------------------------------------------------------------
    def evaluate(self, dom):
        """
        

        Parameters
        ----------
        dom : numpy
            the dom point need to be evaluated.

        Returns
        -------
        disZ_predict : numpy
            The prediction of the ready model.

        """
        dom_tensor = torch.from_numpy(dom).float().to(dev)
        dom_tensor.requires_grad_(True)   
        disZ_predict = self.getUST(dom_tensor)

        disZ_predict = disZ_predict.cpu().detach().numpy()
        return disZ_predict
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
    data = xlrd.open_workbook("悬臂梁y=0的位移的abaqus解.xlsx")
    table = data.sheet_by_index(0)
    x = table.col_values(0)
    xi = np.array(x)
    dis_abqus = table.col_values(1)    
    pred_array = []
    penalty_boundary_loss_array = []
    for i in range(1):
        # ----------------------------------------------------------------------
        #                   STEP 1: SETUP DOMAIN - COLLECT CLEAN DATABASE
        # ----------------------------------------------------------------------
        dom, boundary_neumann, boundary_dirichlet = des.setup_domain() # 返回的boundary value是一个字典，里面不仅仅有坐标还有力大小，力大小没有标准化
        x, y, datatest = des.get_datatest()
        # ----------------------------------------------------------------------
        #                   STEP 2: SETUP MODEL
        # ----------------------------------------------------------------------
        mat = md.EnergyModel('elasticityMP', 2, cf.E, cf.nu)
        dem = DeepEnergyMethod([cf.D_in, cf.H, cf.D_out], 'simpson', mat, 2)
        # ----------------------------------------------------------------------
        #                   STEP 3: TRAINING MODEL
        # ----------------------------------------------------------------------
        start_time = time.time()
        shape = [cf.Nx, cf.Ny]
        dxdy = [cf.hx, cf.hy]
        cf.iteration = 1000
        # cf.filename_out = "./output/dem/P%iBeam2D_minpotential_2Layer_mesh200x50_iter10000_simp_lr_0.000001" % penalty_boundary_loss
        dem.train_particular_soluton(boundary_dirichlet)
#        dem.train_distance(boundary_dirichlet, dom) # 由于悬臂梁的距离网络很容易，所以自己给定了
        penalty_boundary_loss = 100*(cf.particular_tol + cf.distance_tol) / np.sqrt(dem.boundary_dir_n)
        dem.train_model(shape, dxdy, dom, boundary_neumann, boundary_dirichlet, cf.iteration, cf.lr, penalty_boundary_loss)
        end_time = time.time() - start_time
        print("End time: %.5f" % end_time)
        z = np.array([0])
        #U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises, F11, F12, F21, F22 = dem.evaluate_model(x, y, z)
        # util.write_vtk_v2(cf.filename_out, x, y, z, U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises)
        #surUx, surUy, surUz = U
        #L2norm = util.getL2norm2D(surUx, surUy, cf.Nx, cf.Ny, cf.hx, cf.hy)
        #H10norm = util.getH10norm2D(F11, F12, F21, F22, cf.Nx, cf.Ny, cf.hx, cf.hy)
        #print("L2 norm = %.10f" % L2norm)
        #print("H10 norm = %.10f" % H10norm)
        #print('Loss convergence')
        # 对比不同罚函数下的位移曲线
        v = dem.evaluate_model_y0(xi) # 获得扰度值，中性面的扰度值
        pred_array.append(v)
        penalty_boundary_loss_array.append(penalty_boundary_loss)
        #plt.plot(x, v,  color = 'red', label = 'p%i solution' % penalty_boundary_loss )
    #  画曲线评估
    plt.plot(xi, dis_abqus, color = 'red', label = 'Abaqus解',  marker='o')
    for idx, bp in enumerate(penalty_boundary_loss_array):
        plt.plot(xi, pred_array[idx], label = 'DEM', marker='*', ls = ':' )
    plt.xlabel('x坐标')
    plt.ylabel('y方向位移')
    plt.legend()
    plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
