"""
这个距离场是自己构造的
Implements the 2D Hyperelastic beam models (Neo-Hookean)
"""
import sys
sys.path.insert(0, '/home/sg/SeaDrive/My Libraries/开题报告/PINN最小能量原理/dem_hyperelasticity-master') # add路径
from dem_hyperelasticity.crack.quarter_crack import define_structure_quarter as des
from dem_hyperelasticity.MultiLayerNet import *
from dem_hyperelasticity import EnergyModel as md
from dem_hyperelasticity import Utility as util
from dem_hyperelasticity.crack.quarter_crack import config_quarter as cf
from dem_hyperelasticity.IntegrationLoss import *
from dem_hyperelasticity.EnergyModel import *
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import numpy as np
import time
import torch
import xlrd

mpl.rcParams['figure.dpi'] = 1000
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
    def __init__(self, model, numIntType, energy, dim, defect=False, r=0):
        # self.data = data
        self.model = MultiLayerNet(model[0], model[1], model[2]) # 输出的数据结构并没有发生改变
        self.model = self.model.to(dev) # 将神经网络放入相应的设备中，这里可以是GPU
        self.intLoss = IntegrationLoss(numIntType, dim, defect = defect)
        self.energy = energy
        # self.post = PostProcessing(energy, dim)
        self.dim = dim
        self.lossArray = []
        self.r = r
        self.particular = MultiLayerNet_p(model[0], model[1], model[2]).to(dev)  # 定义一个特解的网络结构
        self.distance = MultiLayerNet_d(model[0], model[1], model[2]).to(dev)  # 定义一个距离的网络结构

    def train_particular_soluton(self, dirichletBC, tol=1):
        dirBC_coordinates = {}  # declare a dictionary
        dirBC_values = {}  # declare a dictionary
        dirBC_penalty = {}
        dirBC_normal = {} # 得到一个储存相应tensor的字典{0：tensor}这种形式
        n = 0
        for i, keyi in enumerate(dirichletBC):
            dirBC_coordinates[i] = torch.from_numpy(dirichletBC[keyi]['coord']).float().to(dev)
            dirBC_values[i] = torch.from_numpy(dirichletBC[keyi]['known_value']).float().to(dev)
            dirBC_penalty[i] = torch.tensor(dirichletBC[keyi]['penalty']).float().to(dev)
            n += len(dirichletBC[keyi]['coord'])
        self.num_dir = n
        start_time = time.time()
        particular_loss_array = []
        # 构建输入和输出，将所有边界上面的坐标和位移值都变成一个统一的输入和输出
        x_train = torch.cat([dirBC_coordinates[0], dirBC_coordinates[1]])
        y_train = torch.cat([dirBC_values[0], dirBC_values[1]])
        criterion = torch.nn.L1Loss(reduction = 'sum')
        optimizer = torch.optim.Adam(self.particular.parameters(), lr = 1e-4)
        # 接下来进行特解网络的训练
        t = 0
        while True: # 将最近的两个损失进行对比，如果差不多就可以停止迭代了
            
            it_time = time.time()
            t += 1
            y_pred = self.particular(x_train)
            loss = criterion(y_pred, y_train)
            particular_loss_array.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if t % 10 == 0:
                print('Train particular solution phrase: Iter: %d Loss: %.9e Time: %.3e'
                      % (t, loss.item(), time.time() - it_time))
            if loss<tol: break
        
    def train_distance(self, dirichletBC, dom, proportion=0.05, tol=10):
        '''
        

        Parameters
        ----------
        dirichletBC : diction
            The dictionary include the numpy belong to different Dirlet boundaries.
        dom : numpy
            The domain point, whose shape is n*2 due to the problem is 2 dimensionality.
        proportion : float, optional
            DESCRIPTION. The default is 0.3. Because the domain has too many point, it is not necessary to use all the domain
            point to train the distance neural network.
        tol : float, optional
            DESCRIPTION. The default is 0.3. The MSE loss stop criterion.

        Returns
        -------
        None.

        '''
        dom = dom[:, 0:2]
        dirBC_coordinates = {}  # declare a dictionary
        dirBC_values = {}  # declare a dictionary
        dirBC_penalty = {}
        dirBC_normal = {} # 得到一个储存相应tensor的字典{0：tensor}这种形式
        for i, keyi in enumerate(dirichletBC):
            dirBC_coordinates[i] = torch.from_numpy(dirichletBC[keyi]['coord']).float().to(dev)
            dirBC_values[i] = torch.from_numpy(dirichletBC[keyi]['known_value']).float().to(dev)
            dirBC_penalty[i] = torch.tensor(dirichletBC[keyi]['penalty']).float().to(dev)

        start_time = time.time()
        distance_loss_array = []
        B_train1 = torch.cat([dirBC_coordinates[0]]) # 定义左边界坐标
        B_train2 = torch.cat([dirBC_coordinates[1]]) # 定义下右边界坐标
        B_train1 = B_train1.cpu().numpy() # 分别为两个边界求距离，提供标签
        B_train2 = B_train2.cpu().numpy()
        criterion = torch.nn.L1Loss(reduction = 'sum')
        optimizer = torch.optim.Adam(self.distance.parameters(), lr = 1e-4) # 一个输出，一个左边界的距离
        # 利用KNN的最邻近KDtree的方法进行logn的搜索，这是一种快速搜索的技术，balltree也可以
        kdt1 = KDTree(B_train1, metric='euclidean')
        kdt2 = KDTree(B_train2, metric='euclidean')
        train_d_idx = np.random.choice(len(dom), int(len(dom)*proportion)) # 从全域中均匀取得一部分用来训练位移网络的点,这一步是用来获得相应的训练点的idx的
        D_train = dom[train_d_idx, :] # 获得用来训练距离神经网络的坐标点     
        # 下面定义不同区域的距离，分为6块
        d11, _ = kdt1.query(D_train, k=1, return_distance = True) # d是域内训练点离边界点的最近距离，用来给距离神经网络做标签用
        d12, _ = kdt2.query(D_train, k=1, return_distance = True) # d是域内训练点离边界点的最近距离，用来给距离神经网络做标签用
        d21 =    np.zeros((len(B_train1), 1))
        d22, _ = kdt2.query(B_train1, k=1, return_distance = True)
        d31, _ = kdt1.query(B_train2, k=1, return_distance = True)
        d32 =    np.zeros((len(B_train2), 1))
        # 将上面的距离组合起来
        d1 = np.concatenate((d11, d21, d31))
        d2 = np.concatenate((d12, d22, d32))
        d =np.concatenate((d1, d2), 1)
        D_train = np.concatenate((D_train, B_train1, B_train2))
        y_train= torch.from_numpy(d).float().to(dev) # 训练一个距离网络
        D_train = torch.from_numpy(D_train).float().to(dev)
        t = 0
        
        while True: # 将最近的两个损失进行对比，如果差不多就可以停止迭代了
            
            it_time = time.time()
            t += 1
            y_pred = self.distance(D_train)
            loss = criterion(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            distance_loss_array.append(loss)
            if t % 10 == 0:
                print('Train distance phrase: Iter: %d Loss: %.9e Time: %.3e'
                      % (t, loss.item(), time.time() - it_time))
            if loss<tol: break
    # def distance(self, x):
    #     '''
    #     input:
    #         x: tensor, 2 dimensionality, that should be detach to give to phi
    #     return:
    #         dis: tensor, to dev gpu
    #     '''
    #     x = x.cpu().detach().numpy() # 将x从tensor变成numpy
    #     dis = np.zeros((len(x), 2)) # dis是二维，为了distance乘以广义网络做准备
    #     segt1 = np.array([[0., 0., 0., 20.]])
    #     segt2 = np.array([[10., 0., 20., 0.]])
    #     for idx, item in enumerate(x):
    #         dis[idx, 0] = self.phi(item[0], item[1], segt1) # 获得每一个x点距离本质边界条件的距离
    #         dis[idx, 1] = self.phi(item[0], item[1], segt2) # 获得每一个x点距离本质边界条件的距离
    #     dis = torch.from_numpy(dis).float().to(dev) # 返回一个tensor
    #     return dis
        
    def train_model(self, data, neumannBC, dirichletBC, boundary_id, LHD, iteration, learning_rate, penalty_boundary_loss):
        data_l = data[data[:,0]<10]
        data_r = data[data[:,0]>=10]
        x_l = torch.from_numpy(data_l[:, 0:2]).float()
        x_l = x_l.to(dev)
        x_l.requires_grad_(True)
        Jacob_l = torch.from_numpy(data_l[:, 2, np.newaxis]).float().to(dev).requires_grad_(True)
        x_r = torch.from_numpy(data_r[:, 0:2]).float()
        x_r = x_r.to(dev)
        x_r.requires_grad_(True)
        Jacob_r = torch.from_numpy(data_r[:, 2, np.newaxis]).float().to(dev).requires_grad_(True)
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
                ust_pred_l = self.getUST_l(x_l)
                ust_pred_l.double()
                strainEnergy_l = self.energy.getStoredEnergy(ust_pred_l, x_l) # 最小势能仅仅是应变能**************************
                internal2_l = self.intLoss.ele2d(strainEnergy_l, Jacob_l) # 获得了数值积分值
                
                ust_pred_r = self.getUST_r(x_r)
                ust_pred_r.double()
                strainEnergy_r = self.energy.getStoredEnergy(ust_pred_r, x_r) # 最小势能仅仅是应变能**************************
                internal2_r = self.intLoss.ele2d(strainEnergy_r, Jacob_r) # 获得了数值积分值
                internal2 = internal2_l + internal2_r
                external2 = torch.zeros(len(neuBC_coordinates)) # 外力功,到时候损失要加负号
                for i, vali in enumerate(neuBC_coordinates):
                    neuBC_coordinate = neuBC_coordinates[i] # 将坐标与力左右分开
                    neuBC_value = neuBC_values[i]
                    neuBC_value_l = neuBC_value[neuBC_coordinate[:,0]<10] 
                    neuBC_value_r = neuBC_value[neuBC_coordinate[:,0]>=10]
                    neuBC_coordinates_l = neuBC_coordinate[neuBC_coordinate[:,0]<10]
                    neuBC_coordinates_r = neuBC_coordinate[neuBC_coordinate[:,0]>=10]
                    neu_ust_pred_l = self.getUST_l(neuBC_coordinates_l)
                    neu_ust_pred_r = self.getUST_r(neuBC_coordinates_r)
                    neu_u_pred_l = neu_ust_pred_l[:,(0, 1)]
                    neu_u_pred_r = neu_ust_pred_r[:,(0, 1)]
                    fext_l = torch.bmm((neu_u_pred_l).unsqueeze(1), neuBC_value_l.unsqueeze(2))
                    fext_r = torch.bmm((neu_u_pred_r).unsqueeze(1), neuBC_value_r.unsqueeze(2))
                    external2_l = self.intLoss.montecarlo1D(fext_l, LHD[1]/2)
                    external2_r = self.intLoss.montecarlo1D(fext_r, LHD[1]/2)
                    external2[i] = external2_l + external2_r
                bc_u_crit = torch.zeros((len(dirBC_coordinates))) # 维度是1
                for i, vali in enumerate(dirBC_coordinates):
                    dirBC_coordinate = dirBC_coordinates[i]
                    dirBC_value = dirBC_values[i][:, i].unsqueeze(1) # 将坐标与标签值左右分开
                    dirBC_coordinates_l = dirBC_coordinate[dirBC_coordinate[:,0]<10]
                    dirBC_coordinates_r = dirBC_coordinate[dirBC_coordinate[:,0]>=10]
                    dirBC_value_l = dirBC_value[dirBC_coordinate[:,0]<10] 
                    dirBC_value_r = dirBC_value[dirBC_coordinate[:,0]>=10]
                    if i==0:
                        dirBC_value = dirBC_value_l
                        
                        dir_u_pred_l = self.getUST_l(dirBC_coordinates_l)
                        dir_u_pred_r = self.getUST_r(dirBC_coordinates_r)
                        dir_u_pred = dir_u_pred_l
                        bc_u_crit[i] = self.loss_squared_sum(dir_u_pred[:, i].unsqueeze(1), dirBC_value[:, i].unsqueeze(1)) # 输入维度是50，2的张量
                    if i==1:
                        dirBC_value = dirBC_value_r
                        
                        dir_u_pred_l = self.getUST_l(dirBC_coordinates_l)
                        dir_u_pred_r = self.getUST_r(dirBC_coordinates_r)
                        dir_u_pred = dir_u_pred_r 
                        bc_u_crit[i] = self.loss_squared_sum(dir_u_pred[:, i].unsqueeze(1), dirBC_value[:, 0].unsqueeze(1)) # 输入维度是50，2的张量

                energy_loss = internal2 - torch.sum(external2)
                boundary_loss = torch.sum(bc_u_crit)
                loss = energy_loss  + penalty_boundary_loss * boundary_loss # 加上了强制位移边界损失，相当于罚函数
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
    def getUST_l(self, x):
        '''
        
        左边距离乘以sqrt(x**2+y**2)
        Parameters
        ----------
        x : tensor
            coordinate of 2 dimensionality.

        Returns
        -------
        ust_pred : tensor
            get the z displacement

        '''
        
        ust = torch.stack((x[:,0], torch.sqrt(x[:, 0]**2+x[:, 1]**2)),1) * self.model(x) # 位移是0的时候位移解就是零，所以这里不需要特解，令其为0

        return ust
    
    def getUST_r(self, x):
        '''
        
        右边距离乘以y
        Parameters
        ----------
        x : tensor
            coordinate of 2 dimensionality.

        Returns
        -------
        ust_pred : tensor
            get the z displacement

        '''
        
        ust = torch.stack((x[:,0], x[:, 1]),1) * self.model(x) # 位移是0的时候位移解就是零，所以这里不需要特解，令其为0

        return ust
    def evaluate_model(self, datatest):
        energy_type = self.energy.type
        dim = self.dim
        if dim == 2:
            Nx = len(datatest)
            x = datatest[:, 0].reshape(Nx, 1)
            y = datatest[:, 1].reshape(Nx, 1)
            #z = datatest[:, 2].reshape(Nx, 1)
            
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
                   
    def evaluate_model_c(self, datatest):
        energy_type = self.energy.type
        dim = self.dim
        if dim == 2:
            Nx = len(datatest)
            x = datatest[:, 0].reshape(Nx, 1)
            y = datatest[:, 1].reshape(Nx, 1)
            #z = datatest[:, 2].reshape(Nx, 1)
            
            xy = np.concatenate((x, y), axis=1)
            xy_tensor = torch.from_numpy(xy).float()
            xy_tensor = xy_tensor.to(dev)
            xy_tensor.requires_grad_(True)
            ust_pred_torch = self.getUST_r(xy_tensor) # 裂纹处是y=0的地方，用第二个神经网络预测就行了
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
    
    def evaluate_strain(self, interface): # 分析theta=0的应变
        dom_tensor = torch.from_numpy(interface).float().to(dev)
        dom_tensor.requires_grad_(True)   
        disZ_predict = self.getUST(dom_tensor)
        dudxy = grad(disZ_predict[:, 0].unsqueeze(1), dom_tensor, torch.ones(dom_tensor.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        e31 = dudxy[:, 0].unsqueeze(1)
        e32 = dudxy[:, 1].unsqueeze(1)
        e31 = e31.cpu().detach().numpy() # 将张量变成可以方便处理的numpy
        e32 = e32.cpu().detach().numpy()
        return e31, e32
    
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
        
    def linseg(self, x,y,x1,y1,x2,y2):
        L = self.dist(x1,y1,x2,y2) 
        xc = (x1+x2)/2. 
        yc = (y1+y2)/2.
        f = (1/L)*((x-x1)*(y2-y1) - (y-y1)*(x2-x1))
        t = (1/L)*((L/2.)**2-self.dist(x,y,xc,yc)**2) 
        varphi = np.sqrt(t**2+f**4) 
        phi = np.sqrt(f**2 + (1/4.)*(varphi - t)**2)
        return phi
    def phi(self, x,y,segments):
        m = 1.
        R = 0. 
        for i in range(len(segments[:,0])): 
            phi = self.linseg(x,y,segments[i,0],segments[i,1],segments[i,2],segments[i,3]) 
            R = R + 1./phi**m
        R = 1/R**(1/m) 
        return R
    def dist(self, x1,y1,x2,y2): 
        return np.sqrt((x2-x1)**2+(y2-y1)**2)
    
    
    
        
def exact_solution(dom):
    '''
    输出断裂力学的精确解，因为是从反逆法给出的，所以有精确解 U=sqrt(r)*sin(theta/2)
    '''
    disZ = np.zeros((len(dom), 1))
    up_idx = np.where(dom[:, 1] > 0 )
    down_idx =np.where(dom[:, 1] < 0 )
    
    disZ[up_idx[0], :] = np.sqrt((np.linalg.norm(dom[up_idx, :][0], axis = 1, keepdims = True)-dom[up_idx, 0].T)/2)
    disZ[down_idx[0], :] = -np.sqrt((np.linalg.norm(dom[down_idx, :][0], axis = 1, keepdims = True)-dom[up_idx, 0].T)/2)

    return disZ       
def exact_solution_y(n=cf.Nx): # y=0的解析解
    x = np.linspace(-1, 1, n)# 将-1到1等间隔分割n个点
    x = x[:, np.newaxis] # 将x从一个一位array 变成一个n*1的二维array
    mask = np.concatenate((np.ones((int(n/2+1), 1)), np.zeros((int(n/2), 1))))
    yup = np.sqrt(np.abs(x))*mask # 构造上边界位移解析解
    ydown = -yup # 构造下边界位移解析解
    return yup, ydown
def exact_solution_strain(interface):
    # 获取theta=0应变的解析解
    e31 = np.zeros((100, 1))
    e32 = 1/np.sqrt(interface[:, 0, np.newaxis])/2
    return e31, e32

    
        
        
        
if __name__ == '__main__':
    # 获得abaqus的mise应力的参照解
    datay0 = xlrd.open_workbook("abaqus_mise.xlsx") # 获得x=0的数据，需要提取y方向位移以及mise应力
    tabley0 = datay0.sheet_by_index(0)
    y0 = tabley0.col_values(0)
    y0 = np.array(y0)
    y0_2d = np.zeros((len(y0), 2))
    y0_2d[:, 0] = y0
    mise_abaqusy0 = tabley0.col_values(1)   
    
    penalty_boundary_loss = 10
    # ----------------------------------------------------------------------
    #                   STEP 1: SETUP DOMAIN - COLLECT CLEAN DATABASE
    # ----------------------------------------------------------------------
    dom, dom_dis,  boundary_neumann, boundary_dirichlet, bound_id = des.setup_domain() # 返回的boundary value是一个字典，里面不仅仅有坐标还有力大小，力大小没有标准化
    datatest = des.get_datatest()
    # ----------------------------------------------------------------------
    #                   STEP 2: SETUP MODEL
    # ----------------------------------------------------------------------
    mat = md.EnergyModel('elasticityMP', 2, cf.E, cf.nu)
    dem = DeepEnergyMethod([cf.D_in, cf.H, cf.D_out], 'simpson', mat, 2)
    # ----------------------------------------------------------------------
    #                   STEP 3: TRAINING MODEL
    # ----------------------------------------------------------------------
    start_time = time.time()
    #dem.train_particular_soluton(boundary_dirichlet)
    #dem.train_distance(boundary_dirichlet, dom_dis, proportion=0.05, tol=20.0)    
    shape = [cf.Nx, cf.Ny]
    dxdy = [cf.hx, cf.hy]
    iters = [100, 500, 1000, 2000, 5000]
    for iter_e in iters:
        cf.iteration = iter_e
        cf.filename_out = "./output/P%iCrack_quarter_iteration%i_points%i" % (penalty_boundary_loss, cf.iteration, len(dom))
        dem.train_model(dom, boundary_neumann, boundary_dirichlet, bound_id, [cf.Length, cf.Height, cf.Depth], cf.iteration, cf.lr, penalty_boundary_loss = penalty_boundary_loss)
        end_time = time.time() - start_time
        print("End time: %.5f" % end_time)
        z = np.zeros((datatest.shape[0], 1))
        datatest = np.concatenate((datatest, z), 1)
        #U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises, F11, F12, F21, F22 = dem.evaluate_model(datatest)
        #util.write_vtk_v2p(cf.filename_out, datatest, U, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMises)
        #surUx, surUy, surUz = U
        # 获得模型y=0，x>0的裂纹尖端的mise应力预测解
        U_2dy0_r, S11, S12, S13, S22, S23, S33, E11, E12, E13, E22, E23, E33, SVonMisesy0_r, F11, F12, F21, F22 = dem.evaluate_model_c(y0_2d)
        pred_mise_r = SVonMisesy0_r
        plt.plot(y0, pred_mise_r, label = 'deep Nitsche ritz solution_iter%i' % iter_e, lw = 2)
    plt.plot(y0, mise_abaqusy0, color = 'red', label = 'abaqus solution', lw = 2)    
    plt.xlabel('x coordinate') 
    plt.ylabel('mise')
    plt.title('mise : y=0,x>0, number points : %i' % len(dom))
    plt.legend()
    plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        