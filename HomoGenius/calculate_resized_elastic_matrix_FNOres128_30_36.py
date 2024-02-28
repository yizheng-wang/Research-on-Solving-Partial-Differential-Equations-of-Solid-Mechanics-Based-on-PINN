from joblib import Parallel, delayed
import multiprocessing # 并行化，为了后面的C
import numpy as np
import scipy.io
import scipy.sparse
import os
import math
import time
from scipy.sparse.linalg import *
from scipy.sparse import *
from scipy.sparse.linalg import splu, cg, LinearOperator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utilities3 import *
import scipy

torch.manual_seed(2024)
np.random.seed(2024)

################################################################
# fourier layer
################################################################
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(4, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 18)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x.unsqueeze(-1), grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        grid_all_train = np.linspace(-0.5, 0.5, size_x).reshape(size_x, 1).astype(np.float64)
        grids_train = []
        grids_train.append(grid_all_train)
        grids_train.append(grid_all_train)
        grids_train.append(grid_all_train)
        gridx, gridy, gridz = np.meshgrid(*grids_train)
        grid = np.stack((gridx, gridy, gridz), axis = -1)	
        grid = torch.tensor(grid.astype(np.float32)).repeat([batchsize, 1, 1, 1, 1])
        return grid.to(device)


def hexahedron(a,b,c):
    CMu = np.diag((2.0,2.0,2.0,1.0,1.0,1.0))
    CLambda = np.zeros((6,6))
    CLambda[0:3,0:3] = 1.0
    xx = np.array([-math.sqrt(0.6),0.0,math.sqrt(0.6)])
    yy = xx
    zz = xx
    ww = np.array([5.0/9.0,8.0/9.0,5.0/9.0])
    keLambda = np.zeros((24,24))
    keMu = np.zeros((24,24))
    feLambda = np.zeros((24,6))
    feMu = np.zeros((24,6))

    qx = np.zeros(8)
    qy = np.zeros(8)
    qz = np.zeros(8)

    for ii in range(len(xx)):
        for jj in range(len(yy)):
            for kk in range(len(zz)):

                x = xx[ii]
                y = yy[jj]
                z = zz[kk]

                qx[0] = -(y-1.0)*(z-1.0)/8.0
                qx[1] = (y-1.0)*(z-1.0)/8.0
                qx[2] = -(y+1.0)*(z-1.0)/8.0
                qx[3] = (y+1.0)*(z-1.0)/8.0
                qx[4] = (y-1.0)*(z+1.0)/8.0
                qx[5] = -(y-1.0)*(z+1.0)/8.0
                qx[6] = (y+1.0)*(z+1.0)/8.0
                qx[7] = -(y+1.0)*(z+1.0)/8.0

                qy[0] = -(x-1.0)*(z-1.0)/8.0
                qy[1] = (x+1.0)*(z-1.0)/8.0
                qy[2] = -(x+1.0)*(z-1.0)/8.0
                qy[3] = (x-1.0)*(z-1.0)/8.0
                qy[4] = (x-1.0)*(z+1.0)/8.0
                qy[5] = -(x+1.0)*(z+1.0)/8.0
                qy[6] = (x+1.0)*(z+1.0)/8.0
                qy[7] = -(x-1.0)*(z+1.0)/8.0

                qz[0] = -(x-1.0)*(y-1.0)/8.0
                qz[1] = (x+1.0)*(y-1.0)/8.0
                qz[2] = -(x+1.0)*(y+1.0)/8.0
                qz[3] = (x-1.0)*(y+1.0)/8.0
                qz[4] = (x-1.0)*(y-1.0)/8.0
                qz[5] = -(x+1.0)*(y-1.0)/8.0
                qz[6] = (x+1.0)*(y+1.0)/8.0
                qz[7] = -(x-1.0)*(y+1.0)/8.0

                J = np.dot(np.hstack((qx.reshape(8,1),qy.reshape(8,1),qz.reshape(8,1))).T,\
                    np.array([[-a,a,a,-a,-a,a,a,-a],[-b,-b,b,b,-b,-b,b,b],[-c,-c,-c,-c,c,c,c,c]]).T)
                qxyz = np.dot(np.linalg.inv(J),np.hstack((qx.reshape(8,1),qy.reshape(8,1),qz.reshape(8,1))).T)

                B_e = np.zeros((6,3,8))
                for i_B in range(8):
                    B_e[:,:,i_B] = [[qxyz[0,i_B],   0.0,             0.0],\
                                    [0.0,             qxyz[1,i_B],   0.0],\
                                    [0.0,             0.0,             qxyz[2,i_B]],\
                                    [qxyz[1,i_B],   qxyz[0,i_B],   0.0,],\
                                    [0.0,             qxyz[2,i_B],   qxyz[1,i_B]],\
                                    [qxyz[2,i_B],   0.0,            qxyz[0,i_B]]]
                B = np.hstack(([B_e[:,:,0],B_e[:,:,1],B_e[:,:,2],B_e[:,:,3],B_e[:,:,4],B_e[:,:,5],\
                    B_e[:,:,6],B_e[:,:,7]]))
                weight = np.linalg.det(J)*ww[ii]*ww[jj]*ww[kk]
                keLambda = keLambda+weight*np.dot(np.dot(B.T,CLambda),B)
                keMu = keMu + weight*np.dot(np.dot(B.T,CMu),B)
                feLambda = feLambda + weight*np.dot(B.T,CLambda)
                feMu = feMu + weight*np.dot(B.T,CMu)
    
    return keLambda, keMu, feLambda, feMu

def matrix_multiply(A,B,tol):
    
    loc = []
    row = A.shape[0]
    col = A.shape[1]
    temp = 0.0
    value = []
    
    for i in range(row):
        for j in range(row):
            temp = np.dot(A[i,:],B[:,j])
            if temp > tol:
                value.append(temp)
                loc.append([j,i])


def homo3D_revised(lx,ly,lz,lamda,mu,voxel):
    start = time.time()
    volume = lx*ly*lz
     # for calculating the time of solving X
    voxel_size = voxel.shape
    nelx = voxel_size[0]
    nely = voxel_size[1]
    nelz = voxel_size[2]

    dx = lx/nelx
    dy = ly/nely
    dz = lz/nelz
    nel = nelx*nely*nelz

    keLambda, keMu, feLambda, feMu = hexahedron(dx/2.0,dy/2.0,dz/2.0)

    nodenrs = np.arange(1,(nelx+1)*(nely+1)*(nely+1)+1,1)
    nodenrs = nodenrs.reshape(1+nelx,1+nely,1+nelz)
    edofVec = 3*nodenrs[0:-1,0:-1,0:-1]+1.0
    edofVec = edofVec.reshape(nel,1)
    addx = np.array([0,1,2,3*nelx+3,3*nelx+4,3*nelx+5,3*nelx,3*nelx+1,3*nelx+2,-3,-2,-1])
    addxy = 3*(nely+1)*(nelx+1)+addx
    edof = np.tile(edofVec,(1,24)) + np.tile(np.hstack((addx.reshape(1,12),addxy.reshape(1,12))),(nel,1))
    edof = edof.astype(int)
    
    nn = (nelx+1)*(nely+1)*(nelz+1)
    nnP = nelx*nely*nelz
    dnnpArray = np.arange(1,nnP+1,1)
    dnnpArray = dnnpArray.reshape(nelx,nely,nelz)
    nnpArray = np.zeros((nelx+1,nely+1,nelz+1))
    nnpArray[0:-1,0:-1,0:-1] = dnnpArray
    nnpArray[-1,:,:] = nnpArray[0,:,:]
    nnpArray[:,-1,:] = nnpArray[:,0,:]
    nnpArray[:,:,-1] = nnpArray[:,:,0]

    dofVector = np.zeros((3*nn,1))
    dofVector[0::3] = 3*nnpArray.reshape(nn,1)-2
    dofVector[1::3] = 3*nnpArray.reshape(nn,1)-1
    dofVector[2::3] = 3*nnpArray.reshape(nn,1)
    
    dofVector = dofVector.flatten()
    edof = dofVector[edof-1].astype(int)
    
    
    material = voxel/v
    lamda = lamda*material
    mu = mu*material
    
        
    voxel_flat = voxel.ravel(order='F')
    
    # 找出所有大于0的元素的索引
    active_indices = np.where(voxel_flat > 0.0)[0]
    
    # 使用这些索引从edof中获取对应的行
    activedofs = edof[active_indices, :]
    
    activedofs = np.sort(np.unique(activedofs))
    # activedofs = activedofs.astype(int) # 这个activedofs是用来掩码的
    





    X0 = np.zeros((nel,24,6))
    X0_e = np.zeros((24,6))
    ke = keMu + keLambda
    fe = feMu + feLambda
    ke1 = ke[[3,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23],:]
    ke2 = ke1[:,[3,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23]]
    fe1 = fe[[3,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23],:]
    X0_e[[3,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23],:] = np.dot(np.linalg.inv(ke2),fe1)
    X0[:,:,0] = np.kron(X0_e[:,0].T,np.ones((nel,1)))
    X0[:,:,1] = np.kron(X0_e[:,1].T,np.ones((nel,1)))
    X0[:,:,2] = np.kron(X0_e[:,2].T,np.ones((nel,1)))
    X0[:,:,3] = np.kron(X0_e[:,3].T,np.ones((nel,1)))
    X0[:,:,4] = np.kron(X0_e[:,4].T,np.ones((nel,1)))
    X0[:,:,5] = np.kron(X0_e[:,5].T,np.ones((nel,1)))


    
    # 上面有FEM计算出来的X，我们需要和FNO 比较。所以下面是FNO的计算结果

    
    start_FNO = time.time()
    input_with_norm = x_normalizer.encode(torch.tensor(voxel, dtype=torch.float32).unsqueeze(0))
    out = model(input_with_norm.cuda())
    X_FNO = y_normalizer.decode(out).detach().cpu().squeeze(0).numpy()
    end_FNO = time.time()
    X_FNO = np.transpose(X_FNO, (0, 2, 1, 3)) #  .reshape(nelx*nely*nelz*3 , 6).detach().cpu().numpy(),训练的模型输出已经是正常的了，所以不用转了，但是GT要转。但是
    # 由于后面处理一样，所以要将错就错，因此这里需要转
    print('calculation time of FNO: %f' % (end_FNO-start_FNO))
    # X_FNO变成32**3*3，6的矩阵，然后乘以一个activedof用来掩码
    X_FNO_invreshape = X_FNO.reshape(-1,6)
    X_FNO_invreshape_active = np.zeros(X_FNO_invreshape.shape)
    X_FNO_invreshape_active[activedofs[3:]-1] = X_FNO_invreshape[activedofs[3:]-1]
    
    X_FNO_active = X_FNO_invreshape_active.reshape(128,128,128,18)
    

    
    X_FNO_active = X_FNO_active.reshape(-1, 6) #大小是6291456，6
 
    edof_flat = edof.flatten() - 1  # 如果edof不是从0开始的索引，则需要调整
    X_FNO_active_selected = X_FNO_active[edof_flat]                
    dX_FNO = X_FNO_active_selected.reshape(nel, 24, 6)
 
    

    start_c = time.time()
    #改成并行化
    def cal_c(i, j):
        sum_L_FNO = np.dot((X0[:,:,i]-dX_FNO[:,:,i]),keLambda) * (X0[:,:,j]-dX_FNO[:,:,j]) # lambda算出来的E
        sum_M_FNO = np.dot((X0[:,:,i]-dX_FNO[:,:,i]),keMu) * (X0[:,:,j]-dX_FNO[:,:,j]) #mu算出来的E
        sum_L_FNO = sum_L_FNO.sum(axis=1)
        sum_L_FNO = sum_L_FNO.reshape(nelx,nely,nelz)
        sum_L_FNO = sum_L_FNO.T
        sum_M_FNO = sum_M_FNO.sum(axis=1)
        sum_M_FNO = sum_M_FNO.reshape(nelx,nely,nelz)
        sum_M_FNO = sum_M_FNO.T
        return (lamda*sum_L_FNO+mu*sum_M_FNO).sum()*1.0/volume
    c_list = []
    for i in range(6):
        for j in range(6):
            c_list.append([i,j])
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(cal_c)(*i) for i in c_list)
    CH_FNO = np.array(results).reshape(6,6)
              

    end = time.time()
    print('The time of  CH by FNO is %f' % (end-start_c))
    print('The time of homonization CH by FNO is %f' % (end-start))
    return  CH_FNO, X_FNO_active, X_FNO

if __name__ == "__main__":
    lx = 1;
    ly = 1;
    lz = 1;
    E = 1.0
    v = 0.3
    mu = E/2.0/(1.0+v)
    lamda = E*v/(1.0+v)/(1.0-2.0*v)
    # lamda = 12115.38;
    # mu = 8076.92;
    
    # input_PATH = []
    # output_PATH = []
    # data_number = 360
    # for i in range(data_number):
    #     input_PATH.append(f'../homogenization/data/npy folder_res32_data600/{i+1}.npy')   # input is 128*128*128 resolution
    #     output_PATH.append(f'../homogenization/calculation results/res32_data600/reshaped_displacement_{i+1}.npy')   # input is 128*128*128 resolution
        
    # input_data_list = [np.load(input_PATH[i]).astype(np.float32) for i in range(data_number)]
    # output_data_list = [np.load(output_PATH[i]).astype(np.float32) for i in range(data_number)]

    # input_data = np.array(input_data_list)
    # output_data = np.swapaxes(np.array(output_data_list), 2, 3).reshape(data_number, 32, 32, 32, -1)

    # absolute_mean = np.mean(np.abs(output_data), axis=(1, 2, 3))

    # absolute_percent = absolute_mean/absolute_mean.max(0) *100


    # good = (np.abs(absolute_percent  - absolute_percent.mean(0)) < absolute_percent.std(0)) # 在平均数附近一个std内的数据存下来，其他删除

    # good_data = (good.sum(1) == 18) # true表示是合格的数据，必须是18个维度都合格才行。
    
    # shuffled_indices = np.random.permutation(len(good_data))
    
    
    
    error_list = []
    error_thermal = [] # 做一个矩阵的热力图
    # lamda = 87.22
    # mu = 41.04
    start_train = 3001
    end_train = 3001 # 在训练集中用来算修正矩阵的
    
    start = 3151
    end = 3200
    root_file_path = r'./'  # windows path dir
    # root_file_path = r'/mnt/i/Spindoid/random samples dir'  # linux path dir
    beta = 0.999 #  历史的比重，这个越大，表示历史的比重就越大

    
    x_normalizer = torch.load('./x_normalizer_nu_res128_data3001_3600')
    y_normalizer = torch.load('./y_normalizer_nu_res128_data3001_3600')
    y_normalizer.cuda()
    model = torch.load('./model/TrainRes_128/data_type3/1/FNO_homo_nu')
    # 先通过训练数据的来获得一个修正系数，这个类型是150个训练数据
    P = np.zeros([6, 6])
    for i in range(start_train,end_train+1):
        start_time = time.perf_counter()
        mat_filename = r'data_homogenization/matlab_data/FEM results/res128/3001-3600/nu_%i.mat' % i
        elastic_matrix_path = os.path.join(root_file_path,mat_filename)
        matdata = np.float64(scipy.io.loadmat(elastic_matrix_path)['data_nu'])
        # scipy.io.savemat('test_matdata.mat',{'test_matdata':matdata}) 
        FNO_C,displacement, dis_FNO = homo3D_revised(lx,ly,lz,lamda,mu,matdata) # dis_FNO是和输入对应的
        
        end_time = time.perf_counter()
        print('Running time: %s Seconds'%(end_time-start_time))
        print('model %s has been completed !!!'%(i))
        GT_C_path = r'data_homogenization/matlab_data/FEM results/res128/3001-3600/CH_'+str(i)+'.mat'

        GT_C = np.float64(scipy.io.loadmat(GT_C_path)['Homoed_C'])
        
        P_e = GT_C/FNO_C # 当前的缩小比例
        
        error_C = np.linalg.norm(GT_C - FNO_C)/np.linalg.norm(GT_C)
        error_list.append(error_C)
        P = P + P_e
        print(r'FNO model for prediction of C: error %.4f' % error_C)
    
    # 得到一个总体的P，然后算平均值
    P = P/(end_train+1-start_train) # 训练数据结果用来计算修正矩阵
    P[3:, 3:] = P[3:, 3:]+0.05
    # P_c = P # 做一个expotential average   
    for i in range(start,end+1):
        # P_c = beta*P_c + (1-beta)*P
        
        start_time = time.perf_counter()
        mat_filename = r'data_homogenization/matlab_data/FEM results/res128/3001-3600/nu_%i.mat' % i
        elastic_matrix_path = os.path.join(root_file_path,mat_filename)
        matdata = np.float64(scipy.io.loadmat(elastic_matrix_path)['data_nu'])
        # scipy.io.savemat('test_matdata.mat',{'test_matdata':matdata}) 
        FNO_C,displacement, dis_FNO = homo3D_revised(lx,ly,lz,lamda,mu,matdata) # dis_FNO是和输入对应的
        
        FNO_C = FNO_C * P
        # np.fill_diagonal(FNO_C, np.diag(FNO_C) * np.diag(P_c))
        # 看一看和reshape后的GT位移场的差距
        # dis_filename = r'data_homogenization/matlab_data/FEM results/res128/601-1200/displacement_%i.mat' % i
        # dis_path = os.path.join(root_file_path,dis_filename)
        # dis_GT = np.float64(scipy.io.loadmat(dis_path)['real_X'])
        # output_data = np.swapaxes(dis_GT.reshape(128, 128, 128, -1), 1, 2) # 第二维度和第三维度换了一下才和GT对应上
        # output_data = dis_GT.reshape(128, 128, 128, -1)
        # error_d = []
        # for j in range(18):
        #     error_d.append(np.linalg.norm(output_data[..., j] - dis_FNO[..., j])/np.linalg.norm(output_data[...,j]))
        
        end_time = time.perf_counter()
        print('Running time: %s Seconds'%(end_time-start_time))
        print('model %s has been completed !!!'%(i))
        GT_C_path = r'data_homogenization/matlab_data/FEM results/res128/3001-3600/CH_'+str(i)+'.mat'

        GT_C = np.float64(scipy.io.loadmat(GT_C_path)['Homoed_C'])
        
        # P = GT_C/FNO_C # 当前的缩小比例
        
        error_C = np.linalg.norm(GT_C - FNO_C)/np.linalg.norm(GT_C)
        error_list.append(error_C)
        error_thermal.append(FNO_C/GT_C)
        print(r'FNO model for prediction of C: error %.4f' % error_C)
    np.save('./outdata/res128_data_type3/error_C_data3001_3600.npy', error_list)
    np.save('./outdata/res128_data_type3/error_thermal_data3001_3600.npy', error_thermal)

    # 生成一个6x6的随机矩阵作为示例数据
    error_thermal_array = np.array(error_thermal) # 只取前面3行和3列，还有后面3个对角线，因为其他元素都接近于0
    error_mean = np.mean(error_thermal_array, axis = 0)-np.ones([6, 6])
    data = np.zeros([6,6])
    data[:3, :3] = error_mean[:3, :3]
    data[3, 3] = error_mean[3,3]
    data[4, 4] = error_mean[4,4]
    data[5, 5] = error_mean[5,5]
    data = np.abs(data)
    # 创建热力图
    plt.figure(figsize=(8,6))
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.colorbar()  # 显示颜色条

    plt.xticks(ticks=np.arange(6), labels=np.arange(1, 7))  # 设置x轴刻度为1-6
    plt.yticks(ticks=np.arange(6), labels=np.arange(1, 7))  # 设置y轴刻度为1-6
    
    # 添加标题和坐标轴标签
    plt.title('Error of homogenization')
    plt.xlabel('Tensor of elastic constants')
    plt.savefig('./Image/TPMS_forward_res128_type3/modulus_homo_res128_data3001_3600.png', dpi=1000)
    # 显示图形
    plt.show()