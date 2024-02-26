"""
@author: Zongyi Li
直接用type3预测
"""
import os
import numpy as np
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
def FNO_plot(resolution_image, pic_name, level, label_name):
    '''
    The function is for plotting the image of the input and the output.

    Parameters
    ----------
    resolustion_image : tensor
        the input tensor is cpu, but the output tensor is GPU.

    Returns
    -------
    None.

    '''
    # x = np.linspace(0, 1, resolution_image.size()[0])
    # y = np.linspace(0, 1, resolution_image.size()[1])
    # X, Y = np.meshgrid(x, y)
    res = resolution_image.size()[0]
    x = np.linspace(-0.5, 0.5, resolution_image.size()[0])
    y = np.linspace(-0.5, 0.5, resolution_image.size()[1])
    z = np.linspace(-0.5, 0.5, resolution_image.size()[2])
    X, Y, Z= np.meshgrid(x, y, z)
    
    h11 = plt.contourf(Y[X==-0.5].reshape(res, res).T, Z[X==-0.5].reshape(res, res).T, resolution_image[X==-0.5].reshape(res, res).T, levels = level ,cmap = 'jet')
    plt.xlabel('Y')
    plt.ylabel('Z')
    ax = plt.gca()
    ax.set_aspect(1)    
    clb = plt.colorbar(h11)
    clb.ax.set_title(label_name, size = 10)
    plt.savefig(fname = pic_name+'_x_-05.png', dpi=1000)
    plt.show()

    h11 = plt.contourf(Y[(X > 0) & (X < 1/31/2+0.001)].reshape(res, res).T, Z[(X > 0) & (X < 1/31/2+0.001)].reshape(res, res).T, resolution_image[(X > 0) & (X < 1/31/2+0.001)].reshape(res, res).T, levels = level ,cmap = 'jet')
    plt.xlabel('Y')
    plt.ylabel('Z')
    ax = plt.gca()
    ax.set_aspect(1)    
    clb = plt.colorbar(h11)
    clb.ax.set_title(label_name, size = 10)
    plt.savefig(fname = pic_name+'_x_00.png', dpi=1000)
    plt.show()
    
    h11 = plt.contourf(Y[X==0.5].reshape(res, res).T, Z[X==-0.5].reshape(res, res).T, resolution_image[X==0.5].reshape(res, res).T, levels = level ,cmap = 'jet')
    plt.xlabel('Y')
    plt.ylabel('Z')
    ax = plt.gca()
    ax.set_aspect(1)    
    clb = plt.colorbar(h11)
    clb.ax.set_title(label_name, size = 10)
    plt.savefig(fname = pic_name+'_x_05.png', dpi=1000)
    plt.show()

    h11 = plt.contourf(X[Y==-0.5].reshape(res, res).T, Z[Y==-0.5].reshape(res, res).T, resolution_image[Y==-0.5].reshape(res, res).T, levels = level ,cmap = 'jet')
    plt.xlabel('X')
    plt.ylabel('Z')
    ax = plt.gca()
    ax.set_aspect(1)    
    clb = plt.colorbar(h11)
    clb.ax.set_title(label_name, size = 10)
    plt.savefig(fname = pic_name+'_y_-05.png', dpi=1000)
    plt.show()

    h11 = plt.contourf(X[(Y > 0) & (Y < 1/31/2+0.001)].reshape(res, res).T, Z[(Y > 0) & (Y < 1/31/2+0.001)].reshape(res, res).T, resolution_image[(Y > 0) & (Y < 1/31/2+0.001)].reshape(res, res).T, levels = level ,cmap = 'jet')
    plt.xlabel('X')
    plt.ylabel('Z')
    ax = plt.gca()
    ax.set_aspect(1)    
    clb = plt.colorbar(h11)
    clb.ax.set_title(label_name, size = 10)
    plt.savefig(fname = pic_name+'_y_00.png', dpi=1000)
    plt.show()
    
    h11 = plt.contourf(X[Y==0.5].reshape(res, res).T, Z[Y==0.5].reshape(res, res).T, resolution_image[Y==0.5].reshape(res, res).T, levels = level ,cmap = 'jet')
    plt.xlabel('X')
    plt.ylabel('Z')
    ax = plt.gca()
    ax.set_aspect(1)    
    clb = plt.colorbar(h11)
    clb.ax.set_title(label_name, size = 10)
    plt.savefig(fname = pic_name+'_y_05.png', dpi=1000)
    plt.show()

    h11 = plt.contourf(X[Z==-0.5].reshape(res, res), Y[Z==-0.5].reshape(res, res), resolution_image[Z==-0.5].reshape(res, res), levels = level ,cmap = 'jet')
    plt.xlabel('X')
    plt.ylabel('Y')
    ax = plt.gca()
    ax.set_aspect(1)    
    clb = plt.colorbar(h11)
    clb.ax.set_title(label_name, size = 10)
    plt.savefig(fname = pic_name+'_z_-05.png', dpi=1000)
    plt.show()

    h11 = plt.contourf(X[(Z > 0) & (Z < 1/31/2+0.001)].reshape(res, res), Y[(Z > 0) & (Z < 1/31/2+0.001)].reshape(res, res), resolution_image[(Z > 0) & (Z < 1/31/2+0.001)].reshape(res, res), levels = level ,cmap = 'jet')
    plt.xlabel('X')
    plt.ylabel('Y')
    ax = plt.gca()
    ax.set_aspect(1)    
    clb = plt.colorbar(h11)
    clb.ax.set_title(label_name, size = 10)
    plt.savefig(fname = pic_name+'_z_00.png', dpi=1000)
    plt.show()
    
    h11 = plt.contourf(X[Z==0.5].reshape(res, res), Y[Z==0.5].reshape(res, res), resolution_image[Z==0.5].reshape(res, res), levels = level ,cmap = 'jet')
    plt.xlabel('X')
    plt.ylabel('Y')
    ax = plt.gca()
    ax.set_aspect(1)    
    clb = plt.colorbar(h11)
    clb.ax.set_title(label_name, size = 10)
    plt.savefig(fname = pic_name+'_z_05.png', dpi=1000)
    plt.show()
    
    
def FNO_main(train_data_res, test_data_res, save_index):
    """
    Parameters
    ----------
    train_data_res : resolution of the training data
    save_index : index of the saving folder
    """

    
    ################################################################
    # configs
    ################################################################
    s_train = train_data_res 
    r_train = (64-1) // (s_train-1) # r is for interval
    
    s_test = test_data_res 
    r_test = (64-1) // (s_test-1) # r is for interval   
    
    # 把所有的40个并行的数据放到一起
    input_data_list = [np.float32(scipy.io.loadmat(input_PATH[i])['data_nu']) for i in range(len(data_number))]
    output_data_list = [np.float32(scipy.io.loadmat(output_PATH[i])['real_X']) for i in range(len(data_number))]  
    
    input_data = np.array(input_data_list)
    output_data = np.swapaxes(np.array(output_data_list).reshape(len(data_number), 64, 64, 64, -1), 2, 3)

    del input_data_list # 删除两个list，因为这两个list的大小太大了，而且超算后面提示说储存的空间过大，所以这里把没有用的超级大的变量给删除了
    del output_data_list 
    # 打乱，因为400个data是的体积分数是从小到大的，所以要打乱顺序
    # shuffled_indices = np.random.permutation(len(input_data))

    # 使用这个索引数组来重新组织原数组
    # input_data = input_data[shuffled_indices]
    output_data = output_data.reshape(len(data_number), test_data_res, test_data_res, test_data_res, -1)
    
    
    # input_data = input_data[good_data]
    # output_data = output_data[good_data]
    
    #output_data = output_data.reshape(len(input_data), test_data_res, test_data_res, test_data_res, -1)
    data_num = len(input_data) 
    ntrain = len(data_number) # 128的数据存的不多，所以就3个
    ntest = len(data_number)

    x_train = torch.tensor(input_data[:ntrain,::r_train,::r_train,::r_train][:,:s_train,:s_train, :s_train])
    y_train = torch.tensor(output_data[:ntrain,::r_train,::r_train,::r_train][:,:s_train,:s_train, :s_train])

    
    # y_train_norm = torch.norm(y_train.reshape(y_train.shape[0],-1), 2, 1) # 把一些没有计算结果的文件去除掉
    # x_train = x_train[y_train_norm>1.0][:900]
    # y_train = y_train[y_train_norm>1.0][:900]
    
    x_test = torch.tensor(input_data[-ntest:,::r_test,::r_test,::r_test][:,:s_test,:s_test,:s_test])
    y_test = torch.tensor(output_data[-ntest:,::r_test,::r_test,::r_test][:,:s_test,:s_test,:s_test])

    del input_data # 删除两个array，因为这两个array的大小太大了，而且超算后面提示说储存的空间过大，所以这里把没有用的超级大的变量给删除了
    del output_data 
    
    x_normalizer =  torch.load('x_normalizer_nu_res64_data_type3_material')
    y_normalizer =  torch.load('y_normalizer_nu_res64_data_type3_material')
    # y_normalizer.cuda()
    

    
    x_train = x_normalizer.encode(x_train) # encode to mean 0, and std 1
    x_test = x_normalizer.encode(x_test)
    
    y_train = y_normalizer.encode(y_train.cuda())
    
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    
    ################################################################
    # training and evaluation
    ################################################################
    
    # ====================================
    # load model
    # ====================================
    current_directory = '.'
    resolution = "TrainRes_"+str(train_data_res)
    folder_index = str(save_index)
    
    model_dir = "/model/" + resolution +"/" + 'data_type3_material/'+ folder_index +"/"
    save_models_to = current_directory + model_dir
    if not os.path.exists(save_models_to):
        os.makedirs(save_models_to)    
    
    model = torch.load(save_models_to+'FNO_homo_nu')
    print(count_params(model))
    
    start_time = default_timer()
    myloss = LpLoss(size_average=False)
    y_normalizer.cuda()
    
    
    index_num = 0
    
    nu = 0.3 #  用来归一化，因为几何使用nu=0.3做出来的
    x_test_plot = x_normalizer.decode(x_test)[index_num].detach().cpu().numpy()
    y_test_plot = y_test[index_num].detach().cpu().numpy()*x_test_plot[..., np.newaxis]
    
    out = model(x_test[index_num].unsqueeze(0).cuda()).reshape(1, s_test, s_test, s_test, 18)
    out_plot = y_normalizer.decode(out).detach().cpu().reshape(s_test, s_test, s_test, 18).numpy()*x_test_plot[..., np.newaxis]
    
    x_space = np.linspace(-0.5, 0.5, test_data_res)
    y_space = np.linspace(-0.5, 0.5, test_data_res)
    z_space = np.linspace(-0.5, 0.5, test_data_res)
    
    write_vtk_in('./Image/NTK_res64_material/dis_res64_data1-600_in', x_space, y_space, z_space, x_test_plot) #input
    
    y_test_p = np.ascontiguousarray(y_test_plot)
    out_p = np.ascontiguousarray(out_plot)
    
    write_vtk_dis('./Image/NTK_res64_material/dis_res64_data1-600_gt1', x_space, y_space, z_space, x_test_plot, np.ascontiguousarray(y_test_p[..., 0]), np.ascontiguousarray(y_test_p[..., 6]), np.ascontiguousarray(y_test_p[..., 12])) # GT load 1
    write_vtk_dis('./Image/NTK_res64_material/dis_res64_data1-600_gt2', x_space, y_space, z_space, x_test_plot, np.ascontiguousarray(y_test_p[..., 1]), np.ascontiguousarray(y_test_p[..., 7]), np.ascontiguousarray(y_test_p[..., 13])) # GT load 2
    write_vtk_dis('./Image/NTK_res64_material/dis_res64_data1-600_gt3', x_space, y_space, z_space, x_test_plot, np.ascontiguousarray(y_test_p[..., 2]), np.ascontiguousarray(y_test_p[..., 8]), np.ascontiguousarray(y_test_p[..., 14])) # GT load 3
    write_vtk_dis('./Image/NTK_res64_material/dis_res64_data1-600_gt4', x_space, y_space, z_space, x_test_plot, np.ascontiguousarray(y_test_p[..., 3]), np.ascontiguousarray(y_test_p[..., 9]), np.ascontiguousarray(y_test_p[..., 15])) # GT load 4
    write_vtk_dis('./Image/NTK_res64_material/dis_res64_data1-600_gt5', x_space, y_space, z_space, x_test_plot, np.ascontiguousarray(y_test_p[..., 4]), np.ascontiguousarray(y_test_p[..., 10]), np.ascontiguousarray(y_test_p[..., 16])) # GT load 5
    write_vtk_dis('./Image/NTK_res64_material/dis_res64_data1-600_gt6', x_space, y_space, z_space, x_test_plot, np.ascontiguousarray(y_test_p[..., 5]), np.ascontiguousarray(y_test_p[..., 11]), np.ascontiguousarray(y_test_p[..., 17])) # GT load 6
    
    write_vtk_dis('./Image/NTK_res64_material/dis_res64_data1-600_FNO1', x_space, y_space, z_space, x_test_plot, np.ascontiguousarray(out_p[..., 0]), np.ascontiguousarray(out_p[..., 6]), np.ascontiguousarray(out_p[..., 12])) # FNO load 1
    write_vtk_dis('./Image/NTK_res64_material/dis_res64_data1-600_FNO2', x_space, y_space, z_space, x_test_plot, np.ascontiguousarray(out_p[..., 1]), np.ascontiguousarray(out_p[..., 7]), np.ascontiguousarray(out_p[..., 13])) # FNO load 2
    write_vtk_dis('./Image/NTK_res64_material/dis_res64_data1-600_FNO3', x_space, y_space, z_space, x_test_plot, np.ascontiguousarray(out_p[..., 2]), np.ascontiguousarray(out_p[..., 8]), np.ascontiguousarray(out_p[..., 14])) # FNO load 3
    write_vtk_dis('./Image/NTK_res64_material/dis_res64_data1-600_FNO4', x_space, y_space, z_space, x_test_plot, np.ascontiguousarray(out_p[..., 3]), np.ascontiguousarray(out_p[..., 9]), np.ascontiguousarray(out_p[..., 15])) # FNO load 4
    write_vtk_dis('./Image/NTK_res64_material/dis_res64_data1-600_FNO5', x_space, y_space, z_space, x_test_plot, np.ascontiguousarray(out_p[..., 4]), np.ascontiguousarray(out_p[..., 10]), np.ascontiguousarray(out_p[..., 16])) # FNO load 5
    write_vtk_dis('./Image/NTK_res64_material/dis_res64_data1-600_FNO6', x_space, y_space, z_space, x_test_plot, np.ascontiguousarray(out_p[..., 5]), np.ascontiguousarray(out_p[..., 11]), np.ascontiguousarray(out_p[..., 17])) # FNO load 6    

    print('test_num: %d' % index_num)
    return 

if __name__ == "__main__":
    # the choice of the resolution can be 421, 211, 141, 106, 85, 71, 61, 43, 36, 31, 29, but our experiment is 421, 211, 106, 61, 29
    training_data_resolution = [64]
    test_data_resolution = [64]
    run_index = 0

    input_PATH = []
    output_PATH = []
    data_number = [400]
    for i in data_number:
        input_PATH.append(f'./data_homogenization/matlab_data/FEM results/res64/1-600_material/nu_{i}.mat')   # input is 128*128*128 resolution
        output_PATH.append(f'./data_homogenization/matlab_data/FEM results/res64/1-600_material/displacement_{i}.mat')   # input is 128*128*128 resolution
            

    
    batch_size = 1
    learning_rate = 0.001
    
    epochs = 300
    step_size = 100
    gamma = 0.5
    
    modes = 12
    width = 32
    
    # 找到合格的数据
 
    
    # input_data = np.array(input_data_list)
    # output_data = np.swapaxes(np.array(output_data_list), 2, 3).reshape(data_number, 32, 32, 32, -1)

    # absolute_mean = np.mean(np.abs(output_data), axis=(1, 2, 3))

    # absolute_percent = absolute_mean/absolute_mean.max(0) *100


    # good = (np.abs(absolute_percent  - absolute_percent.mean(0)) < absolute_percent.std(0)) # 在平均数附近一个std内的数据存下来，其他删除

    # good_data = (good.sum(1) == 18) # true表示是合格的数据，必须是18个维度都合格才行。
 

    # r = 5
    # h = int(((421 - 1)/r) + 1)
    # s = h
    
    ################################################################
    # load data and data normalization
    ################################################################
    # reader_train = MatReader(TRAIN_PATH)
    # reader_test = MatReader(TEST_PATH)

    train_error_dic = {}
    test_error_dic = {}
    time_dic = {}

    train_error_dis_x_1_dic = {}
    train_error_dis_y_1_dic = {}
    train_error_dis_z_1_dic = {}
    
    train_error_dis_x_2_dic = {}
    train_error_dis_y_2_dic = {}
    train_error_dis_z_2_dic = {}
    
    train_error_dis_x_3_dic = {}
    train_error_dis_y_3_dic = {}
    train_error_dis_z_3_dic = {}

    train_error_dis_x_4_dic = {}
    train_error_dis_y_4_dic = {}
    train_error_dis_z_4_dic = {}
    
    train_error_dis_x_5_dic = {}
    train_error_dis_y_5_dic = {}
    train_error_dis_z_5_dic = {}
    
    train_error_dis_x_6_dic = {}
    train_error_dis_y_6_dic = {}
    train_error_dis_z_6_dic = {}

    test_error_dis_x_1_dic = {}
    test_error_dis_y_1_dic = {}
    test_error_dis_z_1_dic = {}
    
    test_error_dis_x_2_dic = {}
    test_error_dis_y_2_dic = {}
    test_error_dis_z_2_dic = {}
    
    test_error_dis_x_3_dic = {}
    test_error_dis_y_3_dic = {}
    test_error_dis_z_3_dic = {}

    test_error_dis_x_4_dic = {}
    test_error_dis_y_4_dic = {}
    test_error_dis_z_4_dic = {}
    
    test_error_dis_x_5_dic = {}
    test_error_dis_y_5_dic = {}
    test_error_dis_z_5_dic = {}
    
    test_error_dis_x_6_dic = {}
    test_error_dis_y_6_dic = {}
    test_error_dis_z_6_dic = {}
    
    
    train_error_dic = np.load('./outdata/res64_data_type3_material/train_error_dic.npy', allow_pickle=True).item()
    test_error_dic = np.load('./outdata/res64_data_type3_material/test_error_dic.npy', allow_pickle=True).item()
    time_dic = np.load('./outdata/res64_data_type3_material/time_dic.npy', allow_pickle=True).item()
    
    train_error_dis_x_1_dic = np.load('./outdata/res64_data_type3_material/train_error_dis_x_1_dic.npy', allow_pickle=True).item()
    train_error_dis_x_2_dic =np.load('./outdata/res64_data_type3_material/train_error_dis_x_2_dic.npy', allow_pickle=True).item()
    train_error_dis_x_3_dic =np.load('./outdata/res64_data_type3_material/train_error_dis_x_3_dic.npy', allow_pickle=True).item()
    train_error_dis_x_4_dic =np.load('./outdata/res64_data_type3_material/train_error_dis_x_4_dic.npy', allow_pickle=True).item()
    train_error_dis_x_5_dic =np.load('./outdata/res64_data_type3_material/train_error_dis_x_5_dic.npy', allow_pickle=True).item()
    train_error_dis_x_6_dic =np.load('./outdata/res64_data_type3_material/train_error_dis_x_6_dic.npy', allow_pickle=True).item()
    
    train_error_dis_y_1_dic = np.load('./outdata/res64_data_type3_material/train_error_dis_y_1_dic.npy', allow_pickle=True).item()
    train_error_dis_y_2_dic = np.load('./outdata/res64_data_type3_material/train_error_dis_y_2_dic.npy', allow_pickle=True).item()
    train_error_dis_y_3_dic = np.load('./outdata/res64_data_type3_material/train_error_dis_y_3_dic.npy', allow_pickle=True).item()
    train_error_dis_y_4_dic = np.load('./outdata/res64_data_type3_material/train_error_dis_y_4_dic.npy', allow_pickle=True).item()
    train_error_dis_y_5_dic = np.load('./outdata/res64_data_type3_material/train_error_dis_y_5_dic.npy', allow_pickle=True).item()
    train_error_dis_y_6_dic = np.load('./outdata/res64_data_type3_material/train_error_dis_y_6_dic.npy', allow_pickle=True).item()
    
    train_error_dis_z_1_dic = np.load('./outdata/res64_data_type3_material/train_error_dis_z_1_dic.npy', allow_pickle=True).item()
    train_error_dis_z_2_dic = np.load('./outdata/res64_data_type3_material/train_error_dis_z_2_dic.npy', allow_pickle=True).item()
    train_error_dis_z_3_dic = np.load('./outdata/res64_data_type3_material/train_error_dis_z_3_dic.npy', allow_pickle=True).item()
    train_error_dis_z_4_dic = np.load('./outdata/res64_data_type3_material/train_error_dis_z_4_dic.npy', allow_pickle=True).item()
    train_error_dis_z_5_dic = np.load('./outdata/res64_data_type3_material/train_error_dis_z_5_dic.npy', allow_pickle=True).item()
    train_error_dis_z_6_dic = np.load('./outdata/res64_data_type3_material/train_error_dis_z_6_dic.npy', allow_pickle=True).item()
    
    test_error_dis_x_1_dic = np.load('./outdata/res64_data_type3_material/test_error_dis_x_1_dic.npy', allow_pickle=True).item()
    test_error_dis_x_2_dic = np.load('./outdata/res64_data_type3_material/test_error_dis_x_2_dic.npy', allow_pickle=True).item()
    test_error_dis_x_3_dic = np.load('./outdata/res64_data_type3_material/test_error_dis_x_3_dic.npy', allow_pickle=True).item()
    test_error_dis_x_4_dic = np.load('./outdata/res64_data_type3_material/test_error_dis_x_4_dic.npy', allow_pickle=True).item()
    test_error_dis_x_5_dic = np.load('./outdata/res64_data_type3_material/test_error_dis_x_5_dic.npy', allow_pickle=True).item()
    test_error_dis_x_6_dic = np.load('./outdata/res64_data_type3_material/test_error_dis_x_6_dic.npy', allow_pickle=True).item()
    
    test_error_dis_y_1_dic = np.load('./outdata/res64_data_type3_material/test_error_dis_y_1_dic.npy', allow_pickle=True).item()
    test_error_dis_y_2_dic = np.load('./outdata/res64_data_type3_material/test_error_dis_y_2_dic.npy', allow_pickle=True).item()
    test_error_dis_y_3_dic = np.load('./outdata/res64_data_type3_material/test_error_dis_y_3_dic.npy', allow_pickle=True).item()
    test_error_dis_y_4_dic = np.load('./outdata/res64_data_type3_material/test_error_dis_y_4_dic.npy', allow_pickle=True).item()
    test_error_dis_y_5_dic = np.load('./outdata/res64_data_type3_material/test_error_dis_y_5_dic.npy', allow_pickle=True).item()
    test_error_dis_y_6_dic = np.load('./outdata/res64_data_type3_material/test_error_dis_y_6_dic.npy', allow_pickle=True).item()
    
    test_error_dis_z_1_dic = np.load('./outdata/res64_data_type3_material/test_error_dis_z_1_dic.npy', allow_pickle=True).item()
    test_error_dis_z_2_dic = np.load('./outdata/res64_data_type3_material/test_error_dis_z_2_dic.npy', allow_pickle=True).item()
    test_error_dis_z_3_dic = np.load('./outdata/res64_data_type3_material/test_error_dis_z_3_dic.npy', allow_pickle=True).item()
    test_error_dis_z_4_dic = np.load('./outdata/res64_data_type3_material/test_error_dis_z_4_dic.npy', allow_pickle=True).item()
    test_error_dis_z_5_dic = np.load('./outdata/res64_data_type3_material/test_error_dis_z_5_dic.npy', allow_pickle=True).item()
    test_error_dis_z_6_dic = np.load('./outdata/res64_data_type3_material/test_error_dis_z_6_dic.npy', allow_pickle=True).item()
    
    for training_data_resolution_e in training_data_resolution:
        for test_data_resolution_e in test_data_resolution:
            run_index += 1
            FNO_main(training_data_resolution_e, test_data_resolution_e, run_index)
            print('FNO')

        fig = plt.figure(figsize=(7, 7))        
        epoch = np.array(range(len(train_error_dic[training_data_resolution_e][test_data_resolution_e])))+1
        plt.plot(epoch, train_error_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error %i,%i' % (training_data_resolution_e, training_data_resolution_e))
        plt.plot(epoch,train_error_dis_x_1_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis x %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        plt.plot(epoch,train_error_dis_y_1_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis y %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        plt.plot(epoch,train_error_dis_z_1_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis z %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        for test_plot_re in test_data_resolution:
            plt.plot(epoch,test_error_dic[training_data_resolution_e][test_plot_re], label = 'Test_error %i,%i' % (test_plot_re, test_plot_re))
            plt.plot(epoch,test_error_dis_x_1_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis x %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
            plt.plot(epoch,test_error_dis_y_1_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis y %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
            plt.plot(epoch,test_error_dis_z_1_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis z %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Relative Error', fontsize=14)
        plt.yscale("log")
        plt.legend()
        plt.savefig('./Image/TPMS_forward_res64_type3_material/L2_loss_train_re_dis_1_%i_%i' % (training_data_resolution_e, training_data_resolution_e), dpi=1000)
        plt.show() 
        
        fig = plt.figure(figsize=(7, 7))            
        plt.plot(train_error_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error %i,%i' % (training_data_resolution_e, training_data_resolution_e))
        plt.plot(train_error_dis_x_2_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis x %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        plt.plot(train_error_dis_y_2_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis y %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        plt.plot(train_error_dis_z_2_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis z %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        for test_plot_re in test_data_resolution:
            plt.plot(test_error_dic[training_data_resolution_e][test_plot_re], label = 'Test_error %i,%i' % (test_plot_re, test_plot_re))
            plt.plot(test_error_dis_x_2_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis x %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
            plt.plot(test_error_dis_y_2_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis y %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
            plt.plot(test_error_dis_z_2_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis z %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Relative Error', fontsize=14)
        plt.yscale("log")
        plt.legend()
        plt.savefig('./Image/TPMS_forward_res64_type3_material/L2_loss_train_re_dis_2_%i_%i' % (training_data_resolution_e, training_data_resolution_e), dpi=1000)
        plt.show() 
        
        
        fig = plt.figure(figsize=(7, 7))            
        plt.plot(train_error_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error %i,%i' % (training_data_resolution_e, training_data_resolution_e))
        plt.plot(train_error_dis_x_3_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis x %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        plt.plot(train_error_dis_y_3_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis y %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        plt.plot(train_error_dis_z_3_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis z %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        for test_plot_re in test_data_resolution:
            plt.plot(test_error_dic[training_data_resolution_e][test_plot_re], label = 'Test_error %i,%i' % (test_plot_re, test_plot_re))
            plt.plot(test_error_dis_x_3_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis x %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
            plt.plot(test_error_dis_y_3_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis y %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
            plt.plot(test_error_dis_z_3_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis z %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Relative Error', fontsize=14)
        plt.yscale("log")
        plt.legend()
        plt.savefig('./Image/TPMS_forward_res64_type3_material/L2_loss_train_re_dis_3_%i_%i' % (training_data_resolution_e, training_data_resolution_e), dpi=1000)
        plt.show() 
        
        fig = plt.figure(figsize=(7, 7))            
        plt.plot(train_error_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error %i,%i' % (training_data_resolution_e, training_data_resolution_e))
        plt.plot(train_error_dis_x_4_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis x %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        plt.plot(train_error_dis_y_4_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis y %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        plt.plot(train_error_dis_z_4_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis z %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        for test_plot_re in test_data_resolution:
            plt.plot(test_error_dic[training_data_resolution_e][test_plot_re], label = 'Test_error %i,%i' % (test_plot_re, test_plot_re))
            plt.plot(test_error_dis_x_4_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis x %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
            plt.plot(test_error_dis_y_4_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis y %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
            plt.plot(test_error_dis_z_4_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis z %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Relative Error', fontsize=14)
        plt.yscale("log")
        plt.legend()
        plt.savefig('./Image/TPMS_forward_res64_type3_material/L2_loss_train_re_dis_4_%i_%i' % (training_data_resolution_e, training_data_resolution_e), dpi=1000)
        plt.show() 
        
        fig = plt.figure(figsize=(7, 7))            
        plt.plot(train_error_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error %i,%i' % (training_data_resolution_e, training_data_resolution_e))
        plt.plot(train_error_dis_x_5_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis x %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        plt.plot(train_error_dis_y_5_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis y %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        plt.plot(train_error_dis_z_5_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis z %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        for test_plot_re in test_data_resolution:
            plt.plot(test_error_dic[training_data_resolution_e][test_plot_re], label = 'Test_error %i,%i' % (test_plot_re, test_plot_re))
            plt.plot(test_error_dis_x_5_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis x %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
            plt.plot(test_error_dis_y_5_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis y %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
            plt.plot(test_error_dis_z_5_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis z %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Relative Error', fontsize=14)
        plt.yscale("log")
        plt.legend()
        plt.savefig('./Image/TPMS_forward_res64_type3_material/L2_loss_train_re_dis_5_%i_%i' % (training_data_resolution_e, training_data_resolution_e), dpi=1000)
        plt.show() 
        
        fig = plt.figure(figsize=(7, 7))            
        plt.plot(train_error_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error %i,%i' % (training_data_resolution_e, training_data_resolution_e))
        plt.plot(train_error_dis_x_6_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis x %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        plt.plot(train_error_dis_y_6_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis y %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        plt.plot(train_error_dis_z_6_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis z %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        for test_plot_re in test_data_resolution:
            plt.plot(test_error_dic[training_data_resolution_e][test_plot_re], label = 'Test_error %i,%i' % (test_plot_re, test_plot_re))
            plt.plot(test_error_dis_x_6_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis x %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
            plt.plot(test_error_dis_y_6_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis y %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
            plt.plot(test_error_dis_z_6_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis z %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Relative Error', fontsize=14)
        plt.yscale("log")
        plt.legend()
        plt.savefig('./Image/TPMS_forward_res64_type3_material/L2_loss_train_re_dis_6_%i_%i' % (training_data_resolution_e, training_data_resolution_e), dpi=1000)
        plt.show() 
        
