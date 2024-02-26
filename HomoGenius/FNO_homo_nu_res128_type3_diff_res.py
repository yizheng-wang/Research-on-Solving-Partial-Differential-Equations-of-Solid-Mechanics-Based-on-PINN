"""
601-1200,  1801-2400,  3001-3600 混合一起，用来找垃圾数据的
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
    r_train = (128-1) // (s_train-1) # r is for interval
    
    s_test = test_data_res 
    r_test = (128-1) // (s_test-1) # r is for interval   
    
    # 把所有的40个并行的数据放到一起
    input_data_list = [np.float32(scipy.io.loadmat(input_PATH[i])['data_nu']) for i in range(int(data_number*3))]
    output_data_list = [np.float32(scipy.io.loadmat(output_PATH[i])['real_X']) for i in range(int(data_number*3))]  
    
    input_data = np.array(input_data_list)
    output_data = np.swapaxes(np.array(output_data_list).reshape(len(input_data_list), 128, 128, 128, -1), 2, 3)

    del input_data_list # 删除两个list，因为这两个list的大小太大了，而且超算后面提示说储存的空间过大，所以这里把没有用的超级大的变量给删除了
    del output_data_list 
    # 打乱，因为1800个data是的体积分数是从小到大的，所以要打乱顺序
    shuffled_indices = np.random.permutation(len(input_data))
    np.save('./outdata/res128_data_type3_diff_res/shuffled_indices.npy', shuffled_indices)
    # 使用这个索引数组来重新组织原数组
    input_data = input_data[shuffled_indices]
    output_data = output_data[shuffled_indices].reshape(len(input_data), test_data_res, test_data_res, test_data_res, -1)
    
    # 删除特别的数据，因为发现有些数据特别大
    # rows_to_delete = [311] + list(range(4, 30))
    # input_data = np.delete(input_data, rows_to_delete,axis=0)
    # output_data = np.delete(output_data, rows_to_delete,axis=0)    
    
    # input_data = input_data[good_data]
    # output_data = output_data[good_data]
    
    #output_data = output_data.reshape(len(input_data), test_data_res, test_data_res, test_data_res, -1)

    x_train = torch.tensor(input_data[:ntrain,::r_train,::r_train,::r_train][:,:s_train,:s_train, :s_train])
    y_train = torch.tensor(output_data[:ntrain,::r_train,::r_train,::r_train][:,:s_train,:s_train, :s_train])
    # y_train_norm = torch.norm(y_train.reshape(y_train.shape[0],-1), 2, 1) # 把一些没有计算结果的文件去除掉
    # x_train = x_train[y_train_norm>1.0][:900]
    # y_train = y_train[y_train_norm>1.0][:900]
    
    x_test = torch.tensor(input_data[-ntest:,::r_test,::r_test,::r_test][:,:s_test,:s_test,:s_test])
    y_test = torch.tensor(output_data[-ntest:,::r_test,::r_test,::r_test][:,:s_test,:s_test,:s_test])

    del input_data # 删除两个array，因为这两个array的大小太大了，而且超算后面提示说储存的空间过大，所以这里把没有用的超级大的变量给删除了
    del output_data 
    
    x_normalizer = ChannelGaussianNormalizer(x_train) # create the object of the GaussianNormalizer, every location is normalized by the Gasssian distribution.
    y_normalizer = ChannelGaussianNormalizer(y_train)
    # y_normalizer.cuda()
    

    
    x_train = x_normalizer.encode(x_train) # encode to mean 0, and std 1
    x_test = x_normalizer.encode(x_test)
    
    y_train = y_normalizer.encode(y_train)
    
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    
    ################################################################
    # training and evaluation
    ################################################################
    model = FNO3d(modes, modes, modes, width).cuda()
    print(count_params(model))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    start_time = default_timer()
    myloss = LpLoss(size_average=False)
    y_normalizer.cuda()
    
    torch.save(x_normalizer, 'x_normalizer_nu_train%i_test%i_data_type3_diff_res' % (train_data_res, test_data_res))
    torch.save(y_normalizer, 'y_normalizer_nu_train%i_test%i_data_type3_diff_res' % (train_data_res, test_data_res))
    
    train_error_list = []
    test_error_list = []
    
    train_error_dis_x_1_list = []
    train_error_dis_y_1_list = []
    train_error_dis_z_1_list = []
    
    train_error_dis_x_2_list = []
    train_error_dis_y_2_list = []
    train_error_dis_z_2_list = []

    train_error_dis_x_3_list = []
    train_error_dis_y_3_list = []
    train_error_dis_z_3_list = []
    
    train_error_dis_x_4_list = []
    train_error_dis_y_4_list = []
    train_error_dis_z_4_list = []
    
    train_error_dis_x_5_list = []
    train_error_dis_y_5_list = []
    train_error_dis_z_5_list = []
    
    train_error_dis_x_6_list = []
    train_error_dis_y_6_list = []
    train_error_dis_z_6_list = []

    test_error_dis_x_1_list = []
    test_error_dis_y_1_list = []
    test_error_dis_z_1_list = []
    
    test_error_dis_x_2_list = []
    test_error_dis_y_2_list = []
    test_error_dis_z_2_list = []

    test_error_dis_x_3_list = []
    test_error_dis_y_3_list = []
    test_error_dis_z_3_list = []
    
    test_error_dis_x_4_list = []
    test_error_dis_y_4_list = []
    test_error_dis_z_4_list = []
    
    test_error_dis_x_5_list = []
    test_error_dis_y_5_list = []
    test_error_dis_z_5_list = []
    
    test_error_dis_x_6_list = []
    test_error_dis_y_6_list = []
    test_error_dis_z_6_list = []
    
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        train_l2_ux_1 = 0
        train_l2_uy_1 = 0
        train_l2_uz_1 = 0
        
        train_l2_ux_2 = 0
        train_l2_uy_2 = 0
        train_l2_uz_2 = 0
        
        train_l2_ux_3 = 0
        train_l2_uy_3 = 0
        train_l2_uz_3 = 0
        
        train_l2_ux_4 = 0
        train_l2_uy_4 = 0
        train_l2_uz_4 = 0        
        
        train_l2_ux_5 = 0
        train_l2_uy_5 = 0
        train_l2_uz_5 = 0

        train_l2_ux_6 = 0
        train_l2_uy_6 = 0
        train_l2_uz_6 = 0
        
        train_mse = 0
        

        
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
    
            optimizer.zero_grad()
            out = model(x).reshape(batch_size, s_train, s_train, s_train, 18)
            
            
            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            

            
            mse.backward()
            
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)
            loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
            # loss.backward()
            optimizer.step()
            train_mse += mse.item()
            train_l2 += loss.item()
            
            # every loss for different component,把零的地方归零，乘以输入矩阵
            train_l2_ux_1 += myloss(out[:, :, :, :, 0].view(batch_size,-1), y[:, :, :, :, 0].view(batch_size,-1)).item()
            train_l2_uy_1 += myloss(out[:, :, :, :, 1].view(batch_size,-1), y[:, :, :, :, 1].view(batch_size,-1)).item()
            train_l2_uz_1 += myloss(out[:, :, :, :, 2].view(batch_size,-1), y[:, :, :, :, 2].view(batch_size,-1)).item()
            
            train_l2_ux_2 += myloss(out[:, :, :, :, 3].view(batch_size,-1), y[:, :, :, :, 3].view(batch_size,-1)).item()
            train_l2_uy_2 += myloss(out[:, :, :, :, 4].view(batch_size,-1), y[:, :, :, :, 4].view(batch_size,-1)).item()
            train_l2_uz_2 += myloss(out[:, :, :, :, 5].view(batch_size,-1), y[:, :, :, :, 5].view(batch_size,-1)).item()
        
            train_l2_ux_3 += myloss(out[:, :, :, :, 6].view(batch_size,-1), y[:, :, :, :, 6].view(batch_size,-1)).item()
            train_l2_uy_3 += myloss(out[:, :, :, :, 7].view(batch_size,-1), y[:, :, :, :, 7].view(batch_size,-1)).item()
            train_l2_uz_3 += myloss(out[:, :, :, :, 8].view(batch_size,-1), y[:, :, :, :, 8].view(batch_size,-1)).item()

            train_l2_ux_4 += myloss(out[:, :, :, :, 9].view(batch_size,-1), y[:, :, :, :, 9].view(batch_size,-1)).item()
            train_l2_uy_4 += myloss(out[:, :, :, :, 10].view(batch_size,-1), y[:, :, :, :, 10].view(batch_size,-1)).item()
            train_l2_uz_4 += myloss(out[:, :, :, :, 11].view(batch_size,-1), y[:, :, :, :, 11].view(batch_size,-1)).item()
            
            train_l2_ux_5 += myloss(out[:, :, :, :, 12].view(batch_size,-1), y[:, :, :, :, 12].view(batch_size,-1)).item()
            train_l2_uy_5 += myloss(out[:, :, :, :, 13].view(batch_size,-1), y[:, :, :, :, 13].view(batch_size,-1)).item()
            train_l2_uz_5 += myloss(out[:, :, :, :, 14].view(batch_size,-1), y[:, :, :, :, 14].view(batch_size,-1)).item()
            
            train_l2_ux_6 += myloss(out[:, :, :, :, 15].view(batch_size,-1), y[:, :, :, :, 15].view(batch_size,-1)).item()
            train_l2_uy_6 += myloss(out[:, :, :, :, 16].view(batch_size,-1), y[:, :, :, :, 16].view(batch_size,-1)).item()
            train_l2_uz_6 += myloss(out[:, :, :, :, 17].view(batch_size,-1), y[:, :, :, :, 17].view(batch_size,-1)).item()
            
        
        scheduler.step()
    
        model.eval()
        test_l2 = 0.0
        test_l2_ux_1 = 0
        test_l2_uy_1 = 0
        test_l2_uz_1 = 0
        
        test_l2_ux_2 = 0
        test_l2_uy_2 = 0
        test_l2_uz_2 = 0
        
        test_l2_ux_3 = 0
        test_l2_uy_3 = 0
        test_l2_uz_3 = 0
        
        test_l2_ux_4 = 0
        test_l2_uy_4 = 0
        test_l2_uz_4 = 0        
        
        test_l2_ux_5 = 0
        test_l2_uy_5 = 0
        test_l2_uz_5 = 0

        test_l2_ux_6 = 0
        test_l2_uy_6 = 0
        test_l2_uz_6 = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                
                out = model(x).reshape(batch_size, s_test, s_test, s_test, 18)

                out = y_normalizer.decode(out) # 将0的地方归零，增加精度
                nu = 0.3
                x_norm = (x_normalizer.decode(x.detach().cpu())/nu).cuda()
                
                test_l2 += myloss((out*x_norm[..., np.newaxis]).view(batch_size,-1), (y*x_norm[..., np.newaxis]).view(batch_size,-1)).item()


                # every loss for different component
                test_l2_ux_1 += myloss(((out*x_norm[..., np.newaxis])*x_norm[..., np.newaxis])[:, :, :, :, 0].view(batch_size,-1), (y*x_norm[..., np.newaxis])[:, :, :, :, 0].view(batch_size,-1)).item()
                test_l2_uy_1 += myloss((out*x_norm[..., np.newaxis])[:, :, :, :, 1].view(batch_size,-1), (y*x_norm[..., np.newaxis])[:, :, :, :, 1].view(batch_size,-1)).item()
                test_l2_uz_1 += myloss((out*x_norm[..., np.newaxis])[:, :, :, :, 2].view(batch_size,-1), (y*x_norm[..., np.newaxis])[:, :, :, :, 2].view(batch_size,-1)).item()
                
                test_l2_ux_2 += myloss((out*x_norm[..., np.newaxis])[:, :, :, :, 3].view(batch_size,-1), (y*x_norm[..., np.newaxis])[:, :, :, :, 3].view(batch_size,-1)).item()
                test_l2_uy_2 += myloss((out*x_norm[..., np.newaxis])[:, :, :, :, 4].view(batch_size,-1), (y*x_norm[..., np.newaxis])[:, :, :, :, 4].view(batch_size,-1)).item()
                test_l2_uz_2 += myloss((out*x_norm[..., np.newaxis])[:, :, :, :, 5].view(batch_size,-1), (y*x_norm[..., np.newaxis])[:, :, :, :, 5].view(batch_size,-1)).item()
            
                test_l2_ux_3 += myloss((out*x_norm[..., np.newaxis])[:, :, :, :, 6].view(batch_size,-1), (y*x_norm[..., np.newaxis])[:, :, :, :, 6].view(batch_size,-1)).item()
                test_l2_uy_3 += myloss((out*x_norm[..., np.newaxis])[:, :, :, :, 7].view(batch_size,-1), (y*x_norm[..., np.newaxis])[:, :, :, :, 7].view(batch_size,-1)).item()
                test_l2_uz_3 += myloss((out*x_norm[..., np.newaxis])[:, :, :, :, 8].view(batch_size,-1), (y*x_norm[..., np.newaxis])[:, :, :, :, 8].view(batch_size,-1)).item()
    
                test_l2_ux_4 += myloss((out*x_norm[..., np.newaxis])[:, :, :, :, 9].view(batch_size,-1), (y*x_norm[..., np.newaxis])[:, :, :, :, 9].view(batch_size,-1)).item()
                test_l2_uy_4 += myloss((out*x_norm[..., np.newaxis])[:, :, :, :, 10].view(batch_size,-1), (y*x_norm[..., np.newaxis])[:, :, :, :, 10].view(batch_size,-1)).item()
                test_l2_uz_4 += myloss((out*x_norm[..., np.newaxis])[:, :, :, :, 11].view(batch_size,-1), (y*x_norm[..., np.newaxis])[:, :, :, :, 11].view(batch_size,-1)).item()
                
                test_l2_ux_5 += myloss((out*x_norm[..., np.newaxis])[:, :, :, :, 12].view(batch_size,-1), (y*x_norm[..., np.newaxis])[:, :, :, :, 12].view(batch_size,-1)).item()
                test_l2_uy_5 += myloss((out*x_norm[..., np.newaxis])[:, :, :, :, 13].view(batch_size,-1), (y*x_norm[..., np.newaxis])[:, :, :, :, 13].view(batch_size,-1)).item()
                test_l2_uz_5 += myloss((out*x_norm[..., np.newaxis])[:, :, :, :, 14].view(batch_size,-1), (y*x_norm[..., np.newaxis])[:, :, :, :, 14].view(batch_size,-1)).item()
                
                test_l2_ux_6 += myloss((out*x_norm[..., np.newaxis])[:, :, :, :, 15].view(batch_size,-1), (y*x_norm[..., np.newaxis])[:, :, :, :, 15].view(batch_size,-1)).item()
                test_l2_uy_6 += myloss((out*x_norm[..., np.newaxis])[:, :, :, :, 16].view(batch_size,-1), (y*x_norm[..., np.newaxis])[:, :, :, :, 16].view(batch_size,-1)).item()
                test_l2_uz_6 += myloss((out*x_norm[..., np.newaxis])[:, :, :, :, 17].view(batch_size,-1), (y*x_norm[..., np.newaxis])[:, :, :, :, 17].view(batch_size,-1)).item()
    
        train_mse /= len(train_loader)
        train_l2/= ntrain
        train_l2_ux_1 /= ntrain
        train_l2_uy_1 /= ntrain
        train_l2_uz_1 /= ntrain
        
        train_l2_ux_2 /= ntrain
        train_l2_uy_2 /= ntrain
        train_l2_uz_2 /= ntrain
        
        train_l2_ux_3 /= ntrain
        train_l2_uy_3 /= ntrain
        train_l2_uz_3 /= ntrain
        
        train_l2_ux_4 /= ntrain
        train_l2_uy_4 /= ntrain
        train_l2_uz_4 /= ntrain        
        
        train_l2_ux_5 /= ntrain
        train_l2_uy_5 /= ntrain
        train_l2_uz_5 /= ntrain

        train_l2_ux_6 /= ntrain
        train_l2_uy_6 /= ntrain
        train_l2_uz_6 /= ntrain
        
        
        test_l2 /= ntest
        test_l2_ux_1 /= ntest
        test_l2_uy_1 /= ntest
        test_l2_uz_1 /= ntest
        
        test_l2_ux_2 /= ntest
        test_l2_uy_2 /= ntest
        test_l2_uz_2 /= ntest
        
        test_l2_ux_3 /= ntest
        test_l2_uy_3 /= ntest
        test_l2_uz_3 /= ntest
        
        test_l2_ux_4 /= ntest
        test_l2_uy_4 /= ntest
        test_l2_uz_4 /= ntest        
        
        test_l2_ux_5 /= ntest
        test_l2_uy_5 /= ntest
        test_l2_uz_5 /= ntest

        test_l2_ux_6 /= ntest
        test_l2_uy_6 /= ntest
        test_l2_uz_6 /= ntest
    
        t2 = default_timer()
        # print(ep, t2-t1, train_l2, test_l2)
        print("Epoch: %d, time: %.3f, Train Loss: %.3e, Train l2: %.4f, Test l2: %.4f,  \n \
              Train x1: %.4f,  Train y1: %.4f,  Train z1: %.4f, Test x1: %.4f,  Test y1: %.4f,  Test z1: %.4f, \n \
              Train x2: %.4f,  Train y2: %.4f,  Train z2: %.4f, Test x2: %.4f,  Test y2: %.4f,  Test z2: %.4f, \n \
              Train x3: %.4f,  Train y2: %.3f,  Train z3: %.4f, Test x3: %.4f,  Test y3: %.4f,  Test z3: %.4f,\n \
              Train x4: %.4f,  Train y4: %.4f,  Train z4: %.4f, Test x4: %.4f,  Test y4: %.4f,  Test z4: %.4f,\n \
              Train x5: %.4f,  Train y5: %.4f,  Train z5: %.4f, Test x5: %.4f,  Test y5: %.4f,  Test z5: %.4f,\n \
              Train x6: %.4f,  Train y6: %.4f,  Train z6: %.4f, Test x6: %.4f,  Test y6: %.4f,  Test z6: %.4f," 
              % (ep, t2-t1, train_mse, train_l2, test_l2, train_l2_ux_1, train_l2_uy_1, train_l2_uz_1,  test_l2_ux_1, test_l2_uy_1, test_l2_uz_1,\
                 train_l2_ux_2, train_l2_uy_2, train_l2_uz_2,  test_l2_ux_2, test_l2_uy_2, test_l2_uz_2,\
                 train_l2_ux_3, train_l2_uy_3, train_l2_uz_3,  test_l2_ux_3, test_l2_uy_3, test_l2_uz_3,
                 train_l2_ux_4, train_l2_uy_4, train_l2_uz_4,  test_l2_ux_4, test_l2_uy_4, test_l2_uz_4,
                 train_l2_ux_5, train_l2_uy_5, train_l2_uz_5,  test_l2_ux_5, test_l2_uy_5, test_l2_uz_5,
                 train_l2_ux_6, train_l2_uy_6, train_l2_uz_6,  test_l2_ux_6, test_l2_uy_6, test_l2_uz_6) )
        train_error_list.append(train_l2)
        train_error_dis_x_1_list.append(train_l2_ux_1)
        train_error_dis_y_1_list.append(train_l2_uy_1)
        train_error_dis_z_1_list.append(train_l2_uz_1)
        train_error_dis_x_2_list.append(train_l2_ux_2)
        train_error_dis_y_2_list.append(train_l2_uy_2)
        train_error_dis_z_2_list.append(train_l2_uz_2)
        train_error_dis_x_3_list.append(train_l2_ux_3)
        train_error_dis_y_3_list.append(train_l2_uy_3)
        train_error_dis_z_3_list.append(train_l2_uz_3)
        train_error_dis_x_4_list.append(train_l2_ux_4)
        train_error_dis_y_4_list.append(train_l2_uy_4)
        train_error_dis_z_4_list.append(train_l2_uz_4)
        train_error_dis_x_5_list.append(train_l2_ux_5)
        train_error_dis_y_5_list.append(train_l2_uy_5)
        train_error_dis_z_5_list.append(train_l2_uz_5)
        train_error_dis_x_6_list.append(train_l2_ux_6)
        train_error_dis_y_6_list.append(train_l2_uy_6)
        train_error_dis_z_6_list.append(train_l2_uz_6)        
        
        test_error_list.append(test_l2)
        test_error_dis_x_1_list.append(test_l2_ux_1)
        test_error_dis_y_1_list.append(test_l2_uy_1)
        test_error_dis_z_1_list.append(test_l2_uz_1)
        test_error_dis_x_2_list.append(test_l2_ux_2)
        test_error_dis_y_2_list.append(test_l2_uy_2)
        test_error_dis_z_2_list.append(test_l2_uz_2)
        test_error_dis_x_3_list.append(test_l2_ux_3)
        test_error_dis_y_3_list.append(test_l2_uy_3)
        test_error_dis_z_3_list.append(test_l2_uz_3)
        test_error_dis_x_4_list.append(test_l2_ux_4)
        test_error_dis_y_4_list.append(test_l2_uy_4)
        test_error_dis_z_4_list.append(test_l2_uz_4)
        test_error_dis_x_5_list.append(test_l2_ux_5)
        test_error_dis_y_5_list.append(test_l2_uy_5)
        test_error_dis_z_5_list.append(test_l2_uz_5)
        test_error_dis_x_6_list.append(test_l2_ux_6)
        test_error_dis_y_6_list.append(test_l2_uy_6)
        test_error_dis_z_6_list.append(test_l2_uz_6)     
        
    elapsed = default_timer() - start_time
    print("\n=============================")
    print("Training done...")
    print('Training time: %.3f'%(elapsed))
    print("=============================\n")
    
    # ====================================
    # saving settings
    # ====================================
    current_directory = os.getcwd()
    resolution = "TrainRes_"+str(train_data_res)+'_diff_res'
    folder_index = str(save_index)
    
        
    model_dir = "/model/" + 'TestRes_128_diff_res/' + resolution +"/" + 'data_type3/' + folder_index +"/"
    save_models_to = current_directory + model_dir
    if not os.path.exists(save_models_to):
        os.makedirs(save_models_to)
    torch.save(model, save_models_to+'FNO_homo_nu')
    ################################################################
    # testing
    ################################################################
    pred = torch.zeros(y_test.shape)
    index = 0
    t1 = default_timer()
    test_l2 = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            t1 = default_timer()
            out = model(x).reshape(batch_size, s_test, s_test, s_test, 18)
            t2 = default_timer()
            out = y_normalizer.decode(out)
            
            pred[batch_size*index : batch_size*(index+1),:,:] = out
    
            # test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
            # print(index, test_l2)
            test_l2 += np.linalg.norm(out.view(1, -1).cpu().numpy() 
                                      - y.view(1, -1).cpu().numpy()) / np.linalg.norm(y.view(1, -1).cpu().numpy())
            index = index + 1
        # for plot output prediction and the ground true.
        # if test_data_res == 421:
        #     index = np.random.randint(1, 20)
        #     y_plot = y[index].cpu()
        #     out_plot = out[index].cpu()
#            FNO_plot(y_plot, './Image/FNO_forward/ground_true_T_input(%i_%i)_output(%i_%i).png' % (train_data_res, train_data_res, y_plot.size()[0], y_plot.size()[1]), 100 , 'Temperature')
#            FNO_plot(out_plot.cpu(), './Image/FNO_forward/FNO_prediction_T_input(%i_%i)_output(%i_%i).png' % (train_data_res, train_data_res, out_plot.size()[0], out_plot.size()[1]), 100 , 'Temperature')
    testing_time = t2-t1
    testing_time_every = testing_time/batch_size
    test_l2 = test_l2/index
    print("\n=============================")
    print('Testing error: %.3e'%(test_l2))
    print("=============================\n")
    
    # index_num = np.random.randint(0, ntest)
    
    # x_test_plot = x_normalizer.decode(x_test)[index_num]
    # y_test_plot = y_test[index_num]
    
    # out = model(x_test[index_num].unsqueeze(0).cuda()).reshape(1, s_test, s_test, s_test, 18)
    # out_plot = y_normalizer.decode(out).detach().cpu().reshape(s_test, s_test, s_test, 18)

    
    # FNO_plot(x_test_plot, './Image/TPMS_forward/E_%i_%i_%i' % (x_test[0].size()[0], x_test[0].size()[1], x_test[0].size()[2]), 10, 'E')
    # # GT
    # FNO_plot(y_test_plot[:, :, :, 0], './Image/TPMS_forward/disx_1_train_%i_test_%i_%i_%i_GT' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_x')
    # FNO_plot(y_test_plot[:, :, :, 1], './Image/TPMS_forward/disy_1_train_%i_test_%i_%i_%i_GT' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_y')
    # FNO_plot(y_test_plot[:, :, :, 2], './Image/TPMS_forward/disz_1_train_%i_test_%i_%i_%i_GT' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_z')
    
    # FNO_plot(y_test_plot[:, :, :, 3], './Image/TPMS_forward/disx_2_train_%i_test_%i_%i_%i_GT' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_x')
    # FNO_plot(y_test_plot[:, :, :, 4], './Image/TPMS_forward/disy_2_train_%i_test_%i_%i_%i_GT' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_y')
    # FNO_plot(y_test_plot[:, :, :, 5], './Image/TPMS_forward/disz_2_train_%i_test_%i_%i_%i_GT' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_z')

    # FNO_plot(y_test_plot[:, :, :, 6], './Image/TPMS_forward/disx_3_train_%i_test_%i_%i_%i_GT' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_x')
    # FNO_plot(y_test_plot[:, :, :, 7], './Image/TPMS_forward/disy_3_train_%i_test_%i_%i_%i_GT' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_y')
    # FNO_plot(y_test_plot[:, :, :, 8], './Image/TPMS_forward/disz_3_train_%i_test_%i_%i_%i_GT' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_z')

    # FNO_plot(y_test_plot[:, :, :, 9], './Image/TPMS_forward/disx_4_train_%i_test_%i_%i_%i_GT' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_x')
    # FNO_plot(y_test_plot[:, :, :, 10], './Image/TPMS_forward/disy_4_train_%i_test_%i_%i_%i_GT' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_y')
    # FNO_plot(y_test_plot[:, :, :, 11], './Image/TPMS_forward/disz_4_train_%i_test_%i_%i_%i_GT' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_z')

    # FNO_plot(y_test_plot[:, :, :, 12], './Image/TPMS_forward/disx_5_train_%i_test_%i_%i_%i_GT' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_x')
    # FNO_plot(y_test_plot[:, :, :, 13], './Image/TPMS_forward/disy_5_train_%i_test_%i_%i_%i_GT' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_y')
    # FNO_plot(y_test_plot[:, :, :, 14], './Image/TPMS_forward/disz_5_train_%i_test_%i_%i_%i_GT' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_z')

    # FNO_plot(y_test_plot[:, :, :, 15], './Image/TPMS_forward/disx_6_train_%i_test_%i_%i_%i_GT' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_x')
    # FNO_plot(y_test_plot[:, :, :, 16], './Image/TPMS_forward/disy_6_train_%i_test_%i_%i_%i_GT' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_y')
    # FNO_plot(y_test_plot[:, :, :, 17], './Image/TPMS_forward/disz_6_train_%i_test_%i_%i_%i_GT' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_z')    
    # # prediction
    # FNO_plot(out_plot[:, :, :, 0], './Image/TPMS_forward/disx_1_train_%i_test_%i_%i_%i_FNO' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_x')
    # FNO_plot(out_plot[:, :, :, 1], './Image/TPMS_forward/disy_1_train_%i_test_%i_%i_%i_FNO' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_y')
    # FNO_plot(out_plot[:, :, :, 2], './Image/TPMS_forward/disz_1_train_%i_test_%i_%i_%i_FNO' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_z')
    
    # FNO_plot(out_plot[:, :, :, 3], './Image/TPMS_forward/disx_2_train_%i_test_%i_%i_%i_FNO' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_x')
    # FNO_plot(out_plot[:, :, :, 4], './Image/TPMS_forward/disy_2_train_%i_test_%i_%i_%i_FNO' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_y')
    # FNO_plot(out_plot[:, :, :, 5], './Image/TPMS_forward/disz_2_train_%i_test_%i_%i_%i_FNO' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_z')

    # FNO_plot(out_plot[:, :, :, 6], './Image/TPMS_forward/disx_3_train_%i_test_%i_%i_%i_FNO' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_x')
    # FNO_plot(out_plot[:, :, :, 7], './Image/TPMS_forward/disy_3_train_%i_test_%i_%i_%i_FNO' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_y')
    # FNO_plot(out_plot[:, :, :, 8], './Image/TPMS_forward/disz_3_train_%i_test_%i_%i_%i_FNO' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_z')

    # FNO_plot(out_plot[:, :, :, 9], './Image/TPMS_forward/disx_4_train_%i_test_%i_%i_%i_FNO' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_x')
    # FNO_plot(out_plot[:, :, :, 10], './Image/TPMS_forward/disy_4_train_%i_test_%i_%i_%i_FNO' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_y')
    # FNO_plot(out_plot[:, :, :, 11], './Image/TPMS_forward/disz_4_train_%i_test_%i_%i_%i_FNO' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_z')

    # FNO_plot(out_plot[:, :, :, 12], './Image/TPMS_forward/disx_5_train_%i_test_%i_%i_%i_FNO' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_x')
    # FNO_plot(out_plot[:, :, :, 13], './Image/TPMS_forward/disy_5_train_%i_test_%i_%i_%i_FNO' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_y')
    # FNO_plot(out_plot[:, :, :, 14], './Image/TPMS_forward/disz_5_train_%i_test_%i_%i_%i_FNO' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_z')

    # FNO_plot(out_plot[:, :, :, 15], './Image/TPMS_forward/disx_6_train_%i_test_%i_%i_%i_FNO' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_x')
    # FNO_plot(out_plot[:, :, :, 16], './Image/TPMS_forward/disy_6_train_%i_test_%i_%i_%i_FNO' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_y')
    # FNO_plot(out_plot[:, :, :, 17], './Image/TPMS_forward/disz_6_train_%i_test_%i_%i_%i_FNO' % (train_data_res, y_test[0].size()[0], y_test[0].size()[1], y_test[0].size()[2]), 100, 'Dis_z')  
    
    

    # print('test_num: %d' % index_num)
    return np.array(train_error_list), np.array(test_error_list), testing_time_every, \
        np.array(train_error_dis_x_1_list), np.array(train_error_dis_y_1_list), np.array(train_error_dis_z_1_list),\
            np.array(train_error_dis_x_2_list), np.array(train_error_dis_y_2_list), np.array(train_error_dis_z_2_list),\
                np.array(train_error_dis_x_3_list), np.array(train_error_dis_y_3_list), np.array(train_error_dis_z_3_list),\
				        np.array(train_error_dis_x_4_list), np.array(train_error_dis_y_4_list), np.array(train_error_dis_z_4_list),\
							np.array(train_error_dis_x_5_list), np.array(train_error_dis_y_5_list), np.array(train_error_dis_z_5_list),\
								np.array(train_error_dis_x_6_list), np.array(train_error_dis_y_6_list), np.array(train_error_dis_z_6_list),\
        np.array(test_error_dis_x_1_list), np.array(test_error_dis_y_1_list), np.array(test_error_dis_z_1_list),\
            np.array(test_error_dis_x_2_list), np.array(test_error_dis_y_2_list), np.array(test_error_dis_z_2_list),\
                np.array(test_error_dis_x_3_list), np.array(test_error_dis_y_3_list), np.array(test_error_dis_z_3_list),\
				        np.array(test_error_dis_x_4_list), np.array(test_error_dis_y_4_list), np.array(test_error_dis_z_4_list),\
							np.array(test_error_dis_x_5_list), np.array(test_error_dis_y_5_list), np.array(test_error_dis_z_5_list),\
								np.array(test_error_dis_x_6_list), np.array(test_error_dis_y_6_list), np.array(test_error_dis_z_6_list)\

if __name__ == "__main__":
    # the choice of the resolution can be 421, 211, 141, 106, 85, 71, 61, 43, 36, 31, 29, but our experiment is 421, 211, 106, 61, 29
    training_data_resolution = [32, 64, 128]
    test_data_resolution = [128]
    run_index = 0

    input_PATH = []
    output_PATH = []
    data_number = 150
    for i in range(data_number):
        input_PATH.append(f'./data_homogenization/matlab_data/FEM results/res128/601-1200/nu_{i+601}.mat')   # input is 128*128*128 resolution
        output_PATH.append(f'./data_homogenization/matlab_data/FEM results/res128/601-1200/displacement_{i+601}.mat')   # input is 128*128*128 resolution
            
    for i in range(data_number):
        input_PATH.append(f'./data_homogenization/matlab_data/FEM results/res128/1801-2400/nu_{i+1801}.mat')   # input is 128*128*128 resolution
        output_PATH.append(f'./data_homogenization/matlab_data/FEM results/res128/1801-2400/displacement_{i+1801}.mat')   # input is 128*128*128 resolution
        
    for i in range(data_number):
        input_PATH.append(f'./data_homogenization/matlab_data/FEM results/res128/3001-3600/nu_{i+3001}.mat')   # input is 128*128*128 resolution
        output_PATH.append(f'./data_homogenization/matlab_data/FEM results/res128/3001-3600/displacement_{i+3001}.mat')   # input is 128*128*128 resolution
    
    batch_size = 2
    learning_rate = 0.001
    
    epochs = 1000
    step_size = 100
    gamma = 0.5
    
    modes = 18
    width = 32
    
    # 找到合格的数据
 
    
    # input_data = np.array(input_data_list)
    # output_data = np.swapaxes(np.array(output_data_list), 2, 3).reshape(data_number, 32, 32, 32, -1)

    # absolute_mean = np.mean(np.abs(output_data), axis=(1, 2, 3))

    # absolute_percent = absolute_mean/absolute_mean.max(0) *100


    # good = (np.abs(absolute_percent  - absolute_percent.mean(0)) < absolute_percent.std(0)) # 在平均数附近一个std内的数据存下来，其他删除

    # good_data = (good.sum(1) == 18) # true表示是合格的数据，必须是18个维度都合格才行。
 
    ntrain = 400  # first 1000 of smooth1.mat 
    ntest = 50    # first 100 of smooth1.mat 
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
    
    for training_data_resolution_e in training_data_resolution:
        for test_data_resolution_e in test_data_resolution:
            run_index += 1
            train_error, test_error, test_time, \
            train_error_dis_x_1, train_error_dis_y_1, train_error_dis_z_1, \
            train_error_dis_x_2, train_error_dis_y_2, train_error_dis_z_2, \
            train_error_dis_x_3, train_error_dis_y_3, train_error_dis_z_3, \
            train_error_dis_x_4, train_error_dis_y_4, train_error_dis_z_4, \
            train_error_dis_x_5, train_error_dis_y_5, train_error_dis_z_5, \
            train_error_dis_x_6, train_error_dis_y_6, train_error_dis_z_6, \
            test_error_dis_x_1, test_error_dis_y_1, test_error_dis_z_1, \
            test_error_dis_x_2, test_error_dis_y_2, test_error_dis_z_2, \
            test_error_dis_x_3, test_error_dis_y_3, test_error_dis_z_3, \
            test_error_dis_x_4, test_error_dis_y_4, test_error_dis_z_4, \
            test_error_dis_x_5, test_error_dis_y_5, test_error_dis_z_5, \
            test_error_dis_x_6, test_error_dis_y_6, test_error_dis_z_6    =   FNO_main(training_data_resolution_e, test_data_resolution_e, run_index)
            
            addtwodimdict(train_error_dic, training_data_resolution_e, test_data_resolution_e, train_error)
            addtwodimdict(train_error_dis_x_1_dic, training_data_resolution_e, test_data_resolution_e, train_error_dis_x_1)
            addtwodimdict(train_error_dis_y_1_dic, training_data_resolution_e, test_data_resolution_e, train_error_dis_y_1)
            addtwodimdict(train_error_dis_z_1_dic, training_data_resolution_e, test_data_resolution_e, train_error_dis_z_1)
            addtwodimdict(train_error_dis_x_2_dic, training_data_resolution_e, test_data_resolution_e, train_error_dis_x_2)
            addtwodimdict(train_error_dis_y_2_dic, training_data_resolution_e, test_data_resolution_e, train_error_dis_y_2)
            addtwodimdict(train_error_dis_z_2_dic, training_data_resolution_e, test_data_resolution_e, train_error_dis_z_2)
            addtwodimdict(train_error_dis_x_3_dic, training_data_resolution_e, test_data_resolution_e, train_error_dis_x_3)
            addtwodimdict(train_error_dis_y_3_dic, training_data_resolution_e, test_data_resolution_e, train_error_dis_y_3)
            addtwodimdict(train_error_dis_z_3_dic, training_data_resolution_e, test_data_resolution_e, train_error_dis_z_3)
            addtwodimdict(train_error_dis_x_4_dic, training_data_resolution_e, test_data_resolution_e, train_error_dis_x_4)
            addtwodimdict(train_error_dis_y_4_dic, training_data_resolution_e, test_data_resolution_e, train_error_dis_y_4)
            addtwodimdict(train_error_dis_z_4_dic, training_data_resolution_e, test_data_resolution_e, train_error_dis_z_4)
            addtwodimdict(train_error_dis_x_5_dic, training_data_resolution_e, test_data_resolution_e, train_error_dis_x_5)
            addtwodimdict(train_error_dis_y_5_dic, training_data_resolution_e, test_data_resolution_e, train_error_dis_y_5)
            addtwodimdict(train_error_dis_z_5_dic, training_data_resolution_e, test_data_resolution_e, train_error_dis_z_5)
            addtwodimdict(train_error_dis_x_6_dic, training_data_resolution_e, test_data_resolution_e, train_error_dis_x_6)
            addtwodimdict(train_error_dis_y_6_dic, training_data_resolution_e, test_data_resolution_e, train_error_dis_y_6)
            addtwodimdict(train_error_dis_z_6_dic, training_data_resolution_e, test_data_resolution_e, train_error_dis_z_6)            
            
            addtwodimdict(test_error_dic, training_data_resolution_e, test_data_resolution_e, test_error)
            addtwodimdict(test_error_dis_x_1_dic, training_data_resolution_e, test_data_resolution_e, test_error_dis_x_1)
            addtwodimdict(test_error_dis_y_1_dic, training_data_resolution_e, test_data_resolution_e, test_error_dis_y_1)
            addtwodimdict(test_error_dis_z_1_dic, training_data_resolution_e, test_data_resolution_e, test_error_dis_z_1)
            addtwodimdict(test_error_dis_x_2_dic, training_data_resolution_e, test_data_resolution_e, test_error_dis_x_2)
            addtwodimdict(test_error_dis_y_2_dic, training_data_resolution_e, test_data_resolution_e, test_error_dis_y_2)
            addtwodimdict(test_error_dis_z_2_dic, training_data_resolution_e, test_data_resolution_e, test_error_dis_z_2)
            addtwodimdict(test_error_dis_x_3_dic, training_data_resolution_e, test_data_resolution_e, test_error_dis_x_3)
            addtwodimdict(test_error_dis_y_3_dic, training_data_resolution_e, test_data_resolution_e, test_error_dis_y_3)
            addtwodimdict(test_error_dis_z_3_dic, training_data_resolution_e, test_data_resolution_e, test_error_dis_z_3)
            addtwodimdict(test_error_dis_x_4_dic, training_data_resolution_e, test_data_resolution_e, test_error_dis_x_4)
            addtwodimdict(test_error_dis_y_4_dic, training_data_resolution_e, test_data_resolution_e, test_error_dis_y_4)
            addtwodimdict(test_error_dis_z_4_dic, training_data_resolution_e, test_data_resolution_e, test_error_dis_z_4)
            addtwodimdict(test_error_dis_x_5_dic, training_data_resolution_e, test_data_resolution_e, test_error_dis_x_5)
            addtwodimdict(test_error_dis_y_5_dic, training_data_resolution_e, test_data_resolution_e, test_error_dis_y_5)
            addtwodimdict(test_error_dis_z_5_dic, training_data_resolution_e, test_data_resolution_e, test_error_dis_z_5)
            addtwodimdict(test_error_dis_x_6_dic, training_data_resolution_e, test_data_resolution_e, test_error_dis_x_6)
            addtwodimdict(test_error_dis_y_6_dic, training_data_resolution_e, test_data_resolution_e, test_error_dis_y_6)
            addtwodimdict(test_error_dis_z_6_dic, training_data_resolution_e, test_data_resolution_e, test_error_dis_z_6)     
            
            addtwodimdict(time_dic, training_data_resolution_e, test_data_resolution_e, test_time)
            
            train_error_dic[training_data_resolution_e][test_data_resolution_e] = train_error
            train_error_dis_x_1_dic[training_data_resolution_e][test_data_resolution_e] = train_error_dis_x_1
            train_error_dis_y_1_dic[training_data_resolution_e][test_data_resolution_e] = train_error_dis_y_1
            train_error_dis_z_1_dic[training_data_resolution_e][test_data_resolution_e] = train_error_dis_z_1
            train_error_dis_x_2_dic[training_data_resolution_e][test_data_resolution_e] = train_error_dis_x_2
            train_error_dis_y_2_dic[training_data_resolution_e][test_data_resolution_e] = train_error_dis_y_2
            train_error_dis_z_2_dic[training_data_resolution_e][test_data_resolution_e] = train_error_dis_z_2
            train_error_dis_x_3_dic[training_data_resolution_e][test_data_resolution_e] = train_error_dis_x_3
            train_error_dis_y_3_dic[training_data_resolution_e][test_data_resolution_e] = train_error_dis_y_3
            train_error_dis_z_3_dic[training_data_resolution_e][test_data_resolution_e] = train_error_dis_z_3
            train_error_dis_x_4_dic[training_data_resolution_e][test_data_resolution_e] = train_error_dis_x_4
            train_error_dis_y_4_dic[training_data_resolution_e][test_data_resolution_e] = train_error_dis_y_4
            train_error_dis_z_4_dic[training_data_resolution_e][test_data_resolution_e] = train_error_dis_z_4
            train_error_dis_x_5_dic[training_data_resolution_e][test_data_resolution_e] = train_error_dis_x_5
            train_error_dis_y_5_dic[training_data_resolution_e][test_data_resolution_e] = train_error_dis_y_5
            train_error_dis_z_5_dic[training_data_resolution_e][test_data_resolution_e] = train_error_dis_z_5
            train_error_dis_x_6_dic[training_data_resolution_e][test_data_resolution_e] = train_error_dis_x_6
            train_error_dis_y_6_dic[training_data_resolution_e][test_data_resolution_e] = train_error_dis_y_6
            train_error_dis_z_6_dic[training_data_resolution_e][test_data_resolution_e] = train_error_dis_z_6            
            
            test_error_dic[training_data_resolution_e][test_data_resolution_e] = test_error
            test_error_dis_x_1_dic[training_data_resolution_e][test_data_resolution_e] = test_error_dis_x_1
            test_error_dis_y_1_dic[training_data_resolution_e][test_data_resolution_e] = test_error_dis_y_1
            test_error_dis_z_1_dic[training_data_resolution_e][test_data_resolution_e] = test_error_dis_z_1
            test_error_dis_x_2_dic[training_data_resolution_e][test_data_resolution_e] = test_error_dis_x_2
            test_error_dis_y_2_dic[training_data_resolution_e][test_data_resolution_e] = test_error_dis_y_2
            test_error_dis_z_2_dic[training_data_resolution_e][test_data_resolution_e] = test_error_dis_z_2
            test_error_dis_x_3_dic[training_data_resolution_e][test_data_resolution_e] = test_error_dis_x_3
            test_error_dis_y_3_dic[training_data_resolution_e][test_data_resolution_e] = test_error_dis_y_3
            test_error_dis_z_3_dic[training_data_resolution_e][test_data_resolution_e] = test_error_dis_z_3
            test_error_dis_x_4_dic[training_data_resolution_e][test_data_resolution_e] = test_error_dis_x_4
            test_error_dis_y_4_dic[training_data_resolution_e][test_data_resolution_e] = test_error_dis_y_4
            test_error_dis_z_4_dic[training_data_resolution_e][test_data_resolution_e] = test_error_dis_z_4
            test_error_dis_x_5_dic[training_data_resolution_e][test_data_resolution_e] = test_error_dis_x_5
            test_error_dis_y_5_dic[training_data_resolution_e][test_data_resolution_e] = test_error_dis_y_5
            test_error_dis_z_5_dic[training_data_resolution_e][test_data_resolution_e] = test_error_dis_z_5
            test_error_dis_x_6_dic[training_data_resolution_e][test_data_resolution_e] = test_error_dis_x_6
            test_error_dis_y_6_dic[training_data_resolution_e][test_data_resolution_e] = test_error_dis_y_6
            test_error_dis_z_6_dic[training_data_resolution_e][test_data_resolution_e] = test_error_dis_z_6                    
            
            time_dic[training_data_resolution_e][test_data_resolution_e] = test_time
        # fig = plt.figure(figsize=(7, 7))            
        # plt.plot(train_error_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error %i,%i' % (training_data_resolution_e, training_data_resolution_e))
        # plt.plot(train_error_dis_x_1_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis x %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        # plt.plot(train_error_dis_y_1_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis y %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        # plt.plot(train_error_dis_z_1_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis z %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        # for test_plot_re in test_data_resolution:
        #     plt.plot(test_error_dic[training_data_resolution_e][test_plot_re], label = 'Test_error %i,%i' % (test_plot_re, test_plot_re))
        #     plt.plot(test_error_dis_x_1_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis x %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
        #     plt.plot(test_error_dis_y_1_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis y %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
        #     plt.plot(test_error_dis_z_1_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis z %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
        # plt.xlabel('Epoch')
        # plt.ylabel('Relative Error')
        # plt.yscale("log")
        # plt.legend()
        # plt.savefig('./Image/TPMS_forward/L2_loss_train_re_dis_1_%i_%i' % (training_data_resolution_e, training_data_resolution_e), dpi=1000)
        # plt.show() 
        
        # fig = plt.figure(figsize=(7, 7))            
        # plt.plot(train_error_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error %i,%i' % (training_data_resolution_e, training_data_resolution_e))
        # plt.plot(train_error_dis_x_2_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis x %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        # plt.plot(train_error_dis_y_2_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis y %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        # plt.plot(train_error_dis_z_2_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis z %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        # for test_plot_re in test_data_resolution:
        #     plt.plot(test_error_dic[training_data_resolution_e][test_plot_re], label = 'Test_error %i,%i' % (test_plot_re, test_plot_re))
        #     plt.plot(test_error_dis_x_2_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis x %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
        #     plt.plot(test_error_dis_y_2_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis y %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
        #     plt.plot(test_error_dis_z_2_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis z %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
        # plt.xlabel('Epoch')
        # plt.ylabel('Relative Error')
        # plt.yscale("log")
        # plt.legend()
        # plt.savefig('./Image/TPMS_forward/L2_loss_train_re_dis_2_%i_%i' % (training_data_resolution_e, training_data_resolution_e), dpi=1000)
        # plt.show() 
        
        
        # fig = plt.figure(figsize=(7, 7))            
        # plt.plot(train_error_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error %i,%i' % (training_data_resolution_e, training_data_resolution_e))
        # plt.plot(train_error_dis_x_3_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis x %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        # plt.plot(train_error_dis_y_3_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis y %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        # plt.plot(train_error_dis_z_3_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis z %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        # for test_plot_re in test_data_resolution:
        #     plt.plot(test_error_dic[training_data_resolution_e][test_plot_re], label = 'Test_error %i,%i' % (test_plot_re, test_plot_re))
        #     plt.plot(test_error_dis_x_3_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis x %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
        #     plt.plot(test_error_dis_y_3_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis y %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
        #     plt.plot(test_error_dis_z_3_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis z %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
        # plt.xlabel('Epoch')
        # plt.ylabel('Relative Error')
        # plt.yscale("log")
        # plt.legend()
        # plt.savefig('./Image/TPMS_forward/L2_loss_train_re_dis_3_%i_%i' % (training_data_resolution_e, training_data_resolution_e), dpi=1000)
        # plt.show() 
        
        # fig = plt.figure(figsize=(7, 7))            
        # plt.plot(train_error_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error %i,%i' % (training_data_resolution_e, training_data_resolution_e))
        # plt.plot(train_error_dis_x_4_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis x %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        # plt.plot(train_error_dis_y_4_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis y %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        # plt.plot(train_error_dis_z_4_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis z %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        # for test_plot_re in test_data_resolution:
        #     plt.plot(test_error_dic[training_data_resolution_e][test_plot_re], label = 'Test_error %i,%i' % (test_plot_re, test_plot_re))
        #     plt.plot(test_error_dis_x_4_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis x %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
        #     plt.plot(test_error_dis_y_4_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis y %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
        #     plt.plot(test_error_dis_z_4_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis z %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
        # plt.xlabel('Epoch')
        # plt.ylabel('Relative Error')
        # plt.yscale("log")
        # plt.legend()
        # plt.savefig('./Image/TPMS_forward/L2_loss_train_re_dis_4_%i_%i' % (training_data_resolution_e, training_data_resolution_e), dpi=1000)
        # plt.show() 
        
        # fig = plt.figure(figsize=(7, 7))            
        # plt.plot(train_error_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error %i,%i' % (training_data_resolution_e, training_data_resolution_e))
        # plt.plot(train_error_dis_x_5_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis x %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        # plt.plot(train_error_dis_y_5_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis y %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        # plt.plot(train_error_dis_z_5_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis z %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        # for test_plot_re in test_data_resolution:
        #     plt.plot(test_error_dic[training_data_resolution_e][test_plot_re], label = 'Test_error %i,%i' % (test_plot_re, test_plot_re))
        #     plt.plot(test_error_dis_x_5_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis x %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
        #     plt.plot(test_error_dis_y_5_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis y %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
        #     plt.plot(test_error_dis_z_5_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis z %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
        # plt.xlabel('Epoch')
        # plt.ylabel('Relative Error')
        # plt.yscale("log")
        # plt.legend()
        # plt.savefig('./Image/TPMS_forward/L2_loss_train_re_dis_5_%i_%i' % (training_data_resolution_e, training_data_resolution_e), dpi=1000)
        # plt.show() 
        
        # fig = plt.figure(figsize=(7, 7))            
        # plt.plot(train_error_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error %i,%i' % (training_data_resolution_e, training_data_resolution_e))
        # plt.plot(train_error_dis_x_6_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis x %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        # plt.plot(train_error_dis_y_6_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis y %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        # plt.plot(train_error_dis_z_6_dic[training_data_resolution_e][test_data_resolution_e], label = 'Train_error dis z %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle=':')
        # for test_plot_re in test_data_resolution:
        #     plt.plot(test_error_dic[training_data_resolution_e][test_plot_re], label = 'Test_error %i,%i' % (test_plot_re, test_plot_re))
        #     plt.plot(test_error_dis_x_6_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis x %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
        #     plt.plot(test_error_dis_y_6_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis y %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
        #     plt.plot(test_error_dis_z_6_dic[training_data_resolution_e][test_data_resolution_e], label = 'Test_error dis z %i,%i' % (training_data_resolution_e, training_data_resolution_e), linestyle='-.')
        # plt.xlabel('Epoch')
        # plt.ylabel('Relative Error')
        # plt.yscale("log")
        # plt.legend()
        # plt.savefig('./Image/TPMS_forward/L2_loss_train_re_dis_6_%i_%i' % (training_data_resolution_e, training_data_resolution_e), dpi=1000)
        # plt.show() 
        
    np.save('./outdata/res128_data_type3_diff_res/train_error_dic.npy', train_error_dic)
    np.save('./outdata/res128_data_type3_diff_res/test_error_dic.npy', test_error_dic)
    np.save('./outdata/res128_data_type3_diff_res/time_dic.npy', time_dic)
    
    np.save('./outdata/res128_data_type3_diff_res/train_error_dis_x_1_dic.npy', train_error_dis_x_1_dic)
    np.save('./outdata/res128_data_type3_diff_res/train_error_dis_x_2_dic.npy', train_error_dis_x_2_dic)
    np.save('./outdata/res128_data_type3_diff_res/train_error_dis_x_3_dic.npy', train_error_dis_x_3_dic)
    np.save('./outdata/res128_data_type3_diff_res/train_error_dis_x_4_dic.npy', train_error_dis_x_4_dic)
    np.save('./outdata/res128_data_type3_diff_res/train_error_dis_x_5_dic.npy', train_error_dis_x_5_dic)
    np.save('./outdata/res128_data_type3_diff_res/train_error_dis_x_6_dic.npy', train_error_dis_x_6_dic)
    
    np.save('./outdata/res128_data_type3_diff_res/train_error_dis_y_1_dic.npy', train_error_dis_y_1_dic)
    np.save('./outdata/res128_data_type3_diff_res/train_error_dis_y_2_dic.npy', train_error_dis_y_2_dic)
    np.save('./outdata/res128_data_type3_diff_res/train_error_dis_y_3_dic.npy', train_error_dis_y_3_dic)
    np.save('./outdata/res128_data_type3_diff_res/train_error_dis_y_4_dic.npy', train_error_dis_y_4_dic)
    np.save('./outdata/res128_data_type3_diff_res/train_error_dis_y_5_dic.npy', train_error_dis_y_5_dic)
    np.save('./outdata/res128_data_type3_diff_res/train_error_dis_y_6_dic.npy', train_error_dis_y_6_dic)
    
    np.save('./outdata/res128_data_type3_diff_res/train_error_dis_z_1_dic.npy', train_error_dis_z_1_dic)
    np.save('./outdata/res128_data_type3_diff_res/train_error_dis_z_2_dic.npy', train_error_dis_z_2_dic)
    np.save('./outdata/res128_data_type3_diff_res/train_error_dis_z_3_dic.npy', train_error_dis_z_3_dic)
    np.save('./outdata/res128_data_type3_diff_res/train_error_dis_z_4_dic.npy', train_error_dis_z_4_dic)
    np.save('./outdata/res128_data_type3_diff_res/train_error_dis_z_5_dic.npy', train_error_dis_z_5_dic)
    np.save('./outdata/res128_data_type3_diff_res/train_error_dis_z_6_dic.npy', train_error_dis_z_6_dic)
    
    np.save('./outdata/res128_data_type3_diff_res/test_error_dis_x_1_dic.npy', test_error_dis_x_1_dic)
    np.save('./outdata/res128_data_type3_diff_res/test_error_dis_x_2_dic.npy', test_error_dis_x_2_dic)
    np.save('./outdata/res128_data_type3_diff_res/test_error_dis_x_3_dic.npy', test_error_dis_x_3_dic)
    np.save('./outdata/res128_data_type3_diff_res/test_error_dis_x_4_dic.npy', test_error_dis_x_4_dic)
    np.save('./outdata/res128_data_type3_diff_res/test_error_dis_x_5_dic.npy', test_error_dis_x_5_dic)
    np.save('./outdata/res128_data_type3_diff_res/test_error_dis_x_6_dic.npy', test_error_dis_x_6_dic)
    
    np.save('./outdata/res128_data_type3_diff_res/test_error_dis_y_1_dic.npy', test_error_dis_y_1_dic)
    np.save('./outdata/res128_data_type3_diff_res/test_error_dis_y_2_dic.npy', test_error_dis_y_2_dic)
    np.save('./outdata/res128_data_type3_diff_res/test_error_dis_y_3_dic.npy', test_error_dis_y_3_dic)
    np.save('./outdata/res128_data_type3_diff_res/test_error_dis_y_4_dic.npy', test_error_dis_y_4_dic)
    np.save('./outdata/res128_data_type3_diff_res/test_error_dis_y_5_dic.npy', test_error_dis_y_5_dic)
    np.save('./outdata/res128_data_type3_diff_res/test_error_dis_y_6_dic.npy', test_error_dis_y_6_dic)
    
    np.save('./outdata/res128_data_type3_diff_res/test_error_dis_z_1_dic.npy', test_error_dis_z_1_dic)
    np.save('./outdata/res128_data_type3_diff_res/test_error_dis_z_2_dic.npy', test_error_dis_z_2_dic)
    np.save('./outdata/res128_data_type3_diff_res/test_error_dis_z_3_dic.npy', test_error_dis_z_3_dic)
    np.save('./outdata/res128_data_type3_diff_res/test_error_dis_z_4_dic.npy', test_error_dis_z_4_dic)
    np.save('./outdata/res128_data_type3_diff_res/test_error_dis_z_5_dic.npy', test_error_dis_z_5_dic)
    np.save('./outdata/res128_data_type3_diff_res/test_error_dis_z_6_dic.npy', test_error_dis_z_6_dic)