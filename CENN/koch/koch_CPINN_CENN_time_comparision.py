    # PINN可能位移场的能量形式，不进行分片
import numpy as np
import torch
import matplotlib.pyplot as plt
from  torch.autograd import grad
from sklearn.neighbors import KDTree
import time
from itertools import chain
import koch 
import koch_points
from pyevtk.hl import gridToVTK
from pyevtk.hl import pointsToVTK

torch.set_default_tensor_type(torch.DoubleTensor) # 将tensor的类型变成默认的double
torch.set_default_tensor_type(torch.cuda.DoubleTensor) # 将cuda的tensor的类型变成默认的double
train_p = 0
tol_p = 0.00001
a = 1
a1 = 1/15
a2 = 1
r0 = 0.5
nepoch_inter = 125
nepoch_u1 = 20
nepoch_u2 = 20

b1 = 60
b2 = 1
b3 = 1
b4 = 30
batch_l = [100, 500, 1000, 3000, 5000, 10000]
cost_t_l = []

for batch in batch_l:
    start = time.time()
    def interface(Ni):
        '''
         生成交界面的随机点
        '''
        theta = np.random.rand(Ni)*2*np.pi
        r = r0
        x = np.cos(theta) * r0
        y = np.sin(theta) * r0
        xi = np.stack([x, y], 1)
        xi = torch.tensor(xi, requires_grad=True, device='cuda')
        return xi
    def write_arr2DVTK(filename, coordinates, values, name):
        '''
        这是一个将tensor转化为VTK数据可视化的函数，输入coordinates坐标，是一个二维的二列数据，
        value是相应的值，是一个列向量
        name是转化的名称，比如RBF距离函数，以及特解网路等等
        '''
        # displacement = np.concatenate((values[:, 0:1], values[:, 1:2], values[:, 0:1]), axis=1)
        x = np.array(coordinates[:, 0].flatten(), dtype='float32')
        y = np.array(coordinates[:, 1].flatten(), dtype='float32')
        z = np.zeros(len(coordinates), dtype='float32')
        disX = np.array(values[:, 0].flatten(), dtype='float32')
        pointsToVTK(filename, x, y, z, data={name: disX}) # 二维数据的VTK文件导出
        
    # def NTK(pred):
    #     paranum = sum(p.numel() for p in model_h.parameters())
    #     A = torch.zeros((len(pred), paranum)).cuda()
    #     for index,pred_e in enumerate(pred):
    #         grad_e = grad(pred_e, model_h.parameters(),retain_graph=True,  create_graph=True) # 获得每一个预测对参数的梯度
    #         grad_one = torch.tensor([]).cuda() # 创建一个空的tensor
    #         for i in grad_e:
    #             grad_one = torch.cat((grad_one, i.flatten())) # 将每一个预测对参数的梯度预测都放入grad_one 中
    #         A[index] = grad_one # 存入A中，A的行是样本个数，列是梯度个数
    #     K = torch.mm(A, A.T) # 矩阵的乘法组成K矩阵，这个K矩阵的获得是利用了矩阵的乘法，矩阵的乘法可以加速获得K矩阵，这里要避免使用for循环
    #     L,_ = torch.eig(K.data.cpu()) # torch不能对gpu求特征值
    #     eig = torch.norm(L, dim=1)
    #     return eig
    
    
    # for particular solution 
    class particular(torch.nn.Module):
        def __init__(self, D_in, H, D_out):
            """
            In the constructor we instantiate two nn.Linear modules and assign them as
            member variables.
            """
            super(particular, self).__init__()
            self.linear1 = torch.nn.Linear(D_in, H)
            self.linear2 = torch.nn.Linear(H, H)
            self.linear3 = torch.nn.Linear(H, H)
            self.linear4 = torch.nn.Linear(H, H)
            self.linear5 = torch.nn.Linear(H, D_out)
            
            #self.a1 = torch.nn.Parameter(torch.Tensor([0.2]))
            
    
            self.a1 = torch.Tensor([0.1]).cuda()
            self.a2 = torch.Tensor([0.1])
            self.a3 = torch.Tensor([0.1])
            self.n = 1/self.a1.data.cuda()
    
    
            torch.nn.init.normal_(self.linear1.weight, mean=0, std=np.sqrt(2/(D_in+H)))
            torch.nn.init.normal_(self.linear2.weight, mean=0, std=np.sqrt(2/(H+H)))
            torch.nn.init.normal_(self.linear3.weight, mean=0, std=np.sqrt(2/(H+H)))
            torch.nn.init.normal_(self.linear4.weight, mean=0, std=np.sqrt(2/(H+H)))
            torch.nn.init.normal_(self.linear5.weight, mean=0, std=np.sqrt(2/(H+D_out)))        
            
            torch.nn.init.normal_(self.linear1.bias, mean=0, std=1)
            torch.nn.init.normal_(self.linear2.bias, mean=0, std=1)
            torch.nn.init.normal_(self.linear3.bias, mean=0, std=1)
            torch.nn.init.normal_(self.linear4.bias, mean=0, std=1)
            torch.nn.init.normal_(self.linear5.bias, mean=0, std=1)
        def forward(self, x):
            """
            In the forward function we accept a Tensor of input data and we must return
            a Tensor of output data. We can use Modules defined in the constructor as
            well as arbitrary operators on Tensors.
            """
    
            y1 = torch.tanh(self.n*self.a1*self.linear1(x))
            y2 = torch.tanh(self.n*self.a1*self.linear2(y1))
            y3 = torch.tanh(self.n*self.a1*self.linear3(y2))
            y4 = torch.tanh(self.n*self.a1*self.linear4(y3))
            y = self.n*self.a1*self.linear5(y4)
            return y
    def write_arr2DVTK(filename, coordinates, values, name):
        '''
        这是一个将tensor转化为VTK数据可视化的函数，输入coordinates坐标，是一个二维的二列数据，
        value是相应的值，是一个列向量
        name是转化的名称，比如RBF距离函数，以及特解网路等等
        '''
        # displacement = np.concatenate((values[:, 0:1], values[:, 1:2], values[:, 0:1]), axis=1)
        x = np.array(coordinates[:, 0].flatten(), dtype='float32')
        y = np.array(coordinates[:, 1].flatten(), dtype='float32')
        z = np.zeros(len(coordinates), dtype='float32')
        disX = np.array(values[:, 0].flatten(), dtype='float32')
        pointsToVTK(filename, x, y, z, data={name: disX}) # 二维数据的VTK文件导出
        
        
    
    def pred(xy):
        '''
        
    
        Parameters
        ----------
        输入xytensor，然而输出位移，都是GPU的张量
        '''
        out = torch.zeros((len(xy), 1))
        out[torch.norm(xy, dim=1)<r0] = model_p1(xy[torch.norm(xy, dim=1)<r0]) # 内部的位移场，不需要RBF距离函数以及特解，只需要一个神经网络即可 
        out[torch.norm(xy, dim=1)>=r0] = model_p2(xy[torch.norm(xy, dim=1)>=r0])# 外部的神经网络
        return out
        
    def evaluate():
        N_test = 100
        dom_koch_n = koch_points.get_koch_points_lin(N_test) # 均匀在科赫雪花内部步点
        dom_koch_t = torch.tensor(dom_koch_n, device = 'cuda')
        
        u_pred = pred(dom_koch_t)
        u_pred = u_pred.data.cpu()
        
    
        u_exact = np.zeros((len(dom_koch_n) ,1))
        u_exact[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)<r0**2] = 1/a1*np.linalg.norm(dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)<r0**2], axis=1, keepdims=True)**4 # 现在网格里面赋值
        u_exact[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)>=r0**2] = 1/a2*np.linalg.norm(dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)>=r0**2], axis=1, keepdims=True)**4 + r0**4*(1/a1-1/a2) # 现在网格外面赋值
    
        u_exact = torch.from_numpy(u_exact)
        error = torch.abs(u_pred - u_exact) # get the error in every points
        error_t = torch.norm(error)/torch.norm(u_exact) # get the total relative L2 error
        return error_t
    
    
    model_p1 = particular(2, 20, 1).cuda()
    model_p2 = particular(2, 20, 1).cuda()
    criterion = torch.nn.MSELoss()
    optim_h1 = torch.optim.Adam(params=model_p1.parameters(), lr= 0.001) # 0.001比较好，两个神经网络
    optim_h2 = torch.optim.Adam(params=model_p2.parameters(), lr= 0.001) # 0.001比较好，两个神经网络
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optim_h1, milestones=[1000, 3000, 5000], gamma = 0.1)
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optim_h2, milestones=[1000, 3000, 5000], gamma = 0.1)
    loss_array = []
    loss1_array = [] # 内部加上边界
    loss2_array = [] # 内部加上边界
    error_array = []
    
    nepoch_u1 = int(nepoch_u1)
    nepoch_u2 = int(nepoch_u2)
    start = time.time()
    for epoch_inter in range(nepoch_inter): # 两个神经网络交错优化
        print('the process : %i' % epoch_inter) # 交错优化的步骤数
        for epoch in range(nepoch_u1): # cpinn先优化1
            if epoch ==1000:
                end = time.time()
                consume_time = end-start
                print('time is %f' % consume_time)
            if epoch%100 == 0:
                dom_koch_n = koch_points.get_koch_points(batch) # 获得n_test个koch的随机分布点
                dom_koch_n1 = dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)<r0**2]#定义内部的dom点，即是r<r0的点
                dom_koch_n2 = dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)>=r0**2]#定义内部的dom点，即是r<r0的点
                dom_koch_t1= torch.tensor(dom_koch_n1,  requires_grad=True, device = 'cuda') # 将numpy数据转化为tensor
                dom_koch_t2= torch.tensor(dom_koch_n2,  requires_grad=True, device = 'cuda') # 将numpy数据转化为tensor
                f1 = -16 * torch.norm(dom_koch_t1, dim = 1, keepdim=True).data**2 # 定义体力 
                f2 = -16 * torch.norm(dom_koch_t2, dim = 1, keepdim=True).data**2 # 定义体力
                Xi = interface(batch)
                Xb = koch.point_bound_rand(batch) # 获得边界点，用来优化网络2的外部
                Xb = torch.tensor(Xb).cuda() # 变成张量
                target_b = 1/a2*torch.norm(Xb, dim=1, keepdim=True)**4+(1/a1-1/a2)*r0**4 # 得到相应的标签   
                
            def closure():  
        
                # 构造可能位移场
                u_pred1 = pred(dom_koch_t1)
                u_pred2 = pred(dom_koch_t2)
                
                du1dxy = grad(u_pred1, dom_koch_t1, torch.ones(dom_koch_t1.size()[0], 1).cuda(), create_graph=True)[0]
                du1dx = du1dxy[:, 0].unsqueeze(1)
                du1dy = du1dxy[:, 1].unsqueeze(1)
    
                du1dxxy = grad(du1dx, dom_koch_t1, torch.ones(dom_koch_t1.size()[0], 1).cuda(), create_graph=True)[0]
                du1dxx = du1dxxy[:, 0].unsqueeze(1)
                du1dxy = du1dxxy[:, 1].unsqueeze(1)
    
                du1dyxy = grad(du1dy, dom_koch_t1, torch.ones(dom_koch_t1.size()[0], 1).cuda(), create_graph=True)[0]
                du1dyx = du1dyxy[:, 0].unsqueeze(1)
                du1dyy = du1dyxy[:, 1].unsqueeze(1)
        
                du2dxy = grad(u_pred2, dom_koch_t2, torch.ones(dom_koch_t2.size()[0], 1).cuda(), create_graph=True)[0]
                du2dx = du2dxy[:, 0].unsqueeze(1)
                du2dy = du2dxy[:, 1].unsqueeze(1)
    
                du2dxy = grad(u_pred2, dom_koch_t2, torch.ones(dom_koch_t2.size()[0], 1).cuda(), create_graph=True)[0]
                du2dx = du2dxy[:, 0].unsqueeze(1)
                du2dy = du2dxy[:, 1].unsqueeze(1)
    
                du2dxxy = grad(du2dx, dom_koch_t2, torch.ones(dom_koch_t2.size()[0], 1).cuda(), create_graph=True)[0]
                du2dxx = du2dxxy[:, 0].unsqueeze(1)
                du2dxy = du2dxxy[:, 1].unsqueeze(1)
    
                du2dyxy = grad(du2dy, dom_koch_t2, torch.ones(dom_koch_t2.size()[0], 1).cuda(), create_graph=True)[0]
                du2dyx = du2dyxy[:, 0].unsqueeze(1)
                du2dyy = du2dyxy[:, 1].unsqueeze(1)
        
                J1 =  torch.sum((a1 * (du1dxx + du1dyy)+f1)**2)/len(du1dxx)
                
                J2 =  torch.sum((a2 * (du2dxx + du2dyy)+f2)**2)/len(du2dxx)
        
    
                # 添加交界面的损失函数
                u_i1 = model_p1(Xi)  # 内部网络的交界面预测
                u_i2 = model_p2(Xi)  # 外部网络的交界面预测
                Ji = criterion(u_i1, u_i2)
                
                # 添加交界面的导数的预测
                du1dxyi = grad(u_i1, Xi, torch.ones(Xi.size()[0], 1).cuda(), create_graph=True)[0]
                du1dxi = du1dxyi[:, 0].unsqueeze(1)
                du1dyi = du1dxyi[:, 1].unsqueeze(1)
       
                du2dxyi = grad(u_i2, Xi, torch.ones(Xi.size()[0], 1).cuda(), create_graph=True)[0]
                du2dxi = du2dxyi[:, 0].unsqueeze(1)
                du2dyi = du2dxyi[:, 1].unsqueeze(1)
                
                di1 = a1 * (Xi[:, 0].unsqueeze(1)*du1dxi + Xi[:, 1].unsqueeze(1)*du1dyi)/torch.norm(Xi, dim=1, keepdim=True)
                di2 = a2 * (Xi[:, 0].unsqueeze(1)*du2dxi + Xi[:, 1].unsqueeze(1)*du2dyi)/torch.norm(Xi, dim=1, keepdim=True)
                Jdi = criterion(di1, di2)
                
                u_b = model_p2(Xb)
                Jb = criterion(u_b, target_b)
                
                loss = b1*J1 + b3*(Ji+Jdi)
                error_t = evaluate()
                optim_h1.zero_grad()
                loss.backward()
                loss1_array.append(( b1*J1 + b3*Ji).data.cpu())
                loss_array.append(loss.data.cpu())
                error_array.append(error_t.data.cpu())
        
                if epoch%10==0:
                    print(' epoch : %i, the loss : %f , loss1 : %f, loss2 : %f, inter : %f, bound : %f, error : %f' % (epoch, loss.data, J1.data, J2.data, Ji.data,  Jb.data, error_t.data))
                return loss
            optim_h1.step(closure)
            scheduler1.step()
            # 网络1损失函数下不去，所以我们用交界面来作为本质边界条件训练网络1
            
        for epoch in range(nepoch_u2): #  对第二个网络进行优化
            if epoch ==1000:
                end = time.time()
                consume_time = end-start
                print('time is %f' % consume_time)
            if epoch%100 == 0:
                dom_koch_n = koch_points.get_koch_points(batch) # 获得n_test个koch的随机分布点
                dom_koch_n1 = dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)<r0**2]#定义内部的dom点，即是r<r0的点
                dom_koch_n2 = dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)>=r0**2]#定义内部的dom点，即是r<r0的点
                dom_koch_t1= torch.tensor(dom_koch_n1,  requires_grad=True, device = 'cuda') # 将numpy数据转化为tensor
                dom_koch_t2= torch.tensor(dom_koch_n2,  requires_grad=True, device = 'cuda') # 将numpy数据转化为tensor
                f1 = -16 * torch.norm(dom_koch_t1, dim = 1, keepdim=True).data**2 # 定义体力 
                f2 = -16 * torch.norm(dom_koch_t2, dim = 1, keepdim=True).data**2 # 定义体力
                Xi = interface(batch)
                Xb = koch.point_bound_rand(batch) # 获得边界点，用来优化网络2的外部
                Xb = torch.tensor(Xb).cuda() # 变成张量
                target_b = 1/a2*torch.norm(Xb, dim=1, keepdim=True)**4+(1/a1-1/a2)*r0**4 # 得到相应的标签   
                
            def closure():  
        
                # 构造可能位移场
                u_pred1 = pred(dom_koch_t1)
                u_pred2 = pred(dom_koch_t2)
                
                du1dxy = grad(u_pred1, dom_koch_t1, torch.ones(dom_koch_t1.size()[0], 1).cuda(), create_graph=True)[0]
                du1dx = du1dxy[:, 0].unsqueeze(1)
                du1dy = du1dxy[:, 1].unsqueeze(1)
    
                du1dxxy = grad(du1dx, dom_koch_t1, torch.ones(dom_koch_t1.size()[0], 1).cuda(), create_graph=True)[0]
                du1dxx = du1dxxy[:, 0].unsqueeze(1)
                du1dxy = du1dxxy[:, 1].unsqueeze(1)
    
                du1dyxy = grad(du1dy, dom_koch_t1, torch.ones(dom_koch_t1.size()[0], 1).cuda(), create_graph=True)[0]
                du1dyx = du1dyxy[:, 0].unsqueeze(1)
                du1dyy = du1dyxy[:, 1].unsqueeze(1)
        
                du2dxy = grad(u_pred2, dom_koch_t2, torch.ones(dom_koch_t2.size()[0], 1).cuda(), create_graph=True)[0]
                du2dx = du2dxy[:, 0].unsqueeze(1)
                du2dy = du2dxy[:, 1].unsqueeze(1)
    
                du2dxy = grad(u_pred2, dom_koch_t2, torch.ones(dom_koch_t2.size()[0], 1).cuda(), create_graph=True)[0]
                du2dx = du2dxy[:, 0].unsqueeze(1)
                du2dy = du2dxy[:, 1].unsqueeze(1)
    
                du2dxxy = grad(du2dx, dom_koch_t2, torch.ones(dom_koch_t2.size()[0], 1).cuda(), create_graph=True)[0]
                du2dxx = du2dxxy[:, 0].unsqueeze(1)
                du2dxy = du2dxxy[:, 1].unsqueeze(1)
    
                du2dyxy = grad(du2dy, dom_koch_t2, torch.ones(dom_koch_t2.size()[0], 1).cuda(), create_graph=True)[0]
                du2dyx = du2dyxy[:, 0].unsqueeze(1)
                du2dyy = du2dyxy[:, 1].unsqueeze(1)
        
                J1 =  torch.sum((a1 * (du1dxx + du1dyy)+f1)**2)/len(du1dxx)
                
                J2 =  torch.sum((a2 * (du2dxx + du2dyy)+f2)**2)/len(du2dxx)
        
    
                # 添加交界面的损失函数
                u_i1 = model_p1(Xi)  # 内部网络的交界面预测
                u_i2 = model_p2(Xi)  # 外部网络的交界面预测
                Ji = criterion(u_i1, u_i2)
                
                # 添加交界面的导数的预测
                du1dxyi = grad(u_i1, Xi, torch.ones(Xi.size()[0], 1).cuda(), create_graph=True)[0]
                du1dxi = du1dxyi[:, 0].unsqueeze(1)
                du1dyi = du1dxyi[:, 1].unsqueeze(1)
       
                du2dxyi = grad(u_i2, Xi, torch.ones(Xi.size()[0], 1).cuda(), create_graph=True)[0]
                du2dxi = du2dxyi[:, 0].unsqueeze(1)
                du2dyi = du2dxyi[:, 1].unsqueeze(1)
                
                di1 = a1 * (Xi[:, 0].unsqueeze(1)*du1dxi + Xi[:, 1].unsqueeze(1)*du1dyi)/torch.norm(Xi, dim=1, keepdim=True)
                di2 = a2 * (Xi[:, 0].unsqueeze(1)*du2dxi + Xi[:, 1].unsqueeze(1)*du2dyi)/torch.norm(Xi, dim=1, keepdim=True)
                Jdi = criterion(di1, di2)
                
                u_b = model_p2(Xb)
                Jb = criterion(u_b, target_b)
                
                loss = b2*J2 + b3*(Ji+Jdi) + b4*Jb 
                error_t = evaluate()
                optim_h2.zero_grad()
                loss.backward()
                loss2_array.append(( b2*J2 + b3*Ji + b4*Jb).data.cpu())
                loss_array.append(loss.data.cpu())
                error_array.append(error_t.data.cpu())
        
                if epoch%10==0:
                    print(' epoch : %i, the loss : %f , loss1 : %f, loss2 : %f, inter : %f, bound : %f, error : %f' % (epoch, loss.data, J1.data, J2.data, Ji.data,  Jb.data, error_t.data))
                return loss
            optim_h2.step(closure)
            scheduler2.step()
    
    # %%   
    end = time.time()
    cost_t = end - start # 计算消耗时间
    cost_t_l.append(cost_t)
    n_test = 100
    dom_koch_n = koch_points.get_koch_points_lin(n_test) # 获得n_test个koch的随机分布点
    xy_test= torch.tensor(dom_koch_n,  requires_grad=True, device = 'cuda') # 将numpy数据转化为tensor
    
    
    u_pred = pred(xy_test)
    u_pred = u_pred.data.cpu()
    
    u_exact = np.zeros((len(dom_koch_n) ,1))
    u_exact[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)<r0**2] = 1/a1*np.linalg.norm(dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)<r0**2], axis=1, keepdims=True)**4 # 现在网格里面赋值
    u_exact[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)>=r0**2] = 1/a2*np.linalg.norm(dom_koch_n[(dom_koch_n[:,0]**2+dom_koch_n[:,1]**2)>=r0**2], axis=1, keepdims=True)**4 + r0**4*(1/a1-1/a2) # 现在网格外面赋值
    
    u_exact = torch.from_numpy(u_exact) # 将精确解从array变成tensor
    
    error = torch.abs(u_pred - u_exact) # get the error in every points
    error_t = torch.norm(error)/torch.norm(u_exact) # get the total relative L2 error
    
    # plot the prediction solution
    fig = plt.figure(figsize=(20, 20))
    
    plt.subplot(3, 3, 1)
    Xb =  koch.point_bound(5)
    Xf = koch_points.get_koch_points(10000)
     # 将边界和内部点画图
    plt.scatter(Xb[:, 0], Xb[:, 1], c='r', s = 0.1)
    plt.scatter(Xf[:, 0], Xf[:, 1], c='b', s = 0.1)
    
    write_arr2DVTK('./output_ntk/pred_cpinn', dom_koch_n, u_pred, 'pred_cpinn')
    
    write_arr2DVTK('./output_ntk/error_cpinn', dom_koch_n, error, 'error_cpinn')
    
    plt.subplot(3, 3, 2)
    loss1_array = np.array(loss1_array)
    loss2_array = np.array(loss2_array)
    plt.yscale('log')
    plt.plot(loss1_array)
    plt.plot(loss2_array)
    plt.legend(['NN1', 'NN2'])
    plt.xlabel('the iteration')
    plt.ylabel('loss')
    plt.title('loss evolution') 
    plt.subplot(3, 3, 3)
    error_array = np.array(error_array)
    plt.yscale('log')
    plt.plot(error_array)
    plt.xlabel('the iteration')
    plt.ylabel('error')
    plt.title('relative total error evolution') 
    
    plt.subplot(3, 3, 4) # 做x=0的位移预测
    n_test = 100
    x0 = np.zeros((n_test, 2))
    x0[:, 1] = np.linspace(-np.sqrt(3), np.sqrt(3), 100)
    exactx0 = np.zeros((n_test, 1))
    exactx0[np.linalg.norm(x0, axis=1)<r0] =  1/a1*np.linalg.norm(x0[np.linalg.norm(x0, axis=1)<r0], axis=1, keepdims=True)**4 # 不同的区域进行不同的解析解赋予
    exactx0[np.linalg.norm(x0, axis=1)>=r0] =  1/a2*np.linalg.norm(x0[np.linalg.norm(x0, axis=1)>=r0], axis=1, keepdims=True)**4 + (1/a1-1/a2)*r0**4
    x0t = torch.tensor(x0)
    predx0 = pred(x0t).data.cpu().numpy() # 预测x=0的原函数
    plt.plot(x0[:, 1], exactx0.flatten())
    plt.plot(x0[:, 1], predx0.flatten())
    plt.xlabel('y')
    plt.ylabel('u')
    plt.legend(['exact', 'pred'])
    plt.title('x=0 u') 
    
    plt.subplot(3, 3, 5) # 做y=0的位移预测
    n_test = 100
    y0 = np.zeros((n_test, 2))
    y0[:, 0] = np.linspace(-1, 1, 100)
    exacty0 = np.zeros((n_test, 1))
    exacty0[np.linalg.norm(y0, axis=1)<r0] =  1/a1*np.linalg.norm(y0[np.linalg.norm(y0, axis=1)<r0], axis=1, keepdims=True)**4 # 不同的区域进行不同的解析解赋予
    exacty0[np.linalg.norm(y0, axis=1)>=r0] =  1/a2*np.linalg.norm(y0[np.linalg.norm(y0, axis=1)>=r0], axis=1, keepdims=True)**4 + (1/a1-1/a2)*r0**4
    y0t = torch.tensor(y0)
    predy0 = pred(y0t).data.cpu().numpy() # 预测x=0的原函数
    plt.plot(y0[:, 0], exacty0.flatten())
    plt.plot(y0[:, 0], predy0.flatten())
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend(['exact', 'pred'])
    plt.title('y=0 u') 
    
    plt.subplot(3, 3, 6) # 做x=0的位移导数预测 dudy
    n_test = 100
    x0 = np.zeros((n_test, 2))
    x0[:, 1] = np.linspace(-np.sqrt(3), np.sqrt(3), 100)
    exactdx0 = np.zeros((n_test, 1))
    exactdx0[np.linalg.norm(x0, axis=1)<r0] =  4/a1*np.linalg.norm(x0[np.linalg.norm(x0, axis=1)<r0], axis=1, keepdims=True)**3 # 不同的区域进行不同的解析解赋予
    exactdx0[np.linalg.norm(x0, axis=1)>=r0] =  4/a2*np.linalg.norm(x0[np.linalg.norm(x0, axis=1)>=r0], axis=1, keepdims=True)**3
    x0t = torch.tensor(x0, requires_grad=True) # 将numpy 变成 tensor
    predx0 = pred(x0t) # 预测一下
    dudxy = grad(predx0, x0t, torch.ones(x0t.size()[0], 1).cuda(), create_graph=True)[0]
    dudx = dudxy[:, 0].unsqueeze(1).data.cpu().numpy()
    dudy = dudxy[:, 1].unsqueeze(1).data.cpu().numpy()
    dudy[x0[:, 1]<0] = -dudy[x0[:, 1]<0] # y小于0的导数添加负号
    plt.plot(x0[:, 1], exactdx0.flatten())
    plt.plot(x0[:, 1], dudy.flatten())
    plt.xlabel('y')
    plt.ylabel('u')
    plt.legend(['exact', 'pred'])
    plt.title('x=0 dudy')
    
    plt.subplot(3, 3, 7) # 做y=0的位移导数预测 dudy
    n_test = 100
    y0 = np.zeros((n_test, 2))
    y0[:, 0] = np.linspace(-1, 1, 100)
    exactdy0 = np.zeros((n_test, 1))
    exactdy0[np.linalg.norm(y0, axis=1)<r0] =  4/a1*np.linalg.norm(y0[np.linalg.norm(y0, axis=1)<r0], axis=1, keepdims=True)**3 # 不同的区域进行不同的解析解赋予
    exactdy0[np.linalg.norm(y0, axis=1)>=r0] =  4/a2*np.linalg.norm(y0[np.linalg.norm(y0, axis=1)>=r0], axis=1, keepdims=True)**3
    y0t = torch.tensor(y0, requires_grad=True)
    predy0 = pred(y0t) # 预测x=0的原函数
    dudxy = grad(predy0, y0t, torch.ones(y0t.size()[0], 1).cuda(), create_graph=True)[0]
    dudx = dudxy[:, 0].unsqueeze(1).data.cpu().numpy()
    dudy = dudxy[:, 1].unsqueeze(1).data.cpu().numpy()
    dudx[y0[:, 0]<0] = -dudx[y0[:, 0]<0] # y小于0的导数添加负号
    plt.plot(y0[:, 0], exactdy0.flatten())
    plt.plot(y0[:, 0], dudx.flatten())
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend(['exact', 'pred'])
    plt.title('y=0 dudx') 
    
    
    plt.suptitle("cpinn batch:%i" % batch)
    plt.savefig('./picture/energy_cpinn_plane.png')
    plt.show()
    print('the relative error is %f' % error_t.data)   
        
        
        