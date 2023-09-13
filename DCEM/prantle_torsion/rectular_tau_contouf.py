


import numpy as np
import torch
import matplotlib.pyplot as plt
from  torch.autograd import grad
import time
import matplotlib as mpl
torch.set_default_tensor_type(torch.DoubleTensor) # 将tensor的类型变成默认的double
torch.set_default_tensor_type(torch.cuda.DoubleTensor) # 将cuda的tensor的类型变成默认的double
mpl.rcParams['figure.dpi'] = 1000

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
setup_seed(1994)
a_list = [1.0, 1.2, 1.5,  2.0,  2.5, 3.0, 4.0, 5.0, 10.0] # 矩形板的长
b = 1 # 矩形板的宽
beta_list =  [0.141, 0.166, 0.196, 0.229, 0.249, 0.263, 0.281, 0.291, 0.312]
beta1_list = [0.208, 0.219, 0.231, 0.246, 0.258, 0.267, 0.282, 0.291, 0.312]
nepoch = 2000
nepoch1 = 0
num = 1 # 分析num个a的长度
N_test = 101
G = 1000
M = 10
dom_num = 10000
# =============================================================================
# 最小余能原理
# =============================================================================

alpha = 0.0005 # a=b=1是0.0709标准值


loss_array_mincomplement = {}
error_Dt_array_mincomplement = {}
pred_alpha_array_mincomplement = {}
loss_dom_array_mincomplement = {}
loss_ex_array_mincomplement = {}
pred_taumax_array_mincomplement = {}
for i in range(num):

    
    a = a_list[i]
    beta = beta_list[i]
    beta1 = beta1_list[i]
    def dom_data(Nf):
        '''
        生成内部点
        '''
        
        x = a*np.random.rand(Nf)-a/2
        y = b*np.random.rand(Nf)-b/2 # 角度0到pi
        xy_dom = np.stack([x, y], 1)
        xy_dom = torch.tensor(xy_dom,  requires_grad=True, device='cuda')
        
        return xy_dom
    
    def dom_data_uniform(Nf):
        '''
        generate the uniform points
        '''
        
        x = np.linspace(-a/2, a/2, Nf)
        y = np.linspace(-b/2, b/2, Nf) # y方向N_test个点
        x_mesh, y_mesh = np.meshgrid(x, y)
        xy = np.stack((x_mesh.flatten(), y_mesh.flatten()), 1)
        xy_tensor = torch.tensor(xy,  requires_grad=True, device='cuda')
        
        return xy_tensor
    

    class FNN(torch.nn.Module):
        def __init__(self, D_in, H, D_out):
            """
            In the constructor we instantiate two nn.Linear modules and assign them as
            member variables.
            """
            super(FNN, self).__init__()
            self.linear1 = torch.nn.Linear(D_in, H)
            self.linear2 = torch.nn.Linear(H, H)
            self.linear3 = torch.nn.Linear(H, H)
            self.linear4 = torch.nn.Linear(H, D_out)
            
            self.a1 = torch.nn.Parameter(torch.Tensor([0.2]))
            
            # 有可能用到加速收敛技术
            self.a2 = torch.Tensor([0.1])
            self.a3 = torch.Tensor([0.1])
            self.n = 1/self.a1.data.cuda()
    
    
            torch.nn.init.normal_(self.linear1.weight, mean=0, std=np.sqrt(2/(D_in+H)))
            torch.nn.init.normal_(self.linear2.weight, mean=0, std=np.sqrt(2/(H+H)))
            torch.nn.init.normal_(self.linear3.weight, mean=0, std=np.sqrt(2/(H+H)))
            torch.nn.init.normal_(self.linear4.weight, mean=0, std=np.sqrt(2/(H+D_out)))
            
            
            torch.nn.init.normal_(self.linear1.bias, mean=0, std=1)
            torch.nn.init.normal_(self.linear2.bias, mean=0, std=1)
            torch.nn.init.normal_(self.linear3.bias, mean=0, std=1)
            torch.nn.init.normal_(self.linear4.bias, mean=0, std=1)
    
        def forward(self, x):
            """
            In the forward function we accept a Tensor of input data and we must return
            a Tensor of output data. We can use Modules defined in the constructor as
            well as arbitrary operators on Tensors.
            """
    
            y1 = torch.tanh(self.n*self.a1*self.linear1(x))
            y2 = torch.tanh(self.n*self.a1*self.linear2(y1))
            y3 = torch.tanh(self.n*self.a1*self.linear3(y2))
            y = self.n*self.a1*self.linear4(y3)
            return y
    
    def pred(xy):
        '''
        
    
        Parameters
        ----------
        xy : tensor 
            the coordinate of the input tensor.
    
        Returns
        the prediction of fai
    
        '''
        x = xy[:, 0].unsqueeze(1)
        y = xy[:, 1].unsqueeze(1)
        pred_fai = (x**2 - a**2/4)*(y**2 - b**2/4)*model(xy)
        return pred_fai
    
    def evaluate_Dt(N_test):# 计算Dt
        X_test = dom_data_uniform(N_test)
        fai = pred(X_test)
        M1 = 2*simpson_int(fai, X_test)
        c_n = M/M1 * (-2 * G * alpha) # 对泊松方程进行修正
        alpha_n = M/M1 * alpha
        Dt = M1 / alpha
        return Dt

    def evaluate_tau(N_test):# 计算Dt和Taumax
        xy_tensor = dom_data_uniform(N_test)
        x = np.linspace(-a/2, a/2, N_test)
        y = np.linspace(-b/2, b/2, N_test) # y方向N_test个点
        x_mesh, y_mesh = np.meshgrid(x, y)       
        fai = pred(xy_tensor)
        M1 = 2*simpson_int(fai, xy_tensor)
        dfaidxy = grad(fai, xy_tensor, torch.ones(xy_tensor.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        dfaidx = dfaidxy[:, 0].unsqueeze(1)
        dfaidy = dfaidxy[:, 1].unsqueeze(1)

        tauzx = M/M1*dfaidy # 根据翘楚函数得到应力
        tauzy = -M/M1*dfaidx
        tauzx = tauzx.data.cpu().numpy().reshape(N_test, N_test) # 转成meshgrid,numpy
        tauzy = tauzy.data.cpu().numpy().reshape(N_test, N_test)
        taumax = np.max(np.sqrt(tauzx**2+tauzy**2)) # 秋最大值
        return x_mesh, y_mesh, tauzx, tauzy, taumax    
    # learning the homogenous network
    def simpson_int(y, x,  nx = N_test, ny = N_test):
        '''
        Simpson integration for 2D

        Parameters
        ----------
        y : tensor
            The value of the x.
        x : tensor
            Coordinate of the input.
        nx : int, optional
            The grid node number of x axis. The default is N_test.
        ny : int, optional
            The grid node number of y axis. The default is N_test.
            
        Returns
        -------
        result : tensor
            the result of the integration.

        '''
        weightx = [4, 2] * int((nx-1)/2)
        weightx = [1] + weightx
        weightx[-1] = weightx[-1]-1
        weightx = np.array(weightx)
        
        weighty = [4, 2] * int((ny-1)/2)
        weighty = [1] + weighty
        weighty[-1] = weighty[-1]-1
        weighty = np.array(weighty)

        weightx = weightx.reshape(-1,1)
        weighty = weighty.reshape(1,-1)
        weight = weightx*weighty
        weight = weight.flatten()
        weight = torch.tensor(weight, device='cuda')
        hx = a/(nx-1)
        hy = b/(ny-1)
        y = y.flatten()
        result = torch.sum(weight*y)*hx*hy/9
        return result
    # learning the homogenous network
    
    model = FNN(2, 20, 1).cuda()
    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr= 0.001)
    optim1 = torch.optim.LBFGS(model.parameters(), lr= 0.001)
    loss_array_mincomplement[i] = []
    error_Dt_array_mincomplement[i] = []
    pred_alpha_array_mincomplement[i] = []
    loss_dom_array_mincomplement[i] = []
    loss_ex_array_mincomplement[i] = []
    pred_taumax_array_mincomplement[i] = []
    nepoch = int(nepoch)
    start = time.time()
    Xf = dom_data_uniform(N_test)    
    for epoch in range(nepoch):
        if epoch == nepoch-1:
            end = time.time()
            consume_time = end-start
            print('time is %f' % consume_time)
        if epoch%100 == 0: # 重新分配点   
            Xf = dom_data_uniform(N_test)
            
        def closure():  
            # 区域内部损失
            fai = pred(Xf)  
            dfaidxy = grad(fai, Xf, torch.ones(Xf.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
            dfaidx = dfaidxy[:, 0].unsqueeze(1)
            dfaidy = dfaidxy[:, 1].unsqueeze(1)      
    
            J_dom = simpson_int((dfaidx**2 + dfaidy**2)*0.5/G, Xf) # 计算余能
            J_ex = 2*alpha*simpson_int(fai, Xf) # 计算外力余功
       
            loss = J_dom - J_ex 
            optim.zero_grad()
            loss.backward(retain_graph=True)
            loss_array_mincomplement[i].append(loss.data.cpu())
            loss_dom_array_mincomplement[i].append(J_dom.data.cpu())
            loss_ex_array_mincomplement[i].append(J_ex.data.cpu())
    
    
            Dt_pred = evaluate_Dt(N_test)   
            Dt_exact = beta  * G * a * b**3 
            error_Dt_array_mincomplement[i].append(torch.abs(Dt_pred-Dt_exact).data.cpu())
            alpha_pred = M/Dt_pred
            pred_alpha_array_mincomplement[i].append(alpha_pred.data.cpu())
            x_mesh, y_mesh, tauzx_pred, tauzy_pred, taumax_pred = evaluate_tau(N_test) # 100*100个点
            pred_taumax_array_mincomplement[i].append(taumax_pred)
            if epoch%10==0:
                print(' epoch : %i, the loss : %f, loss dom : %f, loss ex : %f' % \
                      (epoch, loss.data, J_dom.data, J_ex.data))
            return loss
        optim.step(closure)
        
    for epoch in range(nepoch1):
        if epoch == nepoch-1:
            end = time.time()
            consume_time = end-start
            print('time is %f' % consume_time)
        if epoch%100 == 0: # 重新分配点   
            Xf = dom_data(dom_num)
            
        def closure():  
            # 区域内部损失
            fai = pred(Xf)  
            dfaidxy = grad(fai, Xf, torch.ones(Xf.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
            dfaidx = dfaidxy[:, 0].unsqueeze(1)
            dfaidy = dfaidxy[:, 1].unsqueeze(1)      
    
            J_dom = simpson_int((dfaidx**2 + dfaidy**2)*0.5/G, Xf) # 计算余能
            J_ex = 2*alpha*simpson_int(fai, Xf) # 计算外力余功
       
            loss = J_dom - J_ex 
            optim.zero_grad()
            loss.backward(retain_graph=True)
            loss_array_mincomplement[i].append(loss.data.cpu())
            loss_dom_array_mincomplement[i].append(J_dom.data.cpu())
            loss_ex_array_mincomplement[i].append(J_ex.data.cpu())
    
    
            Dt_pred = evaluate_Dt(N_test)   
            Dt_exact = beta  * G * a * b**3 
            error_Dt_array_mincomplement[i].append(torch.abs(Dt_pred-Dt_exact).data.cpu())
            alpha_pred = M/Dt_pred
            pred_alpha_array_mincomplement[i].append(alpha_pred.data.cpu())
            x_mesh, y_mesh, tauzx_pred, tauzy_pred, taumax_pred = evaluate_tau(N_test) # 100*100个点
            pred_taumax_array_mincomplement[i].append(taumax_pred)
            if epoch%10==0:
                print(' epoch : %i, the loss : %f, loss dom : %f, loss ex : %f' % \
                      (epoch, loss.data, J_dom.data, J_ex.data))
            return loss
        optim1.step(closure)

    x_mesh, y_mesh, tauzx_mincomplement, tauzy_mincomplement, taumax_mincomplement = evaluate_tau(N_test) # 100*100个点
    
alpha_relative_error_mincomplement = np.zeros(num) 
taumax_relative_error_mincomplement = np.zeros(num)
for i in range(num):
    a = a_list[i]
    beta = beta_list[i]
    beta1 = beta1_list[i]
    taumax_exact = M/(beta1*a*b**2)
    alpha_exact = M/(beta*G*a*b**3)
    alpha_relative_error_mincomplement[i] = np.abs(pred_alpha_array_mincomplement[i][-1]-alpha_exact)/alpha_exact
    taumax_relative_error_mincomplement[i] = np.abs(pred_taumax_array_mincomplement[i][-1]-taumax_exact)/taumax_exact
    print('alpha relative error in mincomplementary is ' + str(alpha_relative_error_mincomplement[i]))
    print('taumax relative error in mincomplementary is ' + str(taumax_relative_error_mincomplement[i]))
    # 计算应力的级数项
    total_zx = 0
    for j in range(100):
        labda = (2*j+1)*np.pi/b
        term = (-1)**j/(2*j+1)**2*np.cosh(labda*x_mesh)/np.cosh(labda*a/2)*np.sin(labda*y_mesh)
        total_zx = total_zx + term
    tauzx_exact = -2*G*alpha_exact*b*(y_mesh/b - 4/np.pi**2*total_zx)
    
    total_zy = 0
    for j in range(100):
        labda = (2*j+1)*np.pi/b
        term = (-1)**j/(2*j+1)**2*np.sinh(labda*x_mesh)/np.cosh(labda*a/2)*np.cos(labda*y_mesh)
        total_zy = total_zy + term
    tauzy_exact = 2*G*alpha_exact*b*4/np.pi**2*total_zy
    # zx精确剪应力
    h1 = plt.contourf(x_mesh, y_mesh, tauzx_exact,  cmap = 'jet', levels = 100)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.colorbar(h1).ax.set_title(r'$\tau_{zx}$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    # zy精确剪应力
    h2 = plt.contourf(x_mesh, y_mesh, tauzy_exact,  cmap = 'jet', levels = 100)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.colorbar(h2).ax.set_title(r'$\tau_{zy}$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    # zx最小余能预测
    h3 = plt.contourf(x_mesh, y_mesh, tauzx_mincomplement,  cmap = 'jet', levels = 100)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.colorbar(h3).ax.set_title(r'$\tau_{zx}$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    # zy最小余能预测
    h4 = plt.contourf(x_mesh, y_mesh, tauzy_mincomplement,  cmap = 'jet', levels = 100)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.colorbar(h4).ax.set_title(r'$\tau_{zy}$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    # zx最小余能误差
    h5 = plt.contourf(x_mesh, y_mesh, np.abs(tauzx_mincomplement-tauzx_exact),  cmap = 'jet', levels = 100)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.colorbar(h5).ax.set_title(r'$\tau_{zx}$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    # zy最小余能误差
    h6 = plt.contourf(x_mesh, y_mesh, np.abs(tauzy_mincomplement-tauzy_exact),  cmap = 'jet', levels = 100)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.colorbar(h6).ax.set_title(r'$\tau_{zy}$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    
    
    