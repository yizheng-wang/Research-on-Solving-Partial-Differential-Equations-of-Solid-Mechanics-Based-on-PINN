# 不同的方法（4种），不同的长宽比，对扭角的影响


# =============================================================================
# 位移强形式
# =============================================================================
import numpy as np
import torch
import matplotlib.pyplot as plt
from  torch.autograd import grad
import time
import matplotlib as mpl
mpl.rcParams['font.sans-serif']=['SimHei'] # 为了显示中文matplotlib
torch.set_default_tensor_type(torch.DoubleTensor) # 将tensor的类型变成默认的double
torch.set_default_tensor_type(torch.cuda.DoubleTensor) # 将cuda的tensor的类型变成默认的double
mpl.rcParams['figure.dpi'] = 1000

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
setup_seed(0)
a_list = [1, 1.2, 1.5,  2.0,  2.5, 3.0, 4.0, 5.0, 10.0] # 矩形板的长

b = 1 # 矩形板的宽
beta_list =  [0.141, 0.166, 0.196, 0.229, 0.249, 0.263, 0.281, 0.291, 0.312]
beta1_list = [0.208, 0.219, 0.231, 0.246, 0.258, 0.267, 0.282, 0.291, 0.312]
nepoch = 2000
num = 9 # 分析num个a的长度
N_test = 10000
G = 1000
M = 10
b_dom = 1
b_up = 1
b_down = 1
b_left = 1
b_right = 1

dom_num = 10000
up_num = 1000
down_num = 1000
left_num = 1000
right_num = 1000

loss_array_strong_dis = {}
error_Dt_array_strong_dis = {}
pred_alpha_array_strong_dis = {}
pred_taumax_array_strong_dis = {}
loss_dom_array_strong_dis = {}
loss_up_array_strong_dis = {}  
loss_down_array_strong_dis = {}  
loss_left_array_strong_dis = {}   
loss_right_array_strong_dis = {}

for i in range(num):

    
    a = a_list[i]
    beta = beta_list[i]
    beta1 = beta1_list[i]
    def boundary_up(N_up): # 内部的边界点,由于交界面附近的梯度很大，所以多配一些点
        '''
          生成上边界点
        '''
        
        x = a*np.random.rand(N_up)-a/2 # 角度0到pi
        y = b/2*np.ones(N_up)
        xy_up = np.stack([x, y], 1)
        xy_up = torch.tensor(xy_up,  requires_grad=True, device='cuda')
        return xy_up
    
    def boundary_down(N_down): # 内部的边界点,由于交界面附近的梯度很大，所以多配一些点
        '''
          生成下边界点
        '''
        
        x = a*np.random.rand(N_down)-a/2 # 角度0到pi
        y = -b/2*np.ones(N_down)
        xy_down = np.stack([x, y], 1)
        xy_down = torch.tensor(xy_down,  requires_grad=True, device='cuda')
        return xy_down
    
    def boundary_left(N_left): # 内部的边界点,由于交界面附近的梯度很大，所以多配一些点
        '''
          生成左边界点
        '''
        x = -a/2*np.ones(N_left)
        y = b*np.random.rand(N_left)-b/2 # 角度0到pi
        xy_left = np.stack([x, y], 1)
        xy_left = torch.tensor(xy_left,  requires_grad=True, device='cuda')
        return xy_left
    
    def boundary_right(N_right): # 内部的边界点,由于交界面附近的梯度很大，所以多配一些点
        '''
          生成右边界点
        '''
        x = a/2 *np.ones(N_right)
        y = b*np.random.rand(N_right)-b/2 # 角度0到pi
        xy_right = np.stack([x, y], 1)
        xy_right = torch.tensor(xy_right,  requires_grad=True, device='cuda')
        return xy_right
    
    def dom_data(Nf):
        '''
        生成内部点
        '''
        
        x = a*np.random.rand(Nf)-a/2
        y = b*np.random.rand(Nf)-b/2 # 角度0到pi
        xy_dom = np.stack([x, y], 1)
        xy_dom = torch.tensor(xy_dom,  requires_grad=True, device='cuda')
        
        return xy_dom
    
    
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
            
            #self.a1 = torch.nn.Parameter(torch.Tensor([0.2]))
            
            # 有可能用到加速收敛技术
            self.a1 = torch.Tensor([0.1]).cuda()
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
    
    
    def evaluate_Dt(N_test):# 计算Dt和Taumax
        X_test = dom_data(N_test)      
        psai = model(X_test)
        dpsaidxy = grad(psai, X_test, torch.ones(X_test.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        dpsaidx = dpsaidxy[:, 0].unsqueeze(1)
        dpsaidy = dpsaidxy[:, 1].unsqueeze(1)
        Xx_test = X_test[:, 0].unsqueeze(1)
        Xy_test = X_test[:, 1].unsqueeze(1)
        Dt = G * torch.mean(Xx_test**2 + Xy_test**2 + Xx_test * dpsaidy - Xy_test * dpsaidx) * a * b
        return Dt
    
    def evaluate_tau(N_test):# 计算Dt和Taumax
        x = np.linspace(-a/2, a/2, N_test)
        y = np.linspace(-b/2, b/2, N_test) # y方向N_test个点
        x_mesh, y_mesh = np.meshgrid(x, y)
        xy = np.stack((x_mesh.flatten(), y_mesh.flatten()), 1)
        xy_tensor = torch.tensor(xy,  requires_grad=True, device='cuda')
       
        psai = model(xy_tensor)
        dpsaidxy = grad(psai, xy_tensor, torch.ones(xy_tensor.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        dpsaidx = dpsaidxy[:, 0].unsqueeze(1)
        dpsaidy = dpsaidxy[:, 1].unsqueeze(1)
        Xx_test = xy_tensor[:, 0].unsqueeze(1)
        Xy_test = xy_tensor[:, 1].unsqueeze(1)
        Dt_pred = evaluate_Dt(N_test) # 预测得到Dt
        alpha = M / Dt_pred # 得到alpha
        tauzx = G*alpha*(dpsaidx-Xy_test) # 根据翘楚函数得到应力
        tauzy = G*alpha*(dpsaidy+Xx_test)
        tauzx = tauzx.data.cpu().numpy().reshape(N_test, N_test) # 转成meshgrid,numpy
        tauzy = tauzy.data.cpu().numpy().reshape(N_test, N_test)
        tauabs = np.sqrt(tauzx**2+tauzy**2) # 秋最大值
        return x_mesh, y_mesh, tauzx, tauzy, tauabs
    
    def evaluate_taumax(N_test):# 根据几何形状，判断处最大的剪应力在（0， 0.5）
        x = 0
        y = b/2
        x_mesh, y_mesh = np.meshgrid(x, y)
        xy = np.stack((x_mesh.flatten(), y_mesh.flatten()), 1)
        xy_tensor = torch.tensor(xy,  requires_grad=True, device='cuda')
       
        psai = model(xy_tensor)
        dpsaidxy = grad(psai, xy_tensor, torch.ones(xy_tensor.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        dpsaidx = dpsaidxy[:, 0].unsqueeze(1)
        dpsaidy = dpsaidxy[:, 1].unsqueeze(1)
        Xx_test = xy_tensor[:, 0].unsqueeze(1)
        Xy_test = xy_tensor[:, 1].unsqueeze(1)
        Dt_pred = evaluate_Dt(N_test) # 预测得到Dt
        alpha = M / Dt_pred # 得到alpha
        tauzx = G*alpha*(dpsaidx-Xy_test) # 根据翘楚函数得到应力
        tauzy = G*alpha*(dpsaidy+Xx_test)
        tauzx = tauzx.data.cpu().numpy() # 转成meshgrid,numpy
        tauzy = tauzy.data.cpu().numpy()
        taumax = np.max(np.sqrt(tauzx**2+tauzy**2)) # 秋最大值
        return taumax
    # learning the homogenous network
    
    model = FNN(2, 20, 1).cuda()
    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr= 0.001)
    
    loss_array_strong_dis[i] = []
    error_Dt_array_strong_dis[i] = []
    pred_alpha_array_strong_dis[i] = []
    loss_dom_array_strong_dis[i] = []
    loss_up_array_strong_dis[i] = []
    loss_down_array_strong_dis[i] = []
    loss_left_array_strong_dis[i] = [] 
    loss_right_array_strong_dis[i] = []
    pred_taumax_array_strong_dis[i] = []
    
    nepoch = int(nepoch)
    start = time.time()
    
    for epoch in range(nepoch):
        if epoch == nepoch-1:
            end = time.time()
            consume_time = end-start
            print('time is %f' % consume_time)
        if epoch%100 == 0: # 重新分配点   
            Xf = dom_data(dom_num)
            X_up = boundary_up(up_num)
            X_down = boundary_down(down_num)
            X_left = boundary_left(left_num)
            X_right = boundary_right(right_num)
            
        def closure():  
            # 区域内部损失
            psai = model(Xf)  
            dpsaidxy = grad(psai, Xf, torch.ones(Xf.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
            dpsaidx = dpsaidxy[:, 0].unsqueeze(1)
            dpsaidy = dpsaidxy[:, 1].unsqueeze(1)
            dpsaidxxy = grad(dpsaidx, Xf, torch.ones(Xf.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
            dpsaidyxy = grad(dpsaidy, Xf, torch.ones(Xf.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
            dpsaidxx = dpsaidxxy[:, 0].unsqueeze(1)
            dpsaidyy = dpsaidyxy[:, 1].unsqueeze(1)        
    
            J_dom = torch.mean((dpsaidxx + dpsaidyy)**2)
            
            # 上边界的损失
            psai_up = model(X_up) # predict the boundary condition
            dpsaidxy_up = grad(psai_up, X_up, torch.ones(X_up.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
            dpsaidy_up = dpsaidxy_up[:, 1].unsqueeze(1)      
            Xx_up = X_up[:, 0].unsqueeze(1)
            J_up = torch.mean((dpsaidy_up + Xx_up)**2)
      
            # 下边界的损失
            psai_down = model(X_down) # predict the boundary condition
            dpsaidxy_down = grad(psai_down, X_down, torch.ones(X_down.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
            dpsaidy_down = dpsaidxy_down[:, 1].unsqueeze(1)      
            Xx_down = X_down[:, 0].unsqueeze(1)
            J_down = torch.mean((dpsaidy_down + Xx_down)**2)
            
            # 左边界的损失
            psai_left = model(X_left) # predict the boundary condition
            dpsaidxy_left = grad(psai_left, X_left, torch.ones(X_left.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
            dpsaidx_left = dpsaidxy_left[:, 0].unsqueeze(1)      
            Xy_left = X_left[:, 1].unsqueeze(1)
            J_left = torch.mean((dpsaidx_left - Xy_left)**2)
            
            # 右边界的损失
            psai_right = model(X_right) # predict the boundary condition
            dpsaidxy_right = grad(psai_right, X_right, torch.ones(X_right.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
            dpsaidx_right = dpsaidxy_right[:, 0].unsqueeze(1)      
            Xy_right = X_right[:, 1].unsqueeze(1)
            J_right = torch.mean((dpsaidx_right - Xy_right)**2) 
            
            loss = b_dom * J_dom + b_up * J_up + b_down * J_down + b_left * J_left + b_right * J_right
            optim.zero_grad()
            loss.backward(retain_graph=True)
            loss_array_strong_dis[i].append(loss.data.cpu())
            loss_dom_array_strong_dis[i].append(J_dom.data.cpu())
            loss_up_array_strong_dis[i].append(J_up.data.cpu())
            loss_down_array_strong_dis[i].append(J_down.data.cpu())
            loss_left_array_strong_dis[i].append(J_left.data.cpu())
            loss_right_array_strong_dis[i].append(J_right.data.cpu())
            Dt_pred = evaluate_Dt(N_test)
            x_mesh, y_mesh, tauzx_pred, tauzy_pred, tauabs_pred = evaluate_tau(100) # 100*100个点
            taumax_pred = evaluate_taumax(100)
            alpha_pred = M/Dt_pred
            Dt_exact = beta  * G * a * b**3 
            error_Dt_array_strong_dis[i].append(torch.abs(Dt_pred-Dt_exact).data.cpu())
            pred_alpha_array_strong_dis[i].append(alpha_pred.data.cpu())
            pred_taumax_array_strong_dis[i].append(taumax_pred)
            if epoch%10==0:
                print(' epoch : %i, the loss : %f ,  J_dom : %f , J_up : %f , J_down : %f, J_left : %f, J_right: %f' % \
                      (epoch, loss.data, J_dom.data, J_up.data, J_down.data, J_left.data, J_right.data))
            return loss
        optim.step(closure)
    

# =============================================================================
# 最小势能原理
# =============================================================================

# a = 1 # 矩形板的长
# b = 1 # 矩形板的宽
# beta = 0.141 # 不同的长宽比，beta会不同
# nepoch = 500

# dom_num = 10000
# N_test = 10000
# G = 1000
# M = 10
loss_array_minpotential = {}
error_Dt_array_minpotential = {}
pred_alpha_array_minpotential = {}
pred_taumax_array_minpotential = {}
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
            
            #self.a1 = torch.nn.Parameter(torch.Tensor([0.2]))
            
            # 有可能用到加速收敛技术
            self.a1 = torch.Tensor([0.1]).cuda()
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
    
    
    def evaluate_Dt(N_test):# 计算Dt
        X_test = dom_data(N_test)
    
        
        psai = model(X_test)
        dpsaidxy = grad(psai, X_test, torch.ones(X_test.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        dpsaidx = dpsaidxy[:, 0].unsqueeze(1)
        dpsaidy = dpsaidxy[:, 1].unsqueeze(1)
        Xx_test = X_test[:, 0].unsqueeze(1)
        Xy_test = X_test[:, 1].unsqueeze(1)
        Dt = G * torch.mean(Xx_test**2 + Xy_test**2 + Xx_test * dpsaidy - Xy_test * dpsaidx) * a * b
        return Dt

    def evaluate_tau(N_test):# 计算Dt和Taumax
        x = np.linspace(-a/2, a/2, N_test)
        y = np.linspace(-b/2, b/2, N_test) # y方向N_test个点
        x_mesh, y_mesh = np.meshgrid(x, y)
        xy = np.stack((x_mesh.flatten(), y_mesh.flatten()), 1)
        xy_tensor = torch.tensor(xy,  requires_grad=True, device='cuda')
       
        psai = model(xy_tensor)
        dpsaidxy = grad(psai, xy_tensor, torch.ones(xy_tensor.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        dpsaidx = dpsaidxy[:, 0].unsqueeze(1)
        dpsaidy = dpsaidxy[:, 1].unsqueeze(1)
        Xx_test = xy_tensor[:, 0].unsqueeze(1)
        Xy_test = xy_tensor[:, 1].unsqueeze(1)
        Dt_pred = evaluate_Dt(N_test) # 预测得到Dt
        alpha = M / Dt_pred # 得到alpha
        tauzx = G*alpha*(dpsaidx-Xy_test) # 根据翘楚函数得到应力
        tauzy = G*alpha*(dpsaidy+Xx_test)
        tauzx = tauzx.data.cpu().numpy().reshape(N_test, N_test) # 转成meshgrid,numpy
        tauzy = tauzy.data.cpu().numpy().reshape(N_test, N_test)
        tauabs = np.sqrt(tauzx**2+tauzy**2) # 秋最大值
        return x_mesh, y_mesh, tauzx, tauzy, tauabs
    
    def evaluate_taumax(N_test):# 计算Dt和Taumax

        x = 0
        y = b/2
        x_mesh, y_mesh = np.meshgrid(x, y)
        xy = np.stack((x_mesh.flatten(), y_mesh.flatten()), 1)
        xy_tensor = torch.tensor(xy,  requires_grad=True, device='cuda')
       
        psai = model(xy_tensor)
        dpsaidxy = grad(psai, xy_tensor, retain_graph=True, create_graph=True)[0]
        dpsaidx = dpsaidxy[:, 0].unsqueeze(1)
        dpsaidy = dpsaidxy[:, 1].unsqueeze(1)
        Xx_test = xy_tensor[:, 0].unsqueeze(1)
        Xy_test = xy_tensor[:, 1].unsqueeze(1)
        Dt_pred = evaluate_Dt(N_test) # 预测得到Dt
        alpha = M / Dt_pred # 得到alpha
        tauzx = G*alpha*(dpsaidx-Xy_test) # 根据翘楚函数得到应力
        tauzy = G*alpha*(dpsaidy+Xx_test)
        tauzx = tauzx.data.cpu().numpy() # 转成meshgrid,numpy
        tauzy = tauzy.data.cpu().numpy()
        taumax = np.max(np.sqrt(tauzx**2+tauzy**2)) # 秋最大值
        return taumax    

    # learning the homogenous network
    
    model = FNN(2, 20, 1).cuda()
    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr= 0.001)
    loss_array_minpotential[i] = []
    error_Dt_array_minpotential[i] = []
    pred_alpha_array_minpotential[i] = []
    pred_taumax_array_minpotential[i] = []
    nepoch = int(nepoch)
    start = time.time()
    
    for epoch in range(nepoch):
        if epoch == nepoch-1:
            end = time.time()
            consume_time = end-start
            print('time is %f' % consume_time)
        if epoch%100 == 0: # 重新分配点   
            Xf = dom_data(dom_num)
            
        def closure():  
            # 区域内部损失
            psai = model(Xf)  
            dpsaidxy = grad(psai, Xf, torch.ones(Xf.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
            dpsaidx = dpsaidxy[:, 0].unsqueeze(1)
            dpsaidy = dpsaidxy[:, 1].unsqueeze(1)
            Xf_x = Xf[:, 0].unsqueeze(1) # Xf 域内点的x坐标
            Xf_y = Xf[:, 1].unsqueeze(1) # xf 域内点的y坐标
    
            J_dom = torch.mean((dpsaidx - Xf_y)**2+(dpsaidy + Xf_x)**2)*a*b
            
            
            loss =  J_dom 
            optim.zero_grad()
            loss.backward(retain_graph=True)
            loss_array_minpotential[i].append(loss.data.cpu())
            Dt_pred = evaluate_Dt(N_test)
            Dt_exact = beta  * G * a * b**3 
            error_Dt_array_minpotential[i].append(torch.abs(Dt_pred-Dt_exact).data.cpu())
            alpha_pred = M/Dt_pred
            pred_alpha_array_minpotential[i].append(alpha_pred.data.cpu())
            x_mesh, y_mesh, tauzx_pred, tauzy_pred, tauabs_pred = evaluate_tau(100) # 100*100个点
            taumax_pred = evaluate_taumax(100)
            pred_taumax_array_minpotential[i].append(taumax_pred)
            if epoch%10==0:
                print(' epoch : %i, the loss : %f ' % \
                      (epoch, loss.data))
            return loss
        optim.step(closure)

# =============================================================================
# 应力函数强形式
# =============================================================================
loss_array_strong_fai = {}
error_Dt_array_strong_fai = {}
pred_alpha_array_strong_fai = {}
pred_taumax_array_strong_fai = {}
for i in range(num):
    c1 = -1
    
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
            
            #self.a1 = torch.nn.Parameter(torch.Tensor([0.2]))
            
            # 有可能用到加速收敛技术
            self.a1 = torch.Tensor([0.1]).cuda()
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
        X_test = dom_data(N_test)
        fai = pred(X_test)
        M1 = torch.mean(2*fai)*a*b
        c_n = M/M1 * c1 # 对泊松方程进行修正
        alpha = c_n/(-2*G)
        Dt = M / alpha
        return Dt
    
    def evaluate_tau(N_test):# 计算Dt和Taumax
        x = np.linspace(-a/2, a/2, N_test)
        y = np.linspace(-b/2, b/2, N_test) # y方向N_test个点
        x_mesh, y_mesh = np.meshgrid(x, y)
        xy = np.stack((x_mesh.flatten(), y_mesh.flatten()), 1)
        xy_tensor = torch.tensor(xy,  requires_grad=True, device='cuda')
       
        fai = pred(xy_tensor)
        M1 = torch.mean(2*fai)*a*b
        dfaidxy = grad(fai, xy_tensor, torch.ones(xy_tensor.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        dfaidx = dfaidxy[:, 0].unsqueeze(1)
        dfaidy = dfaidxy[:, 1].unsqueeze(1)

        tauzx = M/M1*dfaidy # 根据翘楚函数得到应力
        tauzy = -M/M1*dfaidx
        tauzx = tauzx.data.cpu().numpy().reshape(N_test, N_test) # 转成meshgrid,numpy
        tauzy = tauzy.data.cpu().numpy().reshape(N_test, N_test)
        tauabs = np.sqrt(tauzx**2+tauzy**2) # 绝对值
        return x_mesh, y_mesh, tauzx, tauzy, tauabs
    
    def evaluate_taumax(N_test):# 计算Dt和Taumax

        x = 0
        y = b/2
        x_mesh, y_mesh = np.meshgrid(x, y)
        xy = np.stack((x_mesh.flatten(), y_mesh.flatten()), 1)
        xy_tensor = torch.tensor(xy,  requires_grad=True, device='cuda')
       
        fai = pred(xy_tensor)
        # 计算M1，假设alpha下的力矩
        x_dom = np.linspace(-a/2, a/2, N_test)
        y_dom = np.linspace(-b/2, b/2, N_test) # y方向N_test个点
        x_mesh_dom, y_mesh_dom = np.meshgrid(x_dom, y_dom)
        xy_dom = np.stack((x_mesh_dom.flatten(), y_mesh_dom.flatten()), 1)
        xy_tensor_dom = torch.tensor(xy_dom,  requires_grad=True, device='cuda')
       
        fai_dom = pred(xy_tensor_dom)
        M1 = torch.mean(2*fai_dom)*a*b
        
        dfaidxy = grad(fai, xy_tensor, retain_graph=True, create_graph=True)[0]
        dfaidx = dfaidxy[:, 0].unsqueeze(1)
        dfaidy = dfaidxy[:, 1].unsqueeze(1)

        tauzx = (M/M1*dfaidy).data.cpu().numpy() # 转成meshgrid,numpy
        tauzy = (-M/M1*dfaidx).data.cpu().numpy()
        taumax = np.max(np.sqrt(tauzx**2+tauzy**2)) # 秋最大值
        return taumax
    # learning the homogenous network
    
    model = FNN(2, 20, 1).cuda()
    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr= 0.001)
    loss_array_strong_fai[i] = []
    error_Dt_array_strong_fai[i] = []
    pred_alpha_array_strong_fai[i] = []
    pred_taumax_array_strong_fai[i] = []
    nepoch = int(nepoch)
    start = time.time()
    
    for epoch in range(nepoch):
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
            dfaidxxy = grad(dfaidx, Xf, torch.ones(Xf.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
            dfaidyxy = grad(dfaidy, Xf, torch.ones(Xf.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
            dfaidxx = dfaidxxy[:, 0].unsqueeze(1)
            dfaidyy = dfaidyxy[:, 1].unsqueeze(1)        
    
            J_dom = torch.mean((dfaidxx + dfaidyy - c1)**2)
             
            loss = J_dom 
            optim.zero_grad()
            loss.backward(retain_graph=True)
            loss_array_strong_fai[i].append(loss.data.cpu())
    
            Dt_pred = evaluate_Dt(N_test)
            Dt_exact = beta  * G * a * b**3 
            error_Dt_array_strong_fai[i].append(torch.abs(Dt_pred-Dt_exact).data.cpu())
            alpha_pred = M/Dt_pred
            pred_alpha_array_strong_fai[i].append(alpha_pred.data.cpu())
            x_mesh, y_mesh, tauzx_pred, tauzy_pred, tauabs_pred = evaluate_tau(100) # 100*100个点
            taumax_pred = evaluate_taumax(100)
            pred_taumax_array_strong_fai[i].append(taumax_pred)
            if epoch%10==0:
                print(' epoch : %i, the loss : %f' % \
                      (epoch, loss.data))
            return loss
        optim.step(closure)

# =============================================================================
# 最小余能原理
# =============================================================================

alpha = 0.0005 # a=b=1是0.0709标准值
N_test_uni = 101
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
            
            #self.a1 = torch.nn.Parameter(torch.Tensor([0.2]))
            
            # 有可能用到加速收敛技术
            self.a1 = torch.Tensor([0.1]).cuda()
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
        M1 = simpson_int(2*fai, X_test)
        c_n = M/M1 * (-2 * G * alpha) # 对泊松方程进行修正
        alpha_n = M/M1 * alpha
        Dt = M1 / alpha
        return Dt
    def simpson_int(y, x,  nx = N_test_uni, ny = N_test_uni):
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
    def evaluate_tau(N_test):# 计算Dt和Taumax
        x = np.linspace(-a/2, a/2, N_test)
        y = np.linspace(-b/2, b/2, N_test) # y方向N_test个点
        x_mesh, y_mesh = np.meshgrid(x, y)
        xy = np.stack((x_mesh.flatten(), y_mesh.flatten()), 1)
        xy_tensor = torch.tensor(xy,  requires_grad=True, device='cuda')
       
        fai = pred(xy_tensor)
        M1 = simpson_int(2*fai, xy_tensor)
        dfaidxy = grad(fai, xy_tensor, torch.ones(xy_tensor.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        dfaidx = dfaidxy[:, 0].unsqueeze(1)
        dfaidy = dfaidxy[:, 1].unsqueeze(1)

        tauzx = M/M1*dfaidy # 根据翘楚函数得到应力
        tauzy = -M/M1*dfaidx
        tauzx = tauzx.data.cpu().numpy().reshape(N_test, N_test) # 转成meshgrid,numpy
        tauzy = tauzy.data.cpu().numpy().reshape(N_test, N_test)
        tauabs = np.sqrt(tauzx**2+tauzy**2) # 秋最大值
        return x_mesh, y_mesh, tauzx, tauzy, tauabs   
    
    def evaluate_taumax(N_test):# 计算Dt和Taumax

        x = 0
        y = b/2
        x_mesh, y_mesh = np.meshgrid(x, y)
        xy = np.stack((x_mesh.flatten(), y_mesh.flatten()), 1)
        xy_tensor = torch.tensor(xy,  requires_grad=True, device='cuda')
       
        fai = pred(xy_tensor)
        # 计算M1，假设alpha下的力矩
        x_dom = np.linspace(-a/2, a/2, N_test)
        y_dom = np.linspace(-b/2, b/2, N_test) # y方向N_test个点
        x_mesh_dom, y_mesh_dom = np.meshgrid(x_dom, y_dom)
        xy_dom = np.stack((x_mesh_dom.flatten(), y_mesh_dom.flatten()), 1)
        xy_tensor_dom = torch.tensor(xy_dom,  requires_grad=True, device='cuda')
       
        fai_dom = pred(xy_tensor_dom)
        M1 = simpson_int(2*fai_dom, xy_tensor_dom)
        
        dfaidxy = grad(fai, xy_tensor, retain_graph=True, create_graph=True)[0]
        dfaidx = dfaidxy[:, 0].unsqueeze(1)
        dfaidy = dfaidxy[:, 1].unsqueeze(1)

        tauzx = (M/M1*dfaidy).data.cpu().numpy() # 转成meshgrid,numpy
        tauzy = (-M/M1*dfaidx).data.cpu().numpy()
        taumax = np.max(np.sqrt(tauzx**2+tauzy**2)) # 秋最大值
        return taumax    
    # learning the homogenous network
    
    model = FNN(2, 20, 1).cuda()
    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr= 0.001)
    loss_array_mincomplement[i] = []
    error_Dt_array_mincomplement[i] = []
    pred_alpha_array_mincomplement[i] = []
    loss_dom_array_mincomplement[i] = []
    loss_ex_array_mincomplement[i] = []
    pred_taumax_array_mincomplement[i] = []
    nepoch = int(nepoch)
    start = time.time()
    Xf = dom_data_uniform(N_test_uni)
    for epoch in range(nepoch):
        if epoch == nepoch-1:
            end = time.time()
            consume_time = end-start
            print('time is %f' % consume_time)
            
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
    
    
            Dt_pred = evaluate_Dt(N_test_uni)   
            Dt_exact = beta  * G * a * b**3 
            error_Dt_array_mincomplement[i].append(torch.abs(Dt_pred-Dt_exact).data.cpu())
            alpha_pred = M/Dt_pred
            pred_alpha_array_mincomplement[i].append(alpha_pred.data.cpu())
            x_mesh, y_mesh, tauzx_pred, tauzy_pred, tauabs_pred = evaluate_tau(N_test_uni) # 100*100个点
            taumax_pred = evaluate_taumax(N_test_uni)
            pred_taumax_array_mincomplement[i].append(taumax_pred)
            if epoch%10==0:
                print(' epoch : %i, the loss : %f, loss dom : %f, loss ex : %f' % \
                      (epoch, loss.data, J_dom.data, J_ex.data))
            return loss
        optim.step(closure)
smo = 15 # 光滑系数，越大越好

for i in range(num):
    a = a_list[i]
    beta = beta_list[i]
    beta1 = beta1_list[i]
    taumax_exact = M/(beta1*a*b**2)
    pred_taumax_array_strong_dis_smo = np.array(pred_taumax_array_strong_dis[i])
    pred_taumax_array_minpotential_smo = np.array(pred_taumax_array_minpotential[i])
    pred_taumax_array_strong_fai_smo = np.array(pred_taumax_array_strong_fai[i])
    pred_taumax_array_mincomplement_smo = np.array(pred_taumax_array_mincomplement[i])
    for j in range(int(len(pred_taumax_array_strong_dis[i])/smo-1.1)):
        if j  == int(len(pred_taumax_array_strong_dis[i])/smo-1.1) - 1:
            pred_taumax_array_strong_dis_smo[j*smo:] = np.mean(np.array(pred_taumax_array_strong_dis[i][j*smo:]))
            pred_taumax_array_minpotential_smo[j*smo:] = np.mean(np.array(pred_taumax_array_minpotential[i][j*smo:]))
            pred_taumax_array_strong_fai_smo[j*smo:] = np.mean(np.array(pred_taumax_array_strong_fai[i][j*smo:]))
            pred_taumax_array_mincomplement_smo[j*smo:] = np.mean(np.array(pred_taumax_array_mincomplement[i][j*smo:]))
        pred_taumax_array_strong_dis_smo[j*smo:(j+1)*smo] = np.mean(np.array(pred_taumax_array_strong_dis[i][j*smo:(j+1)*smo]))
        pred_taumax_array_minpotential_smo[j*smo:(j+1)*smo] = np.mean(np.array(pred_taumax_array_minpotential[i][j*smo:(j+1)*smo]))
        pred_taumax_array_strong_fai_smo[j*smo:(j+1)*smo] = np.mean(np.array(pred_taumax_array_strong_fai[i][j*smo:(j+1)*smo]))
        pred_taumax_array_mincomplement_smo[j*smo:(j+1)*smo] = np.mean(np.array(pred_taumax_array_mincomplement[i][j*smo:(j+1)*smo]))

    plt.plot(pred_taumax_array_strong_dis_smo, linestyle=(0,(3.5, 1.5, 1.5)))
    plt.plot(pred_taumax_array_minpotential_smo, linestyle='--')
    plt.plot(pred_taumax_array_strong_fai_smo, linestyle='-.')
    plt.plot(pred_taumax_array_mincomplement_smo, ls = ':')
    plt.axhline(y=taumax_exact, color='r')
    plt.xlabel('Iteration', fontsize=13)
    plt.ylabel('Maximum shear stress', fontsize=13)
    plt.ylim(bottom=0.0, top = taumax_exact*2)
    plt.legend(['Strong: Displacement', 'DEM', 'Strong: Stress function', 'DCEM', 'Exact'])
    plt.show()




# Dt_pred = evaluate_Dt(N_test)
# Dt_exact = beta * G * a * b**3
# error_rel = (M/Dt_pred-M/Dt_exact)/(M/Dt_exact) 
# # plot the prediction solution
# fig = plt.figure(figsize=(14, 7))
# 相对误差
taumax_relative_error_strong_dis = np.zeros(num)
taumax_relative_error_minpotential = np.zeros(num)
taumax_relative_error_strong_fai = np.zeros(num)
taumax_relative_error_mincomplement = np.zeros(num)
for i in range(num):
    a = a_list[i]
    beta = beta_list[i]
    beta1 = beta1_list[i]
    taumax_exact = M/(beta1*a*b**2)
    taumax_relative_error_strong_dis[i] = np.abs(pred_taumax_array_strong_dis[i][-1]-taumax_exact)/taumax_exact
    taumax_relative_error_minpotential[i] = np.abs(pred_taumax_array_minpotential[i][-1]-taumax_exact)/taumax_exact
    taumax_relative_error_strong_fai[i] = np.abs(pred_taumax_array_strong_fai[i][-1]-taumax_exact)/taumax_exact
    taumax_relative_error_mincomplement[i] = np.abs(pred_taumax_array_mincomplement[i][-1]-taumax_exact)/taumax_exact
    
plt.plot(a_list, taumax_relative_error_strong_dis, marker='o', linestyle='-')
plt.plot(a_list, taumax_relative_error_minpotential, marker='*', linestyle=':')
plt.plot(a_list, taumax_relative_error_strong_fai, marker='v', linestyle='--')
plt.plot(a_list, taumax_relative_error_mincomplement, marker='p', linestyle='-.')
plt.xscale('log')
plt.xlabel('a/b', fontsize=13)
plt.ylabel('Relative error', fontsize=13)
plt.legend(['Strong: Displacement', 'DEM', 'Strong: Stress function', 'DCEM'])
plt.show()

    
    
    