import numpy as np
import torch
import matplotlib.pyplot as plt
from  torch.autograd import grad
import time

torch.set_default_tensor_type(torch.DoubleTensor) # 将tensor的类型变成默认的double
torch.set_default_tensor_type(torch.cuda.DoubleTensor) # 将cuda的tensor的类型变成默认的double
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
setup_seed(0)

a_list = [1, 1.2, 1.5,  2.0,  2.5, 3.0, 4.0, 5.0, 10.0] # 矩形板的长
beta_list =  [0.141, 0.166, 0.196, 0.229, 0.249, 0.263, 0.281, 0.291, 0.312]
k = 0
a = a_list[k] # 矩形板的长
b = 1 # 矩形板的宽
beta = beta_list[k]
beta1 = 0.208
nepoch = 5000
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
N_test = 10000
G = 1000
M = 10
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
# learning the homogenous network

model = FNN(2, 20, 1).cuda()
criterion = torch.nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr= 0.001)
loss_array = []
error_Dt_array = []
loss_dom_array = []
loss_up_array = []  
loss_down_array = []  
loss_left_array = []   
loss_right_array = []
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
        loss_array.append(loss.data.cpu())
        loss_dom_array.append(J_dom.data.cpu())
        loss_up_array.append(J_up.data.cpu())
        loss_down_array.append(J_down.data.cpu())
        loss_left_array.append(J_left.data.cpu())
        loss_right_array.append(J_right.data.cpu())
        Dt_pred = evaluate_Dt(N_test)
        alpha_pred = M/Dt_pred
        alpha_exact = M/(beta  * G * a * b**3 )
        relative_alpha_error = (alpha_pred-alpha_exact)/alpha_exact
        Dt_exact = beta  * G * a * b**3 
        error_Dt_array.append(torch.abs(Dt_pred-Dt_exact).data.cpu())
        end_epoch = time.time()
        if epoch%10==0:
            print('T: %f,  epoch : %i, the loss : %f ,  J_dom : %f , J_up : %f , J_down : %f, J_left : %f, J_right: %f, rel A: %f' % \
                  (end_epoch-start, epoch, loss.data, J_dom.data, J_up.data, J_down.data, J_left.data, J_right.data, relative_alpha_error.data))
        return loss
    optim.step(closure)
    
    
    
Dt_pred = evaluate_Dt(N_test)
Dt_exact = beta * G * a * b**3
error_rel = (M/Dt_pred-M/Dt_exact)/(M/Dt_exact) 
print('alpha relative error is ' + str(error_rel))
# plot the prediction solution
fig = plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
loss_array = np.array(loss_array)
loss_dom_array = np.array(loss_dom_array)
loss_up_array = np.array(loss_up_array)
loss_down_array = np.array(loss_down_array)
loss_left_array = np.array(loss_left_array)
loss_right_array = np.array(loss_right_array)
plt.yscale('log')
plt.plot(loss_array) 
plt.plot(loss_dom_array) 
plt.plot(loss_up_array) 
plt.plot(loss_down_array) 
plt.plot(loss_left_array) 
plt.plot(loss_right_array)
plt.legend(['loss', 'loss_dom', 'loss_up', 'loss_down', 'loss_left', 'loss_right'])
plt.xlabel('the iteration')
plt.ylabel('loss')
plt.title('loss evolution')

plt.subplot(1, 2, 2)
error_Dt_array = np.array(error_Dt_array)
h2 = plt.plot(error_Dt_array)
plt.xlabel('the iteration')
plt.ylabel('error_abs')
plt.ylim(bottom=0.0, top = Dt_exact/5)
plt.title('Dt error') 

plt.show()
    
    
    