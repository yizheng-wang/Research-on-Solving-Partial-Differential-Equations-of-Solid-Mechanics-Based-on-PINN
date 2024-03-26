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
cases = 9

b = 1 # 矩形板的宽
nepoch = 5000

dom_num = 10000
N_test = 10000
G = 1000
M = 10


for i in range(cases):
    a = a_list[i] # 矩形板的长
    b = 1 # 矩形板的宽
    beta = beta_list[i]

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
    # learning the homogenous network
    
    model = FNN(2, 20, 1).cuda()
    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr= 0.001)
    loss_array = []
    error_alpha_array = []
    time_array = []
    nepoch = int(nepoch)
    start = time.time()
    alpha_exact = M/(beta  * G * a * b**3 )
    Dt_exact = beta  * G * a * b**3 
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
            Dt_pred = evaluate_Dt(N_test)
            Dt_pred = evaluate_Dt(N_test)
            alpha_pred = M/Dt_pred
            relative_alpha_error = (alpha_pred-alpha_exact)/alpha_exact
            
            end_epoch = time.time()
            loss_array.append(loss.data.cpu())
            error_alpha_array.append(relative_alpha_error.data.cpu())
            time_array.append(end_epoch-start)
            if epoch%10==0:
                print('T: %f,  epoch : %i, the loss : %f , rel A: %f'  % \
                      (end_epoch-start,epoch, loss.data, relative_alpha_error.data))
            return loss
        optim.step(closure)
        
        
        
    plt.plot(time_array, loss_array)
    plt.plot(time_array, error_alpha_array)
    plt.yscale('log')
    plt.legend(['loss','error alpha'])
    plt.show()
        
        
        