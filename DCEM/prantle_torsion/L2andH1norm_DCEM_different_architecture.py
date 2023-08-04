# for different nn architecture, a/b = 1


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

a_list = [1, 1.2, 1.5,  2.0,  2.5, 3.0, 4.0, 5.0, 10.0] # 矩形板的长
beta_list =  [0.141, 0.166, 0.196, 0.229, 0.249, 0.263, 0.281, 0.291, 0.312]
k = 0
a = a_list[k] # 矩形板的长
b = 1 # 矩形板的宽
beta = beta_list[k]
nepoch = 2000
alpha = 0.0005 # a=b=1是0.0709标准值

dom_num = 10000
N_test = 101
G = 1000
M = 10

N_test_norm = 100

# the exact fai 
x = np.linspace(-a/2, a/2, N_test_norm)
y = np.linspace(-b/2, b/2, N_test_norm) # y方向N_test个点
x_mesh, y_mesh = np.meshgrid(x, y)
alpha_exact = M/(beta*G*a*b**3)
total_fai = 0
for j in range(100):
    labda = (2*j+1)*np.pi/b
    term = (-1)**j/(2*j+1)**3*np.cosh(labda*x_mesh)/np.cosh(labda*a/2)*np.cos(labda*y_mesh)
    total_fai = total_fai + term
fai_exact = G*alpha_exact*(b**2/4 - y_mesh**2) - 8*G*alpha_exact*b**2/np.pi**3*total_fai

L2norm_fai_exact = np.sqrt(np.trapz(np.trapz(fai_exact**2, dx=(y[-1]-y[-2])), dx=(x[-1]-x[-2])))

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

tau_exact = (tauzx_exact**2 + tauzy_exact**2)**0.5

H1norm_fai_exact = np.sqrt(np.trapz(np.trapz(tau_exact**2, dx=(y[-1]-y[-2])), dx=(x[-1]-x[-2])))

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

def dom_data(Nf):
    '''
    生成内部点
    '''
    
    x = a*np.random.rand(Nf)-a/2
    y = b*np.random.rand(Nf)-b/2 # 角度0到pi
    xy_dom = np.stack([x, y], 1)
    xy_dom = torch.tensor(xy_dom,  requires_grad=True, device='cuda')
    
    return xy_dom


class FNN1(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(FNN1, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, D_out)
        
        #self.a1 = torch.nn.Parameter(torch.Tensor([0.2]))
        
        # 有可能用到加速收敛技术
        self.a1 = torch.Tensor([0.1])
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
        y = self.n*self.a1*self.linear4(y1)
        return y

class FNN2(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(FNN2, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, D_out)
        
        #self.a1 = torch.nn.Parameter(torch.Tensor([0.2]))
        
        # 有可能用到加速收敛技术
        self.a1 = torch.Tensor([0.1])
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
        y = self.n*self.a1*self.linear4(y2)
        return y
class FNN3(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(FNN3, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, D_out)
        
        #self.a1 = torch.nn.Parameter(torch.Tensor([0.2]))
        
        # 有可能用到加速收敛技术
        self.a1 = torch.Tensor([0.1])
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


class FNN4(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(FNN4, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, D_out)
        
        #self.a1 = torch.nn.Parameter(torch.Tensor([0.2]))
        
        # 有可能用到加速收敛技术
        self.a1 = torch.Tensor([0.1])
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


def pred(xy, model):
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


def evaluate_tau(N_test, model):# 计算Dt和Taumax
    xy_tensor = dom_data_uniform(N_test)
    x = np.linspace(-a/2, a/2, N_test)
    y = np.linspace(-b/2, b/2, N_test) # y方向N_test个点
    x_mesh, y_mesh = np.meshgrid(x, y)       
    fai = pred(xy_tensor, model)
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

def evaluate_Dt(N_test, model):# 计算Dt
    X_test = dom_data_uniform(N_test)
    fai = pred(X_test, model)
    M1 = simpson_int(2*fai, X_test)
    c_n = M/M1 * (-2 * G * alpha) # 对泊松方程进行修正
    alpha_n = M/M1 * alpha
    Dt = M1 / alpha
    return Dt

def evaluate_L2_H1(N_test_norm, model):# 计算Dt
    xy_dom = dom_data_uniform(N_test)
    fai = pred(xy_dom, model)
    M1 = simpson_int(2*fai, xy_dom)
    dfaidxy = grad(fai, xy_dom, torch.ones(xy_dom.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    dfaidx = dfaidxy[:, 0].unsqueeze(1)
    dfaidy = dfaidxy[:, 1].unsqueeze(1)   
    
    tauzx = M/M1*dfaidy # 根据翘楚函数得到应力
    tauzy = -M/M1*dfaidx
    tau_pred = (tauzx**2 + tauzy**2)**0.5 
    
    fai_pred = fai.reshape(N_test_norm, N_test_norm)
    tau_pred = tau_pred.reshape(N_test_norm, N_test_norm)
    L2norm_fai_pred = M/M1.cpu().data.numpy()*np.sqrt(np.trapz(np.trapz(fai_pred.cpu().data**2, dx=(y[-1]-y[-2])), dx=(x[-1]-x[-2])))
    H1norm_fai_pred = np.sqrt(np.trapz(np.trapz(tau_pred.cpu().data**2, dx=(y[-1]-y[-2])), dx=(x[-1]-x[-2])))
    return L2norm_fai_pred, H1norm_fai_pred


# learning the homogenous network
criterion = torch.nn.MSELoss()

model1 = FNN1(2, 20, 1).cuda()
optim1 = torch.optim.Adam(model1.parameters(), lr= 0.001)

model2 = FNN2(2, 20, 1).cuda()
optim2 = torch.optim.Adam(model2.parameters(), lr= 0.001)

model3 = FNN3(2, 20, 1).cuda()
optim3 = torch.optim.Adam(model3.parameters(), lr= 0.001)

model4 = FNN4(2, 20, 1).cuda()
optim4 = torch.optim.Adam(model4.parameters(), lr= 0.001)


L2loss_array_mincomplement1 = []
H1loss_array_mincomplement1 = []

L2loss_array_mincomplement2 = []
H1loss_array_mincomplement2 = []

L2loss_array_mincomplement3 = []
H1loss_array_mincomplement3 = []

L2loss_array_mincomplement4 = []
H1loss_array_mincomplement4 = []

nepoch = int(nepoch)
start = time.time()

for epoch in range(nepoch):
    if epoch == nepoch-1:
        end = time.time()
        consume_time = end-start
        print('time is %f' % consume_time)
    if epoch%100 == 0: # 重新分配点   
        Xf = dom_data_uniform(N_test)
        
    def closure():  
        # 区域内部损失
        fai = pred(Xf, model1)  
        dfaidxy = grad(fai, Xf, torch.ones(Xf.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        dfaidx = dfaidxy[:, 0].unsqueeze(1)
        dfaidy = dfaidxy[:, 1].unsqueeze(1)      

        J_dom = simpson_int((dfaidx**2 + dfaidy**2)*0.5/G, Xf) # 计算余能
        J_ex = 2*alpha*simpson_int(fai, Xf) # 计算外力余功
   
        loss = J_dom - J_ex 
        optim1.zero_grad()
        loss.backward(retain_graph=True)

        if epoch%100==0 and 99<epoch<=1001:
            L2norm_fai_pred, H1norm_fai_pred = evaluate_L2_H1(N_test, model1)    
            L2loss_array_mincomplement1.append((L2norm_fai_pred - L2norm_fai_exact)/L2norm_fai_exact)
            H1loss_array_mincomplement1.append((H1norm_fai_pred - H1norm_fai_exact)/H1norm_fai_exact)        
        if epoch%10==0:
            print(' epoch : %i, the loss : %f, loss dom : %f, loss ex : %f' % \
                  (epoch, loss.data, J_dom.data, J_ex.data))
        return loss
    optim1.step(closure)
    
for epoch in range(nepoch):
    if epoch == nepoch-1:
        end = time.time()
        consume_time = end-start
        print('time is %f' % consume_time)
    if epoch%100 == 0: # 重新分配点   
        Xf = dom_data_uniform(N_test)
        
    def closure():  
        # 区域内部损失
        fai = pred(Xf, model2)  
        dfaidxy = grad(fai, Xf, torch.ones(Xf.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        dfaidx = dfaidxy[:, 0].unsqueeze(1)
        dfaidy = dfaidxy[:, 1].unsqueeze(1)      

        J_dom = simpson_int((dfaidx**2 + dfaidy**2)*0.5/G, Xf) # 计算余能
        J_ex = 2*alpha*simpson_int(fai, Xf) # 计算外力余功
   
        loss = J_dom - J_ex
        optim2.zero_grad()
        loss.backward(retain_graph=True)

        if epoch%100==0 and 99<epoch<=1001:
            L2norm_fai_pred, H1norm_fai_pred = evaluate_L2_H1(N_test, model2)    
            L2loss_array_mincomplement2.append((L2norm_fai_pred - L2norm_fai_exact)/L2norm_fai_exact)
            H1loss_array_mincomplement2.append((H1norm_fai_pred - H1norm_fai_exact)/H1norm_fai_exact)        
        if epoch%10==0:
            print(' epoch : %i, the loss : %f, loss dom : %f, loss ex : %f' % \
                  (epoch, loss.data, J_dom.data, J_ex.data))
        return loss
    optim2.step(closure)


for epoch in range(nepoch):
    if epoch == nepoch-1:
        end = time.time()
        consume_time = end-start
        print('time is %f' % consume_time)
    if epoch%100 == 0: # 重新分配点   
        Xf = dom_data_uniform(N_test)
        
    def closure():  
        # 区域内部损失
        fai = pred(Xf, model3)  
        dfaidxy = grad(fai, Xf, torch.ones(Xf.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        dfaidx = dfaidxy[:, 0].unsqueeze(1)
        dfaidy = dfaidxy[:, 1].unsqueeze(1)      

        J_dom = simpson_int((dfaidx**2 + dfaidy**2)*0.5/G, Xf) # 计算余能
        J_ex = 2*alpha*simpson_int(fai, Xf) # 计算外力余功
   
        loss = J_dom - J_ex 
        optim3.zero_grad()
        loss.backward(retain_graph=True)

        if epoch%100==0 and 99<epoch<=1001:
            L2norm_fai_pred, H1norm_fai_pred = evaluate_L2_H1(N_test, model3)    
            L2loss_array_mincomplement3.append((L2norm_fai_pred - L2norm_fai_exact)/L2norm_fai_exact)
            H1loss_array_mincomplement3.append((H1norm_fai_pred - H1norm_fai_exact)/H1norm_fai_exact)                  
        if epoch%10==0:
            print(' epoch : %i, the loss : %f, loss dom : %f, loss ex : %f' % \
                  (epoch, loss.data, J_dom.data, J_ex.data))
        return loss
    optim3.step(closure)
    
for epoch in range(nepoch):
    if epoch == nepoch-1:
        end = time.time()
        consume_time = end-start
        print('time is %f' % consume_time)
    if epoch%100 == 0: # 重新分配点   
        Xf = dom_data_uniform(N_test)
        
    def closure():  
        # 区域内部损失
        fai = pred(Xf, model4)  
        dfaidxy = grad(fai, Xf, torch.ones(Xf.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        dfaidx = dfaidxy[:, 0].unsqueeze(1)
        dfaidy = dfaidxy[:, 1].unsqueeze(1)      

        J_dom = simpson_int((dfaidx**2 + dfaidy**2)*0.5/G, Xf) # 计算余能
        J_ex = 2*alpha*simpson_int(fai, Xf) # 计算外力余功
   
        loss = J_dom - J_ex
        optim4.zero_grad()
        loss.backward(retain_graph=True)

        if epoch%100==0 and 99<epoch<=1001:
            L2norm_fai_pred, H1norm_fai_pred = evaluate_L2_H1(N_test, model4)    
            L2loss_array_mincomplement4.append((L2norm_fai_pred - L2norm_fai_exact)/L2norm_fai_exact)
            H1loss_array_mincomplement4.append((H1norm_fai_pred - H1norm_fai_exact)/H1norm_fai_exact)            
        if epoch%10==0:
            print(' epoch : %i, the loss : %f, loss dom : %f, loss ex : %f' % \
                  (epoch, loss.data, J_dom.data, J_ex.data))
        return loss
    optim4.step(closure)    
    
    
    
    
#%%
L2norm_fai_pred, H1norm_fai_pred = evaluate_L2_H1(N_test, model1)    
Dt_pred = evaluate_Dt(N_test, model1)
Dt_exact = beta * G * a * b**3
error_rel = (M/Dt_pred-M/Dt_exact)/(M/Dt_exact) 
print('relative error is ' + str(error_rel.data))
print('relative error of L2_fai is ' + str(((L2norm_fai_pred - L2norm_fai_exact)/L2norm_fai_exact)))
print('relative error of H1_fai is ' + str(((H1norm_fai_pred - H1norm_fai_exact)/H1norm_fai_exact)))
train_step = np.linspace(100, 1000, 10)
plt.figure(figsize=(10,6))
plt.plot(train_step, L2loss_array_mincomplement1, marker='*', ls = ':')
plt.plot(train_step, L2loss_array_mincomplement2, linestyle='--', marker = 'v')
plt.plot(train_step, L2loss_array_mincomplement3, linestyle='-.', marker = '^')
plt.plot(train_step, L2loss_array_mincomplement4, marker='o')
plt.xlim((50, 1050))
plt.xlabel('Training steps', fontsize=13)
plt.ylabel(r'$\mathcal{L}_{2}^{rel}$ error', fontsize=13)
plt.yscale('log')
plt.legend(['1HL-DCEM', '2HL-DCEM','3HL-DCEM','4HL-DCEM'], loc="lower left")
plt.show()

train_step = np.linspace(100, 1000, 10)
plt.figure(figsize=(10,6))
plt.plot(train_step, H1loss_array_mincomplement1, marker='*', ls = ':')
plt.plot(train_step, H1loss_array_mincomplement2, linestyle='--', marker = 'v')
plt.plot(train_step, H1loss_array_mincomplement3, linestyle='-.', marker = '^')
plt.plot(train_step, H1loss_array_mincomplement4, marker='o')
plt.xlim((50, 1050))
plt.xlabel('Training steps', fontsize=13)
plt.ylabel(r'$\mathcal{H}_{1}^{rel}$ error', fontsize=13)
plt.yscale('log')
plt.legend(['1HL-DCEM', '2HL-DCEM','3HL-DCEM','4HL-DCEM'], loc="lower left")
plt.show()
#%%

# x_mesh, y_mesh, tauzx_mincomplement, tauzy_mincomplement, tau_mincomplement = evaluate_tau(N_test)

# h1 = plt.contourf(x_mesh, y_mesh, tau_exact,  cmap = 'jet', levels = 100)
# ax = plt.gca()
# ax.set_aspect(1)
# plt.colorbar(h1).ax.set_title(r'$\tau$')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

# # pred剪应力
# h2 = plt.contourf(x_mesh, y_mesh, tau_mincomplement,  cmap = 'jet', levels = 100)
# ax = plt.gca()
# ax.set_aspect(1)
# plt.colorbar(h2).ax.set_title(r'$\tau$')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()