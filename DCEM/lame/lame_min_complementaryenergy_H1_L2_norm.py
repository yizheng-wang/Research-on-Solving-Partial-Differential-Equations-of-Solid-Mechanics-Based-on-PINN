# test different number points 51 101 151 201 251
import numpy as np
import torch
import matplotlib.pyplot as plt
from  torch.autograd import grad
import time
import matplotlib as mpl
torch.set_default_tensor_type(torch.DoubleTensor) # 将tensor的类型变成默认的double
torch.set_default_tensor_type(torch.cuda.DoubleTensor) # 将cuda的tensor的类型变成默认的double

plt.rcParams['font.family'] = ['sans-serif'] # 用来正常显示负号
mpl.rcParams['figure.dpi'] = 1000


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
setup_seed(2023)

Po = 10
Pi = 5
a = 0.5
b = 1.0

nepoch = 1500
dom_num_list = [11, 21, 31, 41, 51]
L2loss_array_mincomplement1 = []
H1loss_array_mincomplement1 = []
L2loss_array_mincomplement2 = []
H1loss_array_mincomplement2 = []
L2loss_array_mincomplement3 = []
H1loss_array_mincomplement3 = []
L2loss_array_mincomplement4 = []
H1loss_array_mincomplement4 = []


def dom_data(Nf):
    '''
    生成内部点，极坐标形式生成
    '''
    
    # r = (b-a)*np.random.rand(Nf)+a
    # theta = 2 * np.pi * np.random.rand(Nf) # 角度0到2*pi
    r = np.linspace(a,b, Nf)
    xy_dom = np.stack([r], 1)
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

    pred_fai = model(xy)
    return pred_fai

def evaluate_sigma(N_test, model):# calculate the prediction of the stress rr and theta
# 分析sigma应力，输入坐标是极坐标r和theta
    r = np.linspace(a, b, N_test)
    theta = np.linspace(0, 2*np.pi, N_test) # y方向N_test个点
    r_mesh, theta_mesh = np.meshgrid(r, theta)
    xy = np.stack((r_mesh.flatten(), theta_mesh.flatten()), 1)
    X_test = torch.tensor(xy,  requires_grad=True, device='cuda')
    r = X_test[:, 0].unsqueeze(1)
    # 将r输入到pred中，输出应力函数
    fai = pred(r, model)
    dfaidr = grad(fai, r, torch.ones(r.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    dfaidrr = grad(dfaidr, r, torch.ones(r.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    pred_sigma_rr = (1/r)*dfaidr # rr方向的正应力
    pred_sigma_theta = dfaidrr # theta方向的正应力
    pred_u_r = 1/E*(r*dfaidrr-nu*dfaidr)
    
    pred_sigma_rr = pred_sigma_rr.data.cpu().numpy().reshape(N_test, N_test)
    pred_sigma_theta = pred_sigma_theta.data.cpu().numpy().reshape(N_test, N_test)
    pred_u_r = pred_u_r.data.cpu().numpy().reshape(N_test, N_test)
    # sigma_r = a**2/(b**2-a**2)*(1-b**2/r**2)*Pi - b**2/(b**2-a**2)*(1-a**2/r**2)*Po
    # sigma_theta = a**2/(b**2-a**2)*(1+b**2/r**2)*Pi - b**2/(b**2-a**2)*(1+a**2/r**2)*Po

    return r_mesh, theta_mesh, pred_sigma_rr, pred_sigma_theta, pred_u_r

def evaluate_sigma_line(N_test, model):# output the prediction of the stress rr and theta along radius in direction of r without theta
    r_numpy = np.linspace(a, b, N_test) # generate the coordinate of r by numpy 
    r = torch.tensor(r_numpy,  requires_grad=True, device='cuda').unsqueeze(1) # change the type of array to tensor used to be AD
    fai = pred(r, model) # Input r to the pred function to get the necessary predition stress function
    dfaidr = grad(fai, r, torch.ones(r.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    dfaidrr = grad(dfaidr, r, torch.ones(r.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    pred_sigma_rr = (1/r)*dfaidr # rr方向的正应力
    pred_sigma_theta = dfaidrr # theta方向的正应力
    pred_u_r = 1/E*(r*dfaidrr-nu*dfaidr)
    pred_sigma_rr = pred_sigma_rr.data.cpu().numpy() # change the type of tensor to the array for the use of plotting
    pred_sigma_theta = pred_sigma_theta.data.cpu().numpy()
    pred_u_r = pred_u_r.data.cpu().numpy()
    return r_numpy, pred_sigma_rr, pred_sigma_theta, pred_u_r
# learning the homogenous network
def evaluate_L2_H1(N_test_norm, model):
    Xf = dom_data(N_test_norm)
    r = Xf    
    fai = pred(r, model)
    # Because Airy stress function can be different as a constant
    c = fai_exact[0] - fai[0]
    dfaidr = grad(fai, r, torch.ones(r.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0] 
    L2norm_fai_pred = np.sqrt(np.trapz((fai+c).flatten().cpu().data**2, dx=(r[-1].cpu().data-r[-2].cpu().data)))
    H1norm_fai_pred = np.sqrt(np.trapz(dfaidr.flatten().cpu().data**2, dx=(r[-1].cpu().data-r[-2].cpu().data)))
    return L2norm_fai_pred, H1norm_fai_pred

for i, dom_num in enumerate(dom_num_list):
    dom_num = dom_num_list[i]
    N_test = dom_num
    r = np.linspace(a,b, N_test)
    fai_exact = a**2/(b**2-a**2)*(r**2/2-b**2*np.log(r))*Pi-b**2/(b**2-a**2)*(r**2/2-a**2*np.log(r))*Po
    L2norm_fai_exact = np.sqrt(np.trapz(fai_exact**2, dx=(r[-1]-r[-2])))
    fai_r_exact = a**2/(b**2-a**2)*(r-b**2/r)*Pi-b**2/(b**2-a**2)*(r-a**2/r)*Po
    H1norm_fai_exact = np.sqrt(np.trapz(fai_r_exact**2, dx=(r[-1]-r[-2])))
    
    E = 1000
    nu = 0.3
    Ura = 1/E*((1-nu)*(a**2*Pi-b**2*Po)/(b**2-a**2)*a + (1+nu)*(a**2*b**2)*(Pi-Po)/(b**2-a**2)/a)
    Urb = 1/E*((1-nu)*(a**2*Pi-b**2*Po)/(b**2-a**2)*b + (1+nu)*(a**2*b**2)*(Pi-Po)/(b**2-a**2)/b)
    

    def simpson_int(y, x,  nx = dom_num):
        '''
        Simpson integration for 1D
    
        Parameters
        ----------
        y : tensor
            The value of the x.
        x : tensor
            Coordinate of the input.
        nx : int, optional
            The grid node number of x axis. The default is N_test.
            
        Returns
        -------
        result : tensor
            the result of the integration.
    
        '''
        weightx = [4, 2] * int((nx-1)/2)
        weightx = [1] + weightx
        weightx[-1] = weightx[-1]-1
        weightx = np.array(weightx)
        
    
        weightx = weightx.reshape(-1,1)
    
    
        weight = torch.tensor(weightx, device='cuda')
        weight = weight.flatten()
        hx = a/(nx-1)
        y = y.flatten()
        result = torch.sum(weight*y)*hx/3
        return result    


    model1 = FNN1(1, 20, 1).cuda()
    optim1 = torch.optim.Adam(model1.parameters(), lr= 0.001)
    
    model2 = FNN2(1, 20, 1).cuda()
    optim2 = torch.optim.Adam(model2.parameters(), lr= 0.001)
    
    model3 = FNN3(1, 20, 1).cuda()
    optim3 = torch.optim.Adam(model3.parameters(), lr= 0.001)
    
    model4 = FNN4(1, 20, 1).cuda()
    optim4 = torch.optim.Adam(model4.parameters(), lr= 0.001)
    
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
            r = Xf
            fai = pred(r, model1)
            dfaidr = grad(fai, r, torch.ones(r.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
            dfaidrr = grad(dfaidr, r, torch.ones(r.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]   
            
            J_dom = simpson_int(0.5/E*(1/r**2*dfaidr**2 + dfaidrr**2 - 2*nu/r*dfaidr*dfaidrr)*2*np.pi*r, r)
            #J_dom = torch.mean(0.5/E*(1/r**2*dfaidr**2 + dfaidrr**2 - 2*nu/r*dfaidr*dfaidrr)*2*np.pi*r) * (b-a) # 计算余能
            # calculate the complementary external energy
            ra = torch.tensor([a],  requires_grad=True, device='cuda').unsqueeze(1)
            rb = torch.tensor([b],  requires_grad=True, device='cuda').unsqueeze(1)
            faia = pred(ra, model1)
            faib = pred(rb, model1)
            dfaidra = grad(faia, ra, retain_graph=True, create_graph=True)[0]
            dfaidrb = grad(faib, rb, retain_graph=True, create_graph=True)[0]
            J_ex = 2*np.pi*(-Ura*dfaidra + Urb*dfaidrb) # 计算外力余势
       
            loss = J_dom - J_ex 
            optim1.zero_grad()
            loss.backward(retain_graph=True)


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
            Xf = dom_data(dom_num)
            
        def closure():  
            # 区域内部损失
            r = Xf
            fai = pred(r, model2)
            dfaidr = grad(fai, r, torch.ones(r.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
            dfaidrr = grad(dfaidr, r, torch.ones(r.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]   
            
            J_dom = simpson_int(0.5/E*(1/r**2*dfaidr**2 + dfaidrr**2 - 2*nu/r*dfaidr*dfaidrr)*2*np.pi*r, r)
            #J_dom = torch.mean(0.5/E*(1/r**2*dfaidr**2 + dfaidrr**2 - 2*nu/r*dfaidr*dfaidrr)*2*np.pi*r) * (b-a) # 计算余能
            # calculate the complementary external energy
            ra = torch.tensor([a],  requires_grad=True, device='cuda').unsqueeze(1)
            rb = torch.tensor([b],  requires_grad=True, device='cuda').unsqueeze(1)
            faia = pred(ra, model2)
            faib = pred(rb, model2)
            dfaidra = grad(faia, ra, retain_graph=True, create_graph=True)[0]
            dfaidrb = grad(faib, rb, retain_graph=True, create_graph=True)[0]
            J_ex = 2*np.pi*(-Ura*dfaidra + Urb*dfaidrb) # 计算外力余势
       
            loss = J_dom - J_ex 
            optim2.zero_grad()
            loss.backward(retain_graph=True)


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
            Xf = dom_data(dom_num)
            
        def closure():  
            # 区域内部损失
            r = Xf
            fai = pred(r, model3)
            dfaidr = grad(fai, r, torch.ones(r.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
            dfaidrr = grad(dfaidr, r, torch.ones(r.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]   
            
            J_dom = simpson_int(0.5/E*(1/r**2*dfaidr**2 + dfaidrr**2 - 2*nu/r*dfaidr*dfaidrr)*2*np.pi*r, r)
            #J_dom = torch.mean(0.5/E*(1/r**2*dfaidr**2 + dfaidrr**2 - 2*nu/r*dfaidr*dfaidrr)*2*np.pi*r) * (b-a) # 计算余能
            # calculate the complementary external energy
            ra = torch.tensor([a],  requires_grad=True, device='cuda').unsqueeze(1)
            rb = torch.tensor([b],  requires_grad=True, device='cuda').unsqueeze(1)
            faia = pred(ra, model3)
            faib = pred(rb, model3)
            dfaidra = grad(faia, ra, retain_graph=True, create_graph=True)[0]
            dfaidrb = grad(faib, rb, retain_graph=True, create_graph=True)[0]
            J_ex = 2*np.pi*(-Ura*dfaidra + Urb*dfaidrb) # 计算外力余势
       
            loss = J_dom - J_ex 
            optim3.zero_grad()
            loss.backward(retain_graph=True)


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
            Xf = dom_data(dom_num)
            
        def closure():  
            # 区域内部损失
            r = Xf
            fai = pred(r, model4)
            dfaidr = grad(fai, r, torch.ones(r.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
            dfaidrr = grad(dfaidr, r, torch.ones(r.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]   
            
            J_dom = simpson_int(0.5/E*(1/r**2*dfaidr**2 + dfaidrr**2 - 2*nu/r*dfaidr*dfaidrr)*2*np.pi*r, r)
            #J_dom = torch.mean(0.5/E*(1/r**2*dfaidr**2 + dfaidrr**2 - 2*nu/r*dfaidr*dfaidrr)*2*np.pi*r) * (b-a) # 计算余能
            # calculate the complementary external energy
            ra = torch.tensor([a],  requires_grad=True, device='cuda').unsqueeze(1)
            rb = torch.tensor([b],  requires_grad=True, device='cuda').unsqueeze(1)
            faia = pred(ra, model4)
            faib = pred(rb, model4)
            dfaidra = grad(faia, ra, retain_graph=True, create_graph=True)[0]
            dfaidrb = grad(faib, rb, retain_graph=True, create_graph=True)[0]
            J_ex = 2*np.pi*(-Ura*dfaidra + Urb*dfaidrb) # 计算外力余势
       
            loss = J_dom - J_ex 
            optim4.zero_grad()
            loss.backward(retain_graph=True)


            if epoch%10==0:
                print(' epoch : %i, the loss : %f, loss dom : %f, loss ex : %f' % \
                      (epoch, loss.data, J_dom.data, J_ex.data))
            return loss
        optim4.step(closure)
        
    L2norm_fai_pred1, H1norm_fai_pred1 = evaluate_L2_H1(dom_num, model1)    
    L2norm_fai_pred2, H1norm_fai_pred2 = evaluate_L2_H1(dom_num, model2) 
    L2norm_fai_pred3, H1norm_fai_pred3 = evaluate_L2_H1(dom_num, model3) 
    L2norm_fai_pred4, H1norm_fai_pred4 = evaluate_L2_H1(dom_num, model4) 
    
    L2loss_array_mincomplement1.append(torch.abs((L2norm_fai_pred1 - L2norm_fai_exact)/L2norm_fai_exact))
    H1loss_array_mincomplement1.append(torch.abs((H1norm_fai_pred1 - H1norm_fai_exact)/H1norm_fai_exact))  
    L2loss_array_mincomplement2.append(torch.abs((L2norm_fai_pred2 - L2norm_fai_exact)/L2norm_fai_exact))
    H1loss_array_mincomplement2.append(torch.abs((H1norm_fai_pred2 - H1norm_fai_exact)/H1norm_fai_exact))   
    L2loss_array_mincomplement3.append(torch.abs((L2norm_fai_pred3 - L2norm_fai_exact)/L2norm_fai_exact))
    H1loss_array_mincomplement3.append(torch.abs((H1norm_fai_pred3 - H1norm_fai_exact)/H1norm_fai_exact))   
    L2loss_array_mincomplement4.append(torch.abs((L2norm_fai_pred4 - L2norm_fai_exact)/L2norm_fai_exact))
    H1loss_array_mincomplement4.append(torch.abs((H1norm_fai_pred4 - H1norm_fai_exact)/H1norm_fai_exact))   

# plot
#%%
plt.figure(figsize=(10,6))
plt.plot(np.array(dom_num_list), L2loss_array_mincomplement1, marker='*', ls = ':')
plt.plot(np.array(dom_num_list), L2loss_array_mincomplement2, linestyle='--', marker = 'v')
plt.plot(np.array(dom_num_list), L2loss_array_mincomplement3, linestyle='-.', marker = '^')
plt.plot(np.array(dom_num_list), L2loss_array_mincomplement4, marker='o')
plt.xlabel('The number of dom points', fontsize=13)
plt.ylabel(r'$\mathcal{L}_{2}^{rel}$ error', fontsize=13)
plt.yscale('log')
plt.legend(['1HL-DCEM', '2HL-DCEM','3HL-DCEM','4HL-DCEM'], loc="lower left")
plt.show()

plt.figure(figsize=(10,6))
plt.plot(np.array(dom_num_list), H1loss_array_mincomplement1, marker='*', ls = ':')
plt.plot(np.array(dom_num_list), H1loss_array_mincomplement2, linestyle='--', marker = 'v')
plt.plot(np.array(dom_num_list), H1loss_array_mincomplement3, linestyle='-.', marker = '^')
plt.plot(np.array(dom_num_list), H1loss_array_mincomplement4, marker='o')
plt.xlabel('The number of dom points', fontsize=13)
plt.ylabel(r'$\mathcal{H}_{1}^{rel}$ error', fontsize=13)
plt.yscale('log')
plt.legend(['1HL-DCEM', '2HL-DCEM','3HL-DCEM','4HL-DCEM'], loc="lower left")
plt.show()        