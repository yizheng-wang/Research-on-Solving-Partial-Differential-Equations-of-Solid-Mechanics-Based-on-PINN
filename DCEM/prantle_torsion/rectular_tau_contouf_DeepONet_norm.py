# DeepONet_energy and DCM comparison, 100 to 1000 L2 and H1 norm


import numpy as np
import torch
import matplotlib.pyplot as plt
from  torch.autograd import grad
import time
import matplotlib as mpl
from itertools import chain

torch.set_default_tensor_type(torch.DoubleTensor) # 将tensor的类型变成默认的double
torch.set_default_tensor_type(torch.cuda.DoubleTensor) # 将cuda的tensor的类型变成默认的double
mpl.rcParams['figure.dpi'] = 1000

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
setup_seed(2023)
a_list = [1.0, 1.2, 1.5,  2.0,  2.5, 3.0, 4.0, 5.0, 10.0] # 矩形板的长
a_list_train = list(np.linspace(0.5, 2, 11))  # for DeepONet training
b = 1 # 矩形板的宽
beta_list =  [0.141, 0.166, 0.196, 0.229, 0.249, 0.263, 0.281, 0.291, 0.312]
beta_list_train =  []
for a in a_list_train:
    beta_train = 0
    for j in range(100):
        labda = (2*j+1)*np.pi/b
        term = np.tanh(labda*a/2)/(2*j+1)**5
        beta_train = beta_train + term
    beta_train = 1/3*(1-192/np.pi**5*b/a*beta_train)
    beta_list_train.append(beta_train)


beta1_list = [0.208, 0.219, 0.231, 0.246, 0.258, 0.267, 0.282, 0.291, 0.312]
nepoch = 2000 # Adam iteration
N_norm = 101
nepoch_DeepONet = 1000 # iteraion number in one dataset
iter_data = 1 # the iteration number of the whole data
num = 1 # 分析num个a的长度
G = 1000.
M = 10.
N_DeepONet_t = 101 # the points in DeepONet trunk net
Train_DeepONet = 0 # whether train the DeepONet




class Trunk_net(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        DeepONet: Trunk Net
        """
        super(Trunk_net, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, H)
        self.linear6 = torch.nn.Linear(H, H)
        self.linear7 = torch.nn.Linear(H, D_out)
        
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
        Input: the coordinate of the cylinder
        Output: the basis of the DeepONet
        """

        y1 = torch.tanh(self.n*self.a1*self.linear1(x))
        y2 = torch.tanh(self.n*self.a1*self.linear2(y1))
        y3 = torch.tanh(self.n*self.a1*self.linear3(y2))
        y4 = torch.tanh(self.n*self.a1*self.linear4(y3))
        y5 = torch.tanh(self.n*self.a1*self.linear5(y4))
        y6 = torch.tanh(self.n*self.a1*self.linear5(y5))
        y = self.n*self.a1*self.linear7(y6)
        return y
  
class Branch_net(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        DeepONet: Branch net
        """
        super(Branch_net, self).__init__()
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
        The input: a, b, Mt, and G
        """

        y1 = torch.tanh(self.n*self.a1*self.linear1(x))
        y2 = torch.tanh(self.n*self.a1*self.linear2(y1))
        y3 = torch.tanh(self.n*self.a1*self.linear3(y2))
        y = self.n*self.a1*self.linear4(y3)
        return y


# =============================================================================
# DeepONet dataset: we try a/b ratio is 1.0 for the test, and the other datasets for training set
# =============================================================================
geo_data_set = []
for a in a_list_train:  
    geo_data_set.append([a, 1.0]) # a, b, Mt, G
geo_data_set = np.array(geo_data_set)
point_data_set = np.zeros([len(a_list_train), N_DeepONet_t**2, 2])
fai_data_set = np.zeros([len(a_list_train), N_DeepONet_t**2])
for i,a in enumerate(a_list_train):
    x = np.linspace(-a/2, a/2, N_DeepONet_t)
    y = np.linspace(-b/2, b/2, N_DeepONet_t) # y方向N_test个点
    x_mesh, y_mesh = np.meshgrid(x, y)
    point_data_set[i] = np.stack([x_mesh.flatten(), y_mesh.flatten()], 1)
   
    
    beta = beta_list_train[i]
    alpha = 0.0005 # maintain the consistency with DCM
    fai = 0
    for j in range(100):
        labda = (2*j+1)*np.pi/b
        term = (-1)**j/(2*j+1)**3*(1-np.cosh(labda*x_mesh)/np.cosh(labda*a/2))*np.cos(labda*y_mesh)
        fai = fai + term
    fai_exact = fai*8*G*alpha*b**2/np.pi**3
    fai_data_set[i] = fai_exact.flatten()

def pred(xy, geo):
    '''
    

    Parameters
    ----------
    xy : tensor 
        the coordinate of the input tensor.
    geo : a, b, Mt, and G

    Returns
    the prediction of fai

    '''
    x = xy[:, 0].unsqueeze(1)
    y = xy[:, 1].unsqueeze(1)
    dis = (x**2 - a**2/4)*(y**2 - b**2/4)
    particular = 0
    DeepONet = torch.sum((model_Branch_net(geo)*model_Trunk_net(xy)),1,keepdims = True)
    pred_fai = dis*DeepONet
    return pred_fai
    
# =============================================================================
#     DeepONet training
# =============================================================================
if Train_DeepONet == 1:
    model_Branch_net = Branch_net(2, 30, 20).cuda()
    model_Trunk_net = Trunk_net(2, 30, 20).cuda()
    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam(params=chain(model_Branch_net.parameters(), model_Trunk_net.parameters()), lr= 0.001)
    loss_array_DeepONet = {}
    for iter_all in range(iter_data): 
        for i,a in enumerate(a_list_train):
            loss_array_DeepONet[i] = []
            geo = torch.tensor(geo_data_set[i]).cuda()
            xy_set = torch.tensor(point_data_set[i]).cuda()
            fai_set = torch.tensor(fai_data_set[i][:, np.newaxis]).cuda()
            def pred(xy, geo):
                '''
                
            
                Parameters
                ----------
                xy : tensor 
                    the coordinate of the input tensor.
                geo : a, b, Mt, and G
            
                Returns
                the prediction of fai
            
                '''
                x = xy[:, 0].unsqueeze(1)
                y = xy[:, 1].unsqueeze(1)
                dis = (x**2 - a**2/4)*(y**2 - b**2/4)
                particular = 0
                DeepONet = torch.sum((model_Branch_net(geo)*model_Trunk_net(xy)),1,keepdims = True)
                pred_fai = dis*DeepONet
                return pred_fai
            
            
            start = time.time()
            for epoch in range(nepoch_DeepONet):
                if epoch == nepoch_DeepONet - 1:
                    end = time.time()
                    consume_time = end-start
                    print('time is %f' % consume_time)
                    
                def closure():  
                    # 区域内部损失
                    pred_fai = pred(xy_set, geo) 
               
                    loss = criterion(pred_fai, fai_set) 
                    optim.zero_grad()
                    loss.backward(retain_graph=True)
                    loss_array_DeepONet[i].append(loss.data.cpu())
        
                    if epoch%10==0:
                        print('whole iter: %i, a : %f,  epoch : %i, the loss : %f' % \
                              (iter_all, a, epoch, loss.data))
                    return loss
                optim.step(closure)
    torch.save(model_Branch_net, './branch_nn')
    torch.save(model_Trunk_net, './trunk_nn')
model_Branch_net = torch.load('./branch_nn')
model_Trunk_net = torch.load('./trunk_nn')
# =============================================================================
# 最小余能原理 DeepONet 100, 300, 500, 1000
# =============================================================================

alpha = 0.0005 # a=b=1是0.0709标准值
alpha = 0.0005# a=b=1是0.0709标准值

loss_array_mincomplement_D = {}
error_Dt_array_mincomplement_D = {}
pred_alpha_array_mincomplement_D = {}
loss_dom_array_mincomplement_D = {}
loss_ex_array_mincomplement_D = {}
pred_taumax_array_mincomplement_D = {}
L2loss_array_mincomplement_D = {}
H1loss_array_mincomplement_D = {}
tauzx_array_mincomplement_D = {}
tauzy_array_mincomplement_D = {}
taumax_array_mincomplement_D = {}
for i in range(num):

    
    a = a_list[i]
    beta = beta_list[i]
    beta1 = beta1_list[i]
    geo = torch.tensor([a, b]).cuda()
    
    # the exact fai 
    x = np.linspace(-a/2, a/2, N_norm)
    y = np.linspace(-b/2, b/2, N_norm) # y方向N_test个点
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
    def simpson_int(y, x,  nx = N_norm, ny = N_norm):
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
    def evaluate_L2_H1(N_test_norm):# 计算Dt
        xy_dom = dom_data_uniform(N_test_norm)
        fai = pred(xy_dom, geo)
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
    
    def evaluate_Dt(N_test):# 计算Dt
        xy_dom = dom_data_uniform(N_test)
        fai = pred(xy_dom, geo)
        M1 = simpson_int(2*fai, xy_dom)
        Dt = M1 / alpha
        return Dt

    def evaluate_tau(N_test):# 计算Dt和Taumax
        x = np.linspace(-a/2, a/2, N_test)
        y = np.linspace(-b/2, b/2, N_test) # y方向N_test个点
        x_mesh, y_mesh = np.meshgrid(x, y)
        xy = np.stack((x_mesh.flatten(), y_mesh.flatten()), 1)
        xy_tensor = torch.tensor(xy,  requires_grad=True, device='cuda')
       
        fai = pred(xy_tensor, geo)
        M1 = simpson_int(2*fai, xy_tensor)
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
    
    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam(params=chain(model_Branch_net.parameters(), model_Trunk_net.parameters()), lr= 0.001)
    loss_array_mincomplement_D[i] = []
    error_Dt_array_mincomplement_D[i] = []
    pred_alpha_array_mincomplement_D[i] = []
    loss_dom_array_mincomplement_D[i] = []
    loss_ex_array_mincomplement_D[i] = []
    pred_taumax_array_mincomplement_D[i] = []
    L2loss_array_mincomplement_D[i] = []
    H1loss_array_mincomplement_D[i] = []
    tauzx_array_mincomplement_D[i] = []
    tauzy_array_mincomplement_D[i] = []
    start = time.time()
    
    for epoch in range(nepoch):
        if epoch == nepoch-1:
            end = time.time()
            consume_time = end-start
            print('time is %f' % consume_time)
        Xf = dom_data_uniform(N_norm)
            
        def closure():  
            # 区域内部损失
            fai = pred(Xf, geo)  
            dfaidxy = grad(fai, Xf, torch.ones(Xf.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
            dfaidx = dfaidxy[:, 0].unsqueeze(1)
            dfaidy = dfaidxy[:, 1].unsqueeze(1)      
    
            J_dom = simpson_int((dfaidx**2 + dfaidy**2)*0.5/G, Xf) # 计算余能
            J_ex = 2*alpha*simpson_int(fai, Xf) # 计算外力余功
       
            loss = J_dom - J_ex 
            optim.zero_grad()
            loss.backward(retain_graph=True)
            loss_array_mincomplement_D[i].append(loss.data.cpu())
            loss_dom_array_mincomplement_D[i].append(J_dom.data.cpu())
            loss_ex_array_mincomplement_D[i].append(J_ex.data.cpu())
    
    
            Dt_pred = evaluate_Dt(N_norm)   
            Dt_exact = beta  * G * a * b**3 
            error_Dt_array_mincomplement_D[i].append(torch.abs(Dt_pred-Dt_exact).data.cpu())
            alpha_pred = M/Dt_pred
            pred_alpha_array_mincomplement_D[i].append(alpha_pred.data.cpu())
            x_mesh, y_mesh, tauzx_pred, tauzy_pred, taumax_pred = evaluate_tau(N_norm) # 100*100个点
            pred_taumax_array_mincomplement_D[i].append(taumax_pred)
            L2norm_fai_pred, H1norm_fai_pred = evaluate_L2_H1(N_norm)
            L2loss_array_mincomplement_D[i].append((L2norm_fai_pred - L2norm_fai_exact)/L2norm_fai_exact)
            H1loss_array_mincomplement_D[i].append((H1norm_fai_pred - H1norm_fai_exact)/H1norm_fai_exact)  
            if epoch%10==0:
                print('DCM-O epoch : %i, the loss : %f, loss dom : %f, loss ex : %f' % \
                      (epoch, loss.data, J_dom.data, J_ex.data))
            if epoch==250 or epoch ==500 or epoch==750:
                x_mesh, y_mesh, tauzx_mincomplement_D, tauzy_mincomplement_D, taumax_mincomplement_D = evaluate_tau(N_norm)
                tauzx_array_mincomplement_D[i].append(tauzx_mincomplement_D)
                tauzy_array_mincomplement_D[i].append(tauzy_mincomplement_D)
            return loss
        optim.step(closure)

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
L2loss_array_mincomplement = {}
H1loss_array_mincomplement = {}
tauzx_array_mincomplement = {}
tauzy_array_mincomplement = {}
taumax_array_mincomplement = {}
for i in range(num):

    
    a = a_list[i]
    beta = beta_list[i]
    beta1 = beta1_list[i]
    
    
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
        Dt = M1 / alpha
        return Dt

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
        taumax = np.max(np.sqrt(tauzx**2+tauzy**2)) # 秋最大值
        return x_mesh, y_mesh, tauzx, tauzy, taumax    
    
    def evaluate_L2_H1(N_test_norm):# 计算Dt
        xy_dom = dom_data_uniform(N_test_norm)
        fai = pred(xy_dom)
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
    L2loss_array_mincomplement[i] = []
    H1loss_array_mincomplement[i] = []
    tauzx_array_mincomplement[i] = []
    tauzy_array_mincomplement[i] = []
    start = time.time()
    
    for epoch in range(nepoch):
        if epoch == nepoch-1:
            end = time.time()
            consume_time = end-start
            print('time is %f' % consume_time)
        Xf = dom_data_uniform(N_norm)
            
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
    
    
            Dt_pred = evaluate_Dt(N_norm)   
            Dt_exact = beta  * G * a * b**3 
            error_Dt_array_mincomplement[i].append(torch.abs(Dt_pred-Dt_exact).data.cpu())
            alpha_pred = M/Dt_pred
            pred_alpha_array_mincomplement[i].append(alpha_pred.data.cpu())
            x_mesh, y_mesh, tauzx_pred, tauzy_pred, taumax_pred = evaluate_tau(N_norm) # 100*100个点
            pred_taumax_array_mincomplement[i].append(taumax_pred)
            L2norm_fai_pred, H1norm_fai_pred = evaluate_L2_H1(N_norm)
            L2loss_array_mincomplement[i].append((L2norm_fai_pred - L2norm_fai_exact)/L2norm_fai_exact)
            H1loss_array_mincomplement[i].append((H1norm_fai_pred - H1norm_fai_exact)/H1norm_fai_exact)  
            if epoch%10==0:
                print('DCM epoch : %i, the loss : %f, loss dom : %f, loss ex : %f' % \
                      (epoch, loss.data, J_dom.data, J_ex.data))
            if epoch==250 or epoch ==500 or epoch==750:
                x_mesh, y_mesh, tauzx_mincomplement, tauzy_mincomplement, taumax_mincomplement = evaluate_tau(N_norm)
                tauzx_array_mincomplement[i].append(tauzx_mincomplement)
                tauzy_array_mincomplement[i].append(tauzy_mincomplement)
            return loss
        optim.step(closure)
#%%
start = 20
for i in range(num):
    train_step = np.array(range(start, nepoch))
# L2 NORM
    plt.plot(train_step, L2loss_array_mincomplement[i][start:nepoch], ls = ':')
    plt.plot(train_step, L2loss_array_mincomplement_D[i][start:nepoch], linestyle='--')
    plt.xlabel('Training steps', fontsize=13)
    plt.ylabel(r'$\mathcal{L}_{2}^{rel}$ error', fontsize=13)
    plt.yscale('log')
    plt.legend(['DCEM', 'DCEM-O'])
    plt.show()
 
# H1 NORM
    plt.plot(train_step, H1loss_array_mincomplement[i][start:nepoch],ls = ':')
    plt.plot(train_step, H1loss_array_mincomplement_D[i][start:nepoch], linestyle='--')
    plt.xlabel('Training steps', fontsize=13)
    plt.ylabel(r'$\mathcal{H}_{1}^{rel}$ error', fontsize=13)
    plt.yscale('log')
    plt.legend(['DCEM', 'DCEM-O'])
    plt.show()

# CONTOUF
    a = a_list[i]
    beta = beta_list[i]
    beta1 = beta1_list[i]
    taumax_exact = M/(beta1*a*b**2)
    alpha_exact = M/(beta*G*a*b**3)
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
# =============================================================================
# 250
# =============================================================================

    h1 = plt.contourf(x_mesh, y_mesh, np.abs(tauzx_array_mincomplement[i][0]-tauzx_exact),  cmap = 'jet', levels = 100)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.colorbar(h1).ax.set_title(r'$\tau_{zx}$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    # zy最小余能误差_100
    h2 = plt.contourf(x_mesh, y_mesh, np.abs(tauzy_array_mincomplement[i][0]-tauzy_exact),  cmap = 'jet', levels = 100)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.colorbar(h2).ax.set_title(r'$\tau_{zy}$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    # zx最小余能_DeepONet误差_100
    h3 = plt.contourf(x_mesh, y_mesh, np.abs(tauzx_array_mincomplement_D[i][0]-tauzx_exact),  cmap = 'jet', levels = 100)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.colorbar(h3).ax.set_title(r'$\tau_{zx}$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    # zy最小余能_DeepONet误差_100
    h4 = plt.contourf(x_mesh, y_mesh, np.abs(tauzy_array_mincomplement_D[i][0]-tauzy_exact),  cmap = 'jet', levels = 100)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.colorbar(h4).ax.set_title(r'$\tau_{zy}$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
# =============================================================================
# 50
# =============================================================================
    # zx最小余能误差_300

    h1 = plt.contourf(x_mesh, y_mesh, np.abs(tauzx_array_mincomplement[i][1]-tauzx_exact),  cmap = 'jet', levels = 100)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.colorbar(h1).ax.set_title(r'$\tau_{zx}$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    # zy最小余能误差_100
    h2 = plt.contourf(x_mesh, y_mesh, np.abs(tauzy_array_mincomplement[i][1]-tauzy_exact),  cmap = 'jet', levels = 100)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.colorbar(h2).ax.set_title(r'$\tau_{zy}$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    # zx最小余能_DeepONet误差_100
    h3 = plt.contourf(x_mesh, y_mesh, np.abs(tauzx_array_mincomplement_D[i][1]-tauzx_exact),  cmap = 'jet', levels = 100)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.colorbar(h3).ax.set_title(r'$\tau_{zx}$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    # zy最小余能_DeepONet误差_100
    h4 = plt.contourf(x_mesh, y_mesh, np.abs(tauzy_array_mincomplement_D[i][1]-tauzy_exact),  cmap = 'jet', levels = 100)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.colorbar(h4).ax.set_title(r'$\tau_{zy}$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
# =============================================================================
# 100    
# =============================================================================
    # zx最小余能误差_500

    h1 = plt.contourf(x_mesh, y_mesh, np.abs(tauzx_array_mincomplement[i][2]-tauzx_exact),  cmap = 'jet', levels = 100)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.colorbar(h1).ax.set_title(r'$\tau_{zx}$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    # zy最小余能误差_100
    h2 = plt.contourf(x_mesh, y_mesh, np.abs(tauzy_array_mincomplement[i][2]-tauzy_exact),  cmap = 'jet', levels = 100)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.colorbar(h2).ax.set_title(r'$\tau_{zy}$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    # zx最小余能_DeepONet误差_100
    h3 = plt.contourf(x_mesh, y_mesh, np.abs(tauzx_array_mincomplement_D[i][2]-tauzx_exact),  cmap = 'jet', levels = 100)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.colorbar(h3).ax.set_title(r'$\tau_{zx}$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    # zy最小余能_DeepONet误差_100
    h4 = plt.contourf(x_mesh, y_mesh, np.abs(tauzy_array_mincomplement_D[i][2]-tauzy_exact),  cmap = 'jet', levels = 100)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.colorbar(h4).ax.set_title(r'$\tau_{zy}$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    