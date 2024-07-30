import numpy as np
import torch
import matplotlib.pyplot as plt
from  torch.autograd import grad
import time
import matplotlib as mpl
from itertools import chain
torch.set_default_tensor_type(torch.DoubleTensor) # 将tensor的类型变成默认的double
torch.set_default_tensor_type(torch.cuda.DoubleTensor) # 将cuda的tensor的类型变成默认的double

plt.rcParams['font.family'] = ['sans-serif'] # 用来正常显示负号
mpl.rcParams['figure.dpi'] = 1000

def settick():
    '''
    对刻度字体进行设置，让上标的符号显示正常
    :return: None
    '''
    ax1 = plt.gca()  # 获取当前图像的坐标轴
 
    # 更改坐标轴字体，避免出现指数为负的情况
    tick_font = mpl.font_manager.FontProperties(family='DejaVu Sans', size=7.0)
    for labelx  in ax1.get_xticklabels():
        labelx.set_fontproperties(tick_font) #设置 x轴刻度字体
    for labely in ax1.get_yticklabels():
        labely.set_fontproperties(tick_font) #设置 y轴刻度字体
    ax1.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))  # x轴刻度设置为整数
    plt.tight_layout()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
setup_seed(2024)

Po = 10.
Pi = 5.
a = 0.5
b = 1.0

nepoch = 4950

dom_num = 101
N_test = 101
E = 1000
nu = 0.5
Ura = 1/E*((1-nu)*(a**2*Pi-b**2*Po)/(b**2-a**2)*a + (1+nu)*(a**2*b**2)*(Pi-Po)/(b**2-a**2)/a)
Urb = 1/E*((1-nu)*(a**2*Pi-b**2*Po)/(b**2-a**2)*b + (1+nu)*(a**2*b**2)*(Pi-Po)/(b**2-a**2)/b)

Po_list_train = list(np.linspace(9, 11, 10))  # for DeepONet training
Pi_list_train = list(np.linspace(4, 6, 10))  # for DeepONet training
poisson_list_train = list(np.linspace(0.3, 0.49, 10))  # for DeepONet training
b_list_train = list(np.linspace(0.9, 1.1, 10))  # for DeepONet training

Time_list_DCEM = []
Time_list_DCEM_O = []
error_tol = 0.01

nepoch_DeepONet = 1000# iteraion number in one dataset
iter_data = 1 # the iteration number of the whole data
num = 1 # 分析num个a的长度

N_DeepONet_t = 101 # the points in DeepONet trunk net
Train_DeepONet = 0 # whether train the DeepONet

def simpson_int(y, x,  nx = N_test):
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
    hx = torch.abs(x[0]-x[-1])/(nx-1)
    y = y.flatten()
    result = torch.sum(weight*y)*hx/3
    return result

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
        
        self.a1 = torch.nn.Parameter(torch.Tensor([0.2]))
        
        # 有可能用到加速收敛技术
        #self.a1 = torch.Tensor([0.1]).cuda()
        # self.a2 = torch.Tensor([0.1])
        # self.a3 = torch.Tensor([0.1])
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
        
        self.a1 = torch.nn.Parameter(torch.Tensor([0.2]))
        
        # 有可能用到加速收敛技术
        # self.a1 = torch.Tensor([0.1]).cuda()
        # self.a2 = torch.Tensor([0.1])
        # self.a3 = torch.Tensor([0.1])
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
stress_data_set = []
for Pi_e in Pi_list_train:  
    for Po_e in Po_list_train:  
        for poisson_e in poisson_list_train:  
            for b_e in b_list_train:  
                Ura_e = 1/E*((1-poisson_e)*(a**2*Pi_e-b_e**2*Po_e)/(b_e**2-a**2)*a + (1+poisson_e)*(a**2*b_e**2)*(Pi_e-Po_e)/(b_e**2-a**2)/a)
                Urb_e = 1/E*((1-poisson_e)*(a**2*Pi_e-b_e**2*Po_e)/(b_e**2-a**2)*b_e + (1+poisson_e)*(a**2*b_e**2)*(Pi_e-Po_e)/(b_e**2-a**2)/b_e)
                stress_data_set.append([Pi_e, Po_e, Ura_e, Urb_e, poisson_e, b_e]) # a, b, Mt, G
stress_data_set = np.array(stress_data_set)
point_data_set = np.zeros([len(stress_data_set), N_DeepONet_t])
fai_data_set = np.zeros([len(stress_data_set), N_DeepONet_t])
for i, Pi_Po in enumerate(stress_data_set):
    Pi_e = Pi_Po[0]
    Po_e = Pi_Po[1]
    b_e = Pi_Po[5]
    
    r = np.linspace(a, b, N_DeepONet_t)
    point_data_set[i] = r
    
    fai_exact = a**2/(b_e**2-a**2)*(r**2/2-b_e**2*np.log(r))*Pi_e-b_e**2/(b_e**2-a**2)*(r**2/2-a**2*np.log(r))*Po_e
    fai_data_set[i] = fai_exact
stress_data_set = stress_data_set[:, 2:] # delete the stress and leave the dis

def pred(xy, stress):
    '''
    

    Parameters
    ----------
    xy : tensor 
        the coordinate of the input tensor.
    stress : Pi and Po

    Returns
    the prediction of fai

    '''

    pred_fai = torch.sum((model_Branch_net(stress)*model_Trunk_net(xy)),1,keepdims = True)

    return pred_fai
    
# =============================================================================
#     DeepONet training
# =============================================================================
if Train_DeepONet == 1:
    model_Branch_net = Branch_net(4, 30, 20).cuda()
    model_Trunk_net = Trunk_net(1, 30, 20).cuda()
    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam(params=chain(model_Branch_net.parameters(), model_Trunk_net.parameters()), lr= 0.001)
    loss_array_DeepONet = {}
    for iter_all in range(iter_data): 
        for i, Pi_Po in enumerate(stress_data_set):
            loss_array_DeepONet[i] = []
            stress_eve = torch.tensor(stress_data_set[i]).cuda()
            xy_set = torch.tensor(point_data_set[i][:, np.newaxis]).cuda()
            fai_set = torch.tensor(fai_data_set[i][:, np.newaxis]).cuda()
            start = time.time()
            for epoch in range(nepoch_DeepONet):
                if epoch == nepoch_DeepONet - 1:
                    end = time.time()
                    consume_time = end-start
                    print('time is %f' % consume_time)
                    
                def closure():  
                    # 区域内部损失
                    pred_fai = pred(xy_set, stress_eve) 
                    loss = criterion(pred_fai, fai_set) 
                    optim.zero_grad()
                    loss.backward(retain_graph=True)
                    loss_array_DeepONet[i].append(loss.data.cpu())
        
                    if epoch%10==0:
                        print('whole iter: %i, pi : %f, po : %f,   epoch : %i, the loss : %f' % \
                              (iter_all, Pi_Po[0], Pi_Po[1], epoch, loss.data))
                    return loss
                optim.step(closure)
    torch.save(model_Branch_net, './branch_nn')
    torch.save(model_Trunk_net, './trunk_nn')
model_Branch_net = torch.load('./branch_nn')
model_Trunk_net = torch.load('./trunk_nn')



# =============================================================================
# DeepONet_ DCM
# =============================================================================
Po = 10.
Pi = 5.
a = 0.5
b = 1.0
E = 1000
nu_para = [0.31+i*0.01 for i in range(18)]

for nu in nu_para:
    Ura = 1/E*((1-nu)*(a**2*Pi-b**2*Po)/(b**2-a**2)*a + (1+nu)*(a**2*b**2)*(Pi-Po)/(b**2-a**2)/a)
    Urb = 1/E*((1-nu)*(a**2*Pi-b**2*Po)/(b**2-a**2)*b + (1+nu)*(a**2*b**2)*(Pi-Po)/(b**2-a**2)/b)

    def pred(xy, stress):
        '''
        
    
        Parameters
        ----------
        xy : tensor 
            the coordinate of the input tensor.
        stress : Pi and Po
    
        Returns
        the prediction of fai
    
        '''
    
        pred_fai = torch.sum((model_Branch_net(stress)*model_Trunk_net(xy)),1,keepdims = True)
    
        return pred_fai
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
    
    
    
    def evaluate_sigma(N_test):# calculate the prediction of the stress rr and theta
    # 分析sigma应力，输入坐标是极坐标r和theta
        r = np.linspace(a, b, N_test)
        theta = np.linspace(0, 2*np.pi, N_test) # y方向N_test个点
        r_mesh, theta_mesh = np.meshgrid(r, theta)
        xy = np.stack((r_mesh.flatten(), theta_mesh.flatten()), 1)
        X_test = torch.tensor(xy,  requires_grad=True, device='cuda')
        r = X_test[:, 0].unsqueeze(1)
        # 将r输入到pred中，输出应力函数
        fai = pred(r, stress)
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
    
    def evaluate_sigma_line(N_test):# output the prediction of the stress rr and theta along radius in direction of r without theta
        r_numpy = np.linspace(a, b, N_test) # generate the coordinate of r by numpy 
        r = torch.tensor(r_numpy,  requires_grad=True, device='cuda').unsqueeze(1) # change the type of array to tensor used to be AD
        fai = pred(r, stress) # Input r to the pred function to get the necessary predition stress function
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
    # learning the homogenous network


    optim = torch.optim.Adam(params=chain(model_Branch_net.parameters(), model_Trunk_net.parameters()), lr= 0.001)
    step_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=2000, gamma=0.1)
    loss_array = []
    loss_dom_array = []
    loss_ex_array = []
    error_sigma_rr_array_D = []
    error_sigma_theta_array_D = []
    error_dis_r_array_D = []
    L2loss_array_mincomplementD = []
    H1loss_array_mincomplementD = []
    
    nepoch = int(nepoch)
    
    stress = torch.tensor([Ura, Urb, nu, b]).cuda()
    Xf = dom_data(dom_num)
    
    
    L2_error = 1.0
    start = time.time()
    epoch = 1
# r_numpy, predline_sigma_rr_0D, predline_sigma_theta_0D, predline_dis_r_0D = evaluate_sigma_line(N_test) # data-driven
    while L2_error >= error_tol:
        epoch = epoch +1
        def closure():  
            global L2_error
            # 区域内部损失
            r = Xf
            fai = pred(r, stress)  
            dfaidr = grad(fai, r, torch.ones(r.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
            dfaidrr = grad(dfaidr, r, torch.ones(r.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]   
    
            J_dom = simpson_int(0.5/E*(1/r**2*dfaidr**2 + dfaidrr**2 - 2*nu/r*dfaidr*dfaidrr)*2*np.pi*r, r)
            # calculate the complementary external energy
            ra = torch.tensor([a],  requires_grad=True, device='cuda').unsqueeze(1)
            rb = torch.tensor([b],  requires_grad=True, device='cuda').unsqueeze(1)
            faia = pred(ra, stress)  
            faib = pred(rb, stress)  
            dfaidra = grad(faia, ra, retain_graph=True, create_graph=True)[0]
            dfaidrb = grad(faib, rb, retain_graph=True, create_graph=True)[0]
            J_ex = 2*np.pi*(-Ura*dfaidra + Urb*dfaidrb) # 计算外力余势
       
            loss = J_dom - J_ex 
            optim.zero_grad()
            loss.backward(retain_graph=True)
            loss_array.append(loss.data.cpu())
            loss_dom_array.append(J_dom.data.cpu())
            loss_ex_array.append(J_ex.data.cpu())
            r_numpy, pred_sigma_rr, pred_sigma_theta, pred_dis_r = evaluate_sigma_line(N_test)    
            exact_sigma_rr = a**2/(b**2-a**2)*(1-b**2/r_numpy**2)*Pi - b**2/(b**2-a**2)*(1-a**2/r_numpy**2)*Po
            exact_sigma_theta = a**2/(b**2-a**2)*(1+b**2/r_numpy**2)*Pi - b**2/(b**2-a**2)*(1+a**2/r_numpy**2)*Po     
            exact_dis_r = 1/E*((1-nu)*(a**2*Pi-b**2*Po)/(b**2-a**2)*r_numpy + (1+nu)*(a**2*b**2)*(Pi-Po)/(b**2-a**2)/r_numpy)
            
            # for L2 H1 norm
            fai_exact = a**2/(b**2-a**2)*(r_numpy**2/2-b**2*np.log(r_numpy))*Pi-b**2/(b**2-a**2)*(r_numpy**2/2-a**2*np.log(r_numpy))*Po
            L2norm_fai_exact = np.sqrt(np.trapz(fai_exact**2, dx=(r_numpy[-1]-r_numpy[-2])))
            fai_r_exact = a**2/(b**2-a**2)*(r_numpy-b**2/r_numpy)*Pi-b**2/(b**2-a**2)*(r_numpy-a**2/r_numpy)*Po
            H1norm_fai_exact = np.sqrt(np.trapz(fai_r_exact**2, dx=(r_numpy[-1]-r_numpy[-2])))
    
            c = fai_exact[0] - fai[0]
            dfaidr = grad(fai, r, torch.ones(r.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0] 
            L2norm_fai_pred = np.sqrt(np.trapz((fai+c).flatten().cpu().data**2, dx=(r[-1].cpu().data-r[-2].cpu().data)))
            H1norm_fai_pred = np.sqrt(np.trapz(dfaidr.flatten().cpu().data**2, dx=(r[-1].cpu().data-r[-2].cpu().data)))
    
            L2_error = torch.abs((L2norm_fai_pred - L2norm_fai_exact)/L2norm_fai_exact).numpy()
            L2loss_array_mincomplementD.append(L2_error)
            H1loss_array_mincomplementD.append(torch.abs((H1norm_fai_pred - H1norm_fai_exact)/H1norm_fai_exact))  
            
            L2_rr_error = np.linalg.norm(pred_sigma_rr.flatten() - exact_sigma_rr.flatten())/np.linalg.norm(exact_sigma_rr.flatten())
            L2_theta_error = np.linalg.norm(pred_sigma_theta.flatten() - exact_sigma_theta.flatten())/np.linalg.norm(exact_sigma_theta.flatten())
            L2_dis_r_error = np.linalg.norm(pred_dis_r.flatten() - exact_dis_r.flatten())/np.linalg.norm(exact_dis_r.flatten())
            
            error_sigma_rr_array_D.append(L2_rr_error)
            error_sigma_theta_array_D.append(L2_theta_error)
            error_dis_r_array_D.append(L2_dis_r_error)
            if epoch%10==0:
                print(' epoch : %i, the loss : %f, loss dom : %f, loss ex : %f, error : %f' % \
                      (epoch, loss.data, J_dom.data, J_ex.data, L2_error))
            return loss
        optim.step(closure)
        step_scheduler
    end = time.time()
    consume_time = end-start
    print('time is %f' % consume_time)  
    Time_list_DCEM_O.append(consume_time)



    # =============================================================================
    # DCM
    # =============================================================================
      
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
    
        pred_fai = model(xy)
        return pred_fai
    
    def evaluate_sigma(N_test):# calculate the prediction of the stress rr and theta
    # 分析sigma应力，输入坐标是极坐标r和theta
        r = np.linspace(a, b, N_test)
        theta = np.linspace(0, 2*np.pi, N_test) # y方向N_test个点
        r_mesh, theta_mesh = np.meshgrid(r, theta)
        xy = np.stack((r_mesh.flatten(), theta_mesh.flatten()), 1)
        X_test = torch.tensor(xy,  requires_grad=True, device='cuda')
        r = X_test[:, 0].unsqueeze(1)
        # 将r输入到pred中，输出应力函数
        fai = pred(r)
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
    
    def evaluate_sigma_line(N_test):# output the prediction of the stress rr and theta along radius in direction of r without theta
        r_numpy = np.linspace(a, b, N_test) # generate the coordinate of r by numpy 
        r = torch.tensor(r_numpy,  requires_grad=True, device='cuda').unsqueeze(1) # change the type of array to tensor used to be AD
        fai = pred(r) # Input r to the pred function to get the necessary predition stress function
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
    
    model = FNN(1, 20, 1).cuda() # input: r; output: Airy stress function
    optim = torch.optim.Adam(model.parameters(), lr= 0.001)
    step_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=2000, gamma=0.1)
    loss_array = []
    loss_dom_array = []
    loss_ex_array = []
    error_sigma_rr_array = []
    error_sigma_theta_array = []
    error_dis_r_array = []
    L2loss_array_mincomplement = []
    H1loss_array_mincomplement = []  
    nepoch = int(nepoch)
    start = time.time()
    Xf = dom_data(dom_num)
    
    L2_error = 1.0
    epoch = 1
    start = time.time()
# r_numpy, predline_sigma_rr_0D, predline_sigma_theta_0D, predline_dis_r_0D = evaluate_sigma_line(N_test) # data-driven
    while L2_error >= error_tol:
        epoch = epoch + 1
        def closure():  
            global L2_error
            # 区域内部损失
            r = Xf
            fai = pred(r)  
            dfaidr = grad(fai, r, torch.ones(r.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
            dfaidrr = grad(dfaidr, r, torch.ones(r.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]   
    
            J_dom = simpson_int(0.5/E*(1/r**2*dfaidr**2 + dfaidrr**2 - 2*nu/r*dfaidr*dfaidrr)*2*np.pi*r, r)
            # calculate the complementary external energy
            ra = torch.tensor([a],  requires_grad=True, device='cuda').unsqueeze(1)
            rb = torch.tensor([b],  requires_grad=True, device='cuda').unsqueeze(1)
            faia = pred(ra)
            faib = pred(rb)
            dfaidra = grad(faia, ra, retain_graph=True, create_graph=True)[0]
            dfaidrb = grad(faib, rb, retain_graph=True, create_graph=True)[0]
            J_ex = 2*np.pi*(-Ura*dfaidra + Urb*dfaidrb) # 计算外力余势
       
            loss = J_dom - J_ex 
            optim.zero_grad()
            loss.backward(retain_graph=True)
            loss_array.append(loss.data.cpu())
            loss_dom_array.append(J_dom.data.cpu())
            loss_ex_array.append(J_ex.data.cpu())
            if epoch == nepoch-1:
                print()
            r_numpy, pred_sigma_rr, pred_sigma_theta, pred_dis_r = evaluate_sigma_line(N_test)    
            exact_sigma_rr = a**2/(b**2-a**2)*(1-b**2/r_numpy**2)*Pi - b**2/(b**2-a**2)*(1-a**2/r_numpy**2)*Po
            exact_sigma_theta = a**2/(b**2-a**2)*(1+b**2/r_numpy**2)*Pi - b**2/(b**2-a**2)*(1+a**2/r_numpy**2)*Po     
            exact_dis_r = 1/E*((1-nu)*(a**2*Pi-b**2*Po)/(b**2-a**2)*r_numpy + (1+nu)*(a**2*b**2)*(Pi-Po)/(b**2-a**2)/r_numpy)
    
            # for L2 H1 norm
            fai_exact = a**2/(b**2-a**2)*(r_numpy**2/2-b**2*np.log(r_numpy))*Pi-b**2/(b**2-a**2)*(r_numpy**2/2-a**2*np.log(r_numpy))*Po
            L2norm_fai_exact = np.sqrt(np.trapz(fai_exact**2, dx=(r_numpy[-1]-r_numpy[-2])))
            fai_r_exact = a**2/(b**2-a**2)*(r_numpy-b**2/r_numpy)*Pi-b**2/(b**2-a**2)*(r_numpy-a**2/r_numpy)*Po
            H1norm_fai_exact = np.sqrt(np.trapz(fai_r_exact**2, dx=(r_numpy[-1]-r_numpy[-2])))
    
            c = fai_exact[0] - fai[0]
            dfaidr = grad(fai, r, torch.ones(r.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0] 
            L2norm_fai_pred = np.sqrt(np.trapz((fai+c).flatten().cpu().data**2, dx=(r[-1].cpu().data-r[-2].cpu().data)))
            H1norm_fai_pred = np.sqrt(np.trapz(dfaidr.flatten().cpu().data**2, dx=(r[-1].cpu().data-r[-2].cpu().data)))
    
            L2_error = torch.abs((L2norm_fai_pred - L2norm_fai_exact)/L2norm_fai_exact)
            L2loss_array_mincomplement.append(L2_error)
            H1loss_array_mincomplement.append(torch.abs((H1norm_fai_pred - H1norm_fai_exact)/H1norm_fai_exact))  
            
            L2_rr_error = np.linalg.norm(pred_sigma_rr.flatten() - exact_sigma_rr.flatten())/np.linalg.norm(exact_sigma_rr.flatten())
            L2_theta_error = np.linalg.norm(pred_sigma_theta.flatten() - exact_sigma_theta.flatten())/np.linalg.norm(exact_sigma_theta.flatten())
            L2_dis_r_error = np.linalg.norm(pred_dis_r.flatten() - exact_dis_r.flatten())/np.linalg.norm(exact_dis_r.flatten())
            
            error_sigma_rr_array.append(L2_rr_error)
            error_sigma_theta_array.append(L2_theta_error)
            error_dis_r_array.append(L2_dis_r_error)
            if epoch%10==0:
                print(' epoch : %i, the loss : %f, loss dom : %f, loss ex : %f, error : %f' % \
                      (epoch, loss.data, J_dom.data, J_ex.data, L2_error.data))
            return loss
        optim.step(closure)
        step_scheduler.step()
    end = time.time()
    consume_time = end-start
    print('time is %f' % consume_time)  
    Time_list_DCEM.append(consume_time)
#%%
plt.plot(nu_para, Time_list_DCEM, 'b--', label='DCEM')
plt.scatter(nu_para, Time_list_DCEM, color='blue')
# 绘制DCEM-O的实线图
plt.plot(nu_para, Time_list_DCEM_O, 'r-', label='DCEM-O')
plt.scatter(nu_para, Time_list_DCEM_O, color='red')

# 添加标题和标签
plt.title('Time Comparison')
plt.xlabel('Poisson')
plt.ylabel('Time (s)')
# 添加图例
plt.legend()
plt.savefig('pic/para_nu_DCEM-O_DCEM.pdf', dpi = 300)
# 显示图形
plt.show()
