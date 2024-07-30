# 转化为一维问题处理，用罚函数
import numpy as np
import torch
import matplotlib.pyplot as plt
from  torch.autograd import grad
import time
import matplotlib as mpl
import numpy as np
from itertools import chain


torch.set_default_tensor_type(torch.DoubleTensor) # 将tensor的类型变成默认的double
torch.set_default_tensor_type(torch.cuda.DoubleTensor) # 将cuda的tensor的类型变成默认的double

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(2023)

order = 2

dom_num = 1001
N_test = 1001
bound_num = 5
E = 1000
nu = 0.3
alpha = np.pi
q = 5 # 均布压力载荷
alpha_q = torch.tensor([alpha, q]).cuda()
l = np.sqrt(3)


G = E/2/(1+nu)
c = q/(np.tan(alpha)-alpha)/2
factor = (4/alpha**2)**order
beta1 = 1
beta2 = 1
beta3 = 1
beta4 = 1
beta5 = 1


alpha_list_train = list(np.linspace(np.pi*0.9, np.pi*1.1, 10))  # for DeepONet training
q_list_train = list(np.linspace(4.1, 6, 11))  # for DeepONet training
nepoch1 = 10 # Adam iteration
nepoch2 = 20
nepoch3 = 20
nepoch = 2500

nepoch_DeepONet = 1000# iteraion number in one dataset
iter_data = 1 # the iteration number of the whole data
num = 1 # 分析num个a的长度

N_DeepONet_t = 100 # the points in DeepONet trunk net
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
        
        self.a1 = torch.nn.Parameter(torch.Tensor([0.2]))
        
        # 有可能用到加速收敛技术
        #self.a1 = torch.Tensor([0.1]).cuda()
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
        
        self.a1 = torch.nn.Parameter(torch.Tensor([0.2]))
        
        # 有可能用到加速收敛技术
        #self.a1 = torch.Tensor([0.1]).cuda()
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
if Train_DeepONet == 1:
    alpha_q_data_set = []
    for alpha_e in alpha_list_train:  
        for q_e in q_list_train:  
            alpha_q_data_set.append([alpha_e, q_e]) # a, b, Mt, G
    alpha_q_data_set = np.array(alpha_q_data_set)
    point_data_set = np.zeros([len(alpha_q_data_set), N_DeepONet_t])
    fai_data_set = np.zeros([len(alpha_q_data_set), N_DeepONet_t])
    for i, alpha_q in enumerate(alpha_q_data_set):
        theta = np.linspace(0, alpha_q[0], N_DeepONet_t)
        point_data_set[i] = theta
        c = alpha_q[1]/(np.tan(alpha_q[0])-alpha_q[0])/2
        fai_exact = c*(alpha_q[0] - theta + np.sin(theta)*np.cos(theta) - np.cos(theta)**2*np.tan(alpha_q[0]))
        fai_data_set[i] = fai_exact

def pred(theta, alpha_q):
    '''
    

    Parameters
    ----------
    xy : tensor 
        the coordinate of the input tensor, theta
    qlpha_q : alpha and q

    Returns
    the prediction of fai

    '''
    alpha = alpha_q[0]
    q = alpha_q[1]
    NNp = -q/alpha**3*theta**3 + 3*q/2/alpha**2*theta**2 - q/2
    NNd = (theta*(alpha-theta))**order*factor    
    NNg = torch.sum((model_Branch_net(alpha_q)*model_Trunk_net(theta)),1,keepdims = True)
    pred_fai = (NNp + NNd * NNg)    
    return pred_fai
    
def evaluate_deeponet(N_test, material):# calculate the prediction of the stress rr and theta
    # 生成theta数据点
    theta_numpy = np.linspace(0 , alpha, N_test)
    theta = np.stack([theta_numpy], 1) # 将mesh的点flatten
    theta = torch.tensor(theta,  requires_grad=True, device='cuda')

    f = pred(theta, material).data.cpu().numpy()


    return  theta_numpy, f
# =============================================================================
#     DeepONet training
# =============================================================================
if Train_DeepONet == 1:
    model_Branch_net = Branch_net(2, 30, 20).cuda()
    model_Trunk_net = Trunk_net(1, 30, 20).cuda()
    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam(params=chain(model_Branch_net.parameters(), model_Trunk_net.parameters()), lr= 0.001)
    loss_array_DeepONet = {}
    for iter_all in range(iter_data): 
        for i, alpha_q in enumerate(alpha_q_data_set):
            loss_array_DeepONet[i] = []
            alpha_q_eve = torch.tensor(alpha_q_data_set[i]).cuda()
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
                    pred_fai = pred(xy_set, alpha_q_eve) 
                    loss = criterion(pred_fai, fai_set) 
                    optim.zero_grad()
                    loss.backward(retain_graph=True)
                    loss_array_DeepONet[i].append(loss.data.cpu())
        
                    if epoch%10==0:
                        print('whole iter: %i, alpha : %f, q : %f,   epoch : %i, the loss : %f' % \
                              (iter_all, alpha_q[0], alpha_q[1], epoch, loss.data))
                    return loss
                optim.step(closure)
            theta_numpy, f_pred = evaluate_deeponet(N_test, alpha_q_eve)
            f_exact = alpha_q[1]/2/(np.tan(alpha_q[0])-alpha_q[0])*(alpha_q[0] - theta_numpy + np.sin(theta_numpy)*np.cos(theta_numpy)-np.cos(theta_numpy)**2*np.tan(alpha_q[0]))
            plt.plot(theta_numpy, f_exact)
            plt.plot(theta_numpy, f_pred)
            plt.legend(['exact', 'pred'])
            plt.show()
    torch.save(model_Branch_net, './branch_nn')
    torch.save(model_Trunk_net, './trunk_nn')
model_Branch_net = torch.load('./branch_nn')
model_Trunk_net = torch.load('./trunk_nn')

model_Branch_net_data = torch.load('./branch_nn')
model_Trunk_net_data = torch.load('./trunk_nn')

def pred_data(theta, alpha_q):
    '''
    

    Parameters
    ----------
    xy : tensor 
        the coordinate of the input tensor, theta
    qlpha_q : alpha and q

    Returns
    the prediction of fai

    '''
    alpha = alpha_q[0]
    q = alpha_q[1]
    NNp = -q/alpha**3*theta**3 + 3*q/2/alpha**2*theta**2 - q/2
    NNd = (theta*(alpha-theta))**order*factor    
    NNg = torch.sum((model_Branch_net_data(alpha_q)*model_Trunk_net_data(theta)),1,keepdims = True)
    pred_fai = (NNp + NNd * NNg)    
    return pred_fai


def evaluate_sigma_data(N_test):# calculate the prediction of the stress rr and theta
    # 生成theta数据点
    theta_numpy = np.linspace(0 , alpha, N_test)
    theta = np.stack([theta_numpy], 1) # 将mesh的点flatten
    theta = torch.tensor(theta,  requires_grad=True, device='cuda')

    f = pred_data(theta, alpha_q)
    dfdtheta = grad(f, theta, torch.ones(theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    dfdthetadtheta = grad(dfdtheta, theta, torch.ones(theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    sigma_r = dfdthetadtheta + 2 * f
    sigma_theta = 2 * f
    sigma_rtheta = -dfdtheta

    pred_sigma_r = sigma_r.data.cpu().numpy() # change the type of tensor to the array for the use of plotting
    pred_sigma_theta = sigma_theta.data.cpu().numpy()
    pred_sigma_rtheta = sigma_rtheta.data.cpu().numpy()

    return  theta_numpy, pred_sigma_r, pred_sigma_theta, pred_sigma_rtheta

# =============================================================================
# 最小余能原理 DeepONet 100, 300, 500, 1000
# =============================================================================

mpl.rcParams['figure.dpi'] = 1000

    



def dom_data(Nf):
    '''
    生成内部点，极坐标形式生成,我转化为一维问题
    '''
    

    theta_dom = np.linspace(0 , alpha, Nf)
    polar_dom = np.stack([theta_dom], 1) # 将mesh的点flatten
    polar_dom = torch.tensor(polar_dom,  requires_grad=True, device='cuda')
    return polar_dom

def boundary_data(Nf):
    '''
    生成边界点，极坐标形式生成
    '''
    
    # r = (b-a)*np.random.rand(Nf)+a
    # theta = 2 * np.pi * np.random.rand(Nf) # 角度0到2*pi
    
    theta_up = np.ones(Nf) * 0
    polar_up = np.stack([theta_up], 1) # 将mesh的点flatten
    polar_up = torch.tensor(polar_up,  requires_grad=True, device='cuda')
    
    theta_down = np.ones(Nf) * alpha
    polar_down = np.stack([theta_down], 1) # 将mesh的点flatten
    polar_down = torch.tensor(polar_down,  requires_grad=True, device='cuda')
    
    theta_mid = np.ones(Nf) * alpha/2
    polar_mid = np.stack([theta_mid], 1) # 将mesh的点flatten
    polar_mid = torch.tensor(polar_mid,  requires_grad=True, device='cuda')
    
    return polar_up, polar_down,  polar_mid

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
    hx = alpha/(nx-1)
    y = y.flatten()
    result = torch.sum(weight*y)*hx/3
    return result

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
        #self.a1 = torch.Tensor([0.1]).cuda()
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



def evaluate_sigma(N_test):# calculate the prediction of the stress rr and theta
    # 生成theta数据点
    theta_numpy = np.linspace(0 , alpha, N_test)
    theta = np.stack([theta_numpy], 1) # 将mesh的点flatten
    theta = torch.tensor(theta,  requires_grad=True, device='cuda')

    f = pred(theta, alpha_q)
    dfdtheta = grad(f, theta, torch.ones(theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    dfdthetadtheta = grad(dfdtheta, theta, torch.ones(theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    sigma_r = dfdthetadtheta + 2 * f
    sigma_theta = 2 * f
    sigma_rtheta = -dfdtheta

    pred_sigma_r = sigma_r.data.cpu().numpy() # change the type of tensor to the array for the use of plotting
    pred_sigma_theta = sigma_theta.data.cpu().numpy()
    pred_sigma_rtheta = sigma_rtheta.data.cpu().numpy()

    return  theta_numpy, pred_sigma_r, pred_sigma_theta, pred_sigma_rtheta


def evaluate(N_test):# calculate the prediction of the stress rr and theta
    # 生成theta数据点
    theta_numpy = np.linspace(0 , alpha, N_test)
    theta = np.stack([theta_numpy], 1) # 将mesh的点flatten
    theta = torch.tensor(theta,  requires_grad=True, device='cuda')

    f = pred(theta, alpha_q).data.cpu().numpy()


    return  theta_numpy, f

# learning the homogenous network

alpha_q = torch.tensor([alpha, q]).cuda()

loss_array = []
loss_dom_array = []
loss_b1_array = []
loss_b2_array = []
loss_b3_array = []
loss_b4_array = []
loss_b5_array = []
loss_ex_array = []
error_sigma_r_array_D = []
error_sigma_theta_array_D = []
error_sigma_rtheta_array_D = []
nepoch = int(nepoch)
start = time.time()
criterion = torch.nn.MSELoss()
optim = torch.optim.Adam(params=chain(model_Branch_net.parameters(), model_Trunk_net.parameters()), lr= 0.001)
step_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1000, gamma=0.1)

alpha_tensor = torch.tensor(alpha, device='cuda')  
theta = dom_data(dom_num)

theta_up, theta_down, theta_mid = boundary_data(bound_num)
f_up_label = (torch.ones(bound_num)*(-0.5*q)).unsqueeze(1).cuda()
dfdtheta_up_label = torch.zeros(bound_num).unsqueeze(1).cuda()
f_down_label = torch.zeros(bound_num).unsqueeze(1).cuda()
dfdtheta_down_label = torch.zeros(bound_num).unsqueeze(1).cuda()
f_mid_label = (torch.ones(bound_num)*(q/2/(torch.tan(alpha_tensor)-alpha_tensor)*(alpha_tensor/2+0.5*torch.sin(alpha_tensor)-torch.cos(alpha_tensor/2)**2*torch.tan(alpha_tensor)))).unsqueeze(1).cuda()


# =============================================================================
# 10
# =============================================================================

for epoch in range(nepoch1):
    if epoch == nepoch1-1:
        end = time.time()
        consume_time = end-start
        print('time is %f' % consume_time)        
    def closure():  
        # 区域内部损失
        f = pred(theta, alpha_q)
        dfdtheta = grad(f, theta, torch.ones(theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        dfdthetadtheta = grad(dfdtheta, theta, torch.ones(theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]


        comple_density = 0.5*dfdthetadtheta**2-2*dfdtheta**2
        int_theta = comple_density
        J_dom = simpson_int(int_theta, theta)
        
        # obtain the penalty term of the boundary condition on theta:0 and theta:alpha
        f_up = pred(theta_up, alpha_q)
        dfdtheta_up = grad(f_up, theta_up, torch.ones(theta_up.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]        
        
        f_down = pred(theta_down, alpha_q)
        dfdtheta_down = grad(f_down, theta_down, torch.ones(theta_down.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]             
        
        f_mid = pred(theta_mid, alpha_q)
        # 上区域的损失用MSE 
        loss_up = criterion(f_up, f_up_label) 
        loss_up_d = criterion(dfdtheta_up, dfdtheta_up_label) 
        # 下区域的损失用MSE
        loss_down = criterion(f_down, f_down_label) 
        loss_down_d = criterion(dfdtheta_down, dfdtheta_down_label)  
        # 中区域的损失用MSE
        loss_mid = criterion(f_mid, f_mid_label) 
        loss = J_dom + beta1 * loss_up + beta2 * loss_up_d + beta3 * loss_down + beta4 * loss_down_d + beta5 * loss_mid # 因为位移边界条件是0，所以没有外力余势
        optim.zero_grad()
        loss.backward(retain_graph=True)
        loss_array.append(loss.data.cpu())
        loss_dom_array.append(J_dom.data.cpu())
        
        loss_b1_array.append(loss_up.data.cpu())
        loss_b2_array.append(loss_up_d.data.cpu())
        loss_b3_array.append(loss_down.data.cpu())
        loss_b4_array.append(loss_down_d.data.cpu())
        loss_b5_array.append(loss_mid.data.cpu())
       
        theta_dom, pred_sigma_r, pred_sigma_theta, pred_sigma_rtheta = evaluate_sigma(N_test)
        
        exact_sigma_r = 2*c*((alpha-theta_dom)-np.sin(theta_dom)**2*np.tan(alpha)-np.sin(theta_dom)*np.cos(theta_dom))
        exact_sigma_theta = 2*c*((alpha-theta_dom) + np.sin(theta_dom)*np.cos(theta_dom)-np.cos(theta_dom)**2*np.tan(alpha))   
        exact_sigma_rtheta = -c*(-1 + np.cos(2*theta_dom) + np.sin(2*theta_dom)*np.tan(alpha))
        
        L2_r_error = np.linalg.norm(pred_sigma_r.flatten() - exact_sigma_r.flatten())/np.linalg.norm(exact_sigma_r.flatten())
        L2_theta_error = np.linalg.norm(pred_sigma_theta.flatten() - exact_sigma_theta.flatten())/np.linalg.norm(exact_sigma_theta.flatten())
        L2_rtheta_error = np.linalg.norm(pred_sigma_rtheta.flatten() - exact_sigma_rtheta.flatten())/np.linalg.norm(exact_sigma_rtheta.flatten())
        error_sigma_r_array_D.append(L2_r_error)
        error_sigma_theta_array_D.append(L2_theta_error)
        error_sigma_rtheta_array_D.append(L2_rtheta_error)
        if epoch % 100 == 0:          
            theta_numpy, fai_pred =  evaluate(N_test)
            fai_exact = q/2/(np.tan(alpha)-alpha)*(alpha - theta_numpy + np.sin(theta_numpy)*np.cos(theta_numpy)-np.cos(theta_numpy)**2*np.tan(alpha))
            plt.plot(theta_numpy, fai_exact)
            plt.plot(theta_numpy, fai_pred)
            plt.legend(['exact', 'pred'])
            plt.show()

        if epoch % 1000 == 0:
            print(L2_r_error)
            print(L2_theta_error)
            print(L2_rtheta_error)
        
        if epoch%10==0:
            print(' epoch : %i, the loss : %f, loss dom : %f, loss up : %f, loss up d : %f, loss down : %f, loss down d : %f, loss mid : %f' % \
                  (epoch, loss.data, J_dom.data, loss_up.data, loss_up_d.data, loss_down.data, loss_down_d.data, loss_mid.data))

        return loss
    optim.step(closure)
    step_scheduler.step()
theta_dom, pred_sigma_r_10D, pred_sigma_theta_10D, pred_sigma_rtheta_10D = evaluate_sigma(N_test)



# =============================================================================
# 30
# =============================================================================

for epoch in range(nepoch2):
    if epoch == nepoch2-1:
        end = time.time()
        consume_time = end-start
        print('time is %f' % consume_time)
    def closure():  
        # 区域内部损失
        f = pred(theta, alpha_q)
        dfdtheta = grad(f, theta, torch.ones(theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        dfdthetadtheta = grad(dfdtheta, theta, torch.ones(theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]

        comple_density = 0.5*dfdthetadtheta**2-2*dfdtheta**2
        int_theta = comple_density
        J_dom = simpson_int(int_theta, theta)
        
        # obtain the penalty term of the boundary condition on theta:0 and theta:alpha
        f_up = pred(theta_up, alpha_q)
        dfdtheta_up = grad(f_up, theta_up, torch.ones(theta_up.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]        
        
        f_down = pred(theta_down, alpha_q)
        dfdtheta_down = grad(f_down, theta_down, torch.ones(theta_down.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]             
        
        f_mid = pred(theta_mid, alpha_q)
        # 上区域的损失用MSE 
        loss_up = criterion(f_up, f_up_label) 
        loss_up_d = criterion(dfdtheta_up, dfdtheta_up_label) 
        # 下区域的损失用MSE
        loss_down = criterion(f_down, f_down_label) 
        loss_down_d = criterion(dfdtheta_down, dfdtheta_down_label)  
        # 中区域的损失用MSE
        loss_mid = criterion(f_mid, f_mid_label) 
        loss = J_dom + beta1 * loss_up + beta2 * loss_up_d + beta3 * loss_down + beta4 * loss_down_d + beta5 * loss_mid # 因为位移边界条件是0，所以没有外力余势
        optim.zero_grad()
        loss.backward(retain_graph=True)
        loss_array.append(loss.data.cpu())
        loss_dom_array.append(J_dom.data.cpu())
        
        loss_b1_array.append(loss_up.data.cpu())
        loss_b2_array.append(loss_up_d.data.cpu())
        loss_b3_array.append(loss_down.data.cpu())
        loss_b4_array.append(loss_down_d.data.cpu())
        loss_b5_array.append(loss_mid.data.cpu())
       
        theta_dom, pred_sigma_r, pred_sigma_theta, pred_sigma_rtheta = evaluate_sigma(N_test)
        
        exact_sigma_r = 2*c*((alpha-theta_dom)-np.sin(theta_dom)**2*np.tan(alpha)-np.sin(theta_dom)*np.cos(theta_dom))
        exact_sigma_theta = 2*c*((alpha-theta_dom) + np.sin(theta_dom)*np.cos(theta_dom)-np.cos(theta_dom)**2*np.tan(alpha))   
        exact_sigma_rtheta = -c*(-1 + np.cos(2*theta_dom) + np.sin(2*theta_dom)*np.tan(alpha))
        
        L2_r_error = np.linalg.norm(pred_sigma_r.flatten() - exact_sigma_r.flatten())/np.linalg.norm(exact_sigma_r.flatten())
        L2_theta_error = np.linalg.norm(pred_sigma_theta.flatten() - exact_sigma_theta.flatten())/np.linalg.norm(exact_sigma_theta.flatten())
        L2_rtheta_error = np.linalg.norm(pred_sigma_rtheta.flatten() - exact_sigma_rtheta.flatten())/np.linalg.norm(exact_sigma_rtheta.flatten())
        error_sigma_r_array_D.append(L2_r_error)
        error_sigma_theta_array_D.append(L2_theta_error)
        error_sigma_rtheta_array_D.append(L2_rtheta_error)
        if epoch % 1000 == 0:
            print(L2_r_error)
            print(L2_theta_error)
            print(L2_rtheta_error)
        
        if epoch%10==0:
            print(' epoch : %i, the loss : %f, loss dom : %f, loss up : %f, loss up d : %f, loss down : %f, loss down d : %f, loss mid : %f' % \
                  (epoch, loss.data, J_dom.data, loss_up.data, loss_up_d.data, loss_down.data, loss_down_d.data, loss_mid.data))

        return loss
    optim.step(closure)
    step_scheduler.step()
theta_dom, pred_sigma_r_30D, pred_sigma_theta_30D, pred_sigma_rtheta_30D = evaluate_sigma(N_test)

# =============================================================================
# 50
# =============================================================================

for epoch in range(nepoch3):
    if epoch == nepoch3-1:
        end = time.time()
        consume_time = end-start
        print('time is %f' % consume_time)
    def closure():  
        # 区域内部损失
        f = pred(theta, alpha_q)
        dfdtheta = grad(f, theta, torch.ones(theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        dfdthetadtheta = grad(dfdtheta, theta, torch.ones(theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]

        comple_density = 0.5*dfdthetadtheta**2-2*dfdtheta**2
        int_theta = comple_density
        J_dom = simpson_int(int_theta, theta)
        
        # obtain the penalty term of the boundary condition on theta:0 and theta:alpha
        f_up = pred(theta_up, alpha_q)
        dfdtheta_up = grad(f_up, theta_up, torch.ones(theta_up.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]        
        
        f_down = pred(theta_down, alpha_q)
        dfdtheta_down = grad(f_down, theta_down, torch.ones(theta_down.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]             
        
        f_mid = pred(theta_mid, alpha_q)
        # 上区域的损失用MSE 
        loss_up = criterion(f_up, f_up_label) 
        loss_up_d = criterion(dfdtheta_up, dfdtheta_up_label) 
        # 下区域的损失用MSE
        loss_down = criterion(f_down, f_down_label) 
        loss_down_d = criterion(dfdtheta_down, dfdtheta_down_label)  
        # 中区域的损失用MSE
        loss_mid = criterion(f_mid, f_mid_label) 
        loss = J_dom + beta1 * loss_up + beta2 * loss_up_d + beta3 * loss_down + beta4 * loss_down_d + beta5 * loss_mid # 因为位移边界条件是0，所以没有外力余势
        optim.zero_grad()
        loss.backward(retain_graph=True)
        loss_array.append(loss.data.cpu())
        loss_dom_array.append(J_dom.data.cpu())
        
        loss_b1_array.append(loss_up.data.cpu())
        loss_b2_array.append(loss_up_d.data.cpu())
        loss_b3_array.append(loss_down.data.cpu())
        loss_b4_array.append(loss_down_d.data.cpu())
        loss_b5_array.append(loss_mid.data.cpu())
       
        theta_dom, pred_sigma_r, pred_sigma_theta, pred_sigma_rtheta = evaluate_sigma(N_test)
        
        exact_sigma_r = 2*c*((alpha-theta_dom)-np.sin(theta_dom)**2*np.tan(alpha)-np.sin(theta_dom)*np.cos(theta_dom))
        exact_sigma_theta = 2*c*((alpha-theta_dom) + np.sin(theta_dom)*np.cos(theta_dom)-np.cos(theta_dom)**2*np.tan(alpha))   
        exact_sigma_rtheta = -c*(-1 + np.cos(2*theta_dom) + np.sin(2*theta_dom)*np.tan(alpha))
        
        L2_r_error = np.linalg.norm(pred_sigma_r.flatten() - exact_sigma_r.flatten())/np.linalg.norm(exact_sigma_r.flatten())
        L2_theta_error = np.linalg.norm(pred_sigma_theta.flatten() - exact_sigma_theta.flatten())/np.linalg.norm(exact_sigma_theta.flatten())
        L2_rtheta_error = np.linalg.norm(pred_sigma_rtheta.flatten() - exact_sigma_rtheta.flatten())/np.linalg.norm(exact_sigma_rtheta.flatten())
        error_sigma_r_array_D.append(L2_r_error)
        error_sigma_theta_array_D.append(L2_theta_error)
        error_sigma_rtheta_array_D.append(L2_rtheta_error)
        if epoch % 1000 == 0:
            print(L2_r_error)
            print(L2_theta_error)
            print(L2_rtheta_error)
        
        if epoch%10==0:
            print(' epoch : %i, the loss : %f, loss dom : %f, loss up : %f, loss up d : %f, loss down : %f, loss down d : %f, loss mid : %f' % \
                  (epoch, loss.data, J_dom.data, loss_up.data, loss_up_d.data, loss_down.data, loss_down_d.data, loss_mid.data))

        return loss
    optim.step(closure)
    step_scheduler.step()
theta_dom, pred_sigma_r_50D, pred_sigma_theta_50D, pred_sigma_rtheta_50D = evaluate_sigma(N_test)


for epoch in range(nepoch):
    if epoch == nepoch-1:
        end = time.time()
        consume_time = end-start
        print('time is %f' % consume_time)
    def closure():  
        # 区域内部损失
        f = pred(theta, alpha_q)
        dfdtheta = grad(f, theta, torch.ones(theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        dfdthetadtheta = grad(dfdtheta, theta, torch.ones(theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]

        comple_density = 0.5*dfdthetadtheta**2-2*dfdtheta**2
        int_theta = comple_density
        J_dom = simpson_int(int_theta, theta)
        
        # obtain the penalty term of the boundary condition on theta:0 and theta:alpha
        f_up = pred(theta_up, alpha_q)
        dfdtheta_up = grad(f_up, theta_up, torch.ones(theta_up.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]        
        
        f_down = pred(theta_down, alpha_q)
        dfdtheta_down = grad(f_down, theta_down, torch.ones(theta_down.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]             
        
        f_mid = pred(theta_mid, alpha_q)
        # 上区域的损失用MSE 
        loss_up = criterion(f_up, f_up_label) 
        loss_up_d = criterion(dfdtheta_up, dfdtheta_up_label) 
        # 下区域的损失用MSE
        loss_down = criterion(f_down, f_down_label) 
        loss_down_d = criterion(dfdtheta_down, dfdtheta_down_label)  
        # 中区域的损失用MSE
        loss_mid = criterion(f_mid, f_mid_label) 
        loss = J_dom + beta1 * loss_up + beta2 * loss_up_d + beta3 * loss_down + beta4 * loss_down_d + beta5 * loss_mid # 因为位移边界条件是0，所以没有外力余势
        optim.zero_grad()
        loss.backward(retain_graph=True)
        loss_array.append(loss.data.cpu())
        loss_dom_array.append(J_dom.data.cpu())
        
        loss_b1_array.append(loss_up.data.cpu())
        loss_b2_array.append(loss_up_d.data.cpu())
        loss_b3_array.append(loss_down.data.cpu())
        loss_b4_array.append(loss_down_d.data.cpu())
        loss_b5_array.append(loss_mid.data.cpu())
       
        theta_dom, pred_sigma_r, pred_sigma_theta, pred_sigma_rtheta = evaluate_sigma(N_test)
        
        exact_sigma_r = 2*c*((alpha-theta_dom)-np.sin(theta_dom)**2*np.tan(alpha)-np.sin(theta_dom)*np.cos(theta_dom))
        exact_sigma_theta = 2*c*((alpha-theta_dom) + np.sin(theta_dom)*np.cos(theta_dom)-np.cos(theta_dom)**2*np.tan(alpha))   
        exact_sigma_rtheta = -c*(-1 + np.cos(2*theta_dom) + np.sin(2*theta_dom)*np.tan(alpha))
        
        L2_r_error = np.linalg.norm(pred_sigma_r.flatten() - exact_sigma_r.flatten())/np.linalg.norm(exact_sigma_r.flatten())
        L2_theta_error = np.linalg.norm(pred_sigma_theta.flatten() - exact_sigma_theta.flatten())/np.linalg.norm(exact_sigma_theta.flatten())
        L2_rtheta_error = np.linalg.norm(pred_sigma_rtheta.flatten() - exact_sigma_rtheta.flatten())/np.linalg.norm(exact_sigma_rtheta.flatten())
        error_sigma_r_array_D.append(L2_r_error)
        error_sigma_theta_array_D.append(L2_theta_error)
        error_sigma_rtheta_array_D.append(L2_rtheta_error)
        if epoch % 1000 == 0:
            print(L2_r_error)
            print(L2_theta_error)
            print(L2_rtheta_error)
        
        if epoch%10==0:
            print(' epoch : %i, the loss : %f, loss dom : %f, loss up : %f, loss up d : %f, loss down : %f, loss down d : %f, loss mid : %f' % \
                  (epoch, loss.data, J_dom.data, loss_up.data, loss_up_d.data, loss_down.data, loss_down_d.data, loss_mid.data))

        return loss
    optim.step(closure)
    step_scheduler.step()

# =============================================================================
# DCM
# =============================================================================

def pred_theta(theta):
    '''
    

    Parameters
    ----------
    xy : tensor 
        the coordinate of the input tensor.

    Returns
    the prediction of fai

    '''
    NNp = -q/alpha**3*theta**3 + 3*q/2/alpha**2*theta**2 - q/2
    NNd = (theta*(alpha-theta))**order*factor
    pred_fai = (NNp + NNd * model(theta))
    return pred_fai

def evaluate_sigma(N_test):# calculate the prediction of the stress rr and theta
    # 生成theta数据点
    theta_numpy = np.linspace(0 , alpha, N_test)
    theta = np.stack([theta_numpy], 1) # 将mesh的点flatten
    theta = torch.tensor(theta,  requires_grad=True, device='cuda')

    f = pred_theta(theta)
    dfdtheta = grad(f, theta, torch.ones(theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    dfdthetadtheta = grad(dfdtheta, theta, torch.ones(theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    sigma_r = dfdthetadtheta + 2 * f
    sigma_theta = 2 * f
    sigma_rtheta = -dfdtheta
    epsilon_r = (1/E*(sigma_r - nu*sigma_theta)).data.cpu().numpy()
    
    r = np.linspace(0, l, N_test)
    # for theta=0, dis
    dis_r = epsilon_r[0]*r

    pred_sigma_r = sigma_r.data.cpu().numpy() # change the type of tensor to the array for the use of plotting
    pred_sigma_theta = sigma_theta.data.cpu().numpy()
    pred_sigma_rtheta = sigma_rtheta.data.cpu().numpy()

    return  theta_numpy, pred_sigma_r, pred_sigma_theta, pred_sigma_rtheta
# learning the homogenous network
model = FNN(1, 20, 1).cuda()
optim = torch.optim.Adam(model.parameters(), \
                         lr= 0.001)
step_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1000, gamma=0.1)
alpha_tensor = torch.tensor(alpha, device='cuda')  
loss_array = []
loss_dom_array = []
loss_b1_array = []
loss_b2_array = []
loss_b3_array = []
loss_b4_array = []
loss_b5_array = []
loss_ex_array = []
error_sigma_r_array = []
error_sigma_theta_array = []
error_sigma_rtheta_array = []
error_dis_r_array = []
nepoch = int(nepoch)
start = time.time()
criterion = torch.nn.MSELoss()

theta = dom_data(dom_num)
r = np.linspace(0, l, N_test)
theta_up, theta_down, theta_mid = boundary_data(bound_num)
f_up_label = (torch.ones(bound_num)*(-0.5*q)).unsqueeze(1).cuda()
dfdtheta_up_label = torch.zeros(bound_num).unsqueeze(1).cuda()
f_down_label = torch.zeros(bound_num).unsqueeze(1).cuda()
dfdtheta_down_label = torch.zeros(bound_num).unsqueeze(1).cuda()
f_mid_label = (torch.ones(bound_num)*(q/2/(torch.tan(alpha_tensor)-alpha_tensor)*(alpha_tensor/2+0.5*torch.sin(alpha_tensor)-torch.cos(alpha_tensor/2)**2*torch.tan(alpha_tensor)))).unsqueeze(1).cuda()

# =============================================================================
# DCM 10
# =============================================================================

for epoch in range(nepoch1):
    if epoch == nepoch1-1:
        end = time.time()
        consume_time_complementary = end-start
        print('time is %f' % consume_time_complementary)
    def closure():  
        # 区域内部损失
        f = pred_theta(theta)
        dfdtheta = grad(f, theta, torch.ones(theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        dfdthetadtheta = grad(dfdtheta, theta, torch.ones(theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]

        comple_density = 0.5*dfdthetadtheta**2-2*dfdtheta**2
        int_theta = comple_density
        J_dom = simpson_int(int_theta, theta)
        
        # obtain the penalty term of the boundary condition on theta:0 and theta:alpha
        f_up = pred_theta(theta_up)
        dfdtheta_up = grad(f_up, theta_up, torch.ones(theta_up.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]        
        
        f_down = pred_theta(theta_down)
        dfdtheta_down = grad(f_down, theta_down, torch.ones(theta_down.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]             
        
        f_mid = pred_theta(theta_mid)
        # 上区域的损失用MSE 
        loss_up = criterion(f_up, f_up_label) 
        loss_up_d = criterion(dfdtheta_up, dfdtheta_up_label) 
        # 下区域的损失用MSE
        loss_down = criterion(f_down, f_down_label) 
        loss_down_d = criterion(dfdtheta_down, dfdtheta_down_label)  
        # 中区域的损失用MSE
        loss_mid = criterion(f_mid, f_mid_label) 
        loss = J_dom + beta1 * loss_up + beta2 * loss_up_d + beta3 * loss_down + beta4 * loss_down_d + beta5 * loss_mid # 因为位移边界条件是0，所以没有外力余势
        optim.zero_grad()
        loss.backward(retain_graph=True)
        loss_array.append(loss.data.cpu())
        loss_dom_array.append(J_dom.data.cpu())
        
        loss_b1_array.append(loss_up.data.cpu())
        loss_b2_array.append(loss_up_d.data.cpu())
        loss_b3_array.append(loss_down.data.cpu())
        loss_b4_array.append(loss_down_d.data.cpu())
        loss_b5_array.append(loss_mid.data.cpu())
       
        theta_dom, pred_sigma_r, pred_sigma_theta, pred_sigma_rtheta = evaluate_sigma(N_test)
        
        exact_sigma_r = 2*c*((alpha-theta_dom)-np.sin(theta_dom)**2*np.tan(alpha)-np.sin(theta_dom)*np.cos(theta_dom))
        exact_sigma_theta = 2*c*((alpha-theta_dom) + np.sin(theta_dom)*np.cos(theta_dom)-np.cos(theta_dom)**2*np.tan(alpha))   
        exact_sigma_rtheta = -c*(-1 + np.cos(2*theta_dom) + np.sin(2*theta_dom)*np.tan(alpha))
        exact_epsilon_r = 1/E*(exact_sigma_r - nu*exact_sigma_theta)
        exact_dis_r = r*exact_epsilon_r[0]
        
        L2_r_error = np.linalg.norm(pred_sigma_r.flatten() - exact_sigma_r.flatten())/np.linalg.norm(exact_sigma_r.flatten())
        L2_theta_error = np.linalg.norm(pred_sigma_theta.flatten() - exact_sigma_theta.flatten())/np.linalg.norm(exact_sigma_theta.flatten())
        L2_rtheta_error = np.linalg.norm(pred_sigma_rtheta.flatten() - exact_sigma_rtheta.flatten())/np.linalg.norm(exact_sigma_rtheta.flatten())
        
        error_sigma_r_array.append(L2_r_error)
        error_sigma_theta_array.append(L2_theta_error)
        error_sigma_rtheta_array.append(L2_rtheta_error)

        if epoch % 500 == 0:
            print(L2_r_error)
            print(L2_theta_error)
            print(L2_rtheta_error)
      
        if epoch%10==0:
            print(' epoch : %i, the loss : %f, loss dom : %f, loss up : %f, loss up d : %f, loss down : %f, loss down d : %f, loss mid : %f' % \
                  (epoch, loss.data, J_dom.data, loss_up.data, loss_up_d.data, loss_down.data, loss_down_d.data, loss_mid.data))
        return loss
    optim.step(closure)
    step_scheduler.step()
theta_dom, pred_sigma_r_10, pred_sigma_theta_10, pred_sigma_rtheta_10 = evaluate_sigma(N_test)

# =============================================================================
# DCM 30
# =============================================================================

for epoch in range(nepoch2):
    if epoch == nepoch2-1:
        end = time.time()
        consume_time_complementary = end-start
        print('time is %f' % consume_time_complementary)
    def closure():  
        # 区域内部损失
        f = pred_theta(theta)
        dfdtheta = grad(f, theta, torch.ones(theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        dfdthetadtheta = grad(dfdtheta, theta, torch.ones(theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]

        comple_density = 0.5*dfdthetadtheta**2-2*dfdtheta**2
        int_theta = comple_density
        J_dom = simpson_int(int_theta, theta)
        
        # obtain the penalty term of the boundary condition on theta:0 and theta:alpha
        f_up = pred_theta(theta_up)
        dfdtheta_up = grad(f_up, theta_up, torch.ones(theta_up.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]        
        
        f_down = pred_theta(theta_down)
        dfdtheta_down = grad(f_down, theta_down, torch.ones(theta_down.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]             
        
        f_mid = pred_theta(theta_mid)
        # 上区域的损失用MSE 
        loss_up = criterion(f_up, f_up_label) 
        loss_up_d = criterion(dfdtheta_up, dfdtheta_up_label) 
        # 下区域的损失用MSE
        loss_down = criterion(f_down, f_down_label) 
        loss_down_d = criterion(dfdtheta_down, dfdtheta_down_label)  
        # 中区域的损失用MSE
        loss_mid = criterion(f_mid, f_mid_label) 
        loss = J_dom + beta1 * loss_up + beta2 * loss_up_d + beta3 * loss_down + beta4 * loss_down_d + beta5 * loss_mid # 因为位移边界条件是0，所以没有外力余势
        optim.zero_grad()
        loss.backward(retain_graph=True)
        loss_array.append(loss.data.cpu())
        loss_dom_array.append(J_dom.data.cpu())
        
        loss_b1_array.append(loss_up.data.cpu())
        loss_b2_array.append(loss_up_d.data.cpu())
        loss_b3_array.append(loss_down.data.cpu())
        loss_b4_array.append(loss_down_d.data.cpu())
        loss_b5_array.append(loss_mid.data.cpu())
       
        theta_dom, pred_sigma_r, pred_sigma_theta, pred_sigma_rtheta = evaluate_sigma(N_test)
        
        exact_sigma_r = 2*c*((alpha-theta_dom)-np.sin(theta_dom)**2*np.tan(alpha)-np.sin(theta_dom)*np.cos(theta_dom))
        exact_sigma_theta = 2*c*((alpha-theta_dom) + np.sin(theta_dom)*np.cos(theta_dom)-np.cos(theta_dom)**2*np.tan(alpha))   
        exact_sigma_rtheta = -c*(-1 + np.cos(2*theta_dom) + np.sin(2*theta_dom)*np.tan(alpha))
        exact_epsilon_r = 1/E*(exact_sigma_r - nu*exact_sigma_theta)
        exact_dis_r = r*exact_epsilon_r[0]
        
        L2_r_error = np.linalg.norm(pred_sigma_r.flatten() - exact_sigma_r.flatten())/np.linalg.norm(exact_sigma_r.flatten())
        L2_theta_error = np.linalg.norm(pred_sigma_theta.flatten() - exact_sigma_theta.flatten())/np.linalg.norm(exact_sigma_theta.flatten())
        L2_rtheta_error = np.linalg.norm(pred_sigma_rtheta.flatten() - exact_sigma_rtheta.flatten())/np.linalg.norm(exact_sigma_rtheta.flatten())

        
        error_sigma_r_array.append(L2_r_error)
        error_sigma_theta_array.append(L2_theta_error)
        error_sigma_rtheta_array.append(L2_rtheta_error)

        if epoch % 500 == 0:
            print(L2_r_error)
            print(L2_theta_error)
            print(L2_rtheta_error)
    
        if epoch%10==0:
            print(' epoch : %i, the loss : %f, loss dom : %f, loss up : %f, loss up d : %f, loss down : %f, loss down d : %f, loss mid : %f' % \
                  (epoch, loss.data, J_dom.data, loss_up.data, loss_up_d.data, loss_down.data, loss_down_d.data, loss_mid.data))
        return loss
    optim.step(closure)
    step_scheduler.step()
theta_dom, pred_sigma_r_30, pred_sigma_theta_30, pred_sigma_rtheta_30 = evaluate_sigma(N_test)

# =============================================================================
# DCM 50
# =============================================================================

for epoch in range(nepoch3):
    if epoch == nepoch3-1:
        end = time.time()
        consume_time_complementary = end-start
        print('time is %f' % consume_time_complementary)
    def closure():  
        # 区域内部损失
        f = pred_theta(theta)
        dfdtheta = grad(f, theta, torch.ones(theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        dfdthetadtheta = grad(dfdtheta, theta, torch.ones(theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]

        comple_density = 0.5*dfdthetadtheta**2-2*dfdtheta**2
        int_theta = comple_density
        J_dom = simpson_int(int_theta, theta)
        
        # obtain the penalty term of the boundary condition on theta:0 and theta:alpha
        f_up = pred_theta(theta_up)
        dfdtheta_up = grad(f_up, theta_up, torch.ones(theta_up.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]        
        
        f_down = pred_theta(theta_down)
        dfdtheta_down = grad(f_down, theta_down, torch.ones(theta_down.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]             
        
        f_mid = pred_theta(theta_mid)
        # 上区域的损失用MSE 
        loss_up = criterion(f_up, f_up_label) 
        loss_up_d = criterion(dfdtheta_up, dfdtheta_up_label) 
        # 下区域的损失用MSE
        loss_down = criterion(f_down, f_down_label) 
        loss_down_d = criterion(dfdtheta_down, dfdtheta_down_label)  
        # 中区域的损失用MSE
        loss_mid = criterion(f_mid, f_mid_label) 
        loss = J_dom + beta1 * loss_up + beta2 * loss_up_d + beta3 * loss_down + beta4 * loss_down_d + beta5 * loss_mid # 因为位移边界条件是0，所以没有外力余势
        optim.zero_grad()
        loss.backward(retain_graph=True)
        loss_array.append(loss.data.cpu())
        loss_dom_array.append(J_dom.data.cpu())
        
        loss_b1_array.append(loss_up.data.cpu())
        loss_b2_array.append(loss_up_d.data.cpu())
        loss_b3_array.append(loss_down.data.cpu())
        loss_b4_array.append(loss_down_d.data.cpu())
        loss_b5_array.append(loss_mid.data.cpu())
       
        theta_dom, pred_sigma_r, pred_sigma_theta, pred_sigma_rtheta = evaluate_sigma(N_test)
        
        exact_sigma_r = 2*c*((alpha-theta_dom)-np.sin(theta_dom)**2*np.tan(alpha)-np.sin(theta_dom)*np.cos(theta_dom))
        exact_sigma_theta = 2*c*((alpha-theta_dom) + np.sin(theta_dom)*np.cos(theta_dom)-np.cos(theta_dom)**2*np.tan(alpha))   
        exact_sigma_rtheta = -c*(-1 + np.cos(2*theta_dom) + np.sin(2*theta_dom)*np.tan(alpha))
        exact_epsilon_r = 1/E*(exact_sigma_r - nu*exact_sigma_theta)
        exact_dis_r = r*exact_epsilon_r[0]
        
        L2_r_error = np.linalg.norm(pred_sigma_r.flatten() - exact_sigma_r.flatten())/np.linalg.norm(exact_sigma_r.flatten())
        L2_theta_error = np.linalg.norm(pred_sigma_theta.flatten() - exact_sigma_theta.flatten())/np.linalg.norm(exact_sigma_theta.flatten())
        L2_rtheta_error = np.linalg.norm(pred_sigma_rtheta.flatten() - exact_sigma_rtheta.flatten())/np.linalg.norm(exact_sigma_rtheta.flatten())
  
        error_sigma_r_array.append(L2_r_error)
        error_sigma_theta_array.append(L2_theta_error)
        error_sigma_rtheta_array.append(L2_rtheta_error)

        if epoch % 500 == 0:
            print(L2_r_error)
            print(L2_theta_error)
            print(L2_rtheta_error)     
        if epoch%10==0:
            print(' epoch : %i, the loss : %f, loss dom : %f, loss up : %f, loss up d : %f, loss down : %f, loss down d : %f, loss mid : %f' % \
                  (epoch, loss.data, J_dom.data, loss_up.data, loss_up_d.data, loss_down.data, loss_down_d.data, loss_mid.data))
        return loss
    optim.step(closure)
    step_scheduler.step()
    
theta_dom, pred_sigma_r_50, pred_sigma_theta_50, pred_sigma_rtheta_50 = evaluate_sigma(N_test)


for epoch in range(nepoch):
    if epoch == nepoch-1:
        end = time.time()
        consume_time_complementary = end-start
        print('time is %f' % consume_time_complementary)
    def closure():  
        # 区域内部损失
        f = pred_theta(theta)
        dfdtheta = grad(f, theta, torch.ones(theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        dfdthetadtheta = grad(dfdtheta, theta, torch.ones(theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]

        comple_density = 0.5*dfdthetadtheta**2-2*dfdtheta**2
        int_theta = comple_density
        J_dom = simpson_int(int_theta, theta)
        
        # obtain the penalty term of the boundary condition on theta:0 and theta:alpha
        f_up = pred_theta(theta_up)
        dfdtheta_up = grad(f_up, theta_up, torch.ones(theta_up.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]        
        
        f_down = pred_theta(theta_down)
        dfdtheta_down = grad(f_down, theta_down, torch.ones(theta_down.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]             
        
        f_mid = pred_theta(theta_mid)
        # 上区域的损失用MSE 
        loss_up = criterion(f_up, f_up_label) 
        loss_up_d = criterion(dfdtheta_up, dfdtheta_up_label) 
        # 下区域的损失用MSE
        loss_down = criterion(f_down, f_down_label) 
        loss_down_d = criterion(dfdtheta_down, dfdtheta_down_label)  
        # 中区域的损失用MSE
        loss_mid = criterion(f_mid, f_mid_label) 
        loss = J_dom + beta1 * loss_up + beta2 * loss_up_d + beta3 * loss_down + beta4 * loss_down_d + beta5 * loss_mid # 因为位移边界条件是0，所以没有外力余势
        optim.zero_grad()
        loss.backward(retain_graph=True)
        loss_array.append(loss.data.cpu())
        loss_dom_array.append(J_dom.data.cpu())
        
        loss_b1_array.append(loss_up.data.cpu())
        loss_b2_array.append(loss_up_d.data.cpu())
        loss_b3_array.append(loss_down.data.cpu())
        loss_b4_array.append(loss_down_d.data.cpu())
        loss_b5_array.append(loss_mid.data.cpu())
       
        theta_dom, pred_sigma_r, pred_sigma_theta, pred_sigma_rtheta = evaluate_sigma(N_test)
        
        exact_sigma_r = 2*c*((alpha-theta_dom)-np.sin(theta_dom)**2*np.tan(alpha)-np.sin(theta_dom)*np.cos(theta_dom))
        exact_sigma_theta = 2*c*((alpha-theta_dom) + np.sin(theta_dom)*np.cos(theta_dom)-np.cos(theta_dom)**2*np.tan(alpha))   
        exact_sigma_rtheta = -c*(-1 + np.cos(2*theta_dom) + np.sin(2*theta_dom)*np.tan(alpha))
        exact_epsilon_r = 1/E*(exact_sigma_r - nu*exact_sigma_theta)

        
        L2_r_error = np.linalg.norm(pred_sigma_r.flatten() - exact_sigma_r.flatten())/np.linalg.norm(exact_sigma_r.flatten())
        L2_theta_error = np.linalg.norm(pred_sigma_theta.flatten() - exact_sigma_theta.flatten())/np.linalg.norm(exact_sigma_theta.flatten())
        L2_rtheta_error = np.linalg.norm(pred_sigma_rtheta.flatten() - exact_sigma_rtheta.flatten())/np.linalg.norm(exact_sigma_rtheta.flatten())
      
        error_sigma_r_array.append(L2_r_error)
        error_sigma_theta_array.append(L2_theta_error)
        error_sigma_rtheta_array.append(L2_rtheta_error)

        if epoch % 500 == 0:
            print(L2_r_error)
            print(L2_theta_error)
            print(L2_rtheta_error)
  
        if epoch%10==0:
            print(' epoch : %i, the loss : %f, loss dom : %f, loss up : %f, loss up d : %f, loss down : %f, loss down d : %f, loss mid : %f' % \
                  (epoch, loss.data, J_dom.data, loss_up.data, loss_up_d.data, loss_down.data, loss_down_d.data, loss_mid.data))
        return loss
    optim.step(closure)
    step_scheduler.step()  
    

#%%
# =============================================================================
# plot line sigma_rr and theta
# =============================================================================
theta_dom, pred_sigma_r_data, pred_sigma_theta_data, pred_sigma_rtheta_data = evaluate_sigma_data(N_test)

internal = 15

exact_sigma_r = 2*c*((alpha-theta_dom)-np.sin(theta_dom)**2*np.tan(alpha)-np.sin(theta_dom)*np.cos(theta_dom))
exact_sigma_theta = 2*c*((alpha-theta_dom) + np.sin(theta_dom)*np.cos(theta_dom)-np.cos(theta_dom)**2*np.tan(alpha))   
exact_sigma_rtheta = -c*(-1 + np.cos(2*theta_dom) + np.sin(2*theta_dom)*np.tan(alpha))
# =============================================================================
# sigma_rr
# =============================================================================
plt.plot(theta_dom, exact_sigma_r, color = 'r')
plt.scatter(theta_dom[::internal], pred_sigma_r_data[::internal], marker = '.', linestyle='-')
plt.scatter(theta_dom[::internal], pred_sigma_r_10[::internal], marker = '*', linestyle='-')
plt.scatter(theta_dom[::internal], pred_sigma_r_10D[::internal], marker = '+', linestyle='-')
plt.xlabel(r'$\theta$', fontsize=13)
plt.ylabel(r'$\sigma_{r}$', fontsize=13)
plt.legend(['Exact', 'Data-driven', 'DCEM',  'DCEM-O'])
plt.savefig('./DCEM_DCEM-O_sigma_r_%i.png' % nepoch1)
plt.show()

plt.plot(theta_dom, exact_sigma_r, color = 'r')
plt.scatter(theta_dom[::internal], pred_sigma_r_data[::internal], marker = '.', linestyle='-')
plt.scatter(theta_dom[::internal], pred_sigma_r_30[::internal], marker = '*', linestyle='-')
plt.scatter(theta_dom[::internal], pred_sigma_r_30D[::internal], marker = '+', linestyle='-')
plt.xlabel(r'$\theta$', fontsize=13)
plt.ylabel(r'$\sigma_{r}$', fontsize=13)
plt.legend(['Exact', 'Data-driven', 'DCEM',  'DCEM-O'])
plt.savefig('./DCEM_DCEM-O_sigma_r_%i.png' % (nepoch1+nepoch2))
plt.show()


plt.plot(theta_dom, exact_sigma_r, color = 'r')
plt.scatter(theta_dom[::internal], pred_sigma_r_data[::internal], marker = '.', linestyle='-')
plt.scatter(theta_dom[::internal], pred_sigma_r_50[::internal], marker = '*', linestyle='-')
plt.scatter(theta_dom[::internal], pred_sigma_r_50D[::internal], marker = '+', linestyle='-')
plt.xlabel(r'$\theta$', fontsize=13)
plt.ylabel(r'$\sigma_{r}$', fontsize=13)
plt.legend(['Exact', 'Data-driven', 'DCEM',  'DCEM-O'])
plt.savefig('./DCEM_DCEM-O_sigma_r_%i.png' % (nepoch1+nepoch2+nepoch3))
plt.show()

# =============================================================================
# sigma_theta
# =============================================================================

plt.plot(theta_dom, exact_sigma_theta, color = 'r')
plt.scatter(theta_dom[::internal], pred_sigma_theta_data[::internal], marker = '.', linestyle='-')
plt.scatter(theta_dom[::internal], pred_sigma_theta_10[::internal], marker = '*', linestyle='-')
plt.scatter(theta_dom[::internal], pred_sigma_theta_10D[::internal], marker = '+', linestyle='-')
plt.xlabel(r'$\theta$', fontsize=13)
plt.ylabel(r'$\sigma_{\theta}$', fontsize=13)
plt.legend(['Exact', 'Data-driven', 'DCEM',  'DCEM-O'])
plt.savefig('./DCEM_DCEM-O_sigma_theta_%i.png' % (nepoch1))
plt.show()

plt.plot(theta_dom, exact_sigma_theta, color = 'r')
plt.scatter(theta_dom[::internal], pred_sigma_theta_data[::internal], marker = '.', linestyle='-')
plt.scatter(theta_dom[::internal], pred_sigma_theta_30[::internal], marker = '*', linestyle='-')
plt.scatter(theta_dom[::internal], pred_sigma_theta_30D[::internal], marker = '+', linestyle='-')
plt.xlabel(r'$\theta$', fontsize=13)
plt.ylabel(r'$\sigma_{\theta}$', fontsize=13)
plt.legend(['Exact', 'Data-driven', 'DCEM',  'DCEM-O'])
plt.savefig('./DCEM_DCEM-O_sigma_theta_%i.png' % (nepoch1+nepoch2))
plt.show()


plt.plot(theta_dom, exact_sigma_theta, color = 'r')
plt.scatter(theta_dom[::internal], pred_sigma_theta_data[::internal], marker = '.', linestyle='-')
plt.scatter(theta_dom[::internal], pred_sigma_theta_50[::internal], marker = '*', linestyle='-')
plt.scatter(theta_dom[::internal], pred_sigma_theta_50D[::internal], marker = '+', linestyle='-')
plt.xlabel(r'$\theta$', fontsize=13)
plt.ylabel(r'$\sigma_{\theta}$', fontsize=13)
plt.legend(['Exact', 'Data-driven', 'DCEM',  'DCEM-O'])
plt.savefig('./DCEM_DCEM-O_sigma_theta_%i.png' % (nepoch1+nepoch2+nepoch3))
plt.show()


# =============================================================================
# sigma_r_theta
# =============================================================================


plt.plot(theta_dom, exact_sigma_rtheta, color = 'r')
plt.scatter(theta_dom[::internal], pred_sigma_rtheta_data[::internal], marker = '.', linestyle='-')
plt.scatter(theta_dom[::internal], pred_sigma_rtheta_10[::internal], marker = '*', linestyle='-')
plt.scatter(theta_dom[::internal], pred_sigma_rtheta_10D[::internal], marker = '+', linestyle='-')
plt.xlabel(r'$\theta$', fontsize=13)
plt.ylabel(r'$\tau_{r\theta}$', fontsize=13)
plt.legend(['Exact', 'Data-driven', 'DCEM',  'DCEM-O'])
plt.savefig('./DCEM_DCEM-O_sigma_rtheta_%i.png' % (nepoch1))
plt.show()

plt.plot(theta_dom, exact_sigma_rtheta, color = 'r')
plt.scatter(theta_dom[::internal], pred_sigma_rtheta_data[::internal], marker = '.', linestyle='-')
plt.scatter(theta_dom[::internal], pred_sigma_rtheta_30[::internal], marker = '*', linestyle='-')
plt.scatter(theta_dom[::internal], pred_sigma_rtheta_30D[::internal], marker = '+', linestyle='-')
plt.xlabel(r'$\theta$', fontsize=13)
plt.ylabel(r'$\tau_{r\theta}$', fontsize=13)
plt.legend(['Exact', 'Data-driven', 'DCEM',  'DCEM-O'])
plt.savefig('./DCEM_DCEM-O_sigma_rtheta_%i.png' % (nepoch1+nepoch2))
plt.show()


plt.plot(theta_dom, exact_sigma_rtheta, color = 'r')
plt.scatter(theta_dom[::internal], pred_sigma_rtheta_data[::internal], marker = '.', linestyle='-')
plt.scatter(theta_dom[::internal], pred_sigma_rtheta_50[::internal], marker = '*', linestyle='-')
plt.scatter(theta_dom[::internal], pred_sigma_rtheta_50D[::internal], marker = '+', linestyle='-')
plt.xlabel(r'$\theta$', fontsize=13)
plt.ylabel(r'$\tau_{r\theta}$', fontsize=13)
plt.legend(['Exact', 'Data-driven', 'DCEM',  'DCEM-O'])
plt.savefig('./DCEM_DCEM-O_sigma_rtheta_%i.png' % (nepoch1+nepoch2+nepoch3))
plt.show()


# =============================================================================
# L2 error
# =============================================================================


smo = 1 # 平滑度
error_sigma_r_array_smo = np.zeros(len(error_sigma_r_array))
error_sigma_theta_array_smo = np.zeros(len(error_sigma_theta_array))
error_sigma_rtheta_array_smo = np.zeros(len(error_sigma_rtheta_array))
error_sigma_r_array_smo_D = np.zeros(len(error_sigma_r_array_D))
error_sigma_theta_array_smo_D = np.zeros(len(error_sigma_theta_array_D))
error_sigma_rtheta_array_smo_D = np.zeros(len(error_sigma_rtheta_array_D))

for j in range(int(len(error_sigma_r_array)/smo)):
    error_sigma_r_array_smo[j*smo:(j+1)*smo] = np.mean(np.array(error_sigma_r_array)[j*smo:(j+1)*smo])
    error_sigma_theta_array_smo[j*smo:(j+1)*smo] = np.mean(np.array(error_sigma_theta_array)[j*smo:(j+1)*smo])
    error_sigma_rtheta_array_smo[j*smo:(j+1)*smo] = np.mean(np.array(error_sigma_rtheta_array)[j*smo:(j+1)*smo])

for j in range(int(len(error_sigma_r_array)/smo)):
    error_sigma_r_array_smo_D[j*smo:(j+1)*smo] = np.mean(np.array(error_sigma_r_array_D)[j*smo:(j+1)*smo])
    error_sigma_theta_array_smo_D[j*smo:(j+1)*smo] = np.mean(np.array(error_sigma_theta_array_D)[j*smo:(j+1)*smo])
    error_sigma_rtheta_array_smo_D[j*smo:(j+1)*smo] = np.mean(np.array(error_sigma_rtheta_array_D)[j*smo:(j+1)*smo])

plt.plot(error_sigma_r_array_smo, linestyle='-.')
plt.plot(error_sigma_r_array_smo_D, linestyle='-')
plt.yscale('log')
plt.xlabel('Iteration', fontsize=13)
plt.ylabel(r'$\mathcal{L}_{2}: \sigma_{r}$', fontsize=13)
plt.legend(['DCEM', 'DCEM-O'])
plt.savefig('DCEM_DCEM-O_sigma_rr_error.png')
plt.show()


plt.plot(error_sigma_theta_array_smo, linestyle='-.')
plt.plot(error_sigma_theta_array_smo_D, linestyle='-')
plt.yscale('log')
plt.xlabel('Iteration', fontsize=13)
plt.ylabel(r'$\mathcal{L}_{2}: \sigma_{\theta}$', fontsize=13)
plt.legend(['DCEM', 'DCEM-O'])
plt.savefig('DCEM_DCEM-O_sigma_theta_error.png')
plt.show()


plt.plot(error_sigma_rtheta_array_smo, linestyle='-.')
plt.plot(error_sigma_rtheta_array_smo_D, linestyle='-')
plt.yscale('log')
plt.xlabel('Iteration', fontsize=13)
plt.ylabel(r'$\mathcal{L}_{2}: \tau_{r \theta}$', fontsize=13)
plt.legend(['DCEM', 'DCEM-O'])
plt.savefig('DCEM_DCEM-O_sigma_rtheta_error.png')
plt.show()










