# 转化为一维问题处理，用罚函数
import numpy as np
import torch
import matplotlib.pyplot as plt
from  torch.autograd import grad
import time
import matplotlib as mpl
import numpy as np

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
    
setup_seed(2023)

q = 5 # 均布压力载荷
alpha = np.pi
l = np.sqrt(3)

nepoch = 1501

dom_num = 1000
bound_num = 5
N_test = 1001
E = 1000
nu = 0.3
G = E/2/(1+nu)
c = q/(np.tan(alpha)-alpha)/2
order = 2
factor = (4/alpha**2)**order
beta0 = 1
beta1 = 1
beta2 = 1
beta3 = 1
beta4 = 1
beta5 = 1
def dom_data(Nf):
    '''
    生成内部点，极坐标形式生成,我转化为一维问题
    '''
    

    theta_dom = np.random.rand(Nf)*alpha
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
    
    return polar_up, polar_down, polar_mid

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

def evaluate(N_test):# calculate the prediction of the stress rr and theta
    # 生成theta数据点
    theta_numpy = np.linspace(0 , alpha, N_test)
    theta = np.stack([theta_numpy], 1) # 将mesh的点flatten
    theta = torch.tensor(theta,  requires_grad=True, device='cuda')

    f = pred_theta(theta).data.cpu().numpy()


    return  theta_numpy, f

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

    return  theta_numpy, pred_sigma_r, pred_sigma_theta, pred_sigma_rtheta, dis_r
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

for epoch in range(nepoch):
    if epoch == nepoch-1:
        end = time.time()
        consume_time = end-start
        print('time is %f' % consume_time)
    if epoch%10 == 0: # 重新分配点   
        theta = dom_data(dom_num)
        r = np.linspace(0, l, N_test)
        theta_up, theta_down, theta_mid = boundary_data(bound_num)
        f_up_label = (torch.ones(bound_num)*(-0.5*q)).unsqueeze(1).cuda()
        dfdtheta_up_label = torch.zeros(bound_num).unsqueeze(1).cuda()
        f_down_label = torch.zeros(bound_num).unsqueeze(1).cuda()
        dfdtheta_down_label = torch.zeros(bound_num).unsqueeze(1).cuda()
        f_mid_label = (torch.ones(bound_num)*(q/2/(torch.tan(alpha_tensor)-alpha_tensor)*(alpha_tensor/2+0.5*torch.sin(alpha_tensor)-torch.cos(alpha_tensor/2)**2*torch.tan(alpha_tensor)))).unsqueeze(1).cuda()
    def closure():  
        # 区域内部损失
        f = pred_theta(theta)
        dfdtheta1 = grad(f, theta, torch.ones(theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        dfdtheta2 = grad(dfdtheta1, theta, torch.ones(theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        dfdtheta3 = grad(dfdtheta2, theta, torch.ones(theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        dfdtheta4 = grad(dfdtheta3, theta, torch.ones(theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
        J_dom = torch.mean((dfdtheta4 + 4*dfdtheta2)**2)
        
        
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
        
        loss = beta0 * J_dom + beta1 * loss_up + beta2 * loss_up_d + beta3 * loss_down + beta4 * loss_down_d + beta5 * loss_mid  # 因为位移边界条件是0，所以没有外力余势
        optim.zero_grad()
        loss.backward(retain_graph=True)
        loss_array.append(loss.data.cpu())
        loss_dom_array.append(J_dom.data.cpu())
        
        loss_b1_array.append(loss_up.data.cpu())
        loss_b2_array.append(loss_up_d.data.cpu())
        loss_b3_array.append(loss_down.data.cpu())
        loss_b4_array.append(loss_down_d.data.cpu())
        loss_b5_array.append(loss_mid.data.cpu())
       
        theta_dom, pred_sigma_r, pred_sigma_theta, pred_sigma_rtheta, pred_dis_r = evaluate_sigma(N_test)
        
        exact_sigma_r = 2*c*((alpha-theta_dom)-np.sin(theta_dom)**2*np.tan(alpha)-np.sin(theta_dom)*np.cos(theta_dom))
        exact_sigma_theta = 2*c*((alpha-theta_dom) + np.sin(theta_dom)*np.cos(theta_dom)-np.cos(theta_dom)**2*np.tan(alpha))   
        exact_sigma_rtheta = -c*(-1 + np.cos(2*theta_dom) + np.sin(2*theta_dom)*np.tan(alpha))
        exact_epsilon_r = 1/E*(exact_sigma_r - nu*exact_sigma_theta)
        exact_dis_r = r*exact_epsilon_r[0]
        
        L2_r_error = np.linalg.norm(pred_sigma_r.flatten() - exact_sigma_r.flatten())/np.linalg.norm(exact_sigma_r.flatten())
        L2_theta_error = np.linalg.norm(pred_sigma_theta.flatten() - exact_sigma_theta.flatten())/np.linalg.norm(exact_sigma_theta.flatten())
        L2_rtheta_error = np.linalg.norm(pred_sigma_rtheta.flatten() - exact_sigma_rtheta.flatten())/np.linalg.norm(exact_sigma_rtheta.flatten())
        L2_dis_r_error = np.linalg.norm(pred_dis_r.flatten() - exact_dis_r.flatten())/np.linalg.norm(exact_dis_r.flatten())
        
        error_sigma_r_array.append(L2_r_error)
        error_sigma_theta_array.append(L2_theta_error)
        error_sigma_rtheta_array.append(L2_rtheta_error)
        error_dis_r_array.append(L2_dis_r_error)
        if epoch % 500 == 0:
            print(L2_r_error)
            print(L2_theta_error)
            print(L2_rtheta_error)
            print(L2_dis_r_error)
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
plt.yscale('log')
window_size =1
def moving_average(interval, window_size):
    '''
    for smooth

    Parameters
    ----------
    interval : TYPE
        DESCRIPTION.
    window_size : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')  # numpy的卷积函数

error_sigma_r_array_smo = moving_average(interval = error_sigma_r_array, window_size = window_size)
error_sigma_theta_array_smo = moving_average(interval = error_sigma_theta_array, window_size = window_size)
error_sigma_rtheta_array_smo = moving_average(interval = error_sigma_rtheta_array, window_size = window_size)
error_dis_r_array_smo = moving_average(interval = error_dis_r_array, window_size = window_size)

plt.plot(error_sigma_r_array_smo, linestyle='-.')
plt.plot(error_sigma_theta_array_smo, linestyle=':')
plt.plot(error_sigma_rtheta_array_smo, linestyle='--')
plt.plot(error_dis_r_array_smo, linestyle='-.')
plt.xlabel('Iteration')
plt.ylabel('Relative error')
plt.legend([r'$\sigma_{r}$', r'$\sigma_{\theta}$', r'$\tau_{r\theta}$',  r'$u_{r}$'])
settick()
plt.show()


print(error_sigma_r_array[-1])
print(error_sigma_theta_array[-1])
print(error_sigma_rtheta_array[-1])
print(error_dis_r_array[-1])
