# 最小余能不同的初始alpha对收敛率和计算精度的影响


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
    
setup_seed(0)
a = 1.0# 矩形板的长
b = 1 # 矩形板的宽
beta = 0.141
beta1 = 0.208
nepoch = 2000
nepoch1 = 0
num = 1 # 分析num个a的长度
N_test = 10000
G = 1000
M = 10000
dom_num = 10000
# =============================================================================
# 最小余能原理
# =============================================================================

alpha_list = [0.0001,   0.01, 0.05, 0.1] # a=b=1是0.0709标准值
alpha = 0.0005# a=b=1是0.0709标准值
ls_list = [':', '--', '-.']
loss_array_mincomplement = {}
error_Dt_array_mincomplement = {}
pred_alpha_array_mincomplement = {}
loss_dom_array_mincomplement = {}
loss_ex_array_mincomplement = {}
pred_taumax_array_mincomplement = {}
for i in range(len(alpha_list)):
    alpha = alpha_list[i]
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
        # c_n = M/M1 * (-2 * G * alpha) # 对泊松方程进行修正
        # alpha_n = M/M1 * alpha
        Dt = M1 / alpha
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
        taumax = np.max(np.sqrt(tauzx**2+tauzy**2)) # 秋最大值
        return x_mesh, y_mesh, tauzx, tauzy, taumax  
    
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
    optim1 = torch.optim.LBFGS(model.parameters(), lr= 0.001)
    loss_array_mincomplement[i] = []
    error_Dt_array_mincomplement[i] = []
    pred_alpha_array_mincomplement[i] = []
    loss_dom_array_mincomplement[i] = []
    loss_ex_array_mincomplement[i] = []
    pred_taumax_array_mincomplement[i] = []
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
    
            J_dom = torch.mean((dfaidx**2 + dfaidy**2)*0.5/G ) * a * b # 计算余能
            J_ex = torch.mean(alpha * 2 * fai) * a * b # 计算外力余功
       
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
            x_mesh, y_mesh, tauzx_pred, tauzy_pred, tauabs_pred = evaluate_tau(100) # 100*100个点
            taumax_pred = evaluate_taumax(100)
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
    
            J_dom = torch.mean((dfaidx**2 + dfaidy**2)*0.5/G ) * a * b # 计算余能
            J_ex = torch.mean(alpha * 2 * fai) * a * b # 计算外力余功
       
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
            x_mesh, y_mesh, tauzx_pred, tauzy_pred, taumax_pred = evaluate_tau(100) # 100*100个点
            pred_taumax_array_mincomplement[i].append(taumax_pred)
            if epoch%10==0:
                print(' epoch : %i, the loss : %f, loss dom : %f, loss ex : %f' % \
                      (epoch, loss.data, J_dom.data, J_ex.data))
            return loss
        optim1.step(closure)
    
    x_mesh, y_mesh, tauzx_mincomplement, tauzy_mincomplement, taumax_mincomplement = evaluate_tau(100) # 100*100个点
#%%
alpha_relative_error_mincomplement = np.zeros(len(alpha_list)) 
taumax_relative_error_mincomplement = np.zeros(len(alpha_list))
for i in range(len(alpha_list)):
    if i == 0 :
        alpha = alpha_list[i]
        taumax_exact = M/(beta1*a*b**2)
        alpha_exact = M/(beta*G*a*b**3)
        alpha_relative_error_mincomplement[i] = np.abs(pred_alpha_array_mincomplement[i][-1]-alpha_exact)/alpha_exact
        taumax_relative_error_mincomplement[i] = np.abs(pred_taumax_array_mincomplement[i][-1]-taumax_exact)/taumax_exact
        print('alpha relative error in mincomplementary is ' + str(alpha_relative_error_mincomplement[i]))
        print('taumax relative error in mincomplementary is ' + str(taumax_relative_error_mincomplement[i]))
        # 计算不同alpha的损失函数的演化规律
        plt.plot(loss_array_mincomplement[i], label = r'$\alpha$ = ' + str(alpha))
    if i != 0:
        alpha = alpha_list[i]
        taumax_exact = M/(beta1*a*b**2)
        alpha_exact = M/(beta*G*a*b**3)
        alpha_relative_error_mincomplement[i] = np.abs(pred_alpha_array_mincomplement[i][-1]-alpha_exact)/alpha_exact
        taumax_relative_error_mincomplement[i] = np.abs(pred_taumax_array_mincomplement[i][-1]-taumax_exact)/taumax_exact
        print('alpha relative error in mincomplementary is ' + str(alpha_relative_error_mincomplement[i]))
        print('taumax relative error in mincomplementary is ' + str(taumax_relative_error_mincomplement[i]))
        # 计算不同alpha的损失函数的演化规律
        plt.plot(loss_array_mincomplement[i], label = r'$\alpha$ = ' + str(alpha), ls = '%s' % ls_list[i-1])    
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Loss')
settick()
plt.show()
    
    