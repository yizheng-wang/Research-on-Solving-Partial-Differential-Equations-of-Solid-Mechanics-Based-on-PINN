# 完全用来拟合奇异应变，应变作为原函数

import numpy as np
import torch
import matplotlib.pyplot as plt
from  torch.autograd import grad
import time
import matplotlib as mpl
torch.set_default_tensor_type(torch.DoubleTensor) # 将tensor的类型变成默认的double
torch.set_default_tensor_type(torch.cuda.DoubleTensor) # 将cuda的tensor的类型变成默认的double

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


a = 1 # 矩形板的长
b = 1 # 矩形板的宽

nepoch = 50000


dom_num_list = [1000]

N_test = 100
lam = 1
G = 0.5
Q = 4
k_list = [1]

loss_array = {}
error_u_array = {}
dis_pred = {}
u_singular_pred = {}
error_eyy_array = {}

for i in range(len(k_list)):
    
    dom_num = dom_num_list[i]
    delta = 1/dom_num
    k = k_list[i]
    def dom_data(Nf):
        '''
        生成内部点
        '''
        
        
        r = np.linspace(delta, a, Nf) 
        r = r**k/(a**k)*a
        
        rt_dom = np.stack([r.flatten()], 1)
        rt_dom = torch.tensor(rt_dom,  requires_grad=True, device='cuda')
        
        return rt_dom
    
    def bound_data(Nf): # 用来评估误差，所以不用太接近0
        '''
        生成奇异应变坐标点
        '''
        
        r = np.linspace(0.01, a, Nf)
        rt_bound = np.stack([r.flatten()], 1)
        rt_bound = torch.tensor(rt_bound,  requires_grad=True, device='cuda')
        
        return rt_bound
    
    def singular_data(Nf):
        '''
        生成奇异应变坐标点
        '''
        
        r = np.array(range(0,Nf))
        r = 1/10**r

        rt_bound = np.stack([r.flatten()], 1)
        rt_bound = torch.tensor(rt_bound,  requires_grad=True, device='cuda')
        
        return rt_bound
    
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
        
    def evaluate_u(N_test): # evaluate the L2 error of the displacement  x and y 
        rt_test = dom_data(N_test)
        u_pred = pred(rt_test).data.flatten().cpu().numpy()
    
        rt_test = rt_test.data.cpu().numpy()
        r = rt_test[:, 0]
        u_exact = 0.5/r**0.5
        erroru = np.linalg.norm(u_exact-u_pred)/np.linalg.norm(u_exact)
    
        return r, u_pred, erroru
    

    
    def evaluate_singular_u(N_test=5):# calculate the prediction of the strain
        rt_test = singular_data(N_test)
        u = pred(rt_test)
    

      
        u_pred = u.data.cpu().numpy()
        rt_test = rt_test.data.cpu().numpy()
        r = rt_test[:, 0]
    
    
        return r, u_pred    
    # learning the homogenous network
    
    model = FNN(1, 30, 1).cuda()
    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr= 0.0001)

    loss_array[i] = []
    error_u_array[i] = []
    
    error_eyy_array[i] = []
    nepoch = int(nepoch)
    start = time.time()
    
    for epoch in range(nepoch):
        if epoch == nepoch-1:
            end = time.time()
            consume_time = end-start
            print('time is %f' % consume_time)
        if epoch%100 == 0: # 重新分配点   
            Xf = dom_data(dom_num)
            Xf_numpy = Xf.data.cpu().numpy()
            Xf_lable = Xf.data.cpu().numpy()
            r = Xf_lable[:, 0]
            u_exact = 0.5/r**0.5
            u_exact = np.stack([u_exact], 1)
            u_exact = torch.tensor(u_exact).cuda()
        def closure():  
            # 区域内部损失
            u_pred = pred(Xf)  
            
            loss = criterion(u_pred, u_exact)
            optim.zero_grad()
            loss.backward(retain_graph=True)
            loss_array[i].append(loss.data.cpu())
    
            _, _, erroru = evaluate_u(N_test)
            
            error_u_array[i].append(erroru)
            if epoch%10==0:
                print(' epoch : %i, the loss : %f' % \
                      (epoch, loss.data))
            if epoch % 1000 == 0:
                print(f'L2 error of dis  is {erroru} ')
            return loss
        optim.step(closure)
    r_singular, u_singular_pred[i] = evaluate_singular_u(4)
    r, dis_pred[i], erroru = evaluate_u(N_test)
mpl.rcParams['figure.dpi'] = 1000
mpl.rcParams['font.sans-serif']=['SimHei'] # 为了显示中文matplotlib
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号



    
# =============================================================================
# strain plotting
# =============================================================================
u_exact = 0.5/r**0.5
    
plt.plot(r, u_exact,  label = r'$\varepsilon_{yy}$解析解')
# for i in range(len(k_list)):
#     plt.plot(r_mesh, e_yy_pred[i], linestyle='--', label = r'$\varepsilon_{yy}$预测解 k=%i' % k_list[i])
for i in range(len(dom_num_list)):
    plt.plot(r, dis_pred[i], linestyle='--', label = r'$\varepsilon_{yy}$预测解 dom=%i' % dom_num_list[i])
plt.xlabel('r')
plt.ylabel(r'$\varepsilon_{yy}$')
plt.legend()
#plt.title('exx_exact')
plt.show()



# =============================================================================
# ux and strain error
# =============================================================================

plt.yscale('log')
for i in range(len(k_list)):
    plt.plot(error_u_array[i], linestyle=':', label = 'dom=%i' % dom_num_list[i])
plt.xlabel('迭代数')
plt.ylabel('误差')
plt.legend()
settick()
plt.show()



e_yy_exact = 0.5/r_singular**0.5
x_pred = -np.log10(r_singular)
y_exact = np.log10(e_yy_exact)
for i in range(len(k_list)):
    y_pred = np.log10(u_singular_pred[i])
    plt.plot(x_pred, y_pred , linestyle=':', marker = '*', label = 'dom=%i' % dom_num_list[i])
plt.plot(x_pred, y_exact , label = r'精确解')
plt.xlabel('-log10(x)')
plt.ylabel('log10(y)')
plt.legend()
settick()
plt.show()