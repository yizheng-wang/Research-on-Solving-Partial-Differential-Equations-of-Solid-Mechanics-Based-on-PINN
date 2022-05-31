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
delta = 0.01

dom_num = 100

N_test = 100
lam = 1
G = 0.5
Q = 4
def dom_data(Nf):
    '''
    生成内部点
    '''
    
    x = np.linspace(-a, a, Nf)
    y = np.linspace(0, b, Nf) 
    x_mesh, y_mesh = np.meshgrid(x, y)
    xy_dom = np.stack([x_mesh.flatten(), y_mesh.flatten()], 1)
    xy_dom = torch.tensor(xy_dom,  requires_grad=True, device='cuda')
    
    return xy_dom

def bound_data(Nf):
    '''
    生成奇异应变坐标点
    '''
    
    x = np.linspace(0+delta, a, Nf)
    y = np.zeros(len(x))
    xy_bound = np.stack([x.flatten(), y.flatten()], 1)
    xy_bound = torch.tensor(xy_bound,  requires_grad=True, device='cuda')
    
    return xy_bound

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
    XY_test = dom_data(N_test)
    u_pred = pred(XY_test).data.flatten().cpu().numpy()

    XY_test = XY_test.data.cpu().numpy()
    x = XY_test[:, 0]
    y = XY_test[:, 1]
    u_exact = np.sqrt(x**2+y**2)**0.5*np.sqrt((1-x/np.sqrt(x**2+y**2))/2)
    erroru = np.linalg.norm(u_exact-u_pred)/np.linalg.norm(u_exact)

    return erroru

def evaluate_e(N_test):# evaluate the L2 error of strain xx, yy, and xy
    XY_test = bound_data(N_test)
    u = pred(XY_test)

    dudxy = grad(u, XY_test, torch.ones(XY_test.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    dudy_pred = dudxy[:, 1]
  
    e_yy_pred = dudy_pred.data.cpu().numpy()

    
    XY_test = XY_test.data.cpu().numpy()
    x = XY_test[:, 0]

    e_yy_exact = 0.5/np.sqrt(x**2)**0.5


    erroreyy = np.linalg.norm(e_yy_exact-e_yy_pred)/np.linalg.norm(e_yy_exact)

    return erroreyy

def evaluate_strain(N_test):# calculate the prediction of the strain
    XY_test = bound_data(N_test)
    u = pred(XY_test)

    dudxy = grad(u, XY_test, torch.ones(XY_test.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    dudy_pred = dudxy[:, 1]
  
    e_yy_pred = dudy_pred.data.cpu().numpy()

    XY_test = XY_test.data.cpu().numpy()
    x = XY_test[:, 0]


    return x, e_yy_pred

def evaluate_dis(N_test):# calculate the prediction of the dis
# 分析sigma应力，输入坐标是极坐标r和theta
    x = np.linspace(-a, a, N_test)
    y = np.linspace(0, b, N_test)
    x_mesh, y_mesh = np.meshgrid(x, y)
    
    XY_test = dom_data(N_test)
    u = pred(XY_test)

    u_pred = u.data.cpu().numpy().reshape(N_test, N_test)

    return x_mesh, y_mesh, u_pred
# learning the homogenous network

model = FNN(2, 30, 1).cuda()
criterion = torch.nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr= 0.0001)
loss_array = []
error_u_array = []

error_eyy_array = []

nepoch = int(nepoch)
start = time.time()

for epoch in range(nepoch):
    if epoch == nepoch-1:
        end = time.time()
        consume_time = end-start
        print('time is %f' % consume_time)
    if epoch%100 == 0: # 重新分配点   
        Xf = dom_data(dom_num)
        Xf_lable = Xf.data.cpu().numpy()
        x = Xf_lable[:, 0]
        y = Xf_lable[:, 1]
        u_exact = np.sqrt(x**2+y**2)**0.5*np.sqrt((1-x/np.sqrt(x**2+y**2))/2)
        u_exact = np.stack([u_exact], 1)
        u_exact = torch.tensor(u_exact).cuda()
    def closure():  
        # 区域内部损失
        u_pred = pred(Xf)  
        
        loss = criterion(u_pred, u_exact)
        if epoch == 5000:
            print()
        optim.zero_grad()
        loss.backward(retain_graph=True)
        loss_array.append(loss.data.cpu())

        erroru = evaluate_u(N_test)
        erroreyy = evaluate_e(N_test)
        
        error_u_array.append(erroru)
        error_eyy_array.append(erroreyy)
        if epoch%10==0:
            print(' epoch : %i, the loss : %f' % \
                  (epoch, loss.data))
        if epoch % 1000 == 0:
            print(f'L2 error of dis  is {erroru} ')
            print(f'L2 error of strain yy is {erroreyy} ')
        return loss
    optim.step(closure)
mpl.rcParams['figure.dpi'] = 1000
mpl.rcParams['font.sans-serif']=['SimHei'] # 为了显示中文matplotlib
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
# =============================================================================
# displacement plotting
# =============================================================================
x_mesh, y_mesh, u_pred = evaluate_dis(N_test)    
u_exact = np.sqrt(x_mesh**2+y_mesh**2)**0.5*np.sqrt((1-x_mesh/np.sqrt(x_mesh**2+y_mesh**2))/2)
    
plt.contourf(x_mesh, y_mesh, u_exact)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
#plt.title('ux_exact')
plt.show()

plt.contourf(x_mesh, y_mesh, u_pred)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
#plt.title('ux_pred')
plt.show()

plt.contourf(x_mesh, y_mesh, np.abs(u_exact - u_pred))
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
#plt.title('ux_abs_error')
plt.show()


    
# =============================================================================
# strain plotting
# =============================================================================
x_mesh,   e_yy_pred= evaluate_strain(N_test)
e_yy_exact = 0.5/np.sqrt(x_mesh**2)**0.5
    
plt.plot(x_mesh, e_yy_exact, linestyle=':')
plt.plot(x_mesh, e_yy_pred, linestyle='--')
plt.xlabel('x')
plt.ylabel(r'$\varepsilon_{yy}$')
plt.legend([r'$\varepsilon_{yy}$解析解', r'$\varepsilon_{yy}$预测解'])
#plt.title('exx_exact')
plt.show()



# =============================================================================
# ux and strain error
# =============================================================================

plt.yscale('log')
plt.plot(error_u_array, linestyle=':')
plt.xlabel('迭代数')
plt.ylabel('误差')
settick()
plt.show()



plt.plot(error_eyy_array, linestyle=':')
plt.xlabel('迭代数')
plt.ylabel('误差')
settick()
plt.show()