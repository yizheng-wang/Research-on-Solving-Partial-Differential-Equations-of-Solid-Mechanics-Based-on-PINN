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


a = 1 # 矩形板的长
b = 1 # 矩形板的宽

nepoch = 50000


dom_num = 100

N_test = 100
lam = 1
G = 0.5
Q = 4
def dom_data(Nf):
    '''
    生成内部点
    '''
    
    x = np.linspace(0, a, Nf)
    y = np.linspace(0, b, Nf) 
    x_mesh, y_mesh = np.meshgrid(x, y)
    xy_dom = np.stack([x_mesh.flatten(), y_mesh.flatten()], 1)
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
    ux = torch.cos(2*np.pi*x)*torch.sin(np.pi*y)
    uy = torch.sin(np.pi*x)*Q*y**4/4
    pred_fai = torch.cat([ux, uy], 1)
    return pred_fai
    
def evaluate_u(N_test): # evaluate the L2 error of the displacement  x and y 
    XY_test = dom_data(N_test)
    u = pred(XY_test).data.cpu().numpy()
    ux_pred = u[:, 0]
    uy_pred = u[:, 1]
    XY_test = XY_test.data.cpu().numpy()
    x = XY_test[:, 0]
    y = XY_test[:, 1]
    ux_exact = np.cos(2*np.pi*x)*np.sin(np.pi*y)
    uy_exact = np.sin(np.pi*x)*Q*y**4/4
    errorux = np.linalg.norm(ux_exact-ux_pred)/np.linalg.norm(ux_exact)
    erroruy = np.linalg.norm(uy_exact-uy_pred)/np.linalg.norm(uy_exact)
    return errorux, erroruy

def evaluate_e(N_test):# evaluate the L2 error of strain xx, yy, and xy
    XY_test = dom_data(N_test)
    u = pred(XY_test)
    ux = u[:, 0].unsqueeze(1)
    uy = u[:, 1].unsqueeze(1)
    duxdxy = grad(ux, XY_test, torch.ones(XY_test.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    duxdx_pred = duxdxy[:, 0]
    duxdy_pred = duxdxy[:, 1]
    
    duydxy = grad(uy, XY_test, torch.ones(XY_test.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    duydx_pred = duydxy[:, 0]
    duydy_pred = duydxy[:, 1]    
    
    e_xx_pred = duxdx_pred.data.cpu().numpy()
    e_yy_pred = duydy_pred.data.cpu().numpy()
    e_xy_pred = duxdy_pred.data.cpu().numpy()  + duydx_pred.data.cpu().numpy() # engineering strain
    
    XY_test = XY_test.data.cpu().numpy()
    x = XY_test[:, 0]
    y = XY_test[:, 1]
    e_xx_exact = -2*np.pi*np.sin(2*np.pi*x)*np.sin(np.pi*y)
    e_yy_exact = np.sin(np.pi*x)*Q*y**3
    e_xy_exact = np.pi*np.cos(2*np.pi*x)*np.cos(np.pi*y) + np.pi*np.cos(np.pi*x)*Q*y**4/4
    errorexx = np.linalg.norm(e_xx_exact-e_xx_pred)/np.linalg.norm(e_xx_exact)
    erroreyy = np.linalg.norm(e_yy_exact-e_yy_pred)/np.linalg.norm(e_yy_exact)
    errorexy = np.linalg.norm(e_xy_exact-e_xy_pred)/np.linalg.norm(e_xy_exact)
    return errorexx, erroreyy, errorexy

def evaluate_strain(N_test):# calculate the prediction of the strain
# 分析sigma应力，输入坐标是极坐标r和theta
    x = np.linspace(0, a, N_test)
    y = np.linspace(0, b, N_test)
    x_mesh, y_mesh = np.meshgrid(x, y)
    
    XY_test = dom_data(N_test)
    u = pred(XY_test)
    ux = u[:, 0].unsqueeze(1)
    uy = u[:, 1].unsqueeze(1)
    duxdxy = grad(ux, XY_test, torch.ones(XY_test.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    duxdx_pred = duxdxy[:, 0]
    duxdy_pred = duxdxy[:, 1]
    
    duydxy = grad(uy, XY_test, torch.ones(XY_test.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    duydx_pred = duydxy[:, 0]
    duydy_pred = duydxy[:, 1]    
    
    e_xx_pred = duxdx_pred.data.cpu().numpy().reshape(N_test, N_test)
    e_yy_pred = duydy_pred.data.cpu().numpy().reshape(N_test, N_test)
    e_xy_pred = (duxdy_pred.data.cpu().numpy()  + duydx_pred.data.cpu().numpy()).reshape(N_test, N_test) # engineering strain

    return x_mesh, y_mesh, e_xx_pred, e_yy_pred, e_xy_pred

def evaluate_dis(N_test):# calculate the prediction of the dis
# 分析sigma应力，输入坐标是极坐标r和theta
    x = np.linspace(0, a, N_test)
    y = np.linspace(0, b, N_test)
    x_mesh, y_mesh = np.meshgrid(x, y)
    
    XY_test = dom_data(N_test)
    u = pred(XY_test)
    ux = u[:, 0]
    uy = u[:, 1]
 
    
    u_x_pred = ux.data.cpu().numpy().reshape(N_test, N_test)
    u_y_pred = uy.data.cpu().numpy().reshape(N_test, N_test)


    return x_mesh, y_mesh, u_x_pred, u_y_pred
# learning the homogenous network

model = FNN(2, 30, 2).cuda()
criterion = torch.nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr= 0.0001)
loss_array = []
error_ux_array = []
error_uy_array = []
error_exx_array = []
error_eyy_array = []
error_exy_array = []
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
        ux_exact = np.cos(2*np.pi*x)*np.sin(np.pi*y)
        uy_exact = np.sin(np.pi*x)*Q*y**4/4
        u_exact = np.stack([ux_exact, uy_exact], 1)
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

        errorux, erroruy = evaluate_u(N_test)
        errorexx, erroreyy, errorexy = evaluate_e(N_test)
        
        error_ux_array.append(errorux)
        error_uy_array.append(erroruy)
        error_exx_array.append(errorexx)
        error_eyy_array.append(erroreyy)
        error_exy_array.append(errorexy)
        if epoch%10==0:
            print(' epoch : %i, the loss : %f' % \
                  (epoch, loss.data))
        if epoch % 1000 == 0:
            print(f'L2 error of dis x is {errorux} ')
            print(f'L2 error of dis y is {erroruy} ')
            print(f'L2 error of strain xx is {errorexx} ')
            print(f'L2 error of strain yy is {erroreyy} ')
            print(f'L2 error of strain xy is {errorexy} ')
        return loss
    optim.step(closure)
  
# =============================================================================
# displacement plotting
# =============================================================================
x_mesh, y_mesh, ux_pred, uy_pred = evaluate_dis(N_test)    
ux_exact = np.cos(2*np.pi*x_mesh)*np.sin(np.pi*y_mesh)
uy_exact = np.sin(np.pi*x_mesh)*Q*y_mesh**4/4 
    
plt.contourf(x_mesh, y_mesh, ux_exact)
plt.colorbar()
plt.title('ux_exact')
plt.show()

plt.contourf(x_mesh, y_mesh, ux_pred)
plt.colorbar()
plt.title('ux_pred')
plt.show()

plt.contourf(x_mesh, y_mesh, np.abs(ux_exact - ux_pred))
plt.colorbar()
plt.title('ux_abs_error')
plt.show()

plt.contourf(x_mesh, y_mesh, uy_exact)
plt.colorbar()
plt.title('uy_exact')
plt.show()

plt.contourf(x_mesh, y_mesh, uy_pred)
plt.colorbar()
plt.title('uy_pred')
plt.show()

plt.contourf(x_mesh, y_mesh, np.abs(uy_exact - uy_pred))
plt.colorbar()
plt.title('uy_abs_error')
plt.show()
    
# =============================================================================
# strain plotting
# =============================================================================
x_mesh, y_mesh, e_xx_pred, e_yy_pred, e_xy_pred = evaluate_strain(N_test)
e_xx_exact = -2*np.pi*np.sin(2*np.pi*x_mesh)*np.sin(np.pi*y_mesh)
e_yy_exact = np.sin(np.pi*x_mesh)*Q*y_mesh**3
e_xy_exact = np.pi*np.cos(2*np.pi*x_mesh)*np.cos(np.pi*y_mesh) + np.pi*np.cos(np.pi*x_mesh)*Q*y_mesh**4
    
plt.contourf(x_mesh, y_mesh, e_xx_exact)
plt.colorbar()
plt.title('exx_exact')
plt.show()

plt.contourf(x_mesh, y_mesh, e_xx_pred)
plt.colorbar()
plt.title('exx_pred')
plt.show()

plt.contourf(x_mesh, y_mesh, np.abs(e_xx_exact - e_xx_pred))
plt.colorbar()
plt.title('exx_abs_error')
plt.show()

plt.contourf(x_mesh, y_mesh, e_yy_exact)
plt.colorbar()
plt.title('eyy_exact')
plt.show()

plt.contourf(x_mesh, y_mesh, e_yy_pred)
plt.colorbar()
plt.title('eyy_pred')
plt.show()

plt.contourf(x_mesh, y_mesh, np.abs(e_yy_exact - e_yy_pred))
plt.colorbar()
plt.title('eyy_abs_error')
plt.show()

plt.contourf(x_mesh, y_mesh, e_xy_exact)
plt.colorbar()
plt.title('exy_exact')
plt.show()

plt.contourf(x_mesh, y_mesh, e_xy_pred)
plt.colorbar()
plt.title('exy_pred')
plt.show()

plt.contourf(x_mesh, y_mesh, np.abs(e_xy_exact - e_xy_pred))
plt.colorbar()
plt.title('exy_abs_error')
plt.show()

# =============================================================================
# ux and strain error
# =============================================================================

plt.yscale('log')
plt.plot(error_ux_array, linestyle=':')
plt.plot(error_uy_array, linestyle='--')
plt.xlabel('迭代数')
plt.ylabel('误差')
plt.legend(['$u_{x}$', '$u_{y}$'])
plt.show()


plt.yscale('log')
plt.plot(error_exx_array, linestyle=':')
plt.plot(error_eyy_array, linestyle='--')
plt.plot(error_exy_array, linestyle='-.')
plt.xlabel('迭代数')
plt.ylabel('误差')
plt.legend([r'$\varepsilon_{xx}$', r'$\varepsilon_{yy}$', r'$\gamma_{xy}$'])
plt.show()