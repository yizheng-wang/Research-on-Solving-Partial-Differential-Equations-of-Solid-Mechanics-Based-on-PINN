# =============================================================================
# 强形式
# =============================================================================

# 转化为一维问题处理，用罚函数
# 转化为一维问题处理，用罚函数
import numpy as np
import torch
import matplotlib.pyplot as plt
from  torch.autograd import grad
import time
import matplotlib as mpl
import numpy as np
import torch.nn.functional as F
import math

plt.rcParams['font.family'] = ['sans-serif'] # 用来正常显示负号
mpl.rcParams['figure.dpi'] = 300

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
order = 2
q = 5 # 均布压力载荷
alpha = np.pi
l = np.sqrt(3)

nepoch = 1500

dom_num = 1001
bound_num = 5
N_test = 1001
N_contour = 100


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

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output



    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features


    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        L1 and the entropy loss is for the feature selection, i.e., let the weight of the activation function be small.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for index, layer in enumerate(self.layers):
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
            if index < len(self.layers)-1: # 如果不是最后一层，拉到-1到1之间，最后一层不需要tanh
                x = torch.tanh(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

def dom_data(Nf):
    '''
    生成内部点，极坐标形式生成,我转化为一维问题
    '''
    

    theta_dom = np.random.rand(Nf).astype(np.float32)*alpha
    polar_dom = np.stack([theta_dom], 1) # 将mesh的点flatten
    polar_dom = torch.tensor(polar_dom,  requires_grad=True, device='cuda')
    return polar_dom

def boundary_data(Nf):
    '''
    生成边界点，极坐标形式生成
    '''
    
    # r = (b-a)*np.random.rand(Nf)+a
    # theta = 2 * np.pi * np.random.rand(Nf) # 角度0到2*pi
    
    theta_up = np.ones(Nf).astype(np.float32) * 0
    polar_up = np.stack([theta_up], 1) # 将mesh的点flatten
    polar_up = torch.tensor(polar_up,  requires_grad=True, device='cuda')
    
    theta_down = np.ones(Nf).astype(np.float32) * alpha
    polar_down = np.stack([theta_down], 1) # 将mesh的点flatten
    polar_down = torch.tensor(polar_down,  requires_grad=True, device='cuda')
 
    theta_mid = np.ones(Nf).astype(np.float32) * alpha/2
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
    theta_scale = theta/np.pi
    NNp = -q/alpha**3*theta**3 + 3*q/2/alpha**2*theta**2 - q/2
    NNd = (theta*(alpha-theta))**order*factor
    pred_fai = (NNp + NNd * model(theta_scale))
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
    theta_numpy = np.linspace(0 , alpha, N_test).astype(np.float32)
    theta = np.stack([theta_numpy], 1) # 将mesh的点flatten
    theta = torch.tensor(theta,  requires_grad=True, device='cuda')

    f = pred_theta(theta)
    dfdtheta = grad(f, theta, torch.ones(theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    dfdthetadtheta = grad(dfdtheta, theta, torch.ones(theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    sigma_r = dfdthetadtheta + 2 * f
    sigma_theta = 2 * f
    sigma_rtheta = -dfdtheta
    epsilon_r = (1/E*(sigma_r - nu*sigma_theta)).data.cpu().numpy()
    
    r = np.linspace(0, l, N_test).astype(np.float32)
    # for theta=0, dis
    dis_r = epsilon_r[0]*r

    pred_sigma_r = sigma_r.data.cpu().numpy() # change the type of tensor to the array for the use of plotting
    pred_sigma_theta = sigma_theta.data.cpu().numpy()
    pred_sigma_rtheta = sigma_rtheta.data.cpu().numpy()

    return  theta_numpy, pred_sigma_r, pred_sigma_theta, pred_sigma_rtheta, dis_r
def evaluate_sigma_contourf(N_test):# calculate the prediction of the stress rr and theta
# 分析sigma应力，输入坐标是极坐标r和theta
    r = np.linspace(0, l, N_test).astype(np.float32)
    theta = np.linspace(0, alpha, N_test).astype(np.float32) # y方向N_test个点
    r_mesh, theta_mesh = np.meshgrid(r, theta)
    xy = np.stack((r_mesh.flatten(), theta_mesh.flatten()), 1)
    X_test = torch.tensor(xy,  requires_grad=True, device='cuda')
    theta = X_test[:, 1].unsqueeze(1)
    # 将r输入到pred中，输出应力函数
    f = pred_theta(theta)
    dfdtheta = grad(f, theta, torch.ones(theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    dfdthetadtheta = grad(dfdtheta, theta, torch.ones(theta.size()[0], 1).cuda(), retain_graph=True, create_graph=True)[0]
    sigma_r = dfdthetadtheta + 2 * f
    sigma_theta = 2 * f
    sigma_rtheta = -dfdtheta
    
    sigma_r = sigma_r.data.cpu().numpy().reshape(N_test, N_test)
    sigma_theta = sigma_theta.data.cpu().numpy().reshape(N_test, N_test)
    sigma_rtheta = sigma_rtheta.data.cpu().numpy().reshape(N_test, N_test)


    return r_mesh, theta_mesh, sigma_r, sigma_theta, sigma_rtheta

    
    
    
    
# =============================================================================
#     KAN strong form
# =============================================================================
model = KAN([1,5,5, 1], base_activation=torch.nn.SiLU, grid_size=5, grid_range=[0, 1.0], spline_order=3).cuda()
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
        f_mid_label = (torch.ones(bound_num).cuda()*(q/2/(torch.tan(alpha_tensor)-alpha_tensor)*(alpha_tensor/2+0.5*torch.sin(alpha_tensor)-torch.cos(alpha_tensor/2)**2*torch.tan(alpha_tensor)))).unsqueeze(1).cuda()
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
internal = 20
size_n = 13
r_dom = np.linspace(0, l, N_test)
r_mesh, theta_mesh, pred_sigma_r_str_KAN, pred_sigma_theta_str_KAN, pred_sigma_rtheta_str_KAN = evaluate_sigma_contourf(N_contour)    



    



exact_sigma_r = 2*c*((alpha-theta_mesh)-np.sin(theta_mesh)**2*np.tan(alpha)-np.sin(theta_mesh)*np.cos(theta_mesh))
exact_sigma_theta = 2*c*((alpha-theta_mesh) + np.sin(theta_mesh)*np.cos(theta_mesh)-np.cos(theta_mesh)**2*np.tan(alpha))   
exact_sigma_rtheta = -c*(-1 + np.cos(2*theta_mesh) + np.sin(2*theta_mesh)*np.tan(alpha))
# =============================================================================
# plot the contouf of the sigma
# =============================================================================

# rr精确剪应力
x_mesh = r_mesh * np.cos(theta_mesh)
y_mesh = r_mesh * np.sin(theta_mesh)
h1 = plt.contourf(x_mesh, y_mesh, exact_sigma_r,  cmap = 'jet', levels = 100)
plt.gca().invert_yaxis()
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h1).ax.set_title(r'$\sigma_{r}$', fontsize=13)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('./pic/sigma_r_exact.pdf')
plt.show()
# theta精确剪应力
h2 = plt.contourf(x_mesh, y_mesh, exact_sigma_theta,  cmap = 'jet', levels = 100)
plt.gca().invert_yaxis()
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h2).ax.set_title(r'$\sigma_{\theta}$', fontsize=13)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('./pic/sigma_theta_exact.pdf')
plt.show()
# rtheta精确剪应力
h3 = plt.contourf(x_mesh, y_mesh, exact_sigma_rtheta,  cmap = 'jet', levels = 100)
plt.gca().invert_yaxis()
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h3).ax.set_title(r'$\tau_{r\theta}$', fontsize=13)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('./pic/sigma_rtheta_exact.pdf')
plt.show()



# prediction
h10 = plt.contourf(x_mesh, y_mesh,  pred_sigma_r_str_KAN,  cmap = 'jet', levels = 100)
plt.gca().invert_yaxis()
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h10).ax.set_title(r'$\sigma_{r}$', fontsize=13)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('./pic/sigma_r_pred_kan.pdf')
plt.show()
# theta误差
h11 = plt.contourf(x_mesh, y_mesh, pred_sigma_theta_str_KAN,  cmap = 'jet', levels = 100)
plt.gca().invert_yaxis()
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h11).ax.set_title(r'$\sigma_{\theta}$', fontsize=13)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('./pic/sigma_theta_pred_kan.pdf')
plt.show()
# rtheta误差
h12 = plt.contourf(x_mesh, y_mesh, pred_sigma_rtheta_str_KAN,  cmap = 'jet', levels = 100)
plt.gca().invert_yaxis()
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h12).ax.set_title(r'$\tau_{r\theta}$', fontsize=13)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('./pic/sigma_rtheta_pred_kan.pdf')
plt.show()




# rr误差
h10 = plt.contourf(x_mesh, y_mesh, np.abs(exact_sigma_r - pred_sigma_r_str_KAN),  cmap = 'jet', levels = 100)
plt.gca().invert_yaxis()
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h10).ax.set_title(r'$\sigma_{r}$', fontsize=13)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('./pic/sigma_r_error_kan.pdf')
plt.show()
# theta误差
h11 = plt.contourf(x_mesh, y_mesh, np.abs(exact_sigma_theta - pred_sigma_theta_str_KAN),  cmap = 'jet', levels = 100)
plt.gca().invert_yaxis()
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h11).ax.set_title(r'$\sigma_{\theta}$', fontsize=13)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('./pic/sigma_theta_error_kan.pdf')
plt.show()
# rtheta误差
h12 = plt.contourf(x_mesh, y_mesh, np.abs(exact_sigma_rtheta - pred_sigma_rtheta_str_KAN),  cmap = 'jet', levels = 100)
plt.gca().invert_yaxis()
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h12).ax.set_title(r'$\tau_{r\theta}$', fontsize=13)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('./pic/sigma_rtheta_error_kan.pdf')
plt.show()


