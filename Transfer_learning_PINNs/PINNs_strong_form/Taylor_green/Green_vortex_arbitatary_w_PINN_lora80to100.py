import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from kan_efficiency import *
import time
import os

def setup_seed(seed):
# random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(2024)

lambda_p = 1.0
lambda_b = 1.0
lambda_i = 1.0
epoch_num = 80000

print_e = 1

w_list = [1.0, 2.0, 3.0]

r_list = list(range(81, 100)) # 1,4,100已经算过了
r_list = [x for x in r_list if x % 4 == 0] # 去除4这个结果
for r in r_list:
    for w_i in w_list:
        w_i = w_i * np.pi
        for w_o in w_list:
            
            w_o = w_o*np.pi
    
            class LoRALinear(nn.Module):
                """
                低秩适配(LoRA)版本的线性层: W_eff = W + alpha * (A @ B)
            
                参数:
                    in_features (int): 输入特征维度
                    out_features (int): 输出特征维度
                    r (int): LoRA 中的秩 (rank)
                    alpha (float): LoRA 缩放系数
                    bias (bool): 是否包含偏置
                """
                def __init__(self, in_features, out_features, r=r, alpha=1.0, bias=True):
                    super().__init__()
                    self.in_features = in_features
                    self.out_features = out_features
                    self.r = r
                    self.alpha = alpha
            
                    # 普通权重
                    self.weight = nn.Parameter(torch.zeros(out_features, in_features))
                    # LoRA权重: 低秩分解的 A、B
                    if r > 0:
                        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
                        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
                    else:
                        # 如果 r=0，就完全退化，不做LoRA
                        self.lora_A = None
                        self.lora_B = None
            
                    # 偏置
                    if bias:
                        self.bias = nn.Parameter(torch.zeros(out_features))
                    else:
                        self.register_parameter('bias', None)
            
                    # 初始化
                    # 这里可以根据实际需要改成更合适的初始化方式
                    nn.init.normal_(self.weight, mean=0.0, std=0.02)
                    if self.lora_A is not None:
                        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
                    if self.lora_B is not None:
                        nn.init.normal_(self.lora_B, mean=0.0, std=0.02)
                    if self.bias is not None:
                        nn.init.zeros_(self.bias)
            
                    # 默认冻结原始权重 W，仅训练 LoRA 的 A、B
                    self.weight.requires_grad = False
            
                def forward(self, x):
                    if self.r > 0:
                        # W_eff = W + alpha * (A @ B)
                        delta_w = (self.lora_B @ self.lora_A) * self.alpha  # shape: [out_features, in_features]
                        w_eff = self.weight + delta_w
                    else:
                        # 没有LoRA
                        w_eff = self.weight
            
                    return F.linear(x, w_eff, self.bias)
            
            def random_points(N_f, N_b, N_i):
                x_domain = torch.rand((N_f, 3), device=device)
                # the position of the boundary points
                boundary_points = torch.rand((N_b), device=device)
            
                boundary_conditions_left = torch.zeros((N_b, 3), device=device)
                boundary_conditions_down = torch.zeros((N_b, 3), device=device)
                boundary_conditions_right = torch.zeros((N_b, 3), device=device)
                boundary_conditions_up = torch.zeros((N_b, 3), device=device)
            
                t_f = torch.rand((N_b), device=device)
            
                # Left boundary (x = 0)
                boundary_conditions_left[:, 0] = 0
                boundary_conditions_left[:, 1] = boundary_points
                boundary_conditions_left[:, 2] = t_f
                psi_left = (1 / w_o) * torch.exp(-2 * w_o**2 * boundary_conditions_left[:, 2:3]/ Re) * torch.cos(w_o * boundary_conditions_left[:, 1:2])
                omega_left = -2 * w_o * torch.exp(-2 * w_o**2 * boundary_conditions_left[:, 2:3] / Re) * torch.cos(w_o * boundary_conditions_left[:, 1:2])
                u_left = -torch.exp(-2 * w_o**2 * boundary_conditions_left[:, 2:3] / Re) * torch.sin(w_o * boundary_conditions_left[:, 1:2])
                v_left = torch.zeros(N_b,1).cuda()
            
                # Down boundary (y = 0)
                boundary_conditions_down[:, 0] = boundary_points
                boundary_conditions_down[:, 1] = 0
                boundary_conditions_down[:, 2] = t_f
                psi_down = (1 / w_o) * torch.exp(-2 * w_o**2 * boundary_conditions_down[:, 2:3] / Re) * torch.cos(w_o * boundary_conditions_down[:, 0:1])
                omega_down = -2 * w_o * torch.exp(-2 * w_o**2 * boundary_conditions_down[:, 2:3] / Re) * torch.cos(w_o * boundary_conditions_down[:, 0:1])
                u_down = torch.zeros(N_b,1).cuda()
                v_down = torch.exp(-2 * w_o**2 * boundary_conditions_down[:, 2:3] / Re) * torch.sin(w_o * boundary_conditions_down[:, 0:1])
            
                # Right boundary (x = 1)
                boundary_conditions_right[:, 0] = 1
                boundary_conditions_right[:, 1] = boundary_points
                boundary_conditions_right[:, 2] = t_f
                psi_right = (1 / w_o) * torch.exp(-2 * w_o**2 * boundary_conditions_right[:, 2:3] / Re) * torch.cos(torch.tensor(w_o)) * torch.cos(w_o * boundary_conditions_right[:, 1:2]) 
                omega_right = -2 * w_o * torch.exp(-2 * w_o**2 * boundary_conditions_right[:, 2:3] / Re) * torch.cos(torch.tensor(w_o)) * torch.cos(w_o * boundary_conditions_right[:, 1:2])
                u_right = -torch.exp(-2 * w_o**2 * boundary_conditions_right[:, 2:3] / Re) * torch.cos(torch.tensor(w_o)) * torch.sin(w_o * boundary_conditions_right[:, 1:2])
                v_right = torch.exp(-2 * w_o**2 * boundary_conditions_up[:, 2:3] / Re) * torch.sin(torch.tensor(w_o)) * torch.cos(w_o * boundary_conditions_right[:, 1:2]) 
            
                # Up boundary (y = 1)
                boundary_conditions_up[:, 0] = boundary_points
                boundary_conditions_up[:, 1] = 1
                boundary_conditions_up[:, 2] = t_f
                psi_up = (1 / w_o) * torch.exp(-2 * w_o**2 * boundary_conditions_up[:, 2:3] / Re) * torch.cos(w_o * boundary_conditions_up[:, 0:1]) * torch.cos(torch.tensor(w_o))
                omega_up = - 2 * w_o * torch.exp(-2 * w_o**2 * boundary_conditions_up[:, 2:3] / Re) * torch.cos(w_o * boundary_conditions_up[:, 0:1]) * torch.cos(torch.tensor(w_o))
                u_up = -torch.exp(-2 * w_o**2 * boundary_conditions_up[:, 2:3] / Re) * torch.cos(w_o * boundary_conditions_up[:, 0:1]) * torch.sin(torch.tensor(w_o))
                v_up = torch.exp(-2 * w_o**2 * boundary_conditions_up[:, 2:3] / Re) * torch.cos(torch.tensor(w_o)) * torch.sin(w_o * boundary_conditions_up[:, 0:1]) 
            
                boundary_dic = {'left': {'position':boundary_conditions_left, 'psi':psi_left, 'omega':omega_left, 'u':u_left, 'v':v_left}, \
                                'down': {'position':boundary_conditions_down, 'psi':psi_down, 'omega':omega_down, 'u':u_down, 'v':v_down}, \
                                'right': {'position':boundary_conditions_right, 'psi':psi_right, 'omega':omega_right, 'u':u_right, 'v':v_right}, \
                                'up': {'position':boundary_conditions_up, 'psi':psi_up, 'omega':omega_up, 'u':u_up, 'v':v_up}}
                    
                # initial condition:
                initial_points = torch.rand((N_i,3), device=device)
                initial_points[:,2] = 0
            
                psi_initial = (1 / w_o) *  torch.cos(w_o * initial_points[:, 0:1]) *  torch.cos(w_o * initial_points[:, 1:2])
                omega_initial = - 2 * w_o *  torch.cos(w_o * initial_points[:, 0:1]) *  torch.cos(w_o * initial_points[:, 1:2])
                u_initial = -torch.cos(w_o * initial_points[:, 0:1]) *  torch.sin(w_o * initial_points[:, 1:2])
                v_initial = torch.sin(w_o * initial_points[:, 0:1]) *  torch.cos(w_o * initial_points[:, 1:2])
            
                initial_dic = {'position':initial_points, 'psi':psi_initial, 'omega':omega_initial, 'u':u_initial, 'v':v_initial}
                
                return x_domain, boundary_dic, initial_dic
            # Define the PINN model
            class MultiLayerNetLoRA(nn.Module):
                def __init__(self, D_in, H, D_out, r=r, alpha=1.0):
                    """
                    LoRA 版本的多层全连接网络:
                    在每一层都使用 LoRALinear, 并设置相同的 rank 和 alpha。
                    如果某层不想使用LoRA, 可以将该层的r传入0, 或者直接用nn.Linear。
                    """
                    super(MultiLayerNetLoRA, self).__init__()
                    # 这里示范性地为每一层都用了 LoRALinear
                    # 如果想要只对某些层使用 LoRA, 直接用nn.Linear去替换即可
                    self.linear1 = LoRALinear(D_in, H, r=r, alpha=alpha)
                    self.linear2 = LoRALinear(H, H, r=r, alpha=alpha)
                    self.linear3 = LoRALinear(H, H, r=r, alpha=alpha)
                    self.linear4 = LoRALinear(H, H, r=r, alpha=alpha)
                    self.linear5 = LoRALinear(H, D_out, r=r, alpha=alpha)
            
                def forward(self, x):
                    y1 = torch.tanh(self.linear1(x))
                    y2 = torch.tanh(self.linear2(y1))
                    # 这里和之前一样的 skip connection
                    y3 = torch.tanh(self.linear3(y2)) + y1
                    y4 = torch.tanh(self.linear4(y3)) + y2
                    y = self.linear5(y4)
                    return y
            
            
            # Physics-Informed Loss Function
            def loss_fn(model, x_domain_xyt, boundary_dic, initial_dic):
                x_domain = torch.tensor(x_domain_xyt, requires_grad=True)
                
                output = model(x_domain)
                
                psi, omega = output[:, 0:1], output[:, 1:2]
            
                psi_dxyt = torch.autograd.grad(psi, x_domain, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
                psi_x = psi_dxyt[:,0:1]
                psi_y = psi_dxyt[:,1:2]
                psi_xdxy = torch.autograd.grad(psi_x, x_domain, grad_outputs=torch.ones_like(psi_x), create_graph=True)[0]
                psi_xx = psi_xdxy[:,0:1]
                
                psi_ydxy = torch.autograd.grad(psi_y, x_domain, grad_outputs=torch.ones_like(psi_y), create_graph=True)[0]
                psi_yy = psi_ydxy[:,1:2]
            
                u = psi_y
                v = -psi_x
            
                omega_dxyt = torch.autograd.grad(omega, x_domain, grad_outputs=torch.ones_like(omega), create_graph=True)[0]
                omega_x = omega_dxyt[:,0:1]
                omega_y = omega_dxyt[:,1:2]
                omega_t = omega_dxyt[:,2:3]
                
                omega_xdxyt = torch.autograd.grad(omega_x, x_domain, grad_outputs=torch.ones_like(omega_x), create_graph=True)[0]
                omega_xx = omega_xdxyt[:, 0:1]
                
                omega_ydxyt = torch.autograd.grad(omega_y, x_domain, grad_outputs=torch.ones_like(omega_y), create_graph=True)[0]
                omega_yy = omega_ydxyt[:, 1:2]
                # Governing equations
                f_omega = omega_t + u * omega_x + v * omega_y - (1 / Re) * (omega_xx + omega_yy)
                f_psi = psi_xx + psi_yy - omega
            
                physics_loss = torch.mean(f_omega**2) + torch.mean(f_psi**2)
            
                # Boundary loss 
                boundary_left =  torch.tensor(boundary_dic['left']['position'], requires_grad=True)
                # left boundary
            
                boundary_output_left = model(boundary_left)
                psi_output_left, omega_output_left = boundary_output_left[:, 0:1], boundary_output_left[:, 1:2]
                psi_xy_output_left = torch.autograd.grad(psi_output_left, boundary_left, grad_outputs=torch.ones_like(psi_output_left), create_graph=True)[0]
                psi_x_output_left = psi_xy_output_left[:,0:1]
                psi_y_output_left = psi_xy_output_left[:,1:2]   
                u_output_left = psi_y_output_left
                v_output_left = -psi_x_output_left
            
                psi_label_left, omega_label_left, u_label_left, v_label_left = boundary_dic['left']['psi'], boundary_dic['left']['omega'], boundary_dic['left']['u'], boundary_dic['left']['v']
            
                boundary_loss_left = torch.mean((psi_label_left - psi_output_left)**2) + \
                                    torch.mean((omega_label_left - omega_output_left)**2) + \
                                    torch.mean((u_label_left - u_output_left)**2) + \
                                    torch.mean((v_label_left - v_output_left)**2)
            
                # Boundary loss 
                boundary_down = torch.tensor(boundary_dic['down']['position'], requires_grad=True)
                # down boundary
            
                boundary_output_down = model(boundary_down)
                psi_output_down, omega_output_down = boundary_output_down[:, 0:1], boundary_output_down[:, 1:2]
                psi_xy_output_down = torch.autograd.grad(psi_output_down, boundary_down, grad_outputs=torch.ones_like(psi_output_down), create_graph=True)[0]
                psi_x_output_down = psi_xy_output_down[:,0:1]
                psi_y_output_down = psi_xy_output_down[:,1:2]  
                u_output_down = psi_y_output_down
                v_output_down = -psi_x_output_down
            
                psi_label_down, omega_label_down, u_label_down, v_label_down = boundary_dic['down']['psi'], boundary_dic['down']['omega'], boundary_dic['down']['u'], boundary_dic['down']['v']
            
                boundary_loss_down = torch.mean((psi_label_down - psi_output_down)**2) + \
                                    torch.mean((omega_label_down - omega_output_down)**2) + \
                                    torch.mean((u_label_down - u_output_down)**2) + \
                                    torch.mean((v_label_down - v_output_down)**2)
            
            
                # Boundary loss 
                boundary_right = torch.tensor(boundary_dic['right']['position'], requires_grad=True)
                # right boundary
                boundary_output_right = model(boundary_right)
                psi_output_right, omega_output_right = boundary_output_right[:, 0:1], boundary_output_right[:, 1:2]
                psi_xy_output_right = torch.autograd.grad(psi_output_right, boundary_right, grad_outputs=torch.ones_like(psi_output_right), create_graph=True)[0]
                psi_x_output_right = psi_xy_output_right[:,0:1]
                psi_y_output_right = psi_xy_output_right[:,1:2]  
                u_output_right = psi_y_output_right
                v_output_right = -psi_x_output_right
            
                psi_label_right, omega_label_right, u_label_right, v_label_right = boundary_dic['right']['psi'], boundary_dic['right']['omega'], boundary_dic['right']['u'], boundary_dic['right']['v']
            
                boundary_loss_right = torch.mean((psi_label_right - psi_output_right)**2) + \
                                    torch.mean((omega_label_right - omega_output_right)**2) + \
                                    torch.mean((u_label_right - u_output_right)**2) + \
                                    torch.mean((v_label_right - v_output_right)**2)
            
            
                # Boundary loss 
                boundary_up = torch.tensor(boundary_dic['up']['position'], requires_grad=True)
                # up boundary
                boundary_output_up = model(boundary_up)
                psi_output_up, omega_output_up = boundary_output_up[:, 0:1], boundary_output_up[:, 1:2]
                psi_xy_output_up = torch.autograd.grad(psi_output_up, boundary_up, grad_outputs=torch.ones_like(psi_output_up), create_graph=True)[0]
                psi_x_output_up = psi_xy_output_up[:,0:1]
                psi_y_output_up = psi_xy_output_up[:,1:2]     
                u_output_up = psi_y_output_up
                v_output_up = -psi_x_output_up
            
                psi_label_up, omega_label_up, u_label_up, v_label_up = boundary_dic['up']['psi'], boundary_dic['up']['omega'], boundary_dic['up']['u'], boundary_dic['up']['v']
            
                boundary_loss_up = torch.mean((psi_label_up - psi_output_up)**2) + \
                                    torch.mean((omega_label_up - omega_output_up)**2) + \
                                    torch.mean((u_label_up - u_output_up)**2) + \
                                    torch.mean((v_label_up - v_output_up)**2)
            
                boundary_loss = boundary_loss_left + boundary_loss_down + boundary_loss_right + boundary_loss_up
            
            
            
            
              # initial loss 
                initial_position = torch.tensor(initial_dic['position'], requires_grad=True)
                x_initial = initial_position[:, 0:1]
                y_initial = initial_position[:, 1:2]
                initial_output = model(initial_position)
                psi_output_initial, omega_output_initial = initial_output[:, 0:1], initial_output[:, 1:2]
                
                psi_dxy_output_initial = torch.autograd.grad(psi_output_initial, initial_position, grad_outputs=torch.ones_like(psi_output_initial), create_graph=True)[0]
                psi_x_output_initial = psi_dxy_output_initial[:,0:1]
                psi_y_output_initial = psi_dxy_output_initial[:,1:2]
                u_output_initial = psi_y_output_initial
                v_output_initial = -psi_x_output_initial    
            
                psi_label_initial, omega_label_initial, u_label_initial, v_label_initial = initial_dic['psi'], initial_dic['omega'], initial_dic['u'], initial_dic['v']
                
                initial_loss = torch.mean((psi_label_initial - psi_output_initial)**2) + \
                                    torch.mean((omega_label_initial - omega_output_initial)**2) + \
                                    torch.mean((u_label_initial - u_output_initial)**2) + \
                                    torch.mean((v_label_initial - v_output_initial)**2)
                                    
                
                return physics_loss, boundary_loss, initial_loss
            
            def evalutate_error(model, t=1.0):
                x_plot = torch.linspace(0, 1, 100, device=device)
                y_plot = torch.linspace(0, 1, 100, device=device)
                x,y = torch.meshgrid(x_plot, y_plot)
                t_plot = t
                t = torch.ones_like(x)*t_plot
                x_domain_plot = torch.stack([x.flatten(), y.flatten(), t.flatten()], dim = 1)
            
                psi_omega_pred = model(x_domain_plot)
                psi_pred = psi_omega_pred[:, 0].detach().cpu().numpy().reshape(100, 100)
                omega_pred = psi_omega_pred[:, 1].detach().cpu().numpy().reshape(100, 100)
            
                x_plot, y_plot = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
            
                # Exact solution for 
                psi_exact = (1 / w_o) *  np.exp(-2 * w_o**2 * t_plot / Re) * np.cos(w_o * x_plot) * np.cos(w_o * y_plot)
                omega_exact = - 2 * w_o *  np.exp(-2 * w_o**2 * t_plot / Re) *  np.cos(w_o * x_plot) *  np.cos(w_o * y_plot)
                u_exact = -np.exp(-2 * w_o**2 * t_plot / Re) * np.cos(w_o * x_plot) * np.sin(w_o * y_plot)
                v_exact = np.exp(-2 * w_o**2 * t_plot / Re) * np.sin(w_o * x_plot) * np.cos(w_o * y_plot)
                p_exact = -0.5*np.exp(-4 * w_o**2 * t_plot / Re) * (np.cos(w_o * x_plot)**2 + np.cos(w_o * y_plot)**2)
            
                psi_error = np.abs(psi_pred - psi_exact) # get the error in every points
                psi_error_t = np.linalg.norm(psi_error)/np.linalg.norm(psi_exact) # get the total relative L2 error
                
                omega_error = np.abs(omega_pred - omega_exact) # get the error in every points
                omega_error_t = np.linalg.norm(omega_error)/np.linalg.norm(omega_exact) # get the total relative L2 error
                
                return psi_error_t, omega_error_t
            # Training data
            Re = 100  # Reynolds number
            N_f = 1000  # Number of collocation points
            N_b = 100  # Number of boundary points
            N_i = 100
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            
            def freeze_lora_layers(model: nn.Module, train_layer_keywords: list, lr=1e-3):
                """
                冻结 model 中除含有指定关键字的 LoRA 参数以外的所有参数，
                并返回一个只更新指定层 LoRA 参数的优化器。
                
                参数:
                    model (nn.Module): 你的 PyTorch 模型
                    train_layer_keywords (list): 需要更新(微调)的层名称关键字的列表。
                    lr (float): 优化器学习率
                    
                返回:
                    optimizer (torch.optim.Optimizer): 只更新指定层 LoRA 参数的优化器
                """
                # 遍历所有参数
                for name, param in model.named_parameters():
                    # 如果关键词命中，就保持 requires_grad=True；否则 requires_grad=False
                    if any(keyword in name for keyword in train_layer_keywords):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                
                # 收集需要训练的参数
                params_to_optimize = [p for p in model.parameters() if p.requires_grad]
                # 构建优化器
                optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
                
                return optimizer
            
            # Model and optimizer
            #model = KAN([3, 5,5,5, 2], base_activation=torch.nn.SiLU, grid_size=15, grid_range=[0,  1.0], spline_order=3).cuda()
            model = MultiLayerNetLoRA(D_in=3, H=100, D_out=2, r=r, alpha=1.0).to(device)
            model.load_state_dict(torch.load(f'./results/PINN/model_taylor_green_w_{w_i:.2f}.pth'), strict = False)
            
            optimizer = freeze_lora_layers(model, train_layer_keywords=["linear2.lora", "linear2.bias","linear3.lora", "linear3.bias","linear4.lora", "linear4.bias"], lr=0.001)
            
            
            results = {
                "epoch": [],
                'time': [],
                "loss": [],
                "physics_loss": [],
                "boundary_loss": [],
                "initial_loss": [],
                "psi_error_t": [],
                "omega_error_t": []
            }
            # Training loop
            start_time = time.time()
            for epoch in range(epoch_num):
                x_domain, boundary_dic, initial_dic = random_points(N_f, N_b, N_i)
                optimizer.zero_grad()
                physics_loss, boundary_loss, initial_loss = loss_fn(model, x_domain, boundary_dic, initial_dic)
                loss = lambda_p * physics_loss + lambda_b * boundary_loss + lambda_i * initial_loss
                loss.backward()
                optimizer.step()
                
                if epoch % print_e == 0:
                    end_time = time.time()
                    consume_time = (end_time - start_time) 
                    psi_error_t, omega_error_t = evalutate_error(model, t=1.0)
                    print(
                                f"Epoch {epoch}, Time: {consume_time:.5f}, Loss: {loss.item():.5f}, phy: {physics_loss.item():.5f}, "
                                f"bound: {boundary_loss.item():.5f}, initial: {initial_loss.item():.5f}, "
                                f"psi_error_t: {psi_error_t:.5f}, omega_error_t: {omega_error_t:.5f}"
                            )
                    results["epoch"].append(epoch)
                    results["time"].append(consume_time)
                    results["loss"].append(loss.item())
                    results["physics_loss"].append(physics_loss.item())
                    results["boundary_loss"].append(boundary_loss.item())
                    results["initial_loss"].append(initial_loss.item())
                    results["psi_error_t"].append(psi_error_t)
                    results["omega_error_t"].append(omega_error_t)
                    
                    start_time = time.time()
            # 定义路径
            path = f'results/PINN_lora/r={r}'
            
            # 检查路径是否存在，如果不存在则创建
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(f'results/PINN_lora/r={r}/loss_error_results_wi_{w_i:.2f}_wo_{w_o:.2f}.npy', results)
            torch.save(model.state_dict(), f'results/PINN_lora/r={r}/model_taylor_green_wi_{w_i:.2f}_wo_{w_o:.2f}.pth')
            
