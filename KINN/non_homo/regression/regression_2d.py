import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
import sys
sys.path.append("../..")
from kan_efficiency import *
def setup_seed(seed):
# random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(2024)
# 定义函数 u(x)
def u(xy):
    r = torch.norm(xy, dim=1)
    alpha1 = 1/15
    alpha2 = 1
    r0 = 0.5
    u_values = torch.where(r < r0,
                           r**4 / alpha1,
                           r**4 / alpha2 + r0**4 * (1/alpha1 - 1/alpha2))
    return u_values

# 生成训练数据
N_u = 100
X_u = np.linspace(-1, 1, N_u)
Y_u = np.linspace(-1, 1, N_u)
X, Y = np.meshgrid(X_u, Y_u)
XY = np.vstack([X.ravel(), Y.ravel()]).T
U = u(torch.tensor(XY, dtype=torch.float32)).numpy()

# 转换为PyTorch张量
XY_tensor = torch.tensor(XY, dtype=torch.float32).cuda()
U_tensor = torch.tensor(U, dtype=torch.float32).reshape(-1, 1).cuda()

# 定义全连接神经网络
class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.layer1 = nn.Linear(2, 100)
        self.layer2 = nn.Linear(100, 100)
        self.layer3 = nn.Linear(100, 1)
    
    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = self.layer3(x)
        return x

# 实例化模型、损失函数和优化器
kan_model = KAN([2, 5, 5, 1], base_activation=torch.nn.SiLU, grid_size=20, grid_range=[-1, 1.0], spline_order=3).cuda()
fcnn_model = FCNN().cuda()
criterion = nn.MSELoss()
kan_optimizer = optim.Adam(kan_model.parameters(), lr=0.001)
fcnn_optimizer = optim.Adam(fcnn_model.parameters(), lr=0.001)

# 训练模型
num_epochs = 30000
kan_error_list = []
fcnn_error_list = []
checkpoint_epochs = [1000, 10000, 30000]

def train_and_evaluate(model1,  model2, optimizer1, optimizer2, error_list1, error_list2):
    for epoch in range(num_epochs):
        model1.train()
        optimizer1.zero_grad()
        outputs1 = model1(XY_tensor)
        loss1 = criterion(outputs1, U_tensor)
        loss1.backward()
        error_list1.append(loss1.detach().data.cpu().numpy())
        optimizer1.step()
    
        model2.train()
        optimizer2.zero_grad()
        outputs2 = model2(XY_tensor)
        loss2 = criterion(outputs2, U_tensor)
        loss2.backward()
        error_list2.append(loss2.detach().data.cpu().numpy())
        optimizer2.step()
        if (epoch+1) % 500 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}')            

        if (epoch+1) in checkpoint_epochs:
            evaluate_model(model1, model2, epoch+1)

def evaluate_model(kan_model, fcnn_model, epoch):
    kan_model.eval()
    fcnn_model.eval()
    n_test = 100
    r0 = 0.5
    alpha1 = 1/15
    alpha2 = 1

    # x=0的位移预测
    y0 = np.linspace(-1, 1, n_test)
    x0 = np.zeros_like(y0)
    x0y0 = np.vstack([x0, y0]).T
    exact_u_x0 = u(torch.tensor(x0y0, dtype=torch.float32)).numpy()
    x0y0_tensor = torch.tensor(x0y0, dtype=torch.float32).cuda()
    pred_u_x0_kan = kan_model(x0y0_tensor).cpu().data.numpy()
    pred_u_x0_fcnn = fcnn_model(x0y0_tensor).cpu().data.numpy()

    plt.plot(y0, exact_u_x0, 'b-', label='Exact u(x=0, y)')
    plt.plot(y0, pred_u_x0_kan, 'r--', label=f'KAN Predicted u(x=0, y)')
    plt.plot(y0, pred_u_x0_fcnn, 'g--', label=f'MLP Predicted u(x=0, y)')
    plt.xlabel('y')
    plt.ylabel('u')
    plt.legend(fontsize=14)
    plt.title(f'Prediction of u at x=0 (Epoch {epoch})')
    plt.savefig(f'./pic/x0_regress2d_heter_{epoch}.pdf', dpi=300)
    plt.show()

    # 在0.5附近放大
    Y_u_05 = np.linspace(0.49, 0.51, N_u)
    x0 = np.zeros_like(Y_u_05)
    x0y0 = np.vstack([x0, Y_u_05]).T
    exact_u_x0 = u(torch.tensor(x0y0, dtype=torch.float32)).numpy()
    x0y0_tensor = torch.tensor(x0y0, dtype=torch.float32).cuda()
    pred_u_x0_kan = kan_model(x0y0_tensor).cpu().data.numpy()
    pred_u_x0_fcnn = fcnn_model(x0y0_tensor).cpu().data.numpy()

    plt.plot(Y_u_05, exact_u_x0, 'b-', label='Exact u(x=0, y)')
    plt.plot(Y_u_05, pred_u_x0_kan, 'r--', label=f'KAN Predicted u(x=0, y)')
    plt.plot(Y_u_05, pred_u_x0_fcnn, 'g--', label=f'MLP Predicted u(x=0, y)')
    plt.xlabel('y')
    plt.ylabel('u')
    plt.legend(fontsize=14)
    plt.title(f'Prediction of u at x=0 (Epoch {epoch})')
    plt.savefig(f'./pic/x0zoom_regress2d_heter_{epoch}.pdf', dpi = 300)
    plt.show()

    x0 = np.zeros((n_test, 2))
    x0[:, 1] = np.linspace(-1, 1, 100)
    exactdx0 = np.zeros((n_test, 1))
    exactdx0[np.linalg.norm(x0, axis=1)<r0] =  4/alpha1*np.linalg.norm(x0[np.linalg.norm(x0, axis=1)<r0], axis=1, keepdims=True)**3
    exactdx0[np.linalg.norm(x0, axis=1)>=r0] =  4/alpha2*np.linalg.norm(x0[np.linalg.norm(x0, axis=1)>=r0], axis=1, keepdims=True)**3
    x0t = torch.tensor(x0.astype(np.float32), requires_grad=True).cuda()
    predx0_kan = kan_model(x0t)
    predx0_fcnn = fcnn_model(x0t)
    dudxy_kan = grad(predx0_kan, x0t, torch.ones(x0t.size()[0], 1).cuda(), create_graph=True)[0]
    dudxy_fcnn = grad(predx0_fcnn, x0t, torch.ones(x0t.size()[0], 1).cuda(), create_graph=True)[0]
    dudx_kan = dudxy_kan[:, 0].unsqueeze(1).data.cpu().numpy()
    dudy_kan = dudxy_kan[:, 1].unsqueeze(1).data.cpu().numpy()
    dudx_fcnn = dudxy_fcnn[:, 0].unsqueeze(1).data.cpu().numpy()
    dudy_fcnn = dudxy_fcnn[:, 1].unsqueeze(1).data.cpu().numpy()
    dudy_kan[x0[:, 1]<0] = -dudy_kan[x0[:, 1]<0]
    dudy_fcnn[x0[:, 1]<0] = -dudy_fcnn[x0[:, 1]<0]

    plt.plot(x0[:, 1], exactdx0.flatten(), 'b-', label='Exact')
    plt.plot(x0[:, 1], dudy_kan.flatten(), 'r--', label='KAN Pred')
    plt.plot(x0[:, 1], dudy_fcnn.flatten(), 'g--', label='MLP Pred')
    plt.xlabel('y')
    plt.ylabel('u')
    plt.legend(fontsize=14)
    plt.title(f'x=0 dudy (Epoch {epoch})')
    plt.savefig(f'./pic/dudy0zoom_regress2d_heter_{epoch}.pdf', dpi = 300)
    plt.show()

# 预测
train_and_evaluate(kan_model,fcnn_model, kan_optimizer, fcnn_optimizer, kan_error_list,fcnn_error_list)

# 绘制误差图
plt.figure(figsize=(10, 6))
plt.plot(kan_error_list, 'b-', label='KAN Error')
plt.plot(fcnn_error_list, 'r--', label='MLP Error')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.title('Training Error')
plt.savefig('./pic/loss_regress2d_heter.pdf', dpi = 300)
plt.show()
