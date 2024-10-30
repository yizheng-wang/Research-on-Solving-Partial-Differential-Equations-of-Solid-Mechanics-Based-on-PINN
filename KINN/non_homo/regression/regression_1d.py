
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../..") 
from kan_efficiency import *
# 定义函数 u(x)
def u(r):
    alpha1 = 1/15
    alpha2 = 1
    r0 = 0.5
    u_values = np.where(r < r0, 
                        r**4 / alpha1, 
                        r**4 / alpha2 + r0**4 * (1/alpha1 - 1/alpha2))
    return u_values

# 生成训练数据
N_u = 1000
X_u = np.linspace(0, 1, N_u)[:, None]
Y_u = u(X_u)

# 转换为PyTorch张量
X_u_tensor = torch.tensor(X_u, dtype=torch.float32)
Y_u_tensor = torch.tensor(Y_u, dtype=torch.float32)

# 定义全连接神经网络
class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.layer1 = nn.Linear(1, 100)
        self.layer2 = nn.Linear(100, 100)
        self.layer3 = nn.Linear(100, 1)
    
    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = self.layer3(x)
        return x

# 实例化模型、损失函数和优化器

#model = KAN([1, 5,5, 1], base_activation=torch.nn.SiLU, grid_size=20, grid_range=[0, 1.0], spline_order=3)#.cuda()
model = FCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 5000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_u_tensor)
    loss = criterion(outputs, Y_u_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 500 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 预测
model.eval()
X_test = np.linspace(0, 1, 1000)[:, None]
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_pred_tensor = model(X_test_tensor)
Y_pred = Y_pred_tensor.detach().numpy()

# 可视化结果
plt.figure(figsize=(10, 6))
plt.plot(X_u, Y_u, 'b-', label='True Function')
plt.plot(X_test, Y_pred, 'r--', label='Fitted Function')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.title('Function Approximation using Fully Connected Neural Network')
plt.show()

# 可视化结果
# 在0.5附近放大
N_u = 1000
X_u_05 = np.linspace(0.49, 0.51, N_u)[:, None]
Y_u_05 = u(X_u_05)

X_test_05 = np.linspace(0.49, 0.51, N_u)[:, None]
X_test_tensor_05 = torch.tensor(X_test_05, dtype=torch.float32)
Y_pred_tensor_05 = model(X_test_tensor_05)
Y_pred_05 = Y_pred_tensor_05.detach().numpy()

plt.figure(figsize=(10, 6))
plt.plot(X_u_05, Y_u_05, 'b-', label='True Function')
plt.plot(X_test_05, Y_pred_05, 'r--', label='Fitted Function')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.title('Function Approximation using Fully Connected Neural Network')
plt.show()

