import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../..") 
from kan_efficiency import *
# Define the neural network
class MultiLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MultiLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, D_out)

        torch.nn.init.normal_(self.linear1.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear2.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear3.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear4.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear5.bias, mean=0, std=1)

        torch.nn.init.normal_(self.linear1.weight, mean=0, std=np.sqrt(2/(D_in+H)))
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear3.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear4.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear5.weight, mean=0, std=np.sqrt(2/(H+D_out)))

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
       # w = torch.tensor([200, 1]).cuda()
        yt = x
        y1 = torch.tanh(self.linear1(yt))
        y2 = torch.tanh(self.linear2(y1))
        y3 = torch.tanh(self.linear3(y2)) + y1
        y4 = torch.tanh(self.linear4(y3)) + y2
        y =  self.linear5(y4)
        # y = torch.sin(500 * np.pi * x[:,0].unsqueeze(1)) * torch.exp(-x[:,1].unsqueeze(1))
        return y

# Define the PDE residual
def pde_residual(model, xt):
    x = xt[:, 0].unsqueeze(1)
    t = xt[:, 1].unsqueeze(1)
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(torch.cat([x, t], dim=1))
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    return u_t - (1 / (fre * np.pi)**2) * u_xx

# Boundary condition
def boundary(x):
    return torch.zeros_like(x)

# Initial condition
def initial_cond(x):
    return torch.sin(fre * np.pi * x)

# Training data
def generate_data():
    x = np.random.rand(4000, 1)
    t = np.random.rand(4000, 1)
    tb = np.linspace(0, 1, 100).reshape(-1, 1)
    xb = np.zeros_like(tb)
    xb1 = np.ones_like(tb)
    xt = np.random.rand(200, 1)
    tt = np.zeros_like(xt)
    return x, t, tb, xb, xb1, xt, tt

# Convert to tensors
def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)

def cal_error(xy):
    u_pred = model(xy).detach().cpu().numpy().flatten()
    u_exact = np.sin(fre * np.pi * xy[:,0].cpu().numpy()) * np.exp(-xy[:,1].cpu().numpy())
    L2 = np.linalg.norm(u_pred-u_exact)/np.linalg.norm(u_exact)
    return L2

# Training the PINN
def train(model, optimizer, epochs):
    for epoch in range(epochs):
        optimizer.zero_grad()

        x, t, tb, xb, xb1, xt, tt = generate_data()
        x = to_tensor(x).cuda()
        t = to_tensor(t).cuda()
        tb = to_tensor(tb).cuda()
        xb = to_tensor(xb).cuda()
        xb1 = to_tensor(xb1).cuda()
        xt = to_tensor(xt).cuda()
        tt = to_tensor(tt).cuda()

        u_pred = model(torch.cat([xb, tb], dim=1))
        bc_loss = torch.mean((u_pred - boundary(xb))**2)

        u_pred1 = model(torch.cat([xb1, tb], dim=1))
        bc_loss1 = torch.mean((u_pred1 - boundary(xb))**2)

        u_pred_ic = model(torch.cat([xt, tt], dim=1))
        ic_loss = torch.mean((u_pred_ic - initial_cond(xt))**2)

        res = pde_residual(model, torch.cat([x, t], dim=1))
        pde_loss = torch.mean(res**2)

        loss = pde_loss + (bc_loss + bc_loss1) + ic_loss
        loss.backward()
        optimizer.step()
        l2_error = cal_error(torch.cat([x, t], dim=1))
        L2_error_list.append(l2_error)
        loss_r_list.append(pde_loss.data.cpu().numpy())
        loss_b_list.append((bc_loss+bc_loss1).data.cpu().numpy())
        loss_i_list.append(ic_loss.data.cpu().numpy())
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss_r: {pde_loss.item()}, Loss_b: {(bc_loss + bc_loss1).item()}, Loss_i: {ic_loss.item()}, Error: {l2_error}')

# Plot the results
def plot_results(model):
    x = np.linspace(0, 1, 100).reshape(-1, 1)
    t = np.linspace(0, 1, 100).reshape(-1, 1)
    X, T = np.meshgrid(x, t)
    X_test = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    X_test = to_tensor(X_test).cuda()
    y_test = model(X_test).detach().cpu().numpy()
    Y_test = y_test.reshape(100, 100)

 
    plt.pcolormesh(T, X, np.sin(fre * np.pi * X) * np.exp(-T), shading='auto')
    plt.colorbar()
    plt.title('Exact $u(x,t)$')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.gca()
    plt.show()

    plt.pcolormesh(T, X, Y_test, shading='auto')
    plt.colorbar()
    plt.title('Predicted $u(x,t)$')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.gca()
    plt.savefig('pic/Pred_u_mlp.pdf', dpi = 300)
    plt.show()

    plt.pcolormesh(T, X, np.abs(Y_test - np.sin(fre * np.pi * X) * np.exp(-T)), shading='auto')
    plt.colorbar()
    plt.title('Absolute error')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.gca()
    plt.savefig('pic/Abs_u_mlp.pdf', dpi = 300)
    plt.show()

# Main execution
fre = 50
L2_error_list = []
loss_r_list = []
loss_b_list = []
loss_i_list = []
model = MultiLayerNet(2, 30, 1).cuda()
# model = KAN([2, 5, 1], base_activation=torch.nn.SiLU, grid_size=100, grid_range=[0, 1.0], spline_order=3).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train(model, optimizer, epochs=5000)
plot_results(model)

#%%
plt.yscale('log')
plt.plot(L2_error_list)
plt.xlabel('the iteration')
plt.ylabel('loss')
plt.title('error evolution') 