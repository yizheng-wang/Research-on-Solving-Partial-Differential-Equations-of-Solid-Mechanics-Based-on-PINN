import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
from PIL import Image
import sys
from scipy.io import loadmat
sys.path.append("../")
from kan_efficiency import *
import time
# Load and process the image

# Set random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(2024)


# Define fully connected neural network
class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.layer1 = nn.Linear(2, 200)
        self.layer2 = nn.Linear(200, 200)
        self.layer3 = nn.Linear(200, 1)
    
    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = self.layer3(x)
        return x

# =============================================================================
# first we should get the value of T and f 
# =============================================================================
T_data = loadmat(f'./FDM_code/contin_T.mat')['T']
F_data = np.ones([256,256])

K_data = loadmat(f'./FDM_code/contin_K.mat')['lognorm_a']

# =============================================================================
# # bulid coordinate to the K
# =============================================================================
model = FCNN().cuda()

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 3000, 5000], gamma = 0.1)

# Training and evaluation
num_epochs = 3000

checkpoint_epochs = [100, 300, 500, 1000, 1500, 3000]
loss_list = []
error_list = []
# fcnn_model = FCNN().cuda()

# =============================================================================
# # physical loss
# =============================================================================
# create the laplacian operator
Kernel_laplacian = torch.tensor([[[[0, 1, 0],
                                    [1, -4, 1],
                                    [0, 1, 0]]]], dtype=torch.float32).cuda() 
# Kernel_laplacian2 = torch.tensor([[[[0.5, 1, 0.5],
#                                     [1, -6, 1],
#                                     [0.5, 1, 0.5]]]], dtype=torch.float32).cuda() 
# Kernel_laplacian3 = torch.tensor([[[[0.5, 0, 0.5],
#                                     [0, -2, 0],
#                                     [0.5, 0, 0.5]]]], dtype=torch.float32).cuda() 
# create the gradient operator 
Kernel_x = torch.tensor([[[[0, 0, 0],
                           [-1, 0, 1],
                           [0, 0, 0]]]], dtype=torch.float32).cuda() 
Kernel_y = torch.tensor([[[[0, 1, 0],
                           [0, 0, 0],
                           [0, -1, 0]]]], dtype=torch.float32).cuda() 
def Finite_diff (T, kernel):
    """
    This function is for the finite differentiation of the output of the FNO, and create the gradient output for physical learning (fine tune)

    Parameters
    ----------
    T : tensor
        output of the operator learning.
    Kernel : tensor
        the kernel for the finite differentiation.

    Returns
    -------
    the output of the gradient operator such as the Laplacian operator and the gradient operation.

    """
    T = T.unsqueeze(0).unsqueeze(0)
    padding = nn.ReplicationPad2d(1) # replicate the input
    T_padding = padding(T)
    output = F.conv2d(T_padding, kernel, padding = 0)
    output = output.squeeze(0).squeeze(0)
    return output

def evaluate_model(model, epoch):
    model.eval()

    
    N_u = 256
    X_u = np.linspace(0, 1, N_u)
    Y_u = np.linspace(0, 1, N_u)
    X, Y = np.meshgrid(X_u, Y_u)
    XY = np.vstack([X.ravel(), Y.ravel()]).T
    K_data = loadmat(f'./FDM_code/contin_K.mat')['lognorm_a']

    # Convert to PyTorch tensors
    XY_tensor = torch.tensor(XY, dtype=torch.float32).cuda()
    
    K_pred = model(XY_tensor)
    K_pred = K_pred.data
    K_pred = K_pred.reshape(N_u, N_u).cpu()
    error = torch.abs(K_pred - K_data)


    
    # reference solution
    h1 = plt.contourf(X, Y, K_data, levels=100 ,cmap = 'jet')
    ax = plt.gca()
    ax.set_aspect(1)
    plt.colorbar(h1).ax.set_title('Reference')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'./PINNs_results/Ref_con_K_epoch{epoch}.pdf', dpi=300)
    plt.show()      
    

    # kan
    h1 = plt.contourf(X, Y, K_pred, levels=100 ,cmap = 'jet')
    ax = plt.gca()
    ax.set_aspect(1)
    plt.colorbar(h1).ax.set_title('MLP_pred')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'./PINNs_results/MLP_pred_con_K_epoch{epoch}.pdf', dpi=300)
    plt.show() 

    h1 = plt.contourf(X, Y, error, levels=100 ,cmap = 'jet')
    ax = plt.gca()
    ax.set_aspect(1)
    plt.colorbar(h1).ax.set_title('MLP_error')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'./PINNs_results/MLP_error_con_K_epoch{epoch}.pdf', dpi=300)
    plt.show() 


# fine-tune
start_time = time.time()




N_u = 256
X_u = np.linspace(0, 1, N_u)
Y_u = np.linspace(0, 1, N_u)
X, Y = np.meshgrid(X_u, Y_u)
XY = np.vstack([X.ravel(), Y.ravel()]).T

# Convert to PyTorch tensors
XY_tensor = torch.tensor(XY, dtype=torch.float32).cuda()

start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    
    optimizer.zero_grad()  

    K_pred_tensor = model(XY_tensor).reshape(N_u, N_u)

    T_tensor = torch.tensor(T_data, dtype=torch.float32).cuda()
    F_tensor = torch.tensor(F_data, dtype=torch.float32).cuda()
    
    dx = 1/(N_u-1)
    
    Tx = Finite_diff(T_tensor, Kernel_x)/2/dx
    Ty = Finite_diff(T_tensor, Kernel_y)/2/dx
    Kx = Finite_diff(K_pred_tensor, Kernel_x)/2/dx
    Ky = Finite_diff(K_pred_tensor, Kernel_y)/2/dx 
    Txxyy = Finite_diff(T_tensor, Kernel_laplacian)/dx**2     
    
    #loss = torch.mean(( K_pred_tensor * Txxyy + F_tensor)**2)
    phy_loss = (Kx*Tx + Ky * Ty + K_pred_tensor * Txxyy + F_tensor)**2
    
    truncate_bound = 1 # 边界上由于是padding的，损失不物理，删除了
    
    
    
    loss = torch.mean(phy_loss[truncate_bound:-truncate_bound, truncate_bound:-truncate_bound])
    loss.backward()
    optimizer.step() # the trainable parameter is being undated.
    # error = myloss(out_gpu.cpu().view(1,-1), y_test_pl.reshape(1,-1))
    scheduler.step()

    
    loss_list.append(loss.detach().data.cpu().numpy())
    L2 = np.linalg.norm(K_pred_tensor.data.cpu().numpy() - K_data) / np.linalg.norm(K_data)
    error_list.append(L2)
    if (epoch + 1) % 10 == 0:
        end_time = time.time()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Phy_loss: {loss.item():.4f}, L2 error: {L2:.4f}, time: {end_time-start_time:.4f}')

    if (epoch + 1) in checkpoint_epochs:
        evaluate_model(model, epoch + 1)
torch.save(model.state_dict(), './models/model_PINNs_MLP.pth')


np.save('PINNs_results/con_K_MLP_loss.npy', loss_list)
np.save('PINNs_results/con_K_MLP_l2.npy', error_list)

# Plot training error
plt.figure(figsize=(10, 6))
plt.plot(loss_list, 'b-', label='MLP Error')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.title('Error evolution')
plt.savefig('./PINNs_results/loss_con_K_MLP.pdf', dpi=300)
plt.show()


