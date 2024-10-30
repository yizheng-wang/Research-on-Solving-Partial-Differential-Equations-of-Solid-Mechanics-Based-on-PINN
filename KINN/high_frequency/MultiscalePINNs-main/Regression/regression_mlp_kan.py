import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models_torch import Sampler, NN_FF, NN_KAN

def setup_seed(seed):
# random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    setup_seed(2024)
    
    # Define solution and its Laplace
    def u(x, a):
        return np.sin(2*np.pi * x) + 0.1*np.cos(np.pi * a * x)

    # Define computational domain
    dom_coords = np.array([[0.0], [1.0]])

    # Training data on u(x) 
    N_u = 100
    X_u = np.linspace(dom_coords[0, 0], dom_coords[1, 0], N_u)[:, None]

    a = 50
    Y_u = u(X_u, a)
    
    
    # Define the model
    layers_MLP = [1, 100, 1]  # MLP architecture
    layers_KAN = [1, 5,5,5, 1]  # KAN architecture
    sigma = 10   # Hyper-parameter of the Fourier features
    model_mlp = NN_FF(layers_MLP, X_u, Y_u, a, u, sigma) # MLP
    model_kan = NN_KAN(layers_KAN, X_u, Y_u, a, u, sigma) # KAN
    # Train the model for different epochs
    epoch_list = [100, 900, 9000]  # 1000 iterations in total
    u_pred_mlp_list = []
    u_pred_kan_list = []
    loss_mlp = []
    loss_kan = []
# MLP 的 计算
    for epoch in epoch_list:
       # Train the model
       model_mlp.train_model(nIter=epoch, log_NTK=True, log_weights=True)
       
       # Predictions
       u_pred_mlp = model_mlp.predict_u(X_u)
       u_pred_mlp_list.append(u_pred_mlp)
       loss_mlp.extend(model_mlp.loss_u_log)
    # Evaluate the relative l2 error
    error_u_mlp = np.linalg.norm(Y_u - u_pred_mlp_list[-1], 2) / np.linalg.norm(Y_u, 2)
    print('Relative L2 error_u in MLP: {:.2e}'.format(error_u_mlp))

    # Create loggers for the eigenvalues of the NTK
    lambda_K_mlp_log = []

    # Restore the NTK
    K_mlp_list = model_mlp.K_log

    for k in range(len(K_mlp_list)):
        K_mlp = (K_mlp_list[k])

        # Compute eigenvalues
        lambda_mlp_K, eigvec_mlp_K = np.linalg.eig(K_mlp)
        
        # Sort in decreasing order
        lambda_mlp_K = np.sort(np.real(lambda_mlp_K))[::-1]
        
        # Store eigenvalues
        lambda_K_mlp_log.append(lambda_mlp_K)
        

# KAN 的 计算
    for epoch in epoch_list:
       # Train the model
       model_kan.train_model(nIter=epoch, log_NTK=True, log_weights=True)
       
       # Predictions
       u_pred_kan = model_kan.predict_u(X_u)
       u_pred_kan_list.append(u_pred_kan)
       loss_kan.extend(model_kan.loss_u_log)
    # Evaluate the relative l2 error
    error_u_kan = np.linalg.norm(Y_u - u_pred_kan_list[-1], 2) / np.linalg.norm(Y_u, 2)
    print('Relative L2 error_u in KAN: {:.2e}'.format(error_u_kan))

    # Create loggers for the eigenvalues of the NTK
    lambda_K_kan_log = []

    # Restore the NTK
    K_kan_list = model_kan.K_log

    for k in range(len(K_kan_list)):
        K_kan = (K_kan_list[k])

        # Compute eigenvalues
        lambda_kan_K, eigvec_kan_K = np.linalg.eig(K_kan)
        
        # Sort in decreasing order
        lambda_kan_K = np.sort(np.real(lambda_kan_K))[::-1]
        
        # Store eigenvalues
        lambda_K_kan_log.append(lambda_kan_K)


#%%
    # Visualize the eigenfunctions of the NTK
    fig = plt.figure(figsize=(12, 10))
    with sns.axes_style("darkgrid"):
        plt.subplot(2, 3, 1)
        plt.plot(X_u, np.real(eigvec_mlp_K[:, 0]), linestyle='--')
        plt.plot(X_u, np.real(eigvec_kan_K[:, 0]), linestyle='-')
        plt.legend(['MLP', 'KAN'])
        plt.tight_layout()
        
        plt.subplot(2, 3, 2)
        plt.plot(X_u, np.real(eigvec_mlp_K[:, 1]), linestyle='--')
        plt.plot(X_u, np.real(eigvec_kan_K[:, 1]), linestyle='-')
        plt.legend(['MLP', 'KAN'])
        plt.tight_layout()
        
        plt.subplot(2, 3, 3)
        plt.plot(X_u, np.real(eigvec_mlp_K[:, 2]), linestyle='--')
        plt.plot(X_u, np.real(eigvec_kan_K[:, 2]), linestyle='-')
        plt.legend(['MLP', 'KAN'])
        plt.tight_layout()
        
        plt.subplot(2, 3, 4)
        plt.plot(X_u, np.real(eigvec_mlp_K[:, 3]), linestyle='--')
        plt.plot(X_u, np.real(eigvec_kan_K[:, 3]), linestyle='-')
        plt.legend(['MLP', 'KAN'])
        plt.tight_layout()
    
        plt.subplot(2, 3, 5)
        plt.plot(X_u, np.real(eigvec_mlp_K[:, 4]), linestyle='--')
        plt.plot(X_u, np.real(eigvec_kan_K[:, 4]), linestyle='-')
        plt.legend(['MLP', 'KAN'])
        plt.tight_layout()
        
        plt.subplot(2, 3, 6)
        plt.plot(X_u, np.real(eigvec_mlp_K[:, 5]), linestyle='--')
        plt.plot(X_u, np.real(eigvec_kan_K[:, 5]), linestyle='-')
        plt.legend(['MLP', 'KAN'])
        plt.tight_layout()
        plt.savefig('./pic/eigenfunction_reg.pdf', dpi = 300)
        plt.show()
    
    # Visualize the eigenvalues of the NTK
    fig = plt.figure(figsize=(6, 5))
    with sns.axes_style("darkgrid"):
        plt.plot(lambda_K_mlp_log[0], linestyle='--', color = 'b', label='MLP: epoch=100')
        plt.plot(lambda_K_kan_log[0], linestyle='--', color = 'r',label='KAN: epoch=100')
        plt.plot(lambda_K_mlp_log[-1],  color = 'b', label='MLP: epoch=10000')
        plt.plot(lambda_K_kan_log[-1], color = 'r', label='KAN: epoch=10000')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('index')
        plt.ylabel(r'$\lambda$')
        plt.title('Spectrum')
        plt.tight_layout()
        plt.legend()
        plt.savefig('./pic/eigenvalue_reg.pdf', dpi = 300)
        plt.show()

    fig = plt.figure(figsize=(6, 5))
    with sns.axes_style("darkgrid"):
        plt.plot(lambda_K_mlp_log[0]/np.sum(lambda_K_mlp_log[0]), linestyle='--', color = 'b', label='MLP: epoch=100')
        plt.plot(lambda_K_kan_log[0]/np.sum(lambda_K_kan_log[0]), linestyle='--', color = 'r',label='KAN: epoch=100')
        plt.plot(lambda_K_mlp_log[-1]/np.sum(lambda_K_mlp_log[-1]),  color = 'b', label='MLP: epoch=10000')
        plt.plot(lambda_K_kan_log[-1]/np.sum(lambda_K_kan_log[-1]), color = 'r', label='KAN: epoch=10000')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('index')
        plt.ylabel(r'$\lambda$')
        plt.title('Spectrum (Relative)')
        plt.tight_layout()
        plt.legend()
        plt.savefig('./pic/relative_eigenvalue_reg.pdf', dpi = 300)
        plt.show()
        
    # Model predictions at different epochs
    fig = plt.figure(figsize=(12, 4))
    with sns.axes_style("darkgrid"):
        plt.subplot(1, 3, 1)
        plt.plot(X_u, Y_u,  color = 'r', label ='exact')
        plt.plot(X_u, u_pred_mlp_list[0], color='green', linestyle='--', label = 'MLP')
        plt.plot(X_u, u_pred_kan_list[0], color='blue', linestyle='-.', label = 'KAN')
        plt.title('Epoch = 100')
        plt.legend()
        plt.tight_layout()
        
        plt.subplot(1, 3, 2)
        plt.plot(X_u, Y_u,  color = 'r', label ='exact')
        plt.plot(X_u, u_pred_mlp_list[1], color='green', linestyle='--', label = 'MLP')
        plt.plot(X_u, u_pred_kan_list[1], color='blue', linestyle='-.', label = 'KAN')
        plt.title('Epoch = 1000')
        plt.legend()
        plt.tight_layout()
        
        plt.subplot(1, 3, 3)
        plt.plot(X_u, Y_u, color = 'r', label ='exact')
        plt.plot(X_u, u_pred_mlp_list[2], color='green', linestyle='--', label = 'MLP')
        plt.plot(X_u, u_pred_kan_list[2], color='blue', linestyle='-.', label = 'KAN')
        plt.title('Epoch = 10000')
        plt.tight_layout()
        plt.legend()
        plt.savefig('./pic/prediction_reg.pdf', dpi = 300)
        plt.show()
        
    # Visualize the loss value
    fig = plt.figure(figsize=(6, 5))
    with sns.axes_style("darkgrid"):
        plt.plot(loss_mlp, linestyle='--', label='MLP')
        plt.plot(loss_kan, label='KAN')
        plt.yscale('log')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Loss evolution')
        plt.tight_layout()
        plt.legend()
        plt.savefig('./pic/loss_reg.pdf', dpi = 300)
        plt.show()
