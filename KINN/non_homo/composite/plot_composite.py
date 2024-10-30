import numpy as np
import matplotlib.pyplot as plt

# Load the error data
error_L2_mlp = np.load('results/DEM_MLP/error_L2.npy')
error_L2_kinn = np.load('results/DEM_KINN/error_L2.npy')

# Load the x=0 prediction data
x0_u_pred_mlp = np.load('results/DEM_MLP/x0_u_pred.npy')
x0_u_exact = np.load('results/DEM_MLP/x0_u_exact.npy')
x0_u_pred_kinn = np.load('results/DEM_KINN/x0_u_pred.npy')

# Load the zoomed x=0 prediction data
x0_u_z_pred_mlp = np.load('results/DEM_MLP/x0_u_z_pred.npy')
x0_u_z_exact = np.load('results/DEM_MLP/x0_u_z_exact.npy')
x0_u_z_pred_kinn = np.load('results/DEM_KINN/x0_u_z_pred.npy')

# Load the dudy prediction data
x0_dudy_pred_mlp = np.load('results/DEM_MLP/x0_dudy_pred.npy')
x0_dudy_exact = np.load('results/DEM_MLP/x0_dudy_exact.npy')
x0_dudy_pred_kinn = np.load('results/DEM_KINN/x0_dudy_pred.npy')

# Plotting the graphs
# Plot 1: L2 error evolution
plt.yscale('log')
plt.plot(error_L2_mlp, label='DEM_MLP', linestyle='--')
plt.plot(error_L2_kinn, label='DEM_KINN', linestyle='-')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('L2 Error', fontsize=14)
plt.title('L2 Error Evolution', fontsize=16)
plt.legend(fontsize=12) 
plt.savefig('pic/L2_error_evolution.pdf', dpi = 300)
plt.show()

# Plot 2: x=0 prediction comparison

plt.plot(x0_u_exact[:, 0], x0_u_exact[:, 1], 'r', label='Exact Solution')
plt.plot(x0_u_pred_mlp[:, 0], x0_u_pred_mlp[:, 1], 'b--', label='DEM_MLP')
plt.plot(x0_u_pred_kinn[:, 0], x0_u_pred_kinn[:, 1], 'g-', label='DEM_KINN')
plt.xlabel('y', fontsize=14)
plt.ylabel('u', fontsize=14)
plt.title('x=0 u', fontsize=16)
plt.legend(fontsize=12) 
plt.savefig('pic/x0_u.pdf', dpi = 300)
plt.show()

# Plot 3: Zoomed x=0 prediction comparison
plt.plot(x0_u_z_exact[:, 0], x0_u_z_exact[:, 1], 'r', label='Exact Solution')
plt.plot(x0_u_z_pred_mlp[:, 0], x0_u_z_pred_mlp[:, 1], 'b--', label='DEM_MLP')
plt.plot(x0_u_z_pred_kinn[:, 0], x0_u_z_pred_kinn[:, 1], 'g-', label='DEM_KINN')
plt.xlabel('y', fontsize=14)
plt.ylabel('u', fontsize=14)
plt.title('x=0 u (Zoomed)', fontsize=16)
plt.legend(fontsize=12) 
plt.savefig('pic/x0_z_u.pdf', dpi = 300)
plt.show()

# Plot 4: dudy prediction comparison
plt.plot(x0_dudy_exact[:, 0], x0_dudy_exact[:, 1], 'r', label='Exact Solution')
plt.plot(x0_dudy_pred_mlp[:, 0], x0_dudy_pred_mlp[:, 1], 'b--', label='DEM_MLP')
plt.plot(x0_dudy_pred_kinn[:, 0], x0_dudy_pred_kinn[:, 1], 'g-', label='DEM_KINN')
plt.xlabel('y', fontsize=14)
plt.ylabel('dudy', fontsize=14)
plt.title('x=0 dudy', fontsize=16)
plt.legend(fontsize=12) 
plt.savefig('pic/x0_dudy.pdf', dpi = 300)
plt.show()