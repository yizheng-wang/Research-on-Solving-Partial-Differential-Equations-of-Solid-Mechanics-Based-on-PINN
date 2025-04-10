# Load the grid data and displacement data
import numpy as np
import matplotlib.pyplot as plt
f_name = "const-dist1"
x_plot_grid = np.load(f'./results/{f_name}/x_plot_grid.npy')
y_plot_grid = np.load(f'./results/{f_name}/y_plot_grid.npy')
u_exact = np.load(f'./results/{f_name}/u_exact.npy')
v_exact = np.load(f'./results/{f_name}/v_exact.npy')

# Create the 2D contour plots for u_exact and v_exact
fig, ax = plt.subplots(1, 2, figsize=(20, 3))

# Plot for u_exact
cp_u = ax[0].contourf(x_plot_grid, y_plot_grid, u_exact, 50, cmap='gist_rainbow_r')
ax[0].set_aspect('equal')
ax[0].set_title('u_exact Displacement')
fig.colorbar(cp_u, ax=ax[0])

# Plot for v_exact
cp_v = ax[1].contourf(x_plot_grid, y_plot_grid, v_exact, 50, cmap='gist_rainbow_r')
ax[1].set_aspect('equal')
ax[1].set_title('v_exact Displacement')
fig.colorbar(cp_v, ax=ax[1])

# Show the plots
plt.tight_layout()
plt.show()
