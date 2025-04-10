import numpy as np
import matplotlib.pyplot as plt

# Define the grid and parameters
x = np.linspace(0, 1, 30)
y = np.linspace(0, 1, 30)
X, Y = np.meshgrid(x, y)

# Set omega values for pi, 2pi, 3pi
omegas = [np.pi, 2*np.pi, 3*np.pi]

# Create a figure for subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Iterate over omega values and plot
for i, omega in enumerate(omegas):
    u = -np.cos(omega * X) * np.sin(omega * Y)
    v = np.sin(omega * X) * np.cos(omega * Y)
    axes[i].quiver(X, Y, u, v, scale=15, color='blue')
    axes[i].set_title(f"Taylor-Green Vortex Velocity Field (w = {i+1}*pi)", fontsize = 15)
    axes[i].set_xlabel('x', fontsize = 20)
    axes[i].set_ylabel('y', fontsize = 20)
    axes[i].set_xlim([0, 1])
    axes[i].set_ylim([0, 1])
    axes[i].set_aspect('equal', adjustable='box')

# Show the plot with all three omega values
plt.tight_layout()
plt.savefig('taylor_green_intro.pdf', dpi = 500)
plt.show()
