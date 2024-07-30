# for comparison of FEM as reference solution, DEM and DCEM
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import torch

class particular(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(particular, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, D_out)
        
        #self.a1 = torch.nn.Parameter(torch.Tensor([0.2]))
        

        self.a1 = torch.Tensor([0.1]).cuda()
        self.a2 = torch.Tensor([0.1])
        self.a3 = torch.Tensor([0.1])
        self.n = 1/self.a1.data.cuda()


        torch.nn.init.normal_(self.linear1.weight, mean=0, std=np.sqrt(2/(D_in+H)))
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear3.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear4.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear5.weight, mean=0, std=np.sqrt(2/(H+D_out)))        
        
        torch.nn.init.normal_(self.linear1.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear2.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear3.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear4.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear5.bias, mean=0, std=1)
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        y1 = torch.tanh(self.n*self.a1*self.linear1(x))
        y2 = torch.tanh(self.n*self.a1*self.linear2(y1))
        y3 = torch.tanh(self.n*self.a1*self.linear3(y2))
        y4 = torch.tanh(self.n*self.a1*self.linear4(y3))
        y = self.n*self.a1*self.linear5(y4)
        return y

from DCEM_rectangle import (nepoch as DCEM_nepoch,
                            dom as DCEM_dom,
                            pred_sigma_x as DCEM_stress11, \
                            pred_sigma_y as DCEM_stress22, \
                            pred_sigma_xy as DCEM_stress12, \
                            pred_von as  DCEM_mise,
                            error_sigma_x_array as DCEM_error_sigma_x_array,
                            error_sigma_y_array as DCEM_error_sigma_y_array,
                            error_sigma_xy_array as DCEM_error_sigma_xy_array,
                            error_sigma_mise_array as DCEM_error_sigma_mise_array,
                            FEM_node_x05 as FEM_node_x05, FEM_mise_x05 as FEM_mise_x05,   mise_pred_x05 as DCEM_mise_x05,
                            FEM_node_x10 as FEM_node_x10, FEM_mise_x10 as FEM_mise_x10,   mise_pred_x10 as DCEM_mise_x10,
                            FEM_node_y05 as FEM_node_y05, FEM_mise_y05 as FEM_mise_y05,   mise_pred_y05 as DCEM_mise_y05)
from DEM_rectangle import ( nepoch as DEM_nepoch,
                            dom as DEM_dom,
                            pred_sigma_x as DEM_stress11, 
                            pred_sigma_y as DEM_stress22, 
                            pred_sigma_xy as DEM_stress12, 
                            pred_mise as  DEM_mise,
                            error_sigma_x_array as DEM_error_sigma_x_array,
                            error_sigma_y_array as DEM_error_sigma_y_array,
                            error_sigma_xy_array as DEM_error_sigma_xy_array,
                            error_sigma_mise_array as DEM_error_sigma_mise_array,
                            FEM_node_x05 as FEM_node_x05, FEM_mise_x05 as FEM_mise_x05,   mise_pred_x05 as DEM_mise_x05,
                            FEM_node_x10 as FEM_node_x10, FEM_mise_x10 as FEM_mise_x10,   mise_pred_x10 as DEM_mise_x10,
                            FEM_node_y05 as FEM_node_y05, FEM_mise_y05 as FEM_mise_y05,   mise_pred_y05 as DEM_mise_y05)
    

    #%%
# =============================================================================
# # FEM result
# =============================================================================
mpl.rcParams['figure.dpi'] = 100
node_coordinate_abaqus_rectangle = np.load("node_coordinate_abaqus_rectangle.npy")
node_stress_abaqus_rectangle = np.load("node_stress_abaqus_rectangle.npy")

FEM_mise = node_stress_abaqus_rectangle[:,0]
FEM_stress11 = node_stress_abaqus_rectangle[:,1]
FEM_stress22 = node_stress_abaqus_rectangle[:,2]
FEM_stress12 = node_stress_abaqus_rectangle[:,3]

# SIGMA mise

plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(5.8, 4.8)) 
surf = ax.scatter(node_coordinate_abaqus_rectangle[:, 0], node_coordinate_abaqus_rectangle[:, 1], c = FEM_mise, cmap=cm.rainbow, vmin=0, vmax=150.0)# vmin=0, vmax=0.14, 
#cbar = fig.colorbar(ax)
cb = fig.colorbar(surf)
cb.ax.locator_params(nbins=7)
cb.ax.tick_params(labelsize=16)
#cb.set_label(label =r'$\sigma_xx (MPa)$', fontsize=16)
#cb.set_label(fontsize=16)
ax.axis('equal')
ax.set_xlabel('X Position (mm)', fontsize=18)
ax.set_ylabel('Y Position (mm)', fontsize=18)
for tick in ax.get_xticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
for tick in ax.get_yticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
plt.savefig('./rectangle_pic/FEM_mise_rectangle.png', dpi=1000, transparent=True)
plt.show()


# SIGMA 11

plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(5.8, 4.8)) 
surf = ax.scatter(node_coordinate_abaqus_rectangle[:, 0], node_coordinate_abaqus_rectangle[:, 1], c = FEM_stress11, cmap=cm.rainbow, vmin=0, vmax=150.0)# vmin=0, vmax=0.14, 
#cbar = fig.colorbar(ax)
cb = fig.colorbar(surf)
cb.ax.locator_params(nbins=7)
cb.ax.tick_params(labelsize=16)
#cb.set_label(label =r'$\sigma_xx (MPa)$', fontsize=16)
#cb.set_label(fontsize=16)
ax.axis('equal')
ax.set_xlabel('X Position (mm)', fontsize=18)
ax.set_ylabel('Y Position (mm)', fontsize=18)
for tick in ax.get_xticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
for tick in ax.get_yticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
plt.savefig('./rectangle_pic/FEM_sigma11_rectangle.png', dpi=1000, transparent=True)
plt.show()


# SIGMA 22

plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(5.8, 4.8)) 
surf = ax.scatter(node_coordinate_abaqus_rectangle[:, 0], node_coordinate_abaqus_rectangle[:, 1], c = FEM_stress22, cmap=cm.rainbow)# vmin=0, vmax=0.14, 
#cbar = fig.colorbar(ax)
cb = fig.colorbar(surf)
cb.ax.locator_params(nbins=7)
cb.ax.tick_params(labelsize=16)
#cb.set_label(label =r'$\sigma_xx (MPa)$', fontsize=16)
#cb.set_label(fontsize=16)
ax.axis('equal')
ax.set_xlabel('X Position (mm)', fontsize=18)
ax.set_ylabel('Y Position (mm)', fontsize=18)
for tick in ax.get_xticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
for tick in ax.get_yticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
plt.savefig('./rectangle_pic/FEM_sigma22_rectangle.png', dpi=1000, transparent=True)
plt.show()


# SIGMA 12

plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(5.8, 4.8)) 
surf = ax.scatter(node_coordinate_abaqus_rectangle[:, 0], node_coordinate_abaqus_rectangle[:, 1], c = FEM_stress12, cmap=cm.rainbow, vmin=-25.0, vmax=25.0)
#cbar = fig.colorbar(ax)
cb = fig.colorbar(surf)
cb.ax.locator_params(nbins=7)
cb.ax.tick_params(labelsize=16)
#cb.set_label(label =r'$\sigma_xx (MPa)$', fontsize=16)
#cb.set_label(fontsize=16)
ax.axis('equal')
ax.set_xlabel('X Position (mm)', fontsize=18)
ax.set_ylabel('Y Position (mm)', fontsize=18)
for tick in ax.get_xticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
for tick in ax.get_yticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
plt.savefig('./rectangle_pic/FEM_sigma12_rectangle.png', dpi=1000, transparent=True)
plt.show()


# =============================================================================
# # DEM result
# =============================================================================


# SIGMA mise

plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(5.8, 4.8)) 
surf = ax.scatter(DEM_dom[:, 0], DEM_dom[:, 1], c = DEM_mise, cmap=cm.rainbow, vmin=0, vmax=150.0)# vmin=0, vmax=0.14, 
#cbar = fig.colorbar(ax)
cb = fig.colorbar(surf)
cb.ax.locator_params(nbins=7)
cb.ax.tick_params(labelsize=16)
#cb.set_label(label =r'$\sigma_xx (MPa)$', fontsize=16)
#cb.set_label(fontsize=16)
ax.axis('equal')
ax.set_xlabel('X Position (mm)', fontsize=18)
ax.set_ylabel('Y Position (mm)', fontsize=18)
for tick in ax.get_xticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
for tick in ax.get_yticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
plt.savefig('./rectangle_pic/DEM_mise_rectangle.png', dpi=1000, transparent=True)
plt.show()


# SIGMA 11

plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(5.8, 4.8)) 
surf = ax.scatter(DEM_dom[:, 0], DEM_dom[:, 1], c = DEM_stress11, cmap=cm.rainbow, vmin=0, vmax=150.0)# vmin=0, vmax=0.14, 
#cbar = fig.colorbar(ax)
cb = fig.colorbar(surf)
cb.ax.locator_params(nbins=7)
cb.ax.tick_params(labelsize=16)
#cb.set_label(label =r'$\sigma_xx (MPa)$', fontsize=16)
#cb.set_label(fontsize=16)
ax.axis('equal')
ax.set_xlabel('X Position (mm)', fontsize=18)
ax.set_ylabel('Y Position (mm)', fontsize=18)
for tick in ax.get_xticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
for tick in ax.get_yticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
plt.savefig('./rectangle_pic/DEM_sigma11_rectangle.png', dpi=1000, transparent=True)
plt.show()


# SIGMA 22

plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(5.8, 4.8)) 
surf = ax.scatter(DEM_dom[:, 0], DEM_dom[:, 1], c = DEM_stress22, cmap=cm.rainbow)# vmin=0, vmax=0.14, 
#cbar = fig.colorbar(ax)
cb = fig.colorbar(surf)
cb.ax.locator_params(nbins=7)
cb.ax.tick_params(labelsize=16)
#cb.set_label(label =r'$\sigma_xx (MPa)$', fontsize=16)
#cb.set_label(fontsize=16)
ax.axis('equal')
ax.set_xlabel('X Position (mm)', fontsize=18)
ax.set_ylabel('Y Position (mm)', fontsize=18)
for tick in ax.get_xticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
for tick in ax.get_yticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
plt.savefig('./rectangle_pic/DEM_sigma22_rectangle.png', dpi=1000, transparent=True)
plt.show()


# SIGMA 12

plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(5.8, 4.8)) 
surf = ax.scatter(DEM_dom[:, 0], DEM_dom[:, 1], c = DEM_stress12, cmap=cm.rainbow, vmin=-25.0, vmax=25.0)
#cbar = fig.colorbar(ax)
cb = fig.colorbar(surf)
cb.ax.locator_params(nbins=7)
cb.ax.tick_params(labelsize=16)
#cb.set_label(label =r'$\sigma_xx (MPa)$', fontsize=16)
#cb.set_label(fontsize=16)
ax.axis('equal')
ax.set_xlabel('X Position (mm)', fontsize=18)
ax.set_ylabel('Y Position (mm)', fontsize=18)
for tick in ax.get_xticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
for tick in ax.get_yticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
plt.savefig('./rectangle_pic/DEM_sigma12_rectangle.png', dpi=1000, transparent=True)
plt.show()


# =============================================================================
# # DCEM result
# =============================================================================


# SIGMA mise

plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(5.8, 4.8)) 
surf = ax.scatter(DCEM_dom[:, 0], DCEM_dom[:, 1], c = DCEM_mise, cmap=cm.rainbow, vmin=0, vmax=150.0)# vmin=0, vmax=0.14, 
#cbar = fig.colorbar(ax)
cb = fig.colorbar(surf)
cb.ax.locator_params(nbins=7)
cb.ax.tick_params(labelsize=16)
#cb.set_label(label =r'$\sigma_xx (MPa)$', fontsize=16)
#cb.set_label(fontsize=16)
ax.axis('equal')
ax.set_xlabel('X Position (mm)', fontsize=18)
ax.set_ylabel('Y Position (mm)', fontsize=18)
for tick in ax.get_xticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
for tick in ax.get_yticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
plt.savefig('./rectangle_pic/DCEM_mise_rectangle.png', dpi=1000, transparent=True)
plt.show()


# SIGMA 11

plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(5.8, 4.8)) 
surf = ax.scatter(DCEM_dom[:, 0], DCEM_dom[:, 1], c = DCEM_stress11, cmap=cm.rainbow, vmin=0, vmax=150.0)# vmin=0, vmax=0.14, 
#cbar = fig.colorbar(ax)
cb = fig.colorbar(surf)
cb.ax.locator_params(nbins=7)
cb.ax.tick_params(labelsize=16)
#cb.set_label(label =r'$\sigma_xx (MPa)$', fontsize=16)
#cb.set_label(fontsize=16)
ax.axis('equal')
ax.set_xlabel('X Position (mm)', fontsize=18)
ax.set_ylabel('Y Position (mm)', fontsize=18)
for tick in ax.get_xticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
for tick in ax.get_yticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
plt.savefig('./rectangle_pic/DCEM_sigma11_rectangle.png', dpi=1000, transparent=True)
plt.show()


# SIGMA 22

plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(5.8, 4.8)) 
surf = ax.scatter(DCEM_dom[:, 0], DCEM_dom[:, 1], c = DCEM_stress22, cmap=cm.rainbow)# vmin=0, vmax=0.14, 
#cbar = fig.colorbar(ax)
cb = fig.colorbar(surf)
cb.ax.locator_params(nbins=7)
cb.ax.tick_params(labelsize=16)
#cb.set_label(label =r'$\sigma_xx (MPa)$', fontsize=16)
#cb.set_label(fontsize=16)
ax.axis('equal')
ax.set_xlabel('X Position (mm)', fontsize=18)
ax.set_ylabel('Y Position (mm)', fontsize=18)
for tick in ax.get_xticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
for tick in ax.get_yticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
plt.savefig('./rectangle_pic/DCEM_sigma22_rectangle.png', dpi=1000, transparent=True)
plt.show()


# SIGMA 12

plt.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(5.8, 4.8)) 
surf = ax.scatter(DCEM_dom[:, 0], DCEM_dom[:, 1], c = DCEM_stress12, cmap=cm.rainbow, vmin=-25.0, vmax=25.0) 
#cbar = fig.colorbar(ax)
cb = fig.colorbar(surf)
cb.ax.locator_params(nbins=7)
cb.ax.tick_params(labelsize=16)
#cb.set_label(label =r'$\sigma_xx (MPa)$', fontsize=16)
#cb.set_label(fontsize=16)
ax.axis('equal')
ax.set_xlabel('X Position (mm)', fontsize=18)
ax.set_ylabel('Y Position (mm)', fontsize=18)
for tick in ax.get_xticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
for tick in ax.get_yticklabels():
    #tick.set_fontname('Times New Roman')
    tick.set_fontsize(16)
plt.savefig('./rectangle_pic/DCEM_sigma12_rectangle.png', dpi=1000, transparent=True)
plt.show()

#%%
# =============================================================================
# X05
# =============================================================================

plt.figure(figsize=(10, 7))
plt.plot(FEM_node_x05[:, 1][:int(len(FEM_node_x05)/2+1)] , FEM_mise_x05[:int(len(FEM_node_x05)/2+1)], c='r')
plt.plot(FEM_node_x05[:, 1][:int(len(FEM_node_x05)/2+1)], DEM_mise_x05[:int(len(FEM_node_x05)/2+1)], linestyle='-.')
plt.plot(FEM_node_x05[:, 1][:int(len(FEM_node_x05)/2+1)], DCEM_mise_x05[:int(len(FEM_node_x05)/2+1)], linestyle=':', marker='*')
plt.title("x=0.5", fontsize=20)
plt.legend([ 'FEM', 'DEM', 'DCEM'], fontsize=15)
plt.xlabel('y', fontsize=20)
plt.ylabel('VonMises', fontsize=20)
plt.savefig('./rectangle_pic/x05_mise_rectangle.png', dpi=1000, transparent=True)
plt.show()

# =============================================================================
# X10
# =============================================================================
plt.figure(figsize=(10, 7))
plt.plot(FEM_node_x10[:, 1][:int(len(FEM_node_x10)/2+1)] , FEM_mise_x10[:int(len(FEM_node_x10)/2+1)], c='r')
plt.plot(FEM_node_x10[:, 1][:int(len(FEM_node_x10)/2+1)], DEM_mise_x10[:int(len(FEM_node_x10)/2+1)], linestyle='-.')
plt.plot(FEM_node_x10[:, 1][:int(len(FEM_node_x10)/2+1)], DCEM_mise_x10[:int(len(FEM_node_x10)/2+1)], linestyle=':', marker='*')
plt.title("x=1.0", fontsize=20)
plt.legend([ 'FEM', 'DEM', 'DCEM'], fontsize=15, loc='upper right')
plt.xlabel('y', fontsize=20)
plt.ylabel('VonMises', fontsize=20)
plt.savefig('./rectangle_pic/x10_mise_rectangle.png', dpi=1000, transparent=True)
plt.show()

# =============================================================================
# y05
# =============================================================================
plt.figure(figsize=(10, 7))
plt.plot(FEM_node_y05[:, 0][:int(len(FEM_node_y05)/2+1)] , FEM_mise_y05[:int(len(FEM_node_y05)/2+1)], c='r')
plt.plot(FEM_node_y05[:, 0][:int(len(FEM_node_y05)/2+1)], DEM_mise_y05[:int(len(FEM_node_y05)/2+1)], linestyle='-.')
plt.plot(FEM_node_y05[:, 0][:int(len(FEM_node_y05)/2+1)], DCEM_mise_y05[:int(len(FEM_node_y05)/2+1)], linestyle=':', marker='*')
plt.title("y=0.5", fontsize=20)
plt.legend([ 'FEM', 'DEM', 'DCEM'], fontsize=15)
plt.xlabel('x', fontsize=20)
plt.ylabel('VonMises', fontsize=20)
plt.savefig('./rectangle_pic/y05_mise_rectangle.png', dpi=1000, transparent=True)
plt.show()
#%%
# error
plt.figure(figsize=(10, 7))
plt.plot(np.array(range(int(DEM_nepoch/10)))*10, DEM_error_sigma_x_array, linestyle='-.')
plt.plot(np.array(range(int(DCEM_nepoch/10)))*10, DCEM_error_sigma_x_array, linestyle='-.')
plt.plot(np.array(range(int(DEM_nepoch/10)))*10, DEM_error_sigma_y_array, linestyle=':')
plt.plot(np.array(range(int(DCEM_nepoch/10)))*10, DCEM_error_sigma_y_array, linestyle=':')
plt.plot(np.array(range(int(DEM_nepoch/10)))*10, DEM_error_sigma_xy_array, linestyle='--')
plt.plot(np.array(range(int(DCEM_nepoch/10)))*10, DCEM_error_sigma_xy_array, linestyle='--')
plt.plot(np.array(range(int(DEM_nepoch/10)))*10, DEM_error_sigma_mise_array)
plt.plot(np.array(range(int(DCEM_nepoch/10)))*10, DCEM_error_sigma_mise_array)
plt.yscale('log')
plt.legend([ r'DEM: $\sigma_{x}$', r'DCEM: $\sigma_{x}$', r'DEM: $\sigma_{y}$', r'DCEM: $\sigma_{y}$', r'DEM: $\tau_{xy}$', r'DCEM: $\tau_{xy}$', r'DEM: VonMises', r'DCEM: VonMises'])
plt.xlabel('Iteration', fontsize=20)
plt.ylabel(r'$\mathcal{L}_{2}^{rel}$ error', fontsize=20)
plt.savefig('./rectangle_pic/error_rectangle.png', dpi=1000, transparent=True)
plt.show()