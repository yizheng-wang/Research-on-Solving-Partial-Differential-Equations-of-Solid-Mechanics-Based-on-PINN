# -*- coding: utf-8 -*-
"""
Created on Sun May 19 17:16:40 2024

@author: admin
"""

import  numpy as np 
import matplotlib.pyplot as plt

loaded_data = np.load('./results/Crack_grid_size.npy', allow_pickle=True).item()
print(loaded_data)

# determine whether the error converge
for error in loaded_data['errors']:
    plt.semilogy(error)
plt.show()

linestyles = ['--', '-', '-.', '-', '-', '-']
index = 0
Xerror_total = []
for index_ls, layer_e in enumerate(loaded_data['layers_list']):
    Xerror_array = np.empty([0],dtype=float)
    for i in range(len(loaded_data['grid_size_list'])):
        error_e = loaded_data['errors'][index][-1]
        index = index + 1
        Xerror_array = np.append(Xerror_array, error_e)
        Xerror_total.append(error_e)
    plt.plot(np.array(loaded_data['grid_size_list']), Xerror_array, 'o', label = str(layer_e), linestyle=linestyles[index_ls], markersize=5)
plt.yscale('log')
plt.xlabel('Grid_size')
plt.ylabel('Relative Error')
plt.legend()
plt.savefig('./results/Grid_size_error.pdf', dpi = 300)
plt.show()



# parameter scaling law
linestyles = ['--', '-', '-.', '-', '-', '-']
index = 0
para = loaded_data['parameters'] 
error_array = np.array(Xerror_total)
# selected_elements_x = []
# selected_elements_y = []

for index_ls, layer_e in enumerate(loaded_data['layers_list']):
    selected_elements_x = para[index_ls*len(loaded_data['grid_size_list']):index_ls*len(loaded_data['grid_size_list'])+3]
    selected_elements_y = error_array[index_ls*len(loaded_data['grid_size_list']):index_ls*len(loaded_data['grid_size_list'])+3]
    plt.plot(selected_elements_x, selected_elements_y, 'o', label = str(layer_e), linestyle=linestyles[index_ls], markersize=5)

# for index_ls, layer_e in enumerate(loaded_data['layers_list']):
#     selected_elements_x.extend(para[i:i+5])
#     selected_elements_y.extend(error_array[i:i+5])
#plt.plot(selected_elements_x, selected_elements_y, label = str(layer_e), linestyle=linestyles[index_ls], markersize=5)
plt.xlabel('N')
plt.ylabel('Relative Error')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('./results/scale_law_error.pdf', dpi = 300)
plt.show()





loaded_data = np.load('./results/Crack_spline_order.npy', allow_pickle=True).item()
print(loaded_data)

# determine whether the error converge
for error in loaded_data['errors']:
    plt.semilogy(error)
plt.show()

linestyles = ['--', '-', '-.', '-', '-', '-']
index = 0
for index_ls, layer_e in enumerate(loaded_data['layers_list']):
    Xerror_array = np.empty([0],dtype=float)
    for i in range(len(loaded_data['spline_order_list'])):
        error_e = loaded_data['errors'][index][-1]
        index = index + 1
        Xerror_array = np.append(Xerror_array, error_e)
    plt.plot(np.array(loaded_data['spline_order_list']), Xerror_array, 'o', label = str(layer_e), linestyle=linestyles[index_ls], markersize=5)
plt.yscale('log')
plt.xlabel('Spline Order')
plt.ylabel('Relative Error')
plt.legend()
plt.savefig('./results/Spline_order_error.pdf', dpi = 300)
plt.show()