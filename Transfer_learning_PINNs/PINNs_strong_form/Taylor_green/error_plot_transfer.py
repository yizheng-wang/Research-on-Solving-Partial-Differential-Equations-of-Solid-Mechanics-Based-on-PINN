import numpy as np
import matplotlib.pyplot as plt

def smooth(data, window_size=1000):
    """Smooth the data using a moving average."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


# Load the data
loss_results_6_28_wo_3_14_wi_full_finetuning = np.load('results/PINN_full_finetuning/loss_error_results_wi_3.14_wo_6.28.npy', allow_pickle=True).item()
loss_results_9_42_wo_3_14_wi_full_finetuning = np.load('results/PINN_full_finetuning/loss_error_results_wi_3.14_wo_9.42.npy', allow_pickle=True).item()

loss_results_3_14_wo_6_28_wi_full_finetuning = np.load('results/PINN_full_finetuning/loss_error_results_wi_6.28_wo_3.14.npy', allow_pickle=True).item()
loss_results_9_42_wo_6_28_wi_full_finetuning = np.load('results/PINN_full_finetuning/loss_error_results_wi_6.28_wo_9.42.npy', allow_pickle=True).item()

loss_results_3_14_wo_9_42_wi_full_finetuning = np.load('results/PINN_full_finetuning/loss_error_results_wi_9.42_wo_3.14.npy', allow_pickle=True).item()
loss_results_6_28_wo_9_42_wi_full_finetuning = np.load('results/PINN_full_finetuning/loss_error_results_wi_9.42_wo_6.28.npy', allow_pickle=True).item()


# Load the light_weight
loss_results_6_28_wo_3_14_wi_light_weight = np.load('results/PINN_light_weight/loss_error_results_wi_3.14_wo_6.28.npy', allow_pickle=True).item()
loss_results_9_42_wo_3_14_wi_light_weight = np.load('results/PINN_light_weight/loss_error_results_wi_3.14_wo_9.42.npy', allow_pickle=True).item()

loss_results_3_14_wo_6_28_wi_light_weight = np.load('results/PINN_light_weight/loss_error_results_wi_6.28_wo_3.14.npy', allow_pickle=True).item()
loss_results_9_42_wo_6_28_wi_light_weight = np.load('results/PINN_light_weight/loss_error_results_wi_6.28_wo_9.42.npy', allow_pickle=True).item()

loss_results_3_14_wo_9_42_wi_light_weight = np.load('results/PINN_light_weight/loss_error_results_wi_9.42_wo_3.14.npy', allow_pickle=True).item()
loss_results_6_28_wo_9_42_wi_light_weight = np.load('results/PINN_light_weight/loss_error_results_wi_9.42_wo_6.28.npy', allow_pickle=True).item()


# Load the lora
loss_results_6_28_wo_3_14_wi_lora = np.load('results/PINN_lora/r=4/loss_error_results_wi_3.14_wo_6.28.npy', allow_pickle=True).item()
loss_results_9_42_wo_3_14_wi_lora = np.load('results/PINN_lora/r=4/loss_error_results_wi_3.14_wo_9.42.npy', allow_pickle=True).item()

loss_results_3_14_wo_6_28_wi_lora = np.load('results/PINN_lora/r=4/loss_error_results_wi_6.28_wo_3.14.npy', allow_pickle=True).item()
loss_results_9_42_wo_6_28_wi_lora = np.load('results/PINN_lora/r=4/loss_error_results_wi_6.28_wo_9.42.npy', allow_pickle=True).item()

loss_results_3_14_wo_9_42_wi_lora = np.load('results/PINN_lora/r=4/loss_error_results_wi_9.42_wo_3.14.npy', allow_pickle=True).item()
loss_results_6_28_wo_9_42_wi_lora = np.load('results/PINN_lora/r=4/loss_error_results_wi_9.42_wo_6.28.npy', allow_pickle=True).item()


loss_results_3_14_w = np.load('results/PINN/loss_error_results_w_3.14.npy', allow_pickle=True).item()
loss_results_6_28_w = np.load('results/PINN/loss_error_results_w_6.28.npy', allow_pickle=True).item()
loss_results_9_42_w = np.load('results/PINN/loss_error_results_w_9.42.npy', allow_pickle=True).item()


# 提取psi_error和omega_error
# full_fintuning
psi_error_6_28_wo_3_14_wi_full_finetuning = loss_results_6_28_wo_3_14_wi_full_finetuning['psi_error_t']
psi_error_9_42_wo_3_14_wi_full_finetuning = loss_results_9_42_wo_3_14_wi_full_finetuning['psi_error_t']

psi_error_3_14_wo_6_28_wi_full_finetuning = loss_results_3_14_wo_6_28_wi_full_finetuning['psi_error_t']
psi_error_9_42_wo_6_28_wi_full_finetuning = loss_results_9_42_wo_6_28_wi_full_finetuning['psi_error_t']

psi_error_3_14_wo_9_42_wi_full_finetuning = loss_results_3_14_wo_9_42_wi_full_finetuning['psi_error_t']
psi_error_6_28_wo_9_42_wi_full_finetuning = loss_results_6_28_wo_9_42_wi_full_finetuning['psi_error_t']

# light_weight
psi_error_6_28_wo_3_14_wi_light_weight = loss_results_6_28_wo_3_14_wi_light_weight['psi_error_t']
psi_error_9_42_wo_3_14_wi_light_weight = loss_results_9_42_wo_3_14_wi_light_weight['psi_error_t']

psi_error_3_14_wo_6_28_wi_light_weight = loss_results_3_14_wo_6_28_wi_light_weight['psi_error_t']
psi_error_9_42_wo_6_28_wi_light_weight = loss_results_9_42_wo_6_28_wi_light_weight['psi_error_t']

psi_error_3_14_wo_9_42_wi_light_weight = loss_results_3_14_wo_9_42_wi_light_weight['psi_error_t']
psi_error_6_28_wo_9_42_wi_light_weight = loss_results_6_28_wo_9_42_wi_light_weight['psi_error_t']

# lora
psi_error_6_28_wo_3_14_wi_lora = loss_results_6_28_wo_3_14_wi_lora['psi_error_t']
psi_error_9_42_wo_3_14_wi_lora = loss_results_9_42_wo_3_14_wi_lora['psi_error_t']

psi_error_3_14_wo_6_28_wi_lora = loss_results_3_14_wo_6_28_wi_lora['psi_error_t']
psi_error_9_42_wo_6_28_wi_lora = loss_results_9_42_wo_6_28_wi_lora['psi_error_t']

psi_error_3_14_wo_9_42_wi_lora = loss_results_3_14_wo_9_42_wi_lora['psi_error_t']
psi_error_6_28_wo_9_42_wi_lora = loss_results_6_28_wo_9_42_wi_lora['psi_error_t']


psi_error_3_14_w = loss_results_3_14_w['psi_error_t']
psi_error_6_28_w = loss_results_6_28_w['psi_error_t']
psi_error_9_42_w = loss_results_9_42_w['psi_error_t']


# full_fintuning
omega_error_6_28_wo_3_14_wi_full_finetuning = loss_results_6_28_wo_3_14_wi_full_finetuning['omega_error_t']
omega_error_9_42_wo_3_14_wi_full_finetuning = loss_results_9_42_wo_3_14_wi_full_finetuning['omega_error_t']

omega_error_3_14_wo_6_28_wi_full_finetuning = loss_results_3_14_wo_6_28_wi_full_finetuning['omega_error_t']
omega_error_9_42_wo_6_28_wi_full_finetuning = loss_results_9_42_wo_6_28_wi_full_finetuning['omega_error_t']

omega_error_3_14_wo_9_42_wi_full_finetuning = loss_results_3_14_wo_9_42_wi_full_finetuning['omega_error_t']
omega_error_6_28_wo_9_42_wi_full_finetuning = loss_results_6_28_wo_9_42_wi_full_finetuning['omega_error_t']

# light_weight
omega_error_6_28_wo_3_14_wi_light_weight = loss_results_6_28_wo_3_14_wi_light_weight['omega_error_t']
omega_error_9_42_wo_3_14_wi_light_weight = loss_results_9_42_wo_3_14_wi_light_weight['omega_error_t']

omega_error_3_14_wo_6_28_wi_light_weight = loss_results_3_14_wo_6_28_wi_light_weight['omega_error_t']
omega_error_9_42_wo_6_28_wi_light_weight = loss_results_9_42_wo_6_28_wi_light_weight['omega_error_t']

omega_error_3_14_wo_9_42_wi_light_weight = loss_results_3_14_wo_9_42_wi_light_weight['omega_error_t']
omega_error_6_28_wo_9_42_wi_light_weight = loss_results_6_28_wo_9_42_wi_light_weight['omega_error_t']

# lora
omega_error_6_28_wo_3_14_wi_lora = loss_results_6_28_wo_3_14_wi_lora['omega_error_t']
omega_error_9_42_wo_3_14_wi_lora = loss_results_9_42_wo_3_14_wi_lora['omega_error_t']

omega_error_3_14_wo_6_28_wi_lora = loss_results_3_14_wo_6_28_wi_lora['omega_error_t']
omega_error_9_42_wo_6_28_wi_lora = loss_results_9_42_wo_6_28_wi_lora['omega_error_t']

omega_error_3_14_wo_9_42_wi_lora = loss_results_3_14_wo_9_42_wi_lora['omega_error_t']
omega_error_6_28_wo_9_42_wi_lora = loss_results_6_28_wo_9_42_wi_lora['omega_error_t']


omega_error_3_14_w = loss_results_3_14_w['omega_error_t']
omega_error_6_28_w = loss_results_6_28_w['omega_error_t']
omega_error_9_42_w = loss_results_9_42_w['omega_error_t']


# Apply smoothing to the error data
psi_error_3_14_w_smooth = smooth(psi_error_3_14_w)
psi_error_6_28_w_smooth = smooth(psi_error_6_28_w)
psi_error_9_42_w_smooth = smooth(psi_error_9_42_w)

psi_error_6_28_wo_3_14_wi_full_finetuning_smooth = smooth(psi_error_6_28_wo_3_14_wi_full_finetuning)
psi_error_9_42_wo_3_14_wi_full_finetuning_smooth = smooth(psi_error_9_42_wo_3_14_wi_full_finetuning)
psi_error_6_28_wo_3_14_wi_light_weight_smooth = smooth(psi_error_6_28_wo_3_14_wi_light_weight)
psi_error_9_42_wo_3_14_wi_light_weight_smooth = smooth(psi_error_9_42_wo_3_14_wi_light_weight)
psi_error_6_28_wo_3_14_wi_lora_smooth = smooth(psi_error_6_28_wo_3_14_wi_lora)
psi_error_9_42_wo_3_14_wi_lora_smooth = smooth(psi_error_9_42_wo_3_14_wi_lora)

psi_error_3_14_wo_6_28_wi_full_finetuning_smooth = smooth(psi_error_3_14_wo_6_28_wi_full_finetuning)
psi_error_9_42_wo_6_28_wi_full_finetuning_smooth = smooth(psi_error_9_42_wo_6_28_wi_full_finetuning)
psi_error_3_14_wo_6_28_wi_light_weight_smooth = smooth(psi_error_3_14_wo_6_28_wi_light_weight)
psi_error_9_42_wo_6_28_wi_light_weight_smooth = smooth(psi_error_9_42_wo_6_28_wi_light_weight)
psi_error_3_14_wo_6_28_wi_lora_smooth = smooth(psi_error_3_14_wo_6_28_wi_lora)
psi_error_9_42_wo_6_28_wi_lora_smooth = smooth(psi_error_9_42_wo_6_28_wi_lora)

psi_error_3_14_wo_9_42_wi_full_finetuning_smooth = smooth(psi_error_3_14_wo_9_42_wi_full_finetuning)
psi_error_6_28_wo_9_42_wi_full_finetuning_smooth = smooth(psi_error_6_28_wo_9_42_wi_full_finetuning)
psi_error_3_14_wo_9_42_wi_light_weight_smooth = smooth(psi_error_3_14_wo_9_42_wi_light_weight)
psi_error_6_28_wo_9_42_wi_light_weight_smooth = smooth(psi_error_6_28_wo_9_42_wi_light_weight)
psi_error_3_14_wo_9_42_wi_lora_smooth = smooth(psi_error_3_14_wo_9_42_wi_lora)
psi_error_6_28_wo_9_42_wi_lora_smooth = smooth(psi_error_6_28_wo_9_42_wi_lora)
# Similarly, you can apply smoothing for omega_error values
omega_error_3_14_w_smooth = smooth(omega_error_3_14_w)
omega_error_6_28_w_smooth = smooth(omega_error_6_28_w)
omega_error_9_42_w_smooth = smooth(omega_error_9_42_w)

omega_error_6_28_wo_3_14_wi_full_finetuning_smooth = smooth(omega_error_6_28_wo_3_14_wi_full_finetuning)
omega_error_9_42_wo_3_14_wi_full_finetuning_smooth = smooth(omega_error_9_42_wo_3_14_wi_full_finetuning)
omega_error_6_28_wo_3_14_wi_light_weight_smooth = smooth(omega_error_6_28_wo_3_14_wi_light_weight)
omega_error_9_42_wo_3_14_wi_light_weight_smooth = smooth(omega_error_9_42_wo_3_14_wi_light_weight)
omega_error_6_28_wo_3_14_wi_lora_smooth = smooth(omega_error_6_28_wo_3_14_wi_lora)
omega_error_9_42_wo_3_14_wi_lora_smooth = smooth(omega_error_9_42_wo_3_14_wi_lora)

omega_error_3_14_wo_6_28_wi_full_finetuning_smooth = smooth(omega_error_3_14_wo_6_28_wi_full_finetuning)
omega_error_9_42_wo_6_28_wi_full_finetuning_smooth = smooth(omega_error_9_42_wo_6_28_wi_full_finetuning)
omega_error_3_14_wo_6_28_wi_light_weight_smooth = smooth(omega_error_3_14_wo_6_28_wi_light_weight)
omega_error_9_42_wo_6_28_wi_light_weight_smooth = smooth(omega_error_9_42_wo_6_28_wi_light_weight)
omega_error_3_14_wo_6_28_wi_lora_smooth = smooth(omega_error_3_14_wo_6_28_wi_lora)
omega_error_9_42_wo_6_28_wi_lora_smooth = smooth(omega_error_9_42_wo_6_28_wi_lora)

omega_error_3_14_wo_9_42_wi_full_finetuning_smooth = smooth(omega_error_3_14_wo_9_42_wi_full_finetuning)
omega_error_6_28_wo_9_42_wi_full_finetuning_smooth = smooth(omega_error_6_28_wo_9_42_wi_full_finetuning)
omega_error_3_14_wo_9_42_wi_light_weight_smooth = smooth(omega_error_3_14_wo_9_42_wi_light_weight)
omega_error_6_28_wo_9_42_wi_light_weight_smooth = smooth(omega_error_6_28_wo_9_42_wi_light_weight)
omega_error_3_14_wo_9_42_wi_lora_smooth = smooth(omega_error_3_14_wo_9_42_wi_lora)
omega_error_6_28_wo_9_42_wi_lora_smooth = smooth(omega_error_6_28_wo_9_42_wi_lora)


fs = 15
#%%
# Now plot the smoothed data
fig, axes = plt.subplots(2, 3, figsize=(20, 10))

# Plot for wi = 3.14 -> 6.28
axes[0, 0].plot(psi_error_6_28_w_smooth, label='No Transfer Learning', color='blue')
axes[0, 0].plot(psi_error_6_28_wo_3_14_wi_full_finetuning_smooth, label='Full_finetuning', color='orange')
axes[0, 0].plot(psi_error_6_28_wo_3_14_wi_light_weight_smooth, label='Light_finetuning', color='green')
axes[0, 0].plot(psi_error_6_28_wo_3_14_wi_lora_smooth, label='LoRA_finetuning', color='red')
axes[0, 0].set_yscale('log')
axes[0, 0].set_title('w: 3.14 -> 6.28', fontsize=fs)
axes[0, 0].set_xlabel('Epochs', fontsize=fs)
axes[0, 0].set_ylabel(r'Relative error: $\mathcal{L}_{2}$', fontsize=fs+5)
axes[0, 0].legend(fontsize=fs)

# Plot for wi = 3.14 -> 9.42
axes[1, 0].plot(psi_error_9_42_w_smooth, label='No Transfer Learning', color='blue')
axes[1, 0].plot(psi_error_9_42_wo_3_14_wi_full_finetuning_smooth, label='Full_finetuning', color='orange')
axes[1, 0].plot(psi_error_9_42_wo_3_14_wi_light_weight_smooth, label='Light_finetuning', color='green')
axes[1, 0].plot(psi_error_9_42_wo_3_14_wi_lora_smooth, label='LoRA_finetuning', color='red')
axes[1, 0].set_yscale('log')
axes[1, 0].set_title('w: 3.14 -> 9.42', fontsize=fs)
axes[1, 0].set_xlabel('Epochs', fontsize=fs)
axes[1, 0].set_ylabel(r'Relative error: $\mathcal{L}_{2}$', fontsize=fs+5)
axes[1, 0].legend(fontsize=fs)

# Plot for wi = 6.28 -> 3.14
axes[0, 1].plot(psi_error_3_14_w_smooth, label='No Transfer Learning', color='blue')
axes[0, 1].plot(psi_error_3_14_wo_6_28_wi_full_finetuning_smooth, label='Full_finetuning', color='orange')
axes[0, 1].plot(psi_error_3_14_wo_6_28_wi_light_weight_smooth, label='Light_finetuning', color='green')
axes[0, 1].plot(psi_error_3_14_wo_6_28_wi_lora_smooth, label='LoRA_finetuning', color='red')
axes[0, 1].set_yscale('log')
axes[0, 1].set_title('w: 6.28 -> 3.14', fontsize=fs)
axes[0, 1].set_xlabel('Epochs', fontsize=fs)
axes[0, 1].set_ylabel(r'Relative error: $\mathcal{L}_{2}$', fontsize=fs+5)
axes[0, 1].legend(fontsize=fs)

# Plot for wi = 6.28 -> 9.42
axes[1, 1].plot(psi_error_9_42_w_smooth, label='No Transfer Learning', color='blue')
axes[1, 1].plot(psi_error_9_42_wo_6_28_wi_full_finetuning_smooth, label='Full_finetuning', color='orange')
axes[1, 1].plot(psi_error_9_42_wo_6_28_wi_light_weight_smooth, label='Light_finetuning', color='green')
axes[1, 1].plot(psi_error_9_42_wo_6_28_wi_lora_smooth, label='LoRA_finetuning', color='red')
axes[1, 1].set_yscale('log')
axes[1, 1].set_title('w: 6.28 -> 9.42', fontsize=fs)
axes[1, 1].set_xlabel('Epochs', fontsize=fs)
axes[1, 1].set_ylabel(r'Relative error: $\mathcal{L}_{2}$', fontsize=fs+5)
axes[1, 1].legend(fontsize=fs)

# Plot for wi = 9.42 -> 3.14
axes[0, 2].plot(psi_error_3_14_w_smooth, label='No Transfer Learning', color='blue')
axes[0, 2].plot(psi_error_3_14_wo_9_42_wi_full_finetuning_smooth, label='Full_finetuning', color='orange')
axes[0, 2].plot(psi_error_3_14_wo_9_42_wi_light_weight_smooth, label='Light_finetuning', color='green')
axes[0, 2].plot(psi_error_3_14_wo_9_42_wi_lora_smooth, label='LoRA_finetuning', color='red')
axes[0, 2].set_yscale('log')
axes[0, 2].set_title('w: 9.42 -> 3.14', fontsize=fs)
axes[0, 2].set_xlabel('Epochs', fontsize=fs)
axes[0, 2].set_ylabel(r'Relative error: $\mathcal{L}_{2}$', fontsize=fs+5)
axes[0, 2].legend(fontsize=fs)

# Plot for wi = 9.42 -> 6.28
axes[1, 2].plot(psi_error_6_28_w_smooth, label='No Transfer Learning', color='blue')
axes[1, 2].plot(psi_error_6_28_wo_9_42_wi_full_finetuning_smooth, label='Full_finetuning', color='orange')
axes[1, 2].plot(psi_error_6_28_wo_9_42_wi_light_weight_smooth, label='Light_finetuning', color='green')
axes[1, 2].plot(psi_error_6_28_wo_9_42_wi_lora_smooth, label='LoRA_finetuning', color='red')
axes[1, 2].set_yscale('log')
axes[1, 2].set_title('w: 9.42 -> 6.28', fontsize=fs)
axes[1, 2].set_xlabel('Epochs', fontsize=fs)
axes[1, 2].set_ylabel(r'Relative error: $\mathcal{L}_{2}$', fontsize=fs+5)
axes[1, 2].legend(fontsize=fs)

# Adjust layout
plt.tight_layout()
plt.savefig('./pic/PINNs_psi_finetuning.pdf')
plt.show()
#%%
# Now do the same for omega errors
fig, axes = plt.subplots(2, 3, figsize=(20, 10))

# Plot for wi = 3.14 -> 6.28
axes[0, 0].plot(omega_error_6_28_w_smooth, label='No Transfer Learning', color='blue')
axes[0, 0].plot(omega_error_6_28_wo_3_14_wi_full_finetuning_smooth, label='Full_finetuning', color='orange')
axes[0, 0].plot(omega_error_6_28_wo_3_14_wi_light_weight_smooth, label='Light_finetuning', color='green')
axes[0, 0].plot(omega_error_6_28_wo_3_14_wi_lora_smooth, label='LoRA_finetuning', color='red')
axes[0, 0].set_yscale('log')
axes[0, 0].set_title('w: 3.14 -> 6.28', fontsize=fs)
axes[0, 0].set_xlabel('Epochs', fontsize=fs)
axes[0, 0].set_ylabel(r'Relative error: $\mathcal{L}_{2}$', fontsize=fs+5)
axes[0, 0].legend(fontsize=fs)

# Plot for wi = 3.14 -> 9.42
axes[1, 0].plot(omega_error_9_42_w_smooth, label='No Transfer Learning', color='blue')
axes[1, 0].plot(omega_error_9_42_wo_3_14_wi_full_finetuning_smooth, label='Full_finetuning', color='orange')
axes[1, 0].plot(omega_error_9_42_wo_3_14_wi_light_weight_smooth, label='Light_finetuning', color='green')
axes[1, 0].plot(omega_error_9_42_wo_3_14_wi_lora_smooth, label='LoRA_finetuning', color='red')
axes[1, 0].set_yscale('log')
axes[1, 0].set_title('w: 3.14 -> 9.42', fontsize=fs)
axes[1, 0].set_xlabel('Epochs', fontsize=fs)
axes[1, 0].set_ylabel(r'Relative error: $\mathcal{L}_{2}$', fontsize=fs+5)
axes[1, 0].legend(fontsize=fs)

# Plot for wi = 6.28 -> 3.14
axes[0, 1].plot(omega_error_3_14_w_smooth, label='No Transfer Learning', color='blue')
axes[0, 1].plot(omega_error_3_14_wo_6_28_wi_full_finetuning_smooth, label='Full_finetuning', color='orange')
axes[0, 1].plot(omega_error_3_14_wo_6_28_wi_light_weight_smooth, label='Light_finetuning', color='green')
axes[0, 1].plot(omega_error_3_14_wo_6_28_wi_lora_smooth, label='LoRA_finetuning', color='red')
axes[0, 1].set_yscale('log')
axes[0, 1].set_title('w: 6.28 -> 3.14', fontsize=fs)
axes[0, 1].set_xlabel('Epochs', fontsize=fs)
axes[0, 1].set_ylabel(r'Relative error: $\mathcal{L}_{2}$', fontsize=fs+5)
axes[0, 1].legend(fontsize=fs)

# Plot for wi = 6.28 -> 9.42
axes[1, 1].plot(omega_error_9_42_w_smooth, label='No Transfer Learning', color='blue')
axes[1, 1].plot(omega_error_9_42_wo_6_28_wi_full_finetuning_smooth, label='Full_finetuning', color='orange')
axes[1, 1].plot(omega_error_9_42_wo_6_28_wi_light_weight_smooth, label='Light_finetuning', color='green')
axes[1, 1].plot(omega_error_9_42_wo_6_28_wi_lora_smooth, label='LoRA_finetuning', color='red')
axes[1, 1].set_yscale('log')
axes[1, 1].set_title('w: 6.28 -> 9.42', fontsize=fs)
axes[1, 1].set_xlabel('Epochs', fontsize=fs)
axes[1, 1].set_ylabel(r'Relative error: $\mathcal{L}_{2}$', fontsize=fs+5)
axes[1, 1].legend(fontsize=fs)

# Plot for wi = 9.42 -> 3.14
axes[0, 2].plot(omega_error_3_14_w_smooth, label='No Transfer Learning', color='blue')
axes[0, 2].plot(omega_error_3_14_wo_9_42_wi_full_finetuning_smooth, label='Full_finetuning', color='orange')
axes[0, 2].plot(omega_error_3_14_wo_9_42_wi_light_weight_smooth, label='Light_finetuning', color='green')
axes[0, 2].plot(omega_error_3_14_wo_9_42_wi_lora_smooth, label='LoRA_finetuning', color='red')
axes[0, 2].set_yscale('log')
axes[0, 2].set_title('w: 9.42 -> 3.14', fontsize=fs)
axes[0, 2].set_xlabel('Epochs', fontsize=fs)
axes[0, 2].set_ylabel(r'Relative error: $\mathcal{L}_{2}$', fontsize=fs+5)
axes[0, 2].legend(fontsize=fs)

# Plot for wi = 9.42 -> 6.28
axes[1, 2].plot(omega_error_6_28_w_smooth, label='No Transfer Learning', color='blue')
axes[1, 2].plot(omega_error_6_28_wo_9_42_wi_full_finetuning_smooth, label='Full_finetuning', color='orange')
axes[1, 2].plot(omega_error_6_28_wo_9_42_wi_light_weight_smooth, label='Light_finetuning', color='green')
axes[1, 2].plot(omega_error_6_28_wo_9_42_wi_lora_smooth, label='LoRA_finetuning', color='red')
axes[1, 2].set_yscale('log')
axes[1, 2].set_title('w: 9.42 -> 6.28', fontsize=fs)
axes[1, 2].set_xlabel('Epochs', fontsize=fs)
axes[1, 2].set_ylabel(r'Relative error: $\mathcal{L}_{2}$', fontsize=fs+5)
axes[1, 2].legend(fontsize=fs)

# Adjust layout
plt.tight_layout()
plt.savefig('./pic/PINNs_omega_finetuning.pdf')
plt.show()

