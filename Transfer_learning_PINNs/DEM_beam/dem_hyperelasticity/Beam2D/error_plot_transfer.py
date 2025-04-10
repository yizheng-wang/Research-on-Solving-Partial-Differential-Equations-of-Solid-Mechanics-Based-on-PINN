import numpy as np
import matplotlib.pyplot as plt

# Load the data
# Assuming the paths provided in the image are correct, adjust as necessary

# DEM data (no transfer learning)
FGM2_MLP_trap_L2_norm = np.load('output/dem/FGM2_MLP_trap_L2_norm.npy')

# Full Fine-Tuning data
FGM1to2_MLP_trap_L2_norm_full_finetuning = np.load('output/dem_full_finetuning/FGM1to2_MLP_trap_L2_norm.npy')

# Lightweight data
FGM1to2_MLP_trap_L2_norm_lightweight = np.load('output/dem_lightweight/FGM1to2_MLP_trap_L2_norm.npy')

# LoRA data
FGM1to2_MLP_trap_L2_norm_lora_r4 = np.load('output/dem_lora/r=4/FGM1to2_MLP_trap_L2_norm.npy')

# FGM1 data for the second plot
FGM1_MLP_trap_L2_norm = np.load('output/dem/FGM1_MLP_trap_L2_norm.npy')

# Full Fine-Tuning data
FGM2to1_MLP_trap_L2_norm_full_finetuning = np.load('output/dem_full_finetuning/FGM2to1_MLP_trap_L2_norm.npy')

# Lightweight data
FGM2to1_MLP_trap_L2_norm_lightweight = np.load('output/dem_lightweight/FGM2to1_MLP_trap_L2_norm.npy')

# LoRA data
FGM2to1_MLP_trap_L2_norm_lora_r4 = np.load('output/dem_lora/r=4/FGM2to1_MLP_trap_L2_norm.npy')

# Smooth the data (if necessary)
def smooth(data, window_size=400):
    """Smooth the data using a moving average."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

FGM2_MLP_trap_L2_norm_smooth = smooth(FGM2_MLP_trap_L2_norm)
FGM1to2_MLP_trap_L2_norm_full_finetuning_smooth = smooth(FGM1to2_MLP_trap_L2_norm_full_finetuning)
FGM1to2_MLP_trap_L2_norm_lightweight_smooth = smooth(FGM1to2_MLP_trap_L2_norm_lightweight)
FGM1to2_MLP_trap_L2_norm_lora_r4_smooth = smooth(FGM1to2_MLP_trap_L2_norm_lora_r4)

FGM1_MLP_trap_L2_norm_smooth = smooth(FGM1_MLP_trap_L2_norm)
FGM2to1_MLP_trap_L2_norm_full_finetuning_smooth = smooth(FGM2to1_MLP_trap_L2_norm_full_finetuning)
FGM2to1_MLP_trap_L2_norm_lightweight_smooth = smooth(FGM2to1_MLP_trap_L2_norm_lightweight)
FGM2to1_MLP_trap_L2_norm_lora_r4_smooth = smooth(FGM2to1_MLP_trap_L2_norm_lora_r4)

# Set the font size for plots
fs = 15

# First plot
fig, axes = plt.subplots(1, 2, figsize=(20, 7))

# Plot for wi = 3.14 -> 6.28
axes[0].plot(FGM2_MLP_trap_L2_norm_smooth, label='No Transfer Learning', color='blue')
axes[0].plot(FGM1to2_MLP_trap_L2_norm_full_finetuning_smooth, label='Full_finetuning', color='orange')
axes[0].plot(FGM1to2_MLP_trap_L2_norm_lightweight_smooth, label='Light_finetuning', color='green')
axes[0].plot(FGM1to2_MLP_trap_L2_norm_lora_r4_smooth, label='LoRA_finetuning', color='red')
axes[0].set_yscale('log')
axes[0].set_title('Sym -> Asym', fontsize=fs)
axes[0].set_xlabel('Epochs', fontsize=fs)
axes[0].set_ylabel(r'Relative error: $\mathcal{L}_{2}$', fontsize=fs+5)
axes[0].legend(fontsize=fs)

# Plot for wi = 3.14 -> 9.42
axes[1].plot(FGM1_MLP_trap_L2_norm_smooth, label='No Transfer Learning', color='blue')
axes[1].plot(FGM2to1_MLP_trap_L2_norm_full_finetuning_smooth, label='Full_finetuning', color='orange')
axes[1].plot(FGM2to1_MLP_trap_L2_norm_lightweight_smooth, label='Light_finetuning', color='green')
axes[1].plot(FGM2to1_MLP_trap_L2_norm_lora_r4_smooth, label='LoRA_finetuning', color='red')
axes[1].set_yscale('log')
axes[1].set_title('Asym -> Sym', fontsize=fs)
axes[1].set_xlabel('Epochs', fontsize=fs)
axes[1].set_ylabel(r'Relative error: $\mathcal{L}_{2}$', fontsize=fs+5)
axes[1].legend(fontsize=fs)

plt.tight_layout()
plt.savefig('./pic/FGM_transfer_L2_comparision.pdf', dpi = 500)
plt.show()



# DEM data (no transfer learning)
FGM2_MLP_trap_H1_norm = np.load('output/dem/FGM2_MLP_trap_H1_norm.npy')

# Full Fine-Tuning data
FGM1to2_MLP_trap_H1_norm_full_finetuning = np.load('output/dem_full_finetuning/FGM1to2_MLP_trap_H1_norm.npy')

# Lightweight data
FGM1to2_MLP_trap_H1_norm_lightweight = np.load('output/dem_lightweight/FGM1to2_MLP_trap_H1_norm.npy')

# LoRA data
FGM1to2_MLP_trap_H1_norm_lora_r4 = np.load('output/dem_lora/r=4/FGM1to2_MLP_trap_H1_norm.npy')

# FGM1 data for the second plot
FGM1_MLP_trap_H1_norm = np.load('output/dem/FGM1_MLP_trap_H1_norm.npy')

# Full Fine-Tuning data
FGM2to1_MLP_trap_H1_norm_full_finetuning = np.load('output/dem_full_finetuning/FGM2to1_MLP_trap_H1_norm.npy')

# Lightweight data
FGM2to1_MLP_trap_H1_norm_lightweight = np.load('output/dem_lightweight/FGM2to1_MLP_trap_H1_norm.npy')

# LoRA data
FGM2to1_MLP_trap_H1_norm_lora_r4 = np.load('output/dem_lora/r=4/FGM2to1_MLP_trap_H1_norm.npy')

# Smooth the data (if necessary)
def smooth(data, window_size=600):
    """Smooth the data using a moving average."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

FGM2_MLP_trap_H1_norm_smooth = smooth(FGM2_MLP_trap_H1_norm)
FGM1to2_MLP_trap_H1_norm_full_finetuning_smooth = smooth(FGM1to2_MLP_trap_H1_norm_full_finetuning)
FGM1to2_MLP_trap_H1_norm_lightweight_smooth = smooth(FGM1to2_MLP_trap_H1_norm_lightweight)
FGM1to2_MLP_trap_H1_norm_lora_r4_smooth = smooth(FGM1to2_MLP_trap_H1_norm_lora_r4)

FGM1_MLP_trap_H1_norm_smooth = smooth(FGM1_MLP_trap_H1_norm)
FGM2to1_MLP_trap_H1_norm_full_finetuning_smooth = smooth(FGM2to1_MLP_trap_H1_norm_full_finetuning)
FGM2to1_MLP_trap_H1_norm_lightweight_smooth = smooth(FGM2to1_MLP_trap_H1_norm_lightweight)
FGM2to1_MLP_trap_H1_norm_lora_r4_smooth = smooth(FGM2to1_MLP_trap_H1_norm_lora_r4)

# Set the font size for plots
fs = 15

# First plot
fig, axes = plt.subplots(1, 2, figsize=(20, 7))

# Plot for wi = 3.14 -> 6.28
axes[0].plot(FGM2_MLP_trap_H1_norm_smooth, label='No Transfer Learning', color='blue')
axes[0].plot(FGM1to2_MLP_trap_H1_norm_full_finetuning_smooth, label='Full_finetuning', color='orange')
axes[0].plot(FGM1to2_MLP_trap_H1_norm_lightweight_smooth, label='Light_finetuning', color='green')
axes[0].plot(FGM1to2_MLP_trap_H1_norm_lora_r4_smooth, label='LoRA_finetuning', color='red')
axes[0].set_yscale('log')
axes[0].set_title('Sym -> Asym', fontsize=fs)
axes[0].set_xlabel('Epochs', fontsize=fs)
axes[0].set_ylabel(r'Relative error: $\mathcal{H}_{1}$', fontsize=fs+5)
axes[0].legend(fontsize=fs)

# Plot for wi = 3.14 -> 9.42
axes[1].plot(FGM1_MLP_trap_H1_norm_smooth, label='No Transfer Learning', color='blue')
axes[1].plot(FGM2to1_MLP_trap_H1_norm_full_finetuning_smooth, label='Full_finetuning', color='orange')
axes[1].plot(FGM2to1_MLP_trap_H1_norm_lightweight_smooth, label='Light_finetuning', color='green')
axes[1].plot(FGM2to1_MLP_trap_H1_norm_lora_r4_smooth, label='LoRA_finetuning', color='red')
axes[1].set_yscale('log')
axes[1].set_title('Asym -> Sym', fontsize=fs)
axes[1].set_xlabel('Epochs', fontsize=fs)
axes[1].set_ylabel(r'Relative error: $\mathcal{H}_{1}$', fontsize=fs+5)
axes[1].legend(fontsize=fs)

plt.tight_layout()
plt.savefig('./pic/FGM_transfer_H1_comparision.pdf', dpi = 500)
plt.show()