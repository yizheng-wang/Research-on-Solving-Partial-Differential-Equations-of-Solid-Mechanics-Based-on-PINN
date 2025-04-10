import subprocess

def run_python_script(filepath):
    print(f"Running {filepath}...")
    subprocess.run(['python', filepath], check=True)
    print(f"Finished running {filepath}.")

file1 = './Beam2D_FGM1_Trip_MLP.py'
file2 = './Beam2D_FGM2_Trip_MLP.py'
# file3 = './Beam2D_FGM_Trip_MLP_full_finetuning.py'
# file4 = './Beam2D_FGM_Trip_MLP_lightweight.py'
# file5 = './Beam2D_FGM_Trip_MLP_lora.py'
# file6 = './Beam2D_FGM_Trip_MLP_lora_r=1.py'
# file7 = './Beam2D_FGM_Trip_MLP_lora_r=100.py'


print("Running PINN_full_finetuning program...")
run_python_script(file1)
print("Running PINN_light_weight program...")
run_python_script(file2)
# print("Running PINN_lora program...")
# run_python_script(file3)
# print("Running PINN_lightweight program...")
# run_python_script(file4)
# print("Running PINN_lora_r=1 program...")
# run_python_script(file5)
# print("Running PINN_lora_r=100 program...")
# run_python_script(file6)
# print("Running PINN_lora_r=100 program again...")
# run_python_script(file7)
# print("All programs completed.")