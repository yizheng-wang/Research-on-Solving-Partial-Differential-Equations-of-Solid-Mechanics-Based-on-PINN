import subprocess

def run_python_script(filepath):
    print(f"Running {filepath}...")
    subprocess.run(['python', filepath], check=True)
    print(f"Finished running {filepath}.")

file1 = './Green_vortex_arbitatary_w_PINN_full_finetuning.py'
file2 = './Green_vortex_arbitatary_w_PINN_light_weight.py'
file3 = './Green_vortex_arbitatary_w_PINN_lora.py'

print("Running PINN_full_finetuning program...")
run_python_script(file1)
print("Running PINN_light_weight program...")
run_python_script(file2)
print("Running PINN_lora program...")
run_python_script(file3)
print("Both programs completed.")