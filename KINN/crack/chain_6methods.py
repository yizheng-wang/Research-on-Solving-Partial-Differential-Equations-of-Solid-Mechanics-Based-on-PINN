import subprocess

def run_python_script(filepath):
    print(f"Running {filepath}...")
    subprocess.run(['python', filepath], check=True)
    print(f"Finished running {filepath}.")



file1 = './BINN_RIZZO_20240516-CUDA/BINN_MLP.py'
file2 = './BINN_RIZZO_20240516-CUDA/BINN_KAN.py'
file3 = './Crack_DEM_rbf.py'
file4 = './Crack_DEM_KINN_rbf.py'
file5 = './Crack_CPINN.py'
file6 = './Crack_CPINN_KINN.py'

run_python_script(file1)
run_python_script(file2)
# run_python_script(file3)
# run_python_script(file4)
# run_python_script(file5)
# run_python_script(file6)