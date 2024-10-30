import subprocess

def run_python_script(filepath):
    print(f"Running {filepath}...")
    subprocess.run(['python', filepath], check=True)
    print(f"Finished running {filepath}.")



file1 = './onehole_DEM_MLP.py'
file2 = './onehole_DEM_KAN.py'

run_python_script(file1)
run_python_script(file2)
# run_python_script(file3)
# run_python_script(file4)
# run_python_script(file5)
# run_python_script(file6)