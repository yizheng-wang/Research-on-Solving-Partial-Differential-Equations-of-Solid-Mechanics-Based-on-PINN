import subprocess
import os

def run_python_script(filepath, working_directory):
    print(f"Running {filepath} in {working_directory}...")
    try:
        result = subprocess.run(
            ['python', filepath], 
            check=True, 
            capture_output=True, 
            text=True,
            cwd=working_directory  # Set the working directory
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {filepath}: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        print(f"Return code: {e.returncode}")
    print(f"Finished running {filepath}.")

# Define working directory and script paths
working_dir = './'


file1 = "Beam2D_4x1_NeoHook_mont_KAN.py"
file2 = "Beam2D_4x1_NeoHook_mont_MLP.py"
file3 = "Beam2D_4x1_NeoHook_Simp_KAN.py"
file4 = "Beam2D_4x1_NeoHook_Simp_MLP.py"
file5 = "Beam2D_4x1_NeoHook_Trip_KAN.py"
file6 = "Beam2D_4x1_NeoHook_Trip_MLP.py"
# Run the scripts
run_python_script(file1, working_dir)
run_python_script(file2, working_dir)
run_python_script(file3, working_dir)
run_python_script(file4, working_dir)
run_python_script(file5, working_dir)
run_python_script(file6, working_dir)
