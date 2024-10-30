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


file1 = 'prediction_phy_elasticity_painting.py'
file2 = 'prediction_phy_elasticity_con_k.py'


# Run the scripts
run_python_script(file1, working_dir)
run_python_script(file2, working_dir)

