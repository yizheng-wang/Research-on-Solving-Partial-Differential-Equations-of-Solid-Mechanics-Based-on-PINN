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
working_dir_1 = 'C:/Users/admin/OneDrive/KINN/src_KINN/Plate_hole/DEM_plate_hole/plate_hole'
working_dir_2 = 'C:/Users/admin/OneDrive/KINN/src_KINN/Plate_hole/PINNs'

file3 = './Plate_hole_DEM_triangle.py'
file4 = './Plate_hole_KINN_DEM_triangle.py'
file5 = './PINNs_plate_hole.py'
file6 = './KINN_PINNs_plate_hole.py'

# Run the scripts
run_python_script(file3, working_dir_1)
run_python_script(file4, working_dir_1)
run_python_script(file5, working_dir_2)
run_python_script(file6, working_dir_2)
