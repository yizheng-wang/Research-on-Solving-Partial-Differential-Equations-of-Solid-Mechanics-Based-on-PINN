import subprocess

def run_python_script(filepath):
    print(f"Running {filepath}...")
    subprocess.run(['python', filepath], check=True)
    print(f"Finished running {filepath}.")

# File paths extracted from the image
files = [
    './Plate_hole_DEM_triangle_circle.py',
    './Plate_hole_DEM_triangle_circle2elipse_full_finetuning.py',
    './Plate_hole_DEM_triangle_circle2elipse_lightweight.py',
    './Plate_hole_DEM_triangle_circle2elipse_lora_r=1.py',
    './Plate_hole_DEM_triangle_circle2elipse_lora_r=100.py',
    './Plate_hole_DEM_triangle_circle2elipse_lora.py',
    './Plate_hole_DEM_triangle_elipse.py',
    './Plate_hole_DEM_triangle_elipse2circle_full_finetuning.py',
    './Plate_hole_DEM_triangle_elipse2circle_lightweight.py',
    './Plate_hole_DEM_triangle_elipse2circle_lora_r=1.py',
    './Plate_hole_DEM_triangle_elipse2circle_lora_r=100.py',
    './Plate_hole_DEM_triangle_elipse2circle_lora.py'
]

# Running the scripts
for file in files:
    run_python_script(file)

print("All programs completed.")
