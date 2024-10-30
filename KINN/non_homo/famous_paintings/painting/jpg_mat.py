from PIL import Image
import numpy as np
from scipy.io import savemat
import os

# Define the paths to the processed images
image_paths = [
    './picasuo_bw.jpg',
    './salvator_bw.jpg',
    './sky_bw.jpg'
]

# Function to load an image, rotate it, flip it horizontally, and return it as a numpy array
def load_and_process_image(image_path):
    image = Image.open(image_path)
    image = image.rotate(180)  # Rotate the image by 180 degrees
    image = image.transpose(Image.FLIP_LEFT_RIGHT)  # Flip the image horizontally
    return np.array(image)/255

# Load images
images = [load_and_process_image(path) for path in image_paths]

# Save images as .mat files
for i, image_array in enumerate(images):
    base_name = os.path.basename(image_paths[i]).split('.')[0]
    mat_file_path = f'./{base_name}.mat'
    savemat(mat_file_path, {base_name: image_array})
    print(f"Saved {mat_file_path}")

# After saving, you can use these .mat files in your MATLAB code.
