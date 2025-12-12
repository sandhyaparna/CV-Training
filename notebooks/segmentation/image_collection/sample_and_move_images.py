# * We need to collect 3000 Images and there are 7 cameras from which we collected images
# * From each camera, we need to select 430 Images
# * Invalid should be around 50 per camera if possible

import os
import random
import shutil

def sample_and_move_images(source_folder, destination_folder, num_samples):
    # Allowed image file types
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}

    # Create destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # List all images in the source folder
    image_files = [
        f for f in os.listdir(source_folder)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]

    if num_samples > len(image_files):
        raise ValueError("Requested more samples than available images.")

    # Randomly sample images
    sampled_images = random.sample(image_files, num_samples)

    # Move sampled images
    for img in sampled_images:
        src_path = os.path.join(source_folder, img)
        dest_path = os.path.join(destination_folder, img)
        shutil.move(src_path, dest_path)
        print(f"Moved: {img}")

    print(f"\nSuccessfully moved {num_samples} images!")

# Example usage
folder_dir = "Crops/"
source_folder = folder_dir + "red_182/182_Q30"
destination_folder = folder_dir + "3000_images"
num_samples = 362 # 50 380 430 - 250/130/50 - 300/80/50

sample_and_move_images(source_folder, destination_folder, num_samples)
