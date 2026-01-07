import os
import random
import shutil

# Set your source and destination folder paths
source_folder = f'background/v1_class'  # Replace with your actual source folder path
destination_folder = 'background/v1_background_250'  # Replace with your actual destination folder path
n_images = 62  # 62 88 125

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Extract the folder name to use as a prefix
folder_name = os.path.basename(os.path.normpath(source_folder))
print(folder_name)

# Define supported image extensions
image_extensions = ['.jpg', '.jpeg', '.png']

# List all image files in the source folder
all_images = [f for f in os.listdir(source_folder)
              if os.path.splitext(f)[1].lower() in image_extensions]

# Sample up to 320 images
sampled_images = random.sample(all_images, min(n_images, len(all_images)))

# Copy and rename the sampled images
for i, image_name in enumerate(sampled_images):
    # print(image_name)
    src_path = os.path.join(source_folder, image_name)
    dst_path = os.path.join(destination_folder, image_name)
    # print(src_path, dst_path)
    shutil.copy(src_path, dst_path)

print(f"Sampled {len(sampled_images)} images and saved them to '{destination_folder}'.")
