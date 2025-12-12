import os
import random
import shutil

# Set the source folder path
source_folder = 'Crops/3000_images'  # Replace with your actual source folder path

# Define the number of splits and destination base folder
num_splits = 5
destination_base = 'Crops/split_folders'  # Base directory to hold the split folders


# Create destination folders
split_folders = [os.path.join(destination_base, f'split_{i+1}') for i in range(num_splits)]
for folder in split_folders:
    os.makedirs(folder, exist_ok=True)

# Define supported image extensions
image_extensions = ['.jpg', '.jpeg', '.png']

# List and shuffle all image files
all_images = [f for f in os.listdir(source_folder)
              if os.path.splitext(f)[1].lower() in image_extensions]
random.shuffle(all_images)

# Split images evenly into the 5 folders
for idx, image_name in enumerate(all_images):
    src_path = os.path.join(source_folder, image_name)
    split_index = idx % num_splits
    dst_path = os.path.join(split_folders[split_index], image_name)
    shutil.copy(src_path, dst_path)

print(f"Shuffled and split {len(all_images)} images into {num_splits} folders under '{destination_base}'.")
