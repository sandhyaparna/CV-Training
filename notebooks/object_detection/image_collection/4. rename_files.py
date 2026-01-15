import os
import shutil

# ------------- CONFIG ----------------
BASE_DIR = "plant/saved_frames"  # existing folders
OUTPUT_DIR = "plant/saved_frames/all_images_captured"  # new single folder

PLANT_NAME = "hr"
VERSION = "v2"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(sorted(os.listdir(BASE_DIR)))
# Loop through each camera folder
for cam_id in ['176', '177', '178', '179', '180', '181', '182', '184']:
    cam_path = os.path.join(BASE_DIR, cam_id)
    print("cam_path:", cam_path)

    if not os.path.isdir(cam_path):
        continue

    print(f"Processing camera {cam_id}...")

    for img in os.listdir(cam_path):
        if not img.lower().endswith(".jpg"):
            continue

        # Extract original image ID (e.g. "27" from "27.jpg")
        image_id = os.path.splitext(img)[0]

        src = os.path.join(cam_path, img)

        new_name = f"{PLANT_NAME}_{VERSION}_{cam_id}_{image_id}.jpg"
        dst = os.path.join(OUTPUT_DIR, new_name)

        # Safety check to avoid overwrite
        if os.path.exists(dst):
            raise FileExistsError(f"File already exists: {dst}")

        shutil.move(src, dst)

print("✅ Renaming and consolidation completed.")
