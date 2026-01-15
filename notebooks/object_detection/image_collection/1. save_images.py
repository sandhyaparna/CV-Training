##### Save images from multiple cameras in the format of PlantName_VersionNumber_CamID_imageID
import cv2
import os
import time
import gc

gc.collect()

# ---------------- CONFIG ----------------
camera_ids = [181, 184, 178, 179, 180, 182, 177, 176]

PLANT_NAME = "PlantA"
VERSION = "v1"

duration = 3 * 60     # 3 minutes
interval = 5          # seconds

# Single output directory
output_dir = "Dir/saved_frames_all"
os.makedirs(output_dir, exist_ok=True)
# ---------------------------------------

# Determine global image index (across all cameras)
existing_files = [f for f in os.listdir(output_dir) if f.endswith(".jpg")]
existing_numbers = []

for f in existing_files:
    try:
        image_id = int(f.split("_")[-1].split(".")[0])
        existing_numbers.append(image_id)
    except Exception:
        continue

global_image_index = max(existing_numbers, default=0) + 1

while True:
    for cam_id in camera_ids:
        print(f"\nStarting capture for camera {cam_id}")
        camera_url = f"rtsp://user:pass@cam{cam_id}/axis-media/media.amp"
        cap = cv2.VideoCapture(camera_url)

        if not cap.isOpened():
            print(f"Failed to connect to camera {cam_id}. Skipping...")
            continue

        start_time = time.time()
        last_saved_time = start_time

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to grab frame from camera {cam_id}.")
                break

            current_time = time.time()

            if current_time - last_saved_time >= interval:
                filename = (
                    f"{PLANT_NAME}_{VERSION}_{cam_id}_{global_image_index:06d}.jpg"
                )
                filepath = os.path.join(output_dir, filename)

                cv2.imwrite(filepath, frame)
                print(f"[Camera {cam_id}] Saved {filename}")

                last_saved_time = current_time
                global_image_index += 1

            if current_time - start_time >= duration:
                print(f"Finished {duration} secs capture for camera {cam_id}")
                break

        cap.release()

print("All camera captures completed.")


#### Save images in the corresponding camid folder #####
import cv2
import os
import time

# List of camera IDs to loop through
camera_ids = [181, 184, 178, 179, 180, 182, 177, 176,] 

# Duration and interval settings
duration = 3 * 60  # 3 minutes
interval = 5  # Save image every 5 seconds

import gc
gc.collect()

while True:
    for cam_id in camera_ids:
        print(f"\nStarting capture for camera {cam_id}")
        camera_url = f"rtsp://user:pass@cam{cam_id}/axis-media/media.amp"
        cap = cv2.VideoCapture(camera_url)

        if not cap.isOpened():
            print(f"Failed to connect to camera {cam_id}. Skipping...")
            continue

        output_dir = os.path.join("Dir/saved_frames", f"{cam_id}")
        os.makedirs(output_dir, exist_ok=True)

        # Determine starting image number
        existing_files = [f for f in os.listdir(output_dir) if f.endswith(".jpg")]
        existing_numbers = []
        for f in existing_files:
            try:
                num = int(os.path.splitext(f)[0])
                existing_numbers.append(num)
            except ValueError:
                continue
        start_index = max(existing_numbers, default=0) + 1

        start_time = time.time()
        last_saved_time = start_time
        image_index = start_index

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to grab frame from camera {cam_id}.")
                break

            current_time = time.time()

            if current_time - last_saved_time >= interval:
                filename = os.path.join(output_dir, f"{image_index}.jpg")
                cv2.imwrite(filename, frame)
                print(f"[Camera {cam_id}] Saved frame as {image_index}.jpg")
                last_saved_time = current_time
                image_index += 1

            if current_time - start_time >= duration:
                print(f"Finished {duration} secs capture for camera {cam_id}")
                break

        cap.release()

print("All camera captures completed.")

