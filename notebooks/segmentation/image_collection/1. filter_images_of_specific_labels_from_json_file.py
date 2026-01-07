import json
import os
import shutil

def extract_matching_files(data):
    matching_files = []

    for item in data:
        annotations = item.get("annotations", [])
        if not annotations:
            continue

        result_list = annotations[0].get("result", [])
        if not result_list:
            continue

        choices = result_list[0].get("value", {}).get("choices", [])

        # Check for exact match
        if choices == ["label1", "label3"]: # ["label5"]:  
            file_upload = item.get("file_upload")

            if file_upload and "-" in file_upload:
                # Extract part after first dash
                after_dash = file_upload.split("-", 1)[1]
                matching_files.append(after_dash)

    return matching_files

def copy_images(file_list, source_folder, destination_folder):
    """
    Copy images whose names are in file_list from the source folder to the destination folder.
    """

    # Ensure destination exists
    os.makedirs(destination_folder, exist_ok=True)

    copied = []
    missing = []

    for filename in file_list:
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(destination_folder, filename)

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            copied.append(filename)
        else:
            missing.append(filename)

    return copied, missing


# ---- Example Usage ----
# Suppose your JSON is stored in a file named "input.json"
with open("v1-crops-01-04-2025-combined.json", "r") as f:
    json_data = json.load(f)

filtered_files = extract_matching_files(json_data)
# print(filtered_files)
print(len(filtered_files))

# Copy images
copied, missing = copy_images(
    filtered_files,
    source_folder="./3000_images",
    destination_folder="./background/v1_class"
)

# print("Copied:", copied)
# print("Missing:", missing)
print("Copied:", len(copied))
print("Missing:", len(missing))







