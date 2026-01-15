import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def crop_image(img: np.ndarray, hot_spot: dict[str, int]) -> np.ndarray:
    """
    Crops an image based on given coordinates from a hotspot dictionary.

    Args:
        img: The input image as a NumPy array.
        hot_spot: A dictionary containing coordinates of the hotspot. If None, no cropping is performed.
            - top: The starting y-coordinate of the crop.
            - bottom: The ending y-coordinate of the crop.
            - left: The starting x-coordinate of the crop.
            - right: The ending x-coordinate of the crop.

    Returns:
        The cropped image as a NumPy array.
    """
    # Check if the image is valid
    if img is None or img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError("Invalid image: Image is empty or has incorrect dimensions.")

    # Check for valid hotspot dictionary
    required_keys = {"top", "left", "bottom", "right"}
    missing_keys = required_keys - set(hot_spot.keys())
    if missing_keys:
        raise ValueError(
            f"Invalid hotspot dictionary: Missing required keys: {sorted(missing_keys)}"
        )

    # Validate hotspot coordinates
    if hot_spot["left"] >= hot_spot["right"] or hot_spot["top"] >= hot_spot["bottom"]:
        raise ValueError("Starting coordinates must be less than ending coordinates.")

    # Ensure coordinates are within image bounds
    x1 = max(0, hot_spot["left"])
    y1 = max(0, hot_spot["top"])
    x2 = min(img.shape[1], hot_spot["right"])
    y2 = min(img.shape[0], hot_spot["bottom"])

    return img[y1:y2, x1:x2]


def get_entropy(frame: np.ndarray, hot_spot: dict) -> tuple[float, np.ndarray]:
    """
    Calculates the entropy of a cropped region from an image.

    Args:
        frame: The input image.
        hot_spot: A dictionary containing coordinates of the hot spot. If None, no cropping is performed.
            - top: The starting y-coordinate of the crop.
            - bottom: The ending y-coordinate of the crop.
            - left: The starting x-coordinate of the crop.
            - right: The ending x-coordinate of the crop.

    Returns:
        Entropy of the hot spot, and the cropped image in RGB color space.
    """
    # crop the image
    hot_spot_img = crop_image(frame, hot_spot)

    # converts image from BGR color space to RGB color space
    hot_spot_img = Image.fromarray(cv2.cvtColor(hot_spot_img, cv2.COLOR_BGR2RGB))

    return hot_spot_img.entropy(), cv2.cvtColor(
        np.array(hot_spot_img), cv2.COLOR_BGR2RGB
    )

# Process images in folder
def cal_entropy_in_folder(image_folder, hotspot):
    image_folder = Path(image_folder)
    entropy_dict = {}

    for image_file in image_folder.glob("*.jpg"):  # You can add other formats if needed
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"Failed to load image: {image_file}")
            continue

        # _, label_names = visualize_detections(image, model, class_names, return_only=True)
        entropy, hot_spot_img = get_entropy(image, hotspot)
        entropy_dict[image_file] = entropy
        print(f"Processing {image_file}, Entropy: {entropy}")

    entropy_list = list(entropy_dict.values())

    print("5th percentile value:", np.percentile(entropy_list, 5))
    print("10th percentile value:", np.percentile(entropy_list, 10))
    print("15th percentile value:", np.percentile(entropy_list, 15))
    print("20th percentile value:", np.percentile(entropy_list, 20))
    print("25th percentile value:", np.percentile(entropy_list, 25))
    print("min", min(entropy_list))
    print("max", max(entropy_list))


# Example usage
if __name__ == "__main__":
    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # x1 → left
    # y1 → top
    # x2 → right
    # y2 → bottom
    hotspots = {
        "176": {"top": 1225, "bottom": 1850, "left": 1050, "right": 3200},
        "177": {"top": 1170, "bottom": 1830, "left": 650, "right": 2930},
        "178": {"top": 1100, "bottom": 1650, "left": 820, "right": 2600},
        "179": {"top": 1350, "bottom": 2000, "left": 650, "right": 3080},
        "180": {"top": 1120, "bottom": 1750, "left": 830, "right": 3000},
        "181": {"top": 1170, "bottom": 1630, "left": 1350, "right": 2250},
        "182": {"top": 1300, "bottom": 2000, "left": 650, "right": 2930},
        "184": {"top": 1350, "bottom": 2000, "left": 920, "right": 3230},
    }

    # Update entropy thresholds to work in entropy_thres file
    entropy_thres = {
        "176": 7.99,  # 8.1
        "177": 7.36,  # 7.48
        "178": 7.05,  # 7.31
        "179": 7.77,  # 7.86
        "180": 7.65,  # 7.72
        "181": 7.81,  # 7.91
        "182": 7.63,  # 7.73
        "184": 7.78,  # 7.87
    }

    cam_id = 184
    cal_entropy_in_folder(
        f"HighRiver/saved_frames/{cam_id}",
        hotspots[f"{cam_id}"],
        # entropy_thres[f"{cam_id}"],
    )

    # 5th percentile value: 7.838592096280794
    # 10th percentile value: 7.9238629471418
    # 15th percentile value: 7.99349995398908
    # 20th percentile value: 8.054114944364827
    # 25th percentile value: 8.09682838855974
    # min 7.588278534986033
    # max 8.629118902162583



    # for cam_id in [ 179, 180, 181, 182, 184]: # 176, 177, 178,
    #     process_images_in_folder(
    #         f"HighRiver/saved_frames/{cam_id}",
    #         hotspots[f"{cam_id}"],
    #         entropy_thres[f"{cam_id}"]
    #     )

