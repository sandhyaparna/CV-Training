import os
from pathlib import Path
from typing import Tuple

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


def resize_and_save_image(
    input_path: str, output_path: str, new_width: int, new_height: int
) -> None:
    """
    Opens an image, resizes it to the specified dimensions, and saves it.

    Args:
        input_path: Path to the input image file.
        output_path: Path to save the resized image.
        new_width: Desired width of the resized image.
        new_height: Desired height of the resized image.
    """
    try:
        with Image.open(input_path) as img:
            resized_img = img.resize((new_width, new_height))
            resized_img.save(output_path)

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def resize_images_in_folder(
    input_folder: Path,
    output_folder: Path,
    new_width: int,
    new_height: int,
    supported_formats: Tuple = (".png", ".jpg", ".jpeg", ".gif"),
) -> None:
    """
    Resizes all images in the input folder and saves them to the output folder.

    Args:
        input_folder: Path to the input folder containing images.
        output_folder: Path to the output folder to save resized images.
        new_width: Desired width of the resized images.
        new_height: Desired height of the resized images.
        supported_formats: Tuple of supported image formats.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Create the output folder if it doesn't exist
    # save images to new path
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            resize_and_save_image(input_path, output_path, new_width, new_height)
