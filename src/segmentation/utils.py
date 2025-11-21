import os
from pathlib import Path

import torch
import torchvision
from torchvision.transforms import v2

from utils.logging.logger import LoggingConfigurator

configurator = LoggingConfigurator()


def load_transform_images(
    folder_path: Path,
    mask_list: list,
    image_extensions: tuple = (".jpg", ".jpeg", ".png"),
    resize: tuple = (512, 512),
):
    """
    Loads and preprocesses images from a specified folder based on a mask list.

    Args:
      folder_path: Path to the folder containing images.
      mask_list: List of mask filenames for filtering.
      image_extensions: Tuple of valid image file extensions (default: JPG, JPEG, PNG).
      resize: Target size for image resizing (default: 512).

    Returns:
      A list of resized and normalized PyTorch tensors representing images in both the folder and mask list.

    Raises:
      ValueError: If the provided folder path doesn't exist.
      RuntimeError: If an error occurs while reading an image.
    """
    transforms = v2.Compose(
        [
            v2.Resize(size=(resize)),
            v2.ToDtype(torch.float32, scale=False),
        ]
    )

    # Extract mask names from mask filenames (without extensions)
    img_masks = [os.path.splitext(m)[0] for m in mask_list]

    if not img_masks:
        raise ValueError("No masks provided. Function will return empty list.")

    # Check if the folder path exists
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder path '{folder_path}' does not exist!")

    # check if there no images in the folder
    if not os.listdir(folder_path):
        raise ValueError(
            f"Image folder '{folder_path}' appears to be empty. Function might return empty list."
        )

    # empty list to store images
    images = []
    for filename in os.listdir(folder_path):
        if (
            filename.lower().endswith(image_extensions)
            and os.path.splitext(filename)[0] in img_masks
        ):
            try:
                # Read image using torchvision.io.read_image
                image = torchvision.io.read_image(str(folder_path / filename))
                # transform image
                image = transforms(image)
                # Add the processed image to the images list
                images.append(image)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Error reading image '{folder_path / filename}': {e}"
                )
    return images


def calculate_mean_std(images):
    """Calculates the mean and standard deviation of a list of images.

    Args:
      images: A list of PyTorch tensors representing images.

    Returns:
      A tuple of mean and standard deviation tensors.

    Raises:
      ValueError: If the input list is empty or contains tensors of different shapes.
    """

    if not images:
        raise ValueError("Input list of images is empty.")

    # Check if all images have the same shape
    if not all(img.shape == images[0].shape for img in images):
        raise ValueError("All images must have the same shape.")

    # Concatenate images into a single tensor
    images = torch.stack(images, dim=0)

    # Calculate mean and standard deviation across batch, height, and width
    mean = images.mean([0, 2, 3])
    std = images.std([0, 2, 3])

    return mean, std
