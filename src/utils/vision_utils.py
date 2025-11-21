from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms import v2

from image_helper import crop_image


def preprocess_image(
    img: np.ndarray,
    hot_spot: dict[str, int] = None,
    resize: tuple = (512, 512),
    device: str = "cpu",
) -> tuple[np.ndarray, torch.Tensor]:
    """
    Preprocesses an image.

    Args:
      img: The input image.
      hot_spot: A dictionary containing coordinates of the hot spot. If None, no cropping is performed.
            - top: The starting y-coordinate of the crop.
            - bottom: The ending y-coordinate of the crop.
            - left: The starting x-coordinate of the crop.
            - right: The ending x-coordinate of the crop.
      resize: The target size for resizing the image. Defaults to (512, 512).
      device: The device to send the preprocessed image to (e.g., "cuda" for GPU).

    Returns:
      A tuple containing the resized image as a numpy array and the preprocessed image tensor.
    """
    # Crop based on hot spot coordinates
    if hot_spot:
        img = crop_image(img, hot_spot)

    # Convert to PyTorch tensor and convert to [channels, height, width]
    img = torch.from_numpy(img).permute(2, 0, 1).to(device)  # BGR -> RGB

    # Resize (optional)
    img = v2.Resize(size=resize)(img)

    # convert image from [channels, height, width] to [height, width, channels]
    resized_img = np.array(img.permute(1, 2, 0).type(torch.int))

    # Convert to float tensor and send to device
    img = img.type(torch.FloatTensor)

    return resized_img, img.unsqueeze(0)


def save_image(img: np.ndarray, output_img_path: Path):
    """
    Saves an image to a specified file.

    Args:
        img: The input image.
        output_img_path: The path to save the image to.
    """

    # Convert the image from BGR to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create the directory if it doesn't exist
    output_img_path.parent.mkdir(exist_ok=True)

    # Save the image to the specified path
    cv2.imwrite(str(output_img_path), rgb_img)
