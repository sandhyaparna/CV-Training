import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def _get_overlay(
    original_img: np.ndarray,
    mask: np.ndarray,
    img_wt: float = 0.7,
    mask_wt: float = 0.3,
) -> np.ndarray:
    """
    Overlays a mask onto an original image.

    Args:
        original_img: The original image.
        mask: The mask to overlay.
        img_wt: The weight of the original image (default: 0.7).
        mask_wt: The weight of the mask (default: 0.3).

    Returns:
        The overlay image.
    """

    # Create a color mask (range of values [0, 255] represents color)
    color_mask = np.stack([mask * 255, mask, mask], axis=2)

    # Overlay the mask
    return cv2.addWeighted(
        original_img, img_wt, color_mask, mask_wt, 0, dtype=cv2.CV_8U
    )


def img_inference(
    model, original_img: np.ndarray, processed_img: np.ndarray, device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs inference on an image using a segmentation model.

    Args:
        model: Trained Segmentation model.
        original_img: The original image.
        processed_img: The pre-processed image for inference.
        device: The device to send the preprocessed image to (e.g., "cuda" for GPU).

    Returns:
        The overlay image.
    """
    # Perform inference using the model
    with torch.no_grad():
        if type(model).__name__.lower() == "unet":
            y_pred = model(processed_img)[0]
        else:
            y_pred = model(processed_img)["out"][0]

    # Calculate the probability difference between classes
    # TODO: Need to adapt to > 1 classes.
    y_pred_1d = y_pred[0] - y_pred[1]

    # Create a binary mask based on the probability difference
    mask = np.array((y_pred_1d > 0.0).to(device))

    # Generate the overlay image by combining the resized image and mask
    return _get_overlay(original_img, mask)


def display_results(original_img: np.ndarray, overlay: np.ndarray):
    """
    Displays the original image, mask, and overlay in a single figure.

    Args:
        original_img: The original image.
        overlay: The overlaid image.
    """
    # Create a figure with specified size
    plt.figure(figsize=(12, 4))

    # Subplot 1: Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis("off")

    # Subplot 2: Overlay
    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    # Adjust layout to prevent overlapping elements
    plt.tight_layout()
    # Display the figure
    plt.show()
