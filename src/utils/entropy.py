from collections import deque

import cv2
import numpy as np
from PIL import Image

from cds_vision_tools.utils.image_helper import crop_image


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


def entropy_calculator(
    entropy: float, entropy_list: deque[float]
) -> tuple[deque[float], float]:
    """
    Calculates the entropy change percentage relative to the median of a list of entropy values.

    Args:
        entropy: The current entropy value.
        entropy_list: A deque with a defined maximum length containing a history of entropy values.

    Returns:
        A tuple containing the updated entropy list and the calculated entropy change percentage.
    """
    if not isinstance(entropy_list, deque):
        raise TypeError("entropy_list must be a deque.")

    # Append the current entropy
    entropy_list.append(entropy)

    # Calculate the median entropy
    mean_entropy = np.median(entropy_list)

    # Handle division by zero
    if mean_entropy == 0:
        return entropy_list, 0

    # Calculate the entropy change percentage
    entropy_change_percent = (entropy - mean_entropy) * 100 / mean_entropy

    return entropy_list, entropy_change_percent


def get_event_status(
    entropy_change_percent: float, prev_entropy_change_percent: float, thresh: float
) -> str:
    """
    Determines the event status based on entropy change.

    Args:
        entropy_change_percent: The entropy change percent from the current frame.
        prev_entropy_change_percent: The entropy change percent from the previous frame.
        thresh: The threshold for entropy change to consider an event.

    Returns:
        str: The event status ("event_start", "image_collection", "event_end", "no_event").
    """

    if entropy_change_percent > thresh:
        if prev_entropy_change_percent <= thresh:
            return "event_start"
        else:
            return "image_collection"
    elif prev_entropy_change_percent > thresh:
        return "event_end"
    else:
        return "no_event"
