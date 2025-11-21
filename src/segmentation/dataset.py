import os
from pathlib import Path
from typing import List, Optional

import torch
import torchvision
from torchvision.transforms import v2

from utils.logging.logger import LoggingConfigurator

configurator = LoggingConfigurator()


class SegmentationDataset:
    """
    Custom dataset for loading and preprocessing segmentation images and masks.

    This class loads image and mask pairs from specified directories, performs
    resizing and normalization (if enabled), and returns them as PyTorch tensors.

    Eg:
    path_to_annotation = "./TestProject/annotations-unittest.json"
    image_path = "./TestProject/images"
    mask_path = "./TestProject/masks"
    augmented_images_path = "./TestProject/augmented_images_path"
    mask_list = [m for m in os.listdir(Path(mask_path) / "train")]

    coco2mask(path_to_annotation, image_path, mask_path, augmented_images_path)
    images = utils.load_transform_images(image_path, mask_list)
    mean, std = utils.calculate_mean_std(images)
    dataset = SegmentationDataset([image_path, augmented_images_path],
                                  mask_path,
                                  mask_list,
                                  normalize=True,
                                  mean=mean,
                                  std=std)
    """

    def __init__(
        self,
        image_dirs: List[Path],
        mask_dir: Path,
        mask_list: list,
        device: str = "cpu",
        resize: tuple = (512, 512),
        normalize: bool = False,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        """
        Args:
            image_dirs: List of paths to folders containing images.
            mask_dir: Path to the folder containing masks.
            mask_list: List of mask filenames for filtering.
            device: Device to move tensors to (default: cpu)
            resize: Target size for image resizing (default: 512).
            normalize: Flag to enable image normalization (default: False).
            mean: List of channel means for normalization (optional).
            std: List of channel standard deviations for normalization (optional).
        """
        self.image_dirs = image_dirs
        self.mask_dir = mask_dir
        self.mask_list = mask_list
        self.device = device
        self.resize = resize
        self.normalize = normalize
        self.mean = mean
        self.std = std

        if self.normalize and (self.mean is None or self.std is None):
            raise ValueError("Normalization requires both mean and std lists.")

        self.base_names, self.img_extensions = self._preprocess_image_info()

    def _preprocess_image_info(self):
        """Extracts valid image base names & extensions from directories."""
        # Check if the mask directory exists
        if not os.path.exists(self.mask_dir):
            raise ValueError(f"Directory '{self.mask_dir}' does not exist!")

        # Extract mask base names from mask list
        mask_list_names = {os.path.splitext(m)[0] for m in self.mask_list}

        # Extract mask base names from mask directory
        mask_base_names = {
            os.path.splitext(file)[0] for file in os.listdir(self.mask_dir)
        }

        # Identify valid image extensions
        valid_image_extensions = {".jpg", ".jpeg", ".bmp", ".png", ".gif"}

        # Collect image names and extensions from all image directories
        img_extensions = set()
        img_base_names = set()
        for image_dir in self.image_dirs:
            if not os.path.exists(image_dir):
                raise ValueError(f"Directory '{image_dir}' does not exist!")

            # Get image base names and extensions from current directory
            img_file_names, file_extensions = _extract_file_info(image_dir)
            img_base_names.update(img_file_names)
            img_extensions |= file_extensions

        return list(mask_list_names & mask_base_names & img_base_names), list(
            img_extensions & valid_image_extensions
        )

    def __getitem__(self, index):
        """
        Retrieves a pair of image and mask tensors at the specified index.

        Args:
            index: Index.

        Returns:
            A tuple containing a resized and normalized image tensor (float)
            and a resized mask tensor (float).
        """
        transforms = v2.Compose(
            [
                v2.Resize(size=(self.resize)),
                v2.ToDtype(torch.float32, scale=False),
            ]
        )

        # Find the image path with the corresponding extension
        for ext in self.img_extensions:
            for image_dir in self.image_dirs:
                img_path = image_dir / (self.base_names[index] + ext)
                if os.path.exists(img_path):
                    # Read image and mask
                    img = torchvision.io.read_image(str(img_path))
                    mask = torch.load(self.mask_dir / (self.base_names[index] + ".pt"))

                    # transform image and mask
                    img = transforms(img)
                    mask = transforms(mask)

                    # Apply normalization if enabled
                    if self.normalize:
                        img = v2.Normalize(mean=self.mean, std=self.std)(img)

                    return img.to(self.device), mask.to(self.device)

    def __len__(self):
        return len(self.base_names)


def _extract_file_info(directory: Path) -> tuple[set[str], set[str]]:
    """Extracts file extensions and base filenames from a directory.

    Args:
        directory: The path to the directory.

    Returns:
        A tuple containing a set of file extensions and a set of base filenames.
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        raise ValueError(f"Directory '{directory}' does not exist!")

    # Create empty sets to store file extensions and base filenames
    file_extensions = set()
    base_filenames = set()

    # Iterate over files in the directory
    for file in os.listdir(directory):
        # Split the filename into base and extension
        base, ext = os.path.splitext(file)
        # Add base & extension to their respective sets
        base_filenames.add(base)
        file_extensions.add(ext)

    return base_filenames, file_extensions
