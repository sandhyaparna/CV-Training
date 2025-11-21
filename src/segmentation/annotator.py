import json
import os
import random
from pathlib import Path
from typing import Any, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image, ImageDraw
from torchvision.transforms import v2
from tqdm import tqdm

from utils.logging.logger import LoggingConfigurator

configurator = LoggingConfigurator()


def coco2mask(
    paths_to_annotations: List[Union[str, Path]],
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    augmented_images_path: Union[str, Path],
    background_images_path: Union[str, Path] = None,
    background_images_sample: int = 500,
    train_portion: float = 0.7,
    val_portion: float = 0.2,
    shuffle: bool = True,
    augmentation_portion: float = 0.0,
    custom_augmentations: list = None,
) -> None:
    """
    This method converts coco polygon annotations to pytorch mask files. Optionally performs data augmentation as well.

    Ideal folder structure:
    project_name
        - annotations.json
        - image_path (keep images here)
        - mask_path
            - train (keep it empty, to be populated by this function)
            - val (keep it empty, to be populated by this function)
            - test (keep it empty, to be populated by this function)
        - augmented_images_path (keep it empty, to be populated by this function)

    Args:
        paths_to_annotations (list of str or pathlib.Path): paths to the annotations json files
        path_to_annotation (str or pathlib.Path): path to the annotations json file (eg. "carve/annotations.json")
        image_path (str or pathlib.Path): path to the folder where the images are
        mask_path (str or pathlib.Path): path to the folder where the mask files will get generated
        augmented_images_path (str or pathlib.Path): path to the folder where the augmented images will get saved
        background_images_path (str or pathlib.Path): Path to the folder containing background images. Defaults to None.
        background_images_sample (int): Number of background images to sample.
        train_portion (float): portion of the data to be set as train dataset. Default: 0.7
        val_portion (float): portion of the data to be set as validation dataset. Default: 0.2
        shuffle (bool): if the dataset should be shuffled before splitting into train-val-test. Default: True
        augmentation_portion (float): portion of the training dataset to be augmented (without replacement). Default: 0.0 (no image will get augmented)
        custom_augmentations (list): list of torchvision.transforms.V2 transformations

    Returns:
        N/A
    """
    annotations_df = pd.DataFrame()
    # Load and merge annotations from multiple files
    for annotation_path in paths_to_annotations:
        annotation_data = read_coco_annotation_json_file(Path(annotation_path))
        annotations_df = pd.concat([annotations_df, annotation_data], ignore_index=True)

    if background_images_path:
        annotations_df = add_background_images(
            background_images_path,
            annotations_df,
            background_images_sample,
            "file_name",
        )
    create_masks(
        image_path,
        mask_path,
        annotations_df,
        background_images_path,
        train_portion,
        val_portion,
        shuffle,
    )
    augment_data(
        image_path,
        mask_path,
        augmented_images_path,
        background_images_path,
        augmentation_portion,
        custom_augmentations,
    )
    return None


def read_coco_annotation_json_file(
    path_to_annotation: Union[str, Path]
) -> pd.DataFrame:
    """
    Reads annotation file and returns a dataframe with the columns:
    'file_name', 'image_id', 'category_id', 'segmentation'

    Args:
        path_to_annotation (str or pathlib.Path): path to the annotations json file (eg. "carve/annotations.json")

    Returns:
        A dataframe with all the annotations
    """
    try:
        with open(Path(path_to_annotation)) as f:
            annotations = json.load(f)
    except Exception as error:
        configurator.logger.error(error)
        raise

    configurator.logger.info("Successfully read input annotations file!")

    # Create DataFrames from the annotations
    image_df = pd.DataFrame.from_dict(annotations["images"])
    annotations_df = pd.DataFrame.from_dict(annotations["annotations"])

    # Merge DataFrames on image_id
    final_df = pd.merge(
        image_df, annotations_df, left_on="id", right_on="image_id", how="inner"
    )
    # Drop unnecessary columns
    final_df = final_df.drop(
        columns=["id_x", "width", "height", "id_y", "bbox", "area", "iscrowd"]
    )
    # Group by file_name and image_id, aggregating category_id and segmentation into lists
    return (
        final_df.groupby(by=["file_name", "image_id"])
        .agg({"category_id": list, "segmentation": list})
        .reset_index()
    )


def add_background_images(
    background_images_path: Union[str, Path],
    annotations_df: pd.DataFrame,
    sample_size: int = 500,
    column: str = "file_name",
    image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
) -> pd.DataFrame:
    """
    Adds image names from the background images folder to the annotations DataFrame.

    Parameters:
        background_images_path: Path to the folder containing background images.
        annotations_df: Existing DataFrame with annotations.
        sample_size: Number of images to sample.
        column: Column with image name in the annotations_df.
        image_extensions: Allowed image file extensions.

    Returns:
        Updated DataFrame with specified sample of background image names added.
    """
    # Get the list of image names in the specified folder
    image_names = [
        f
        for f in os.listdir(background_images_path)
        if f.lower().endswith(image_extensions)
    ]

    if not image_names:
        return annotations_df

    # Sample the image names
    sampled_image_names = random.sample(image_names, min(sample_size, len(image_names)))

    # Create a new DataFrame with the image names
    image_df = pd.DataFrame({column: sampled_image_names})

    # Concatenate the new DataFrame with the existing DataFrame
    return pd.concat([annotations_df, image_df], ignore_index=True)


def load_and_display_image_with_mask(
    image_path: Union[str, Path], mask_path: Union[str, Path]
) -> None:
    """
    Loads an image and a mask from their respective paths, and displays the image with the mask overlay.

    Args:
        image_path (str or pathlib.Path): path to the image file
        mask_path (str or pathlib.Path): path to the mask file

    Returns:
        Displays the image
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Load mask
    mask_t = torch.load(mask_path)

    # Convert mask to numpy array
    mask_np = mask_t[0].numpy()  # Assuming the first channel is the mask

    # Create a color mask
    color_mask = np.zeros_like(image_np)
    color_mask[:, :, 0] = mask_np * 255  # Red channel

    # Overlay the mask on the image
    overlay = image_np.copy()
    overlay[color_mask[:, :, 0] > 0] = [255, 0, 0]  # Red color for mask

    # Display the image and mask
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_np)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Image with Mask")
    axes[1].axis("off")

    plt.show()


def _load_image(
    image_name: str,
    image_path: Union[str, Path],
    background_images_path: Union[str, Path] = None,
) -> torch.Tensor:
    """
    Attempts to load an image from the primary image path.
    If unsuccessful, it tries to load from a background images path (if provided).

    Args:
        image_name: Name of the image file.
        image_path: Primary directory where the image is expected to be.
        background_images_path: Fallback directory for loading the image if the first attempt fails.

    Returns:
        torch.Tensor: Loaded image as a tensor.
    """
    try:
        # Attempt to read the image from the primary path
        img = torchvision.io.read_image(str(Path(image_path).joinpath(image_name)))
    except:
        if background_images_path is not None:
            # Attempt to read from the background path
            img = torchvision.io.read_image(
                str(Path(background_images_path).joinpath(image_name))
            )

    return img


def get_image_mask_tensor(
    image_path: Union[str, Path],
    file_name: Union[str, Path],
    polygon_points: list,
    background_images_path: Union[str, Path] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Function to create a mask (pytorch tensor) from a list of polygon vertices.

    Args:
        image_path (str or pathlib.Path): path to the folder where the images are
        file_name (str or pathlib.Path): image file name which is present in image_path
        polygon_points (list): a nested python list. The polygon_points[i][j] is a list which represents the vertices of a polygon. Example polygon_points[i][j] = [x1,y1,x2,y2,x3,y3,...,xn,yn]

    Returns:
        img (tensor): the input image as a tensor
        mask_img (tensor): the corresponding mask as a tensor
    """
    image_name = Path(file_name).name
    img = _load_image(image_name, image_path, background_images_path)

    # Missing support for pathlib.Path in older versions of torchvision
    c, h, w = img.shape
    # create a blank image with all pixel values set to 0
    mask_img = Image.new("L", (w, h), 0)
    if isinstance(polygon_points, list):
        for i in range(len(polygon_points)):
            for j in range(len(polygon_points[i])):
                # draw polygon with interior pixel values set to 1 on the blank image
                ImageDraw.Draw(mask_img, "L").polygon(polygon_points[i][j], fill=(1))
    mask_img = torchvision.transforms.functional.pil_to_tensor(mask_img)
    return img, mask_img


def save_mask(
    mask_t: Any, mask_path: Union[str, Path], file_name: Union[str, Path]
) -> None:
    """
    Saves a mask as a .pt file

    Args:
        mask_t (tensor): a tensor which represents a mask
        mask_path (str or pathlib.Path): path to the folder where the mask is to be saved
        file_name (str or pathlib.Path): file name to be used to save the mask

    Returns:
        N/A
    """
    file_name = Path(Path(file_name).stem + ".pt")
    path_to_save_mask = Path(mask_path).joinpath(file_name)
    # add background channel
    torch.save(torch.cat((mask_t, 1 - mask_t)), path_to_save_mask)
    return None


def transform_and_save(
    augmented_images_path: Union[str, Path],
    mask_path: Union[str, Path],
    img_t: Any,
    mask_t: Any,
    file_name: Union[str, Path],
    custom_augmentations: list = None,
) -> None:
    """
    Creates augmented image and it's corresponding mask and saves them.

    Args:
        augmented_images_path (str or pathlib.Path): path to the folder where the augmented images will get saved
        mask_path (str or pathlib.Path): path to the folder where the masks are
        img_t (tensor): the input image as a tensor
        mask_t (tensor): the input mask as a tensor
        file_name (str or pathlib.Path): image file name
        custom_augmentations (list): list of torchvision.transforms.V2 transformations

    Returns:
        N/A
    """
    aug_file_name = Path(Path(file_name).stem + "_augmented" + Path(file_name).suffix)
    if custom_augmentations:
        transforms = v2.Compose(custom_augmentations)
    else:
        transforms = v2.Compose(
            [
                v2.RandomResizedCrop(size=(512, 512), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
    ci, h1, wi = img_t.shape
    cm, hm, wm = mask_t.shape
    if ci > cm:
        mask_t_new = mask_t[0].expand(ci, hm, wm)
    both_images = transforms(
        torch.cat((img_t.unsqueeze(0), mask_t_new.unsqueeze(0)), 0)
    )
    img, mask = both_images[0], both_images[1]
    # TODO: Need to adapt to > 1 classes.
    mask = mask[0].unsqueeze(0)
    # mask = torch.cat((mask[0], 1 - mask[0]))
    torchvision.utils.save_image(
        img,
        fp=Path(augmented_images_path).joinpath(aug_file_name),
    )
    save_mask(mask, mask_path, aug_file_name)
    return None


def create_masks(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    annotations_df: Any,
    background_images_path: Union[str, Path] = None,
    train_portion: float = 0.7,
    val_portion: float = 0.2,
    shuffle: bool = True,
) -> None:
    """
    Reads annotations as dataframe and creates a mask file (.pt) for each image. Splits the dataset in train-val-test.

    Ideal folder structure:
    project_name
        - annotations.json
        - image_path (keep images here)
        - mask_path
            - train (keep it empty, to be populated by this function)
            - val (keep it empty, to be populated by this function)
            - test (keep it empty, to be populated by this function)

    Args:
        image_path (str or pathlib.Path): path to the folder where the images are
        mask_path (str or pathlib.Path): path to the folder where the mask files will get generated
        annotations_df (pandas dataframe): a pandas dataframe with all the annotations. Look at the output of the function read_coco_annotation_json_file
                Example dataframe:
                    data = {'file_name': ["images/0009f625-20230604-165506.png", "images/00610f9b-20230604-160513.png"],
                            'image_id': [1, 2],
                            'category_id': [[0], [0,0]],
                            'segmentation': [[[[9,7,31,45,6,9]]] , [[[1,4,67,34,8,4,23,45],[2,1,3,4,5,7]]]]}
                    pd.DataFrame.from_dict(data)
        background_images_path (str or pathlib.Path): Path to the folder containing background images. Defaults to None.
        train_portion (float): portion of the data to be set as train dataset. Default: 0.7
        val_portion (float): portion of the data to be set as validation dataset. Default: 0.2
        shuffle (bool): if the dataset should be shuffled before splitting into train-val-test. Default: True

    Returns:
        N/A
    """
    try:
        assert (
            train_portion + val_portion <= 1.0
        ), f"Sum of train_portion and val_portion should be less than or equal to 1. Got; {train_portion + val_portion}"
    except Exception as error:
        configurator.logger.error(error)
        raise

    if shuffle:
        annotations_df = annotations_df.sample(frac=1).reset_index(drop=True)

    mask_path = Path(mask_path)
    sub_dirs = ["train", "val", "test"]
    dir_paths = {}

    for sub_dir in sub_dirs:
        dir_path = mask_path.joinpath(sub_dir)
        dir_path.mkdir(parents=True, exist_ok=True)
        dir_paths[sub_dir] = dir_path

    count = 0

    configurator.logger.info("Starting creation of pytorch masks!")

    for index, row in tqdm(annotations_df.iterrows(), total=annotations_df.shape[0]):
        try:
            img_t, mask_t = get_image_mask_tensor(
                image_path,
                row["file_name"],
                row["segmentation"],
                background_images_path,
            )
            if count < round(len(annotations_df) * train_portion):
                save_mask(mask_t, dir_paths["train"], row["file_name"])
            elif count < round(len(annotations_df) * (train_portion + val_portion)):
                save_mask(mask_t, dir_paths["val"], row["file_name"])
            else:
                save_mask(mask_t, dir_paths["test"], row["file_name"])
        except Exception as error:
            configurator.logger.warning(error)
            pass

        count += 1

    configurator.logger.info("Successfully created pytorch masks!")
    return None


def augment_data(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    augmented_images_path: Union[str, Path],
    background_images_path: Union[str, Path] = None,
    augmentation_portion: float = 1.0,
    custom_augmentations: list = None,
    mask_extensions: list = ["pt"],
    image_extensions: list = ["jpg", "jpeg", "bmp", "png", "gif"],
) -> None:
    """
    Creates augmented images and their corresponding masks

    Ideal folder structure:
    project_name
        - annotations.json
        - image_path (keep original images here)
        - mask_path
            - train
            - val
            - test
        - augmented_images_path (keep it empty, to be populated by this function)

    Args:
        image_path (str or pathlib.Path): path to the folder where the images are
        mask_path (str or pathlib.Path): path to the folder where the masks are
        augmented_images_path (str or pathlib.Path): path to the folder where the augmented images will get saved
        background_images_path (str or pathlib.Path): Path to the folder containing background images. Defaults to None.
        augmentation_portion (float): portion of the training dataset to be augmented (without replacement). Default: 1.0 (all images will get augmented)
        custom_augmentations (list): list of torchvision.transforms.V2 transformations
        mask_extensions (list): extensions associated with the masks
        image_extensions (list): extensions associated with the images

    Returns:
        N/A
    """

    if not os.path.exists(Path(augmented_images_path)):
        os.makedirs(Path(augmented_images_path))

    mask_file_names = [
        fn
        for fn in os.listdir(Path(mask_path).joinpath("train"))
        if any(fn.endswith(ext) for ext in mask_extensions)
    ]
    train_masks = sorted(mask_file_names)

    img_file_names = [
        fn
        for folder in [image_path, background_images_path]
        if folder is not None
        for fn in os.listdir(Path(folder))
        if any(fn.endswith(ext) for ext in image_extensions)
    ]

    sample_indexes = random.sample(
        [i for i in range(len(train_masks))],
        round(len(train_masks) * augmentation_portion),
    )

    configurator.logger.info(
        f"Starting data augmentation for {len(sample_indexes)} images!"
    )

    for sample_index in tqdm(sample_indexes):
        mask_name = train_masks[sample_index]
        image_file_name = [
            fn for fn in img_file_names if Path(fn).stem == Path(mask_name).stem
        ][0]
        # Missing support for pathlib.Path in older versions of torchvision
        img_t = _load_image(image_file_name, image_path, background_images_path)
        mask_t = torch.load(Path(mask_path).joinpath("train").joinpath(mask_name))
        transform_and_save(
            augmented_images_path,
            Path(mask_path).joinpath("train"),
            img_t,
            mask_t,
            image_file_name,
            custom_augmentations,
        )

    configurator.logger.info(
        f"Successfully completed data augmentation for {len(sample_indexes)} images!"
    )
    return None
