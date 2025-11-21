import json
import os
import warnings
from collections import OrderedDict, defaultdict
from datetime import date, datetime
from functools import reduce
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple

import torch
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms

def _read_json_file(file_path: Path) -> dict:
    """
    Reads a JSON file and returns its content as a dictionary.

    Args:
        file_path: The path to the JSON file.

    Returns:
        A dictionary containing the JSON data.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
    return data
    
def remap_coco_annotations(coco_file_path: Path, selected_labels: list) -> dict:
    """
    Remaps COCO annotation file based on a list of selected labels.

    Args:
        coco_file_path: Path to the original COCO file.
        selected_labels: A list of labels to filter and remap.

    Returns:
        A new COCO dataset dictionary with filtered and remapped data.
    """
    # sort selected labels
    selected_labels = sorted(selected_labels)

    # read coco dictionary
    coco_data = _read_json_file(coco_file_path)

    # Create a mapping of category_id to new id and name
    category_mapping = {}
    for idx, cat_id in enumerate(selected_labels):
        category = next(
            (cat for cat in coco_data["categories"] if cat["id"] == cat_id), None
        )
        if category:
            category_mapping[cat_id] = {"new_id": idx + 1, "name": category["name"]}

    # Filter annotations based on selected labels and update category_id
    filtered_annotations = [
        ann for ann in coco_data["annotations"] if ann["category_id"] in selected_labels
    ]

    # Get the set of image_ids that have annotations in the selected labels
    filtered_image_ids = {ann["image_id"] for ann in filtered_annotations}

    # Filter images and update file names
    filtered_images = [
        image for image in coco_data["images"] if image["id"] in filtered_image_ids
    ]

    # Assign new sequential IDs to images
    image_id_mapping = {img["id"]: idx + 1 for idx, img in enumerate(filtered_images)}
    images = [
        {
            "width": img["width"],
            "height": img["height"],
            "id": image_id_mapping[img["id"]],
            "file_name": os.path.basename(img["file_name"]),
        }
        for img in filtered_images
    ]

    # Filter categories based on selected labels
    categories = [
        {
            "id": category_mapping[cat_id]["new_id"],
            "name": category_mapping[cat_id]["name"],
        }
        for cat_id in selected_labels
        if cat_id in category_mapping
    ]

    annotations = [
        {
            "id": idx + 1,
            "image_id": image_id_mapping[ann["image_id"]],
            "category_id": category_mapping[ann["category_id"]]["new_id"],
            "segmentation": ann["segmentation"],
            "bbox": ann["bbox"],
            "ignore": ann["ignore"],
            "iscrowd": ann["iscrowd"],
            "area": ann["area"],
        }
        for idx, ann in enumerate(filtered_annotations)
        if ann["category_id"] in category_mapping
    ]

    # Create the new dictionary
    remapped_coco_data = {
        "images": images,
        "categories": categories,
        "annotations": annotations,
        "info": coco_data["info"],
    }

    return remapped_coco_data

def combine_two_coco_files(coco_file_1: Dict, coco_file_2: Dict) -> Dict:
    """
    Combines a list of COCO annotation files into a single dictionary

    Args:
        coco_file_1: The first COCO file
        coco_file_2: The second COCO file

    Returns:
        A dictionary containing the combined COCO annotation data.
    """

    # Modifies the second COCO file based on the ID values of the first COCO file
    # Get the image and annotation counts from the first COCO file.
    image_count_1 = len(coco_file_1["images"])
    annot_count_1 = len(coco_file_1["annotations"])

    # Increment the ID values of the images and annotations in the second COCO file.
    for img in coco_file_2["images"]:
        img["id"] = img["id"] + image_count_1 - 1
    for ant in coco_file_2["annotations"]:
        ant["image_id"] = ant["image_id"] + image_count_1 - 1
        ant["id"] = ant["id"] + annot_count_1 - 1

    # Create a new dictionary to store the combined COCO data.
    coco_dict = {}

    # Iterate over the keys in the second COCO file and add the data to the new dictionary.
    for key, val in coco_file_2.items():
        if key not in ["categories", "info"]:
            # For the "images" and "annotations" keys, combine the data from the two COCO files.
            coco_dict[key] = coco_file_1[key] + coco_file_2[key]
        else:
            # For the "categories" and "info" keys, copy the data from the first COCO file.
            coco_dict[key] = coco_file_1[key]

    return coco_dict


def combine_coco_files(coco_files_to_combine: List[Dict]) -> Dict:
    """
    Combines a list of COCO annotation files into a single dictionary

    Args:
        coco_files_to_combine: A list of file paths to COCO annotated dictionaries

    Returns:
        A dictionary containing the combined COCO annotation data.
    """
    # combine files recursively
    combined_coco_file = reduce(combine_two_coco_files, coco_files_to_combine)
    return combined_coco_file
    
