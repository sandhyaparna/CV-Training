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


def manifest2coco(manifest_file: Path, selected_labels: list) -> Dict:
    """Converts a groundtruth manifest file to COCO format.

    Args:
        manifest_file: Path to the groundtruth manifest file.
        selected_labels:
        Eg:
           {
            "source-ref":"s3://<bucket-name>/<prefix>/<image_name>">,
            "<job_name>":{
                "image_size":[{"width":640,"height":480,"depth":3}],
                "annotations":[
                            {"class_id":0,"top":65,"left":85,"height":302,"width":244},
                            {"class_id":0,"top":68,"left":334,"height":323,"width":255}]
                            },
                "<job_name>-metadata":{"objects":[{"confidence":0},{"confidence":0}],
                "class-map":{"0":"class-1", "1":"class-2"},
                "type":"groundtruth/object-detection",
                "human-annotated":"yes",
                "creation-date":"2021-08-09T16:00:18.327433",
                "job-name":"labeling-job/<job_name>"
            }
        }

    Returns:
        A dictionary representing the COCO data
        Eg: {'images': [{'width': 1280,
           'height': 720,
           'id': 1,
           'file_name': 'images/3/10-29-22-61_20230926-093043-16.png'},],
         'categories': [{'id': 0, 'name': 'bone-discard'},
          {'id': 1, 'name': 'bone-discard-down'},
          {'id': 2, 'name': 'chuck'},
          {'id': 3, 'name': 'worker'}],
         'annotations': [{'id': 1,
           'image_id': 1,
           'category_id': 2,
           'segmentation': [],
           'bbox': [688, 481, 191, 132],
           'ignore': 0,
           'iscrowd': 0,
           'area': 25212},
          {'id': 2,
           'image_id': 1,
           'category_id': 2,
           'segmentation': [],
           'bbox': [989, 390, 164, 130],
           'ignore': 0,
           'iscrowd': 0,
           'area': 21320},],
         'info': {'year': 2024,
          'version': '1.0',
          'description': '',
          'contributor': 'groundtruth',
          'url': '',
          'date_created': '2023-10-19'}}
    """
    # sort selected labels
    selected_labels = sorted(selected_labels)

    # Load the groundtruth manifest file
    groundtruth_manifest = _read_aws_manifest(manifest_file)

    # Extract job name from the groundtruth manifest
    groundtruth_manifest_keys = list(groundtruth_manifest[0].keys())
    try:
        job_name = next(
            manifest_key
            for manifest_key in groundtruth_manifest_keys
            if manifest_key.endswith("adjustment")
        )
    except StopIteration:
        job_name = groundtruth_manifest_keys[1]

    images, annotations, categories = _process_images_and_annotations(
        groundtruth_manifest, job_name, selected_labels
    )

    # process categories info of selected labels
    categories = OrderedDict(sorted(categories.items()))
    categories = _process_category_data(categories, selected_labels)

    # Create the COCO data dictionary.
    coco_dict = {
        "images": images,
        "categories": categories,
        "annotations": annotations,
        "info": {
            "year": date.today().year,
            "version": "1.0",
            "description": "",
            "contributor": "groundtruth",
            "url": "",
            "date_created": _extract_date_as_str(
                groundtruth_manifest[0][job_name + "-metadata"]["creation-date"]
            ),
        },
    }

    return coco_dict


def manifest2pascalvoc(
    manifest_file: Path,
    root: Path,
    job_name: str,
    transform=None,
) -> List:
    """
    This function converts groundtruth manifest to PASCAL VOC format that is compatible with PyTorch.

    This function iterates through entries in a COCO-style manifest dictionary and
    processes each entry to generate a Pascal VOC annotation format.

    Args:
        manifest_file: Path to the groundtruth manifest file.
        Eg:
           {
            "source-ref":"s3://<bucket-name>/<prefix>/<image_name>">,
            "<job_name>":{
                "image_size":[{"width":640,"height":480,"depth":3}],
                "annotations":[
                            {"class_id":0,"top":65,"left":85,"height":302,"width":244},
                            {"class_id":0,"top":68,"left":334,"height":323,"width":255}]
                            },
                "<job_name>-metadata":{"objects":[{"confidence":0},{"confidence":0}],
                "class-map":{"0":"class-1", "1":"class-2"},
                "type":"groundtruth/object-detection",
                "human-annotated":"yes",
                "creation-date":"2021-08-09T16:00:18.327433",
                "job-name":"labeling-job/<job_name>"
            }
        }
        root: Root directory containing the images.
        job_name: The job name key used to identify the annotations within the
                  manifest data.
        transform: Optional transformations to apply to the image (using torchvision transforms).

    Returns:
        A list of tuples containing the processed images and annotations for all images in PASCAL VOC format.
    """
    # Load the groundtruth manifest file
    groundtruth_manifest = _read_aws_manifest(manifest_file)

    # Initialize an empty list to store processed data (image, annotation) tuples
    pascalvoc_data = []
    # Keep track of the image ID for PASCAL VOC annotations
    img_id = 1
    # Iterate through groundtruth manifest
    for i_dict in groundtruth_manifest:
        # Extract the image filename from the S3 path
        img_path = _extract_filename_from_s3_path(i_dict["source-ref"])
        # Load and preprocess image (handles potential missing images)
        img = _load_and_preprocess_image(root, img_path, transform)

        # Skip to the next iteration if image loading fails
        if img is None:
            continue
        # Process the current image entry to generate a PASCAL VOC annotation dictionary
        pascalvoc_annotation = _manifest_to_pascalvoc_annotations(
            i_dict, job_name, img_id
        )
        # Append processed image and annotations to the list
        pascalvoc_data.append((img, pascalvoc_annotation))
        # Increment the image ID for the next iteration
        img_id += 1

    return pascalvoc_data


def coco2pascalvoc(
    root: Path, annotation_file: Path, transform=None, relative_path: bool = True
) -> List[Tuple[torch.Tensor, dict]]:
    """
    This function converts COCO annotations to PASCAL VOC format that is compatible with PyTorch.

    Args:
        root: Root directory containing the images.
        annotation_file: Path to the COCO annotation file.
            Eg:
              {
               "images": [{"width": 1280, "height": 720, "id": 1, "file_name": "images/3/10-29-22-61_20231201-100120-6.png"},
                          {"width": 1280, "height": 720, "id": 2, "file_name": "images/3/10-29-22-61_20231201-103017-10.png"}],
               "categories": [{"id": 0, "name": "bone-discard"},
                              {"id": 1, "name": "bone-discard-down"},
                              {"id": 2, "name": "chuck"},
                              {"id": 3, "name": "person"}],
               "annotations": [{"id": 1, "image_id": 1, "category_id": 2, "segmentation": [], "bbox": [497, 73, 111, 100], "ignore": 0, "iscrowd": 0, "area": 11100},
                               {"id": 2, "image_id": 1, "category_id": 2, "segmentation": [], "bbox": [358, 245, 84, 112], "ignore": 0, "iscrowd": 0, "area": 9408}],
               "info": {"year": 2024, "version": "1.0", "description": "", "contributor": "groundtruth", "url": "", "date_created": "2023-12-13"}
              }
        transform: Optional transformations to apply to the image (using torchvision transforms).
        relative_path: Boolean flag indicating whether to return relative paths (True) or absolute paths (False).

    Returns:
        A list of tuples containing the processed images and annotations for all images in PASCAL VOC format.
    """
    # Load COCO API object
    try:
        coco_instances = COCO(annotation_file)
    except FileNotFoundError:
        raise ValueError(
            f"Annotation file '{annotation_file}' not found."
        ) from FileNotFoundError
    except json.JSONDecodeError:
        raise ValueError(f"Annotation file '{annotation_file}' is empty or corrupt.")

    # Extract image IDs to extract associated image file names & annotations
    ids = list(coco_instances.imgs.keys())

    # Loop through each image ID
    pascalvoc_data = []
    for img_id in ids:
        # Get image path based on COCO data and relative path flag
        img_path = _get_image_path(coco_instances, img_id, relative_path)
        # Load and preprocess image (handles potential missing images)
        img = _load_and_preprocess_image(root, img_path, transform)
        if img is not None:
            # Process COCO annotations for the current image
            pascalvoc_annotation = _coco_to_pascalvoc_annotations(
                coco_instances, img_id
            )
            # Append processed image and annotations to the list
            pascalvoc_data.append((img, pascalvoc_annotation))

    return pascalvoc_data


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


def _read_aws_manifest(file_path: Path) -> List[Dict]:
    """
    Reads aws manifest file generated from AWS Ground Truth and stores it as a list
    Args:
        file_path: Path
            file_path to .manifest file
    Returns:
        list of dictionaries of metadata annotations
        Eg:
           {
            "source-ref":"s3://<bucket-name>/<prefix>/<image_name>">,
            "<job_name>":{
                "image_size":[{"width":640,"height":480,"depth":3}],
                "annotations":[
                            {"class_id":0,"top":65,"left":85,"height":302,"width":244},
                            {"class_id":0,"top":68,"left":334,"height":323,"width":255}]
                            },
                "<job_name>-metadata":{"objects":[{"confidence":0},{"confidence":0}],
                "class-map":{"0":"class-1", "1":"class-2"},
                "type":"groundtruth/object-detection",
                "human-annotated":"yes",
                "creation-date":"2021-08-09T16:00:18.327433",
                "job-name":"labeling-job/<job_name>"
            }
        }
    """
    try:
        with file_path.open("r") as f:
            # Read lines and strip whitespaces
            lines = [line.strip() for line in f]

            # Validate overall structure (assuming one entry per line)
            parsed_data = []
            for line in lines:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON format in line: {line}. Error: {e}")
                # Append to parsed data after individual entry validation
                parsed_data.append(data)

            return parsed_data

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")


def _has_selected_labels(
    i_dict: Dict, job_name: str, selected_labels: List[int]
) -> bool:
    """
    Checks if the given dictionary contains any annotations with the specified class labels.

    Args:
        i_dict: The dictionary containing image data and annotations.
        job_name: The job name used to access annotations within the dictionary.
        selected_labels: A list of class labels to check for in the annotations.

    Returns:
        True if any annotation has a class label in class_labels, False otherwise.
    """
    # Iterate through each annotation in the specified job
    for annotation in i_dict[job_name]["annotations"]:
        # Check if the class_id of the annotation is in the list of selected labels
        if annotation["class_id"] in selected_labels:
            return True  # Return True if a matching class_id is found
    return False  # Return False if no matching class_id is found


def _organize_annotations_by_label(
    i_dict: Dict, job_name: str, selected_labels: List
) -> List[List[Dict]]:
    """
    Organizes annotations by the specified class labels.

    Args:
        i_dict: The dictionary containing image data and annotations.
        job_name: The job name used to access annotations within the dictionary.
        selected_labels: A list of class labels to organize annotations by.

    Returns:
        A list of lists, where each sublist contains annotations for a specific class label.
    """
    # Create a dictionary to hold lists of annotations for each label
    label_to_annotations: Dict[int, List] = {label: [] for label in selected_labels}

    # Iterate through each annotation and add it to the corresponding list in the dictionary
    for annotation in i_dict[job_name]["annotations"]:
        class_id = annotation["class_id"]
        if class_id in label_to_annotations:
            label_to_annotations[class_id].append(annotation)

    return list(label_to_annotations.values())


def _extract_all_category_data(
    categories, i_index: int, i_dict: Dict, job_name: str
) -> DefaultDict:
    """
    Processes category data from an image dictionary and updates the categories dictionary.

    Args:
        categories: Dictionary of category information.
        i_index: Index of the current image entry.
        i_dict: Dictionary containing image metadata.
        job_name: Job name associated with the image.

    Returns:
        Updated dictionary containing category information
    """

    try:
        # Access category data using the job name and metadata key
        category = i_dict[job_name + "-metadata"]["class-map"]

        # Process each category entry
        for c_id, c_name in category.items():
            if c_id not in categories.keys():
                # Add new category to the dictionary
                categories[c_id] = {"id": int(c_id), "name": c_name}

        return categories

    except KeyError:
        raise KeyError(f"Missing 'class-map' key in image id - {i_index + 1}")


def _process_images_and_annotations(
    groundtruth_manifest: List[Dict], job_name: str, selected_labels: List
) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Processes image data from the provided dictionary and updates the image list.

    Args:
        groundtruth_manifest: List of dictionaries containing image data and annotations.
        job_name: Job name associated with the image.
        selected_labels: A list of class labels to filter and organize annotations by.

    Returns:
        A tuple containing two lists:
        - Updated list of image metadata dictionaries.
        - List of annotation dictionaries.
        - Dictionary of all labels
    """

    images: List[Dict] = []
    annotations: List[Dict] = []
    categories: DefaultDict[str, List] = defaultdict()
    counter_img_id = 0

    # Process each entry in the manifest data
    for i_index, i_dict in enumerate(groundtruth_manifest):
        # Extract image width and height
        width, height = _extract_image_size_from_image_dict(i_dict, job_name)

        # Extract filename from image URL with optional prefix
        try:
            file_name = _extract_filename_from_s3_path(i_dict["source-ref"])
        except KeyError:
            raise KeyError(f"Missing 'source-ref' key in object - {i_index}")

        # Skip images that do not have the selected labels
        if not _has_selected_labels(i_dict, job_name, selected_labels):
            continue

        # Organize annotations by the specified class labels
        ann_super_list = _organize_annotations_by_label(
            i_dict, job_name, selected_labels
        )

        # Process annotations
        counter_img_id += 1
        for counter_class, parent_ann in enumerate(ann_super_list):
            for child_ann in parent_ann:
                annotation_dict = {
                    "id": len(annotations) + 1,
                    "image_id": counter_img_id,
                    "category_id": counter_class,
                    "segmentation": [],
                    "bbox": [
                        child_ann["left"],
                        child_ann["top"],
                        child_ann["width"],
                        child_ann["height"],
                    ],
                    "ignore": 0,
                    "iscrowd": 0,
                    "area": child_ann["width"] * child_ann["height"],
                }
                annotations.append(annotation_dict)

        # Add the image metadata to the list
        images.append(
            {
                "width": width,
                "height": height,
                "id": len(images) + 1,
                "file_name": file_name,
            }
        )

        # Process categories information
        categories = _extract_all_category_data(categories, i_index, i_dict, job_name)

    return images, annotations, categories


def _process_category_data(
    categories: OrderedDict, selected_labels: list[int]
) -> list[Dict]:
    """
    Processes category data from the provided dictionary and returns a list of categories.

    Args:
        categories: Dictionary of category information.
        selected_labels: A list of class labels to filter categories by.

    Returns:
        A list of dictionaries, each representing a category with an ID and name.
    """
    # Initialize the list to store selected category dictionaries
    selected_categories = []
    # Initialize the category class counter
    cat_class_counter = 0

    # Iterate through the category items
    for c_id, c_name in categories.items():
        # Check if the class ID is in the list of selected class labels
        if int(c_name["id"]) in selected_labels:
            # Create a dictionary for the category with an ID and name
            category_dict = {
                "id": cat_class_counter,
                "name": c_name["name"],
            }
            cat_class_counter += 1
            # Add the category dictionary to the list
            selected_categories.append(category_dict)

    return selected_categories


def _extract_filename_from_s3_path(s3_path: str, filename_prefix: str = "") -> str:
    """
    Extracts the filename from a provided S3 path string.

    Args:
        s3_path: The S3 path string (e.g., "s3://bucket/folder/image.jpg").
        filename_prefix: Optional prefix to prepend to the extracted filename (default "").

    Returns:
        The extracted filename with the optional prefix.
    """
    if not s3_path:
        raise ValueError("S3 path cannot be empty")

    # Split the S3 path on the delimiter "/" and extract the last element (filename)
    filename = os.path.basename(s3_path)

    if not filename:
        raise ValueError("S3 path must contain a filename")
    # Combine the optional prefix with the extracted filename
    return filename_prefix + filename


def _extract_image_size_from_image_dict(
    image_dict: dict, job_name: str
) -> Tuple[int, int]:
    """
    Extracts image width and height from the image dictionary based on the job name.

    Args:
        image_dict: The dictionary containing image metadata.
        job_name: The job name associated with the image size information.

    Returns:
        A tuple containing the extracted image width and height.
    """

    # Check if the job name exists in the image dictionary
    if job_name not in image_dict:
        raise KeyError(f"Job name '{job_name}' not found in image dictionary")

    # Get the image size information associated with the job name
    image_size = image_dict[job_name].get("image_size")

    # Check if the 'image_size' key is present
    if not image_size:
        raise KeyError(f"Missing image_size key for job '{job_name}'")

    # Validate the format of the image size information
    if not all(key in image_size[0] for key in ("width", "height")):
        raise KeyError(
            f"Missing 'width' or 'height' in image_size key for job '{job_name}'"
        )
    # Extract and return the image width and height
    return image_size[0]["width"], image_size[0]["height"]


def _extract_date_as_str(
    datetime_str: str,
    input_datetime_format: str = "%Y-%m-%dT%H:%M:%S.%f",
    output_date_format: str = "%Y-%m-%d",
) -> str:
    """Extracts the date as a string from the given date string.

    Args:
        datetime_str: A string containing the date string.
        input_datetime_format: The format of the input date string (default: "%Y-%m-%dT%H:%M:%S.%f").
        output_date_format: The desired format for the output date string (default: "%Y-%m-%d").

    Returns:
        The date string in the specified output format, or raises ValueError for parsing errors.
    """

    try:
        # Parse the date string using the input format
        date = datetime.strptime(datetime_str, input_datetime_format)

        # Convert the parsed date to a string in the output format
        date_str = date.strftime(output_date_format)

        return date_str

    except ValueError as e:
        raise ValueError(f"Error parsing datetime string: {e}")


def _load_and_preprocess_image(
    root: Path, img_path: str | None, transform=None
) -> torch.Tensor:
    """
    This function loads an image, opens it, and applies transformations if provided.

    Args:
        root: Root directory containing the images.
        img_path: Image path (relative or absolute) to the root directory.
        transform: Optional transformations to apply to the image (default: None).

    Returns:
        The preprocessed image as a PyTorch tensor.
    """
    # Combine root and image path to form the full image path
    if img_path is not None:
        full_img_path = root / img_path
        # Attempt to open the image using Pillow's Image.open
        try:
            img = Image.open(full_img_path)
            # convert image to tensor
            img = transforms.ToTensor()(img)
        except OSError as e:
            # Log a warning if the image file is not found
            warnings.warn(f"{e}. Skipping...")
            return None  # Return None to indicate missing image
    # apply transformations
    if transform is not None:
        try:
            img = transform(img)
        except TypeError as e:
            # Handle any errors that might occur during transformation
            warnings.warn(f"Error applying transformation: {e}")
            return None
    return img


def _manifest_to_pascalvoc_annotations(
    i_dict: dict, job_name: str, img_id: int = 1
) -> Dict:
    """
    Processes annotation data from the provided image dictionary to Pascal VOC format for a given image ID.

    Args:
        i_dict: Dictionary containing image metadata.
        job_name: Job name associated with the image.
        img_id: Image ID. Defaults to 1.

    Returns:
        A dictionary containing processed annotations in Pascal VOC format:
          - boxes (torch.float32): Bounding boxes of detected objects.
          - labels (torch.int64): Class labels of detected objects.
          - image_id (torch.int64): Image ID associated with the annotations.
          - area (torch.float32): Area of each bounding box.
          - iscrowd (torch.int64): Crowd instance flag (0 for not crowd).
    """

    # Check if the required keys are present
    if job_name not in i_dict:
        raise KeyError(f"{job_name} not found in {i_dict}")
    if "annotations" not in i_dict[job_name]:
        raise KeyError(f"Annotations not found in image dict under: {job_name}")

    boxes, labels, areas = [], [], []
    # process annotations of each image
    for annt in i_dict[job_name]["annotations"]:
        # Extract individual values from each annotation
        # height & width are used to calculate xmax & ymax co-ordinates in addition to area
        class_id, ymin, xmin, h, w = annt.values()

        # Calculate xmax and ymax
        xmax = xmin + w
        ymax = ymin + h

        # Append values
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(class_id)
        areas.append(w * h)

    # Create Pascal VOC annotation dictionary containing processed annotations
    pascalvoc_annotation = {
        "boxes": torch.as_tensor(boxes, dtype=torch.float32),
        "labels": torch.as_tensor(labels, dtype=torch.int64),
        "image_id": torch.tensor([img_id]),
        "area": torch.as_tensor(areas, dtype=torch.float32),
        "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
    }

    return pascalvoc_annotation


def _get_image_path(
    coco_instances: COCO, img_id: int, relative_path: bool = True
) -> Optional[str]:
    """
    This function retrieves the image path based on the image ID and a flag for relative path.

    Args:
        coco_instances: COCO API object.
        img_id: Image ID.
        relative_path: Boolean flag indicating whether to return a relative path (True) or absolute path (False).

    Returns:
        The image path (either relative or absolute) as a string.
    """
    try:
        # Extract file name associated with the image ID
        img_path = coco_instances.loadImgs(img_id)[0]["file_name"]
        # If relative path is requested, extract only the filename from the full path
        if relative_path:
            img_path = Path(img_path).name
        return img_path
    except AttributeError:
        # Handle the case where the image is not found in the COCO annotations
        warnings.warn(f"Image with ID {img_id} not found in COCO annotations.")
        return None


def _coco_to_pascalvoc_annotations(coco_instances: COCO, img_id: int) -> dict:
    """
    This function processes COCO annotations to Pascal VOC format for a given image ID.

    Args:
        coco_instances: COCO API object.
        img_id: Image ID.

    Returns:
        A dictionary containing processed annotations in Pascal VOC format:
          - boxes (torch.float32): Bounding boxes of detected objects.
          - labels (torch.int64): Class labels of detected objects.
          - image_id (torch.int64): Image ID associated with the annotations.
          - area (torch.float32): Area of each bounding box.
          - iscrowd (torch.int64): Crowd instance flag (0 for not crowd).
    """
    # Get annotation IDs for the given image ID
    ann_ids = coco_instances.getAnnIds(imgIds=img_id)
    # Load COCO annotations for the retrieved IDs
    coco_annotation = coco_instances.loadAnns(ann_ids)

    # Initialize empty lists to store processed data
    boxes = []
    labels = []
    areas = []

    # Process each annotation
    for annt in coco_annotation:
        xmin, ymin, w, h = annt["bbox"]
        xmax = xmin + w
        ymax = ymin + h
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(annt["category_id"])
        areas.append(annt["area"])

    # Create the final dictionary containing processed annotations
    pascalvoc_annotation = {
        "boxes": torch.as_tensor(boxes, dtype=torch.float32),
        "labels": torch.as_tensor(labels, dtype=torch.int64),
        "image_id": torch.tensor([img_id]),
        "area": torch.as_tensor(areas, dtype=torch.float32),
        "iscrowd": torch.zeros((len(coco_annotation),), dtype=torch.int64),
    }
    return pascalvoc_annotation


def resize_coco_annotations(
    input_coco_file_path: str,
    output_coco_file_path: str,
    new_width: int,
    new_height: int,
) -> None:
    """
    Resize the annotations in a COCO dataset to new image dimensions.

    Args:
        input_coco_file_path: Path to the input JSON file containing COCO data.
        output_coco_file_path: Path to save the resized COCO data.
        new_width: Desired width for the images.
        new_height: Desired height for the images.
    """
    # Load the COCO data from the provided JSON file
    with open(input_coco_file_path, "r") as f:
        coco_data = json.load(f)

    # Iterate through each image in the COCO data
    for image in coco_data["images"]:
        orig_width, orig_height = image["width"], image["height"]
        # Update the image dimensions to the new width and height
        image["width"], image["height"] = new_width, new_height
        image["file_name"] = os.path.basename(image["file_name"])

        # Calculate the scaling factors for width and height
        x_scale = new_width / orig_width
        y_scale = new_height / orig_height

        # Iterate through each annotation in the COCO data
        for annotation in coco_data["annotations"]:
            # Check if the annotation corresponds to the current image
            if annotation["image_id"] == image["id"]:
                # Update the bounding box coordinates based on the scaling factors
                bbox = annotation["bbox"]
                resized_width = bbox[2] * x_scale
                resized_height = bbox[3] * y_scale
                annotation["bbox"] = [
                    bbox[0] * x_scale,
                    bbox[1] * y_scale,
                    resized_width,
                    resized_height,
                ]
                # cal area
                annotation["area"] = resized_width * resized_height

    # Save the resized COCO data back to a JSON file
    with open(output_coco_file_path, "w") as f:
        json.dump(coco_data, f)


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
