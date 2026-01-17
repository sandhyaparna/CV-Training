import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_V2_Weights
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms

import os
import numpy as np
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO
# import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2

# In my case, just added ToTensor
def get_transform(rotate=False, color_jitter=False):
    custom_transforms = []
    custom_transforms.append(transforms.ToTensor())
    transform = transforms.Compose([
        transforms.Resize((720,1280)),
        # transforms.RandomRotation(degrees=20),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        # add mean and std transformation
        transforms.ToTensor()
    ])
    return torchvision.transforms.Compose(custom_transforms)

class ResizeTransformv2:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, bboxes, orig_w, orig_h):
        # Resize image
        # Calculate new dimensions (lower half)
        new_h = orig_h * 0.6
        new_w = orig_w
        new_size = (orig_w, new_h)
        
        # Resize the image
        img = img.crop((0, new_h, orig_w, orig_h))
        transform = transforms.Compose([#transforms.RandomHorizontalFlip(0.5), 
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = transform(img)
        # Resize bboxes
        #new_w, new_h = self.size
        w_ratio = new_w / orig_w
        h_ratio = new_h / orig_h
        
        # bboxes[:, [0, 2]] *= w_ratio
        # bboxes[:, [1, 3]] *= h_ratio
        
        # Transform bounding boxes
        new_boxes = []
        for box in bboxes:
            x_min, y_min, x_max, y_max = box
            
            # Adjust y-coordinates based on the crop
            if y_min >= new_h:  # Box is fully in the lower half
                new_boxes.append([x_min, y_min - new_h, x_max, y_max - new_h])
            elif y_max >= new_h:  # Box is partially in the lower half
                new_boxes.append([x_min, 0, x_max, y_max - new_h])

        bboxes = torch.FloatTensor(new_boxes)
        len_bbox = [len(i) for i in bboxes]
        #print(len_bbox)
        return img, bboxes
    

class RandomRotateWithBBox:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, image, bboxes):
        angle = np.random.uniform(-self.degrees, self.degrees)
        image = transforms.functional.rotate(image, angle)
        
        # Rotate bounding boxes
        w, h = image.size
        cx, cy = w / 2, h / 2
        new_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            corners = corners - [cx, cy]
            corners = np.dot(corners, self.rotation_matrix(angle))
            corners = corners + [cx, cy]
            new_x1, new_y1 = corners.min(axis=0)
            new_x2, new_y2 = corners.max(axis=0)
            new_bboxes.append([new_x1, new_y1, new_x2, new_y2])
        
        return image, new_bboxes

    def rotation_matrix(self, angle):
        angle = np.deg2rad(angle)
        return np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])


class ResizeTransform:
    
    def __init__(self, size):
        self.size = size

    def __call__(self, img, bboxes, orig_w, orig_h):
        # Resize image
        img = img.resize(self.size, Image.BILINEAR)
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        img = transform(img)
        # Resize bboxes
        new_w, new_h = self.size
        w_ratio = new_w / orig_w
        h_ratio = new_h / orig_h
        bboxes[:, [0, 2]] *= w_ratio
        bboxes[:, [1, 3]] *= h_ratio

        return img, bboxes


class ResizeColorTransform:
    
    def __init__(self, size):
        self.size = size

    def __call__(self, img, bboxes, orig_w, orig_h):
        # print("original bboxes:::", bboxes)
        # Resize image
        img = img.resize(self.size, Image.BILINEAR)
        new_w, new_h = self.size
        w_ratio = new_w / orig_w
        h_ratio = new_h / orig_h
        
        # Resize bounding boxes
        bboxes[:, [0, 2]] *= w_ratio
        bboxes[:, [1, 3]] *= h_ratio
        
        # Perform transformations
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
        ])
        
        # Convert to PIL Image for consistency with transformations
        img = transform(img)
        
        return img, bboxes


class ResizeHorzTransform:
    
    def __init__(self, size):
        self.size = size

    def __call__(self, img, bboxes, orig_w, orig_h):
        # print("original bboxes:::", bboxes)
        # Resize image
        img = img.resize(self.size, Image.BILINEAR)
        new_w, new_h = self.size
        w_ratio = new_w / orig_w
        h_ratio = new_h / orig_h
        
        # Resize bounding boxes
        bboxes[:, [0, 2]] *= w_ratio
        bboxes[:, [1, 3]] *= h_ratio
        
        # Perform transformations
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor(),
        ])

        # Convert to PIL Image for consistency with transformations
        img = transform(img)

        # if isinstance(img, torch.Tensor):
        #     img = transforms.ToPILImage()(img)

        # Flip bounding boxes horizontally
        w = self.size[0]
        bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]  # Flip bbox x-coordinates
        # print("resized bboxes:::", bboxes)
        
        return img, bboxes


class ResizeHorzColorTransform:
    
    def __init__(self, size):
        self.size = size

    def __call__(self, img, bboxes, orig_w, orig_h):
        # print("original bboxes:::", bboxes)
        # Resize image
        img = img.resize(self.size, Image.BILINEAR)
        new_w, new_h = self.size
        w_ratio = new_w / orig_w
        h_ratio = new_h / orig_h
        
        # Resize bounding boxes
        bboxes[:, [0, 2]] *= w_ratio
        bboxes[:, [1, 3]] *= h_ratio
        
        # Perform transformations
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
        ])
        
        # Convert to PIL Image for consistency with transformations
        img = transform(img)

        # Flip bounding boxes horizontally
        w = self.size[0]
        bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]  # Flip bbox x-coordinates
        # print("resized bboxes:::", bboxes)
        
        return img, bboxes


class ResizeRotateTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, bboxes, orig_w, orig_h):
        # Resize image
        img = img.resize(self.size, Image.BILINEAR)
        angle = np.random.uniform(-20, 20)
        img = transforms.functional.rotate(img, angle)
        w, h = img.size

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        img = transform(img)
        # Resize bboxes
        new_w, new_h = self.size
        w_ratio = new_w / orig_w
        h_ratio = new_h / orig_h
        bboxes[:, [0, 2]] *= w_ratio
        bboxes[:, [1, 3]] *= h_ratio

        # Rotate bounding boxes
        cx, cy = w / 2, h / 2
        new_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            corners = corners - [cx, cy]
            corners = np.dot(corners, self.rotation_matrix(angle))
            corners = corners + [cx, cy]
            new_x1, new_y1 = corners.min(axis=0)
            new_x2, new_y2 = corners.max(axis=0)
            new_bboxes.append([new_x1, new_y1, new_x2, new_y2])

        return img, new_bboxes

    def rotation_matrix(self, angle):
        angle = np.deg2rad(angle)
        return np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])


class ResizeRotateColorTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, bboxes, orig_w, orig_h):
        # Resize image
        img = img.resize(self.size, Image.BILINEAR)
        angle = np.random.uniform(-20, 20)
        img = transforms.functional.rotate(img, angle)
        w, h = img.size

        transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
        ])
        img = transform(img)
        # Resize bboxes
        new_w, new_h = self.size
        w_ratio = new_w / orig_w
        h_ratio = new_h / orig_h
        bboxes[:, [0, 2]] *= w_ratio
        bboxes[:, [1, 3]] *= h_ratio

        # Rotate bounding boxes
        cx, cy = w / 2, h / 2
        new_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            corners = corners - [cx, cy]
            corners = np.dot(corners, self.rotation_matrix(angle))
            corners = corners + [cx, cy]
            new_x1, new_y1 = corners.min(axis=0)
            new_x2, new_y2 = corners.max(axis=0)
            new_bboxes.append([new_x1, new_y1, new_x2, new_y2])

        return img, new_bboxes

    def rotation_matrix(self, angle):
        angle = np.deg2rad(angle)
        return np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])


# ------------------------------------------------------------
# Resize + Horizontal Flip + Color
# ------------------------------------------------------------
class ResizeHorzColorTransform(ResizeHorzTransform):
    def __call__(self, img, bboxes, orig_w, orig_h):
        img, bboxes = super().__call__(img, bboxes, orig_w, orig_h)
        img = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
        )(img)
        return img, bboxes


class CustomDataset(torch.utils.data.Dataset):
    """
    Args:
        root (str): Path to image directory.
        annotation (str): Path to COCO annotation file.
        transforms (callable, optional): Transform function.
    """
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))
        # Get image size dynamically
        width, height = img.size
        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(coco_annotation[i]['category_id'])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Tensorise img_id - comment this out to avoid the error
        img_id = torch.tensor([img_id])

        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            # disable this when resizing
            #img = self.transforms(img)
            # disable comments for image resizing
            img, bboxes = self.transforms(img, boxes, width, height))
            my_annotation['boxes'] = torch.Tensor(bboxes)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)

        

# class CustomAugDataset(torch.utils.data.Dataset):
#     def __init__(self, root, annotation, transforms=None):
#         self.root = root
#         self.transforms = transforms
#         self.coco = COCO(annotation)
#         self.ids = list(sorted(self.coco.imgs.keys()))

#     def __getitem__(self, index):
#         # Own coco file
#         coco = self.coco
#         # Image ID
#         img_id = self.ids[index]
#         # List: get annotation id from coco
#         ann_ids = coco.getAnnIds(imgIds=img_id)
#         # Dictionary: target coco_annotation file for an image
#         coco_annotation = coco.loadAnns(ann_ids)
#         # path for input image
#         path = coco.loadImgs(img_id)[0]['file_name']
#         # open the input image
#         img = Image.open(os.path.join(self.root, path))
#         # number of objects in the image
#         num_objs = len(coco_annotation)

#         # Bounding boxes for objects
#         # In coco format, bbox = [xmin, ymin, width, height]
#         # In pytorch, the input should be [xmin, ymin, xmax, ymax]
#         boxes = []
#         labels = []
#         for i in range(num_objs):
#             xmin = coco_annotation[i]['bbox'][0]
#             ymin = coco_annotation[i]['bbox'][1]
#             xmax = xmin + coco_annotation[i]['bbox'][2]
#             ymax = ymin + coco_annotation[i]['bbox'][3]
#             boxes.append([xmin, ymin, xmax, ymax])
#             labels.append(coco_annotation[i]['category_id'])
#         boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         labels = torch.as_tensor(labels, dtype=torch.int64)

#         # Tensorise img_id - comment this out to avoid the error
#         img_id = torch.tensor([img_id])

#         # Size of bbox (Rectangular)
#         areas = []
#         for i in range(num_objs):
#             areas.append(coco_annotation[i]['area'])
#         areas = torch.as_tensor(areas, dtype=torch.float32)
#         # Iscrowd
#         iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

#         # Annotation is in dictionary format
#         my_annotation = {}
#         my_annotation["boxes"] = boxes
#         my_annotation["labels"] = labels
#         my_annotation["image_id"] = img_id
#         my_annotation["area"] = areas
#         my_annotation["iscrowd"] = iscrowd

#         if self.transforms is not None:
#             # disable this when resizing
#             #img = self.transforms(img)
#             # disable comments for image resizing
#             img, bboxes = self.transforms(img, np.array(boxes), np.array(labels))
#             my_annotation['boxes'] = torch.Tensor(bboxes)

#         return img, my_annotation

#     def __len__(self):
#         return len(self.ids)
