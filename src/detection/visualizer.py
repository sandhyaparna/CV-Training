import torch
import torchvision 
from torchvision.utils import draw_bounding_boxes 

from torchvision.io import read_image 
import torchvision.transforms as transforms

from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from PIL import Image 
import cv2
import matplotlib.pyplot as plt


def visualize_detections(img, model, class_names):
    transform = transforms.Compose([
    transforms.ToTensor()])
    
    img_tensor = transform(img)
    
    outputs = model(img_tensor.unsqueeze(0))
    
    result = outputs[0]
    score_threshold = 0.5
    
    boxes = result['boxes'][result['scores'] > score_threshold]
    labels = result['labels'][result['scores'] > score_threshold]
    
    label_name = [class_names[cat.item()] for cat in labels]
    
    img_vision = draw_bounding_boxes(img_tensor, boxes, labels = label_name,
                        width=6, colors = "red")
    
    img_vision = torchvision.transforms.ToPILImage()(img_vision)
    
    display(img_vision)