from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet50,
    deeplabv3_resnet101,
    fcn_resnet50,
    fcn_resnet101,
)

# Vision architectures
supported_architectures = {"deeplabv3", "fcn", "unet"}


# Backbones for Vision segmentation architectures
supported_backbones = {
    "fcn": {"resnet50", "resnet101"},
    "deeplabv3": {"resnet50", "resnet101", "mobilenet"},
}

# Models with architecture and backbone combination
model_builders = {
    "fcn": {
        "resnet50": fcn_resnet50,
        "resnet101": fcn_resnet101,
    },
    "deeplabv3": {
        "resnet50": deeplabv3_resnet50,
        "resnet101": deeplabv3_resnet101,
        "mobilenet": deeplabv3_mobilenet_v3_large,
    },
}
