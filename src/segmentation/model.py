import segmentation_models_pytorch as smp
import torch
from torch import nn

import config


class UNet(nn.Module):
    """
    Implements a U-Net architecture for semantic segmentation tasks.

    Args:
        in_channels (int, optional): Number of input channels (default: 3 for RGB images).
        out_channels (int, optional): Number of output channels (default: 2 for segmentation masks).
        features_start (int, optional): Number of features in the first encoder block (default: 64).
    """

    def __init__(
        self, in_channels: int = 3, out_channels: int = 2, features_start: int = 64
    ):
        super(UNet, self).__init__()
        # Encoder (down-sampling path)
        self.encoder = nn.ModuleList(
            [
                self._conv_block(in_channels, features_start),
                self._conv_block(features_start, features_start * 2),
                self._conv_block(features_start * 2, features_start * 4),
                self._conv_block(features_start * 4, features_start * 8),
            ]
        )
        # Bottleneck layer
        self.bottleneck = self._conv_block(features_start * 8, features_start * 16)
        # Decoder (up-sampling path)
        self.decoder = nn.ModuleList(
            [
                self._conv_block(
                    features_start * 16 + features_start * 8, features_start * 8
                ),
                self._conv_block(
                    features_start * 8 + features_start * 4, features_start * 4
                ),
                self._conv_block(
                    features_start * 4 + features_start * 2, features_start * 2
                ),
                self._conv_block(features_start * 2 + features_start, features_start),
            ]
        )
        # Output layer
        self.output_layer = nn.Conv2d(features_start, out_channels, kernel_size=(1, 1))

    def _conv_block(self, in_features: int, out_features: int):
        """
        Defines a single convolutional block with two 3x3 convolutional layers,
        batch normalization, and ReLU activations.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.

        Returns:
            nn.Sequential: A sequential container with the defined convolutional block.
        """
        return nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_features, out_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Defines the forward pass of the U-Net model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        # Encoder path
        encoder_features = []
        for module in self.encoder:
            x = module(x)
            encoder_features.append(x)
            x = nn.functional.max_pool2d(x, 2, stride=2)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for i, module in enumerate(self.decoder):
            x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
            x = torch.cat([x, encoder_features[-(i + 1)]], dim=1)
            x = module(x)

        # Output layer
        x = self.output_layer(x)
        return x


class LoadSegmentationModel:
    """
    Loads a segmentation model based on specified parameters.

    Returns:
        torch.nn.Module: The loaded segmentation model.
    """

    def __init__(
        self,
        architecture: str = "deeplabv3",
        backbone: str = "mobilenet",
        out_channels: int = 2,
        in_channels: int = 3,
        pretrained: bool = True,
        pretrained_weights: str = "imagenet",
        device: str = "cpu",
    ):
        """
        Args:
            architecture: Architecture of the model ("unet", "deeplabv3", or "fcn").
            backbone: Backbone network.
            out_channels: Number of output channels (number of classes).
            in_channels: Number of input channels for the model. (1 for gray-scale images, 3 for RGB, etc.)
            pretrained: Whether to load pre-trained weights (default: True).
            pretrained_weights: Weights for the pre-trained model (only for U-Net).
            device: Device to load the model on ("cpu" or "mps" or "cuda").
        """
        self.architecture = architecture
        self.backbone = backbone
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.pretrained_weights = pretrained_weights
        self.device = device

        # load model
        self.model = self._load_segmentation_model()

    def _load_segmentation_model(self):
        """
        Loads the appropriate segmentation model based on the specified architecture.
        """
        # Validate architecture
        if self.architecture not in config.supported_architectures:
            raise ValueError(
                f"Unsupported architecture:'{self.architecture}'. Valid options are: {sorted(config.supported_architectures)}"
            )

        if self.architecture == "unet":
            return self._load_unet_model()
        else:
            return self._load_vision_model()

    def _load_unet_model(self):
        """Loads a pre-trained or custom unet model.

        Raises:
        ValueError:
            - If `in_channels` is not 1 (grayscale) or 3 (RGB).
            - If `out_channels` is less than 1.
            - If `backbone` is not supported.
        """
        # Validate input channels
        if self.in_channels not in [1, 3]:
            raise ValueError("in_channels must be either 1 (grayscale) or 3 (RGB).")

        # Validate output channels (should be at least 1)
        if self.out_channels < 1:
            raise ValueError("out_channels must be at least 1 (number of outputs).")

        # Create U-Net model
        if self.backbone:
            unet_model = smp.Unet(
                encoder_name=self.backbone,
                encoder_weights=self.pretrained_weights,
                in_channels=self.in_channels,
                classes=self.out_channels,
            )
        else:
            # Custom UNet
            unet_model = UNet(
                in_channels=self.in_channels, out_channels=self.out_channels
            )

        return unet_model.to(self.device)

    def _load_vision_model(self):
        """Loads a pre-trained vision model.

        This function loads a pre-trained segmentation model from the torchvision library.
        It supports two architectures: "deeplabv3" and "fcn" with various backbones
        like "resnet50", "resnet101", and "mobilenet".

        Raises:
        ValueError: If backbone is not supported.
        """
        # Validate backbone based on architecture
        if self.backbone not in config.supported_backbones[self.architecture]:
            raise ValueError(
                f"Backbone '{self.backbone}' is not supported for '{self.architecture}' architecture. Valid options are: {sorted(config.supported_backbones[self.architecture])}"
            )

        # Define model selection logic based on architecture and backbone
        try:
            model = config.model_builders[self.architecture][self.backbone](
                pretrained=self.pretrained
            )
        except (KeyError, ModuleNotFoundError) as e:
            raise ValueError(f"Failed to load model: {e}") from e

        # Modify number of classes
        if self.out_channels != model.classifier[4].out_channels:
            model.classifier[4] = nn.Conv2d(
                model.classifier[4].in_channels,
                self.out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
            )
            model.aux_classifier[4] = nn.Conv2d(
                model.aux_classifier[4].in_channels,
                self.out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
            )

        return model.to(self.device)
