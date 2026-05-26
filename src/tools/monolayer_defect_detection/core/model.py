"""
PyTorch semantic segmentation models.

This file provides:
- ResNet encoder
- UNet
- UNet++
- Model factory with an interface similar to segmentation_models_pytorch

Supported models:
- unet
- unet++
- unetplusplus
- nested_unet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights


# =============================================================================
# Basic building blocks
# =============================================================================

class ConvBnRelu(nn.Module):
    """Convolution + BatchNorm + ReLU block."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """UNet decoder block: upsampling, skip fusion, and two convolutions."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.conv1 = ConvBnRelu(in_channels + skip_channels, out_channels)
        self.conv2 = ConvBnRelu(out_channels, out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )

        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )

            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)

        return x


# =============================================================================
# ResNet encoder
# =============================================================================

class ResNetEncoder(nn.Module):
    """
    ResNet feature encoder.

    Output feature levels:
    - features[0]: 1/2 resolution
    - features[1]: 1/4 resolution
    - features[2]: 1/8 resolution
    - features[3]: 1/16 resolution
    - features[4]: 1/32 resolution
    """

    CHANNELS = {
        "resnet18": [64, 64, 128, 256, 512],
        "resnet34": [64, 64, 128, 256, 512],
        "resnet50": [64, 256, 512, 1024, 2048],
        "resnet101": [64, 256, 512, 1024, 2048],
        "resnet152": [64, 256, 512, 1024, 2048],
    }

    def __init__(self, name="resnet34", in_channels=3, pretrained=True):
        super().__init__()

        self.name = name.lower()
        self.out_channels = self.CHANNELS.get(self.name, self.CHANNELS["resnet34"])

        resnet = self._load_resnet(self.name, pretrained)

        self.conv1 = self._make_first_conv(
            resnet=resnet,
            in_channels=in_channels,
            pretrained=pretrained,
        )

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def _load_resnet(self, name, pretrained):
        """Load a ResNet backbone. If pretrained weights fail, use random initialization."""

        def safe_load(model_fn, weights_class):
            if pretrained:
                try:
                    return model_fn(weights=weights_class.IMAGENET1K_V1)
                except Exception as exc:
                    print(f"Warning: failed to load pretrained weights: {exc}")
                    print("Using randomly initialized weights instead.")

            return model_fn(weights=None)

        if name == "resnet18":
            return safe_load(models.resnet18, ResNet18_Weights)

        if name == "resnet34":
            return safe_load(models.resnet34, ResNet34_Weights)

        if name == "resnet50":
            return safe_load(models.resnet50, ResNet50_Weights)

        self.name = "resnet34"
        self.out_channels = self.CHANNELS["resnet34"]
        return safe_load(models.resnet34, ResNet34_Weights)

    @staticmethod
    def _make_first_conv(resnet, in_channels, pretrained):
        """Adapt the first convolution layer to the required input channel number."""
        if in_channels == 3:
            return resnet.conv1

        conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        if pretrained:
            with torch.no_grad():
                pretrained_weight = resnet.conv1.weight.data

                if in_channels == 1:
                    conv1.weight.data = pretrained_weight.mean(dim=1, keepdim=True)
                elif in_channels < 3:
                    conv1.weight.data = pretrained_weight[:, :in_channels, :, :]
                else:
                    repeat_count = int(torch.ceil(torch.tensor(in_channels / 3)).item())
                    expanded_weight = pretrained_weight.repeat(1, repeat_count, 1, 1)
                    conv1.weight.data = expanded_weight[:, :in_channels, :, :] / repeat_count

        return conv1

    def forward(self, x):
        features = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)

        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)

        x = self.layer2(x)
        features.append(x)

        x = self.layer3(x)
        features.append(x)

        x = self.layer4(x)
        features.append(x)

        return features


# =============================================================================
# UNet
# =============================================================================

class Unet(nn.Module):
    """
    UNet segmentation model.

    Architecture:
    ResNet encoder + decoder with skip connections.
    """

    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
        decoder_channels=(256, 128, 64, 32, 16),
    ):
        super().__init__()

        pretrained = encoder_weights is not None and encoder_weights.lower() == "imagenet"

        self.encoder = ResNetEncoder(
            name=encoder_name,
            in_channels=in_channels,
            pretrained=pretrained,
        )

        encoder_channels = self.encoder.out_channels

        self.decoder_blocks = nn.ModuleList()

        in_ch = encoder_channels[-1]

        for i, out_ch in enumerate(decoder_channels):
            skip_ch = encoder_channels[-(i + 2)] if i < len(encoder_channels) - 1 else 0

            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                )
            )

            in_ch = out_ch

        self.segmentation_head = nn.Conv2d(
            decoder_channels[-1],
            classes,
            kernel_size=1,
        )

    def forward(self, x):
        input_size = x.shape[2:]

        features = self.encoder(x)
        x = features[-1]

        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = features[-(i + 2)] if i < len(features) - 1 else None
            x = decoder_block(x, skip)

        x = self.segmentation_head(x)

        if x.shape[2:] != input_size:
            x = F.interpolate(
                x,
                size=input_size,
                mode="bilinear",
                align_corners=True,
            )

        return x


# =============================================================================
# UNet++
# =============================================================================

class UnetPlusPlus(nn.Module):
    """
    UNet++ segmentation model.

    This implementation uses a ResNet encoder and nested decoder connections.
    """

    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
        deep_supervision=False,
    ):
        super().__init__()

        self.classes = classes
        self.deep_supervision = deep_supervision

        pretrained = encoder_weights is not None and encoder_weights.lower() == "imagenet"

        self.encoder = ResNetEncoder(
            name=encoder_name,
            in_channels=in_channels,
            pretrained=pretrained,
        )

        nb_filter = self.encoder.out_channels

        self.conv0_1 = self._make_decoder_block(nb_filter[1] + nb_filter[0], nb_filter[0])
        self.conv1_1 = self._make_decoder_block(nb_filter[2] + nb_filter[1], nb_filter[1])
        self.conv2_1 = self._make_decoder_block(nb_filter[3] + nb_filter[2], nb_filter[2])
        self.conv3_1 = self._make_decoder_block(nb_filter[4] + nb_filter[3], nb_filter[3])

        self.conv0_2 = self._make_decoder_block(nb_filter[1] + nb_filter[0] * 2, nb_filter[0])
        self.conv1_2 = self._make_decoder_block(nb_filter[2] + nb_filter[1] * 2, nb_filter[1])
        self.conv2_2 = self._make_decoder_block(nb_filter[3] + nb_filter[2] * 2, nb_filter[2])

        self.conv0_3 = self._make_decoder_block(nb_filter[1] + nb_filter[0] * 3, nb_filter[0])
        self.conv1_3 = self._make_decoder_block(nb_filter[2] + nb_filter[1] * 3, nb_filter[1])

        self.conv0_4 = self._make_decoder_block(nb_filter[1] + nb_filter[0] * 4, nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], classes, kernel_size=1)

    @staticmethod
    def _make_decoder_block(in_channels, out_channels):
        """Create a two-layer decoder block."""
        return nn.Sequential(
            ConvBnRelu(in_channels, out_channels),
            ConvBnRelu(out_channels, out_channels),
        )

    @staticmethod
    def _upsample_cat(x, *skips):
        """Upsample one tensor and concatenate it with skip tensors."""
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )

        tensors = [x]

        for skip in skips:
            if skip.shape[2:] != x.shape[2:]:
                skip = F.interpolate(
                    skip,
                    size=x.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )

            tensors.append(skip)

        return torch.cat(tensors, dim=1)

    def forward(self, x):
        input_size = x.shape[2:]

        x0_0, x1_0, x2_0, x3_0, x4_0 = self.encoder(x)

        x0_1 = self.conv0_1(self._upsample_cat(x1_0, x0_0))
        x1_1 = self.conv1_1(self._upsample_cat(x2_0, x1_0))
        x2_1 = self.conv2_1(self._upsample_cat(x3_0, x2_0))
        x3_1 = self.conv3_1(self._upsample_cat(x4_0, x3_0))

        x0_2 = self.conv0_2(self._upsample_cat(x1_1, x0_0, x0_1))
        x1_2 = self.conv1_2(self._upsample_cat(x2_1, x1_0, x1_1))
        x2_2 = self.conv2_2(self._upsample_cat(x3_1, x2_0, x2_1))

        x0_3 = self.conv0_3(self._upsample_cat(x1_2, x0_0, x0_1, x0_2))
        x1_3 = self.conv1_3(self._upsample_cat(x2_2, x1_0, x1_1, x1_2))

        x0_4 = self.conv0_4(self._upsample_cat(x1_3, x0_0, x0_1, x0_2, x0_3))

        if self.deep_supervision:
            outputs = [
                self.final1(x0_1),
                self.final2(x0_2),
                self.final3(x0_3),
                self.final4(x0_4),
            ]

            return [
                F.interpolate(
                    output,
                    size=input_size,
                    mode="bilinear",
                    align_corners=True,
                )
                for output in outputs
            ]

        output = self.final(x0_4)

        if output.shape[2:] != input_size:
            output = F.interpolate(
                output,
                size=input_size,
                mode="bilinear",
                align_corners=True,
            )

        return output


# =============================================================================
# Model factory
# =============================================================================

class SMPModelFactory:
    """
    Model factory with an interface similar to segmentation_models_pytorch.
    """

    def __init__(
        self,
        encoder="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
    ):
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.classes = classes

    def get_model(self, model_name: str):
        """Create a segmentation model by name."""
        model_name = model_name.lower()

        if model_name == "unet":
            return Unet(
                encoder_name=self.encoder,
                encoder_weights=self.encoder_weights,
                in_channels=self.in_channels,
                classes=self.classes,
            )

        if model_name in ["unet++", "unetplusplus", "nested_unet"]:
            return UnetPlusPlus(
                encoder_name=self.encoder,
                encoder_weights=self.encoder_weights,
                in_channels=self.in_channels,
                classes=self.classes,
            )

        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Supported models: unet, unet++, unetplusplus, nested_unet."
        )