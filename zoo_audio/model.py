"""Model architectures for animal sound classification."""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class AnimalSoundCNN(nn.Module):
    """4-block CNN for classifying mel spectrograms of animal sounds.

    Architecture:
        4 convolutional blocks: Conv2d(3x3) -> BatchNorm -> ReLU -> MaxPool(2x2) -> Dropout2d
        Channel progression: 1 -> 32 -> 64 -> 128 -> 256
        AdaptiveAvgPool2d to flatten spatial dimensions
        FC head: Linear(256, 128) -> ReLU -> Dropout -> Linear(128, num_classes)

    Returns raw logits (no softmax).
    """

    def __init__(self, num_classes: int, dropout: float = 0.2, fc_dropout: float = 0.3):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            # Block 1: 1 -> 32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
            # Block 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input shape: (batch, 1, n_mels, T). Output: (batch, num_classes)."""
        x = self.conv_blocks(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class AnimalSoundResNet(nn.Module):
    """ResNet18-based classifier for mel spectrograms of animal sounds.

    Wraps torchvision's ResNet18. The first conv layer is adapted from 3
    channels to 1 channel by averaging the pretrained weights across the
    channel dimension.

    Same contract as AnimalSoundCNN: input (batch, 1, n_mels, T) -> (batch, num_classes).
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()

        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = tv_models.resnet18(weights=weights)

        # Adapt first conv from 3 channels to 1 channel
        old_conv = self.resnet.conv1
        new_conv = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        if pretrained:
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        self.resnet.conv1 = new_conv

        # Replace final FC layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


# ---------------------------------------------------------------------------
# Factory + architecture-agnostic utilities
# ---------------------------------------------------------------------------

def create_model(model_type: str, num_classes: int, **kwargs) -> nn.Module:
    """Create a model by type string.

    Args:
        model_type: ``"cnn"`` or ``"resnet18"``.
        num_classes: Number of output classes.
        **kwargs: Forwarded to the chosen constructor (e.g. dropout, pretrained).
    """
    if model_type == "cnn":
        return AnimalSoundCNN(
            num_classes=num_classes,
            dropout=kwargs.get("dropout", 0.2),
            fc_dropout=kwargs.get("fc_dropout", 0.3),
        )
    if model_type == "resnet18":
        return AnimalSoundResNet(
            num_classes=num_classes,
            pretrained=kwargs.get("pretrained", True),
        )
    raise ValueError(f"Unknown model_type: {model_type!r}")


def get_final_fc_layer(model: nn.Module) -> nn.Linear:
    """Return the final classification Linear layer, regardless of architecture."""
    if isinstance(model, AnimalSoundCNN):
        return model.fc[3]
    if isinstance(model, AnimalSoundResNet):
        return model.resnet.fc
    raise TypeError(f"Unsupported model type: {type(model)}")


def set_final_fc_layer(model: nn.Module, layer: nn.Linear) -> None:
    """Replace the final classification Linear layer, regardless of architecture."""
    if isinstance(model, AnimalSoundCNN):
        model.fc[3] = layer
    elif isinstance(model, AnimalSoundResNet):
        model.resnet.fc = layer
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")


def freeze_backbone(model: nn.Module, freeze_conv_blocks: int) -> None:
    """Freeze the first *freeze_conv_blocks* convolutional blocks/groups.

    - CNN: each block is 5 sequential layers in ``model.conv_blocks``.
    - ResNet: block 0 = conv1+bn1; blocks 1-4 = layer1..layer4.
    """
    if isinstance(model, AnimalSoundCNN):
        layers_per_block = 5
        freeze_until = freeze_conv_blocks * layers_per_block
        for i, layer in enumerate(model.conv_blocks):
            if i < freeze_until:
                for param in layer.parameters():
                    param.requires_grad = False
    elif isinstance(model, AnimalSoundResNet):
        resnet = model.resnet
        layer_groups = [
            [resnet.conv1, resnet.bn1],  # block 0
            [resnet.layer1],              # block 1
            [resnet.layer2],              # block 2
            [resnet.layer3],              # block 3
            [resnet.layer4],              # block 4
        ]
        for idx in range(min(freeze_conv_blocks, len(layer_groups))):
            for module in layer_groups[idx]:
                for param in module.parameters():
                    param.requires_grad = False
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")
