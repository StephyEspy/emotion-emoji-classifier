import torch
import torch.nn as nn
import torchvision.models as models


def get_resnet18(num_classes=7, pretrained=False):
    """
    Returns a modified ResNet-18 model for FER2013.
    - Accepts 1-channel (grayscale) input instead of RGB.
    - Replaces final FC layer with output dimension = num_classes.

    Args:
        num_classes (int): number of emotion categories.
        pretrained (bool): whether to load ImageNet pretrained weights.
                           If True, first conv layer will be modified accordingly.

    Returns:
        torch.nn.Module: modified ResNet-18 model.
    """

    # Load base architecture
    model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)

    # Modify input layer: 1-channel instead of 3-channel
    original_conv = model.conv1

    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False
    )

    # If using pretrained weights, average RGB weights to initialize grayscale weights
    if pretrained:
        with torch.no_grad():
            model.conv1.weight[:] = original_conv.weight.mean(dim=1, keepdim=True)

    # Modify output layer for 7 classes instead of a 1000
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


if __name__ == "__main__":
    # Quick sanity check
    model = get_resnet18(num_classes=7, pretrained=False)
    x = torch.randn(2, 1, 48, 48)  # batch of 2 grayscale images
    out = model(x)
    print("Output shape:", out.shape)  # Expect: [2, 7]
