import torchvision
from torch import nn

from image_multiclass_classification.utils.aux import set_seeds


def efficient_net_b0(num_classes: int) -> torchvision.models.EfficientNet:
    """
    Returns an instance of the `EfficientNet_B0` PyTorch model with its
    pretrained weights.

    All the base layers have been frozen and only the classifier
    layer is trainable.

    Args:
        num_classes (int):
            The number of output classes for the classifier head.

    Returns:
        torchvision.models.EfficientNet
            An instance of the `EfficientNet_BO` model.
    """
    # Initialize EfficientNet with pretrained weights
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights)

    # Freeze the feature layers of EfficientNet architecture
    for param in model.features.parameters():
        param.requires_grad = False

    set_seeds(seed=42)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=num_classes),
    )
    return model


def efficient_net_b2(num_classes: int) -> torchvision.models.EfficientNet:
    """
    Returns an instance of the `EfficientNet_B2` PyTorch model with its
    pretrained weights.

    All the base layers have been frozen and only the classifier
    layer is trainable.

    Args:
        num_classes (int):
            The number of output classes for the classifier head.

    Returns:
        torchvision.models.EfficientNet
            An instance of the `EfficientNet_B2` model.
    """
    # Initialize EfficientNet with pretrained weights
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights=weights)

    # Freeze the feature layers of EfficientNet architecture
    for param in model.features.parameters():
        param.requires_grad = False

    set_seeds(seed=42)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=num_classes),
    )
    return model
