import torchvision


def create_tinyvgg_transforms() -> torchvision.transforms.Compose:
    """Returns the transformations required for training
    data with the `TinyVGG` model.
    """
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=(64, 64)),
            torchvision.transforms.ToTensor(),
        ]
    )


def create_vit_transforms() -> torchvision.transforms.Compose:
    """Returns the transformations for training data with
    the `ViT` model."""
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.ToTensor(),
        ]
    )


def create_efficient_net_b0_transforms() -> torchvision.transforms.Compose:
    """Returns the transformations required for training
    data with the `EfficientNet_B0` model."""
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    return weights.transforms()


def create_efficient_net_b2_transforms() -> torchvision.transforms.Compose:
    """Returns the transformations required for training
    data with the `EfficientNet_B2` model."""
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    return weights.transforms()
