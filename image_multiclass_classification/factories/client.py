from typing import Iterator

import torch
import torchvision

from image_multiclass_classification import logger
from image_multiclass_classification.factories.factories import (
    ModelsFactory,
    OptimizersFactory,
    TransformsFactory,
)
from image_multiclass_classification.models import efficientnet
from image_multiclass_classification.models.tinyvgg import TinyVGG
from image_multiclass_classification.models.vit import ViT
from image_multiclass_classification.transforms.custom_transforms import (
    create_efficient_net_b0_transforms,
    create_efficient_net_b2_transforms,
    create_tinyvgg_transforms,
    create_vit_transforms,
)
from image_multiclass_classification.utils import (
    constants,
    custom_exceptions,
    error_messages,
)


class Client:
    """
    A client for obtaining neural network models, optimizers, and data
    transformations.

    This class provides methods to obtain neural network models, optimizers,
    and data transformations based on specified names. It internally uses
    factories to manage the creation and retrieval of these components.

    Attributes:
        models_factory (ModelsFactory):
            A factory for registering and obtaining neural network models.
        optimizers_factory (OptimizersFactory):
            A factory for registering and obtaining optimizers.
        transforms_factory (TransformsFactory):
            A factory for registering and obtaining data transformations.
    """

    def __init__(self) -> None:
        self.models_factory: ModelsFactory = ModelsFactory()
        self.optimizers_factory: OptimizersFactory = OptimizersFactory()
        self.transforms_factory: TransformsFactory = TransformsFactory()

    def models_client(self, model_name: str, num_classes: int) -> torch.nn.Module:
        """
        Returns a neural network model based on the provided model name.

        This function serves as a client to obtain different pre-defined
        neural network models with specific numbers of output classes.
        The model_name argument determines which specific model will be
        instantiated.

        Supported model names:
            - 'tiny_vgg': A tiny VGG-style convolutional neural network.
            - 'efficient_net_b0': An EfficientNet-B0 model.
            - 'efficient_net_b2': An EfficientNet-B2 model.

        Args:
            model_name (str): The name of the model to be instantiated.
            num_classes (int): The number of output classes for the model.

        Returns:
            torch.nn.Module: An instance of the specified neural network model.

        Raises:
            UnsupportedModelNameError:
                If the specified model name is not supported.

        Examples:
            >>> from image_multiclass_classification.factories import client
            >>> client = client.Client()
            >>> model_instance = client.models_client(
            ...    name='efficient_net_b0', num_classes=1000
            ... )
            >>> print(model_instance)
            EfficientNet(
              ...
              (classifier): Sequential(
                (dropout): Dropout(p=0.2, inplace=False)
                (fc): Linear(in_features=1280, out_features=1000, bias=True)
              )
            )
        """
        match model_name.lower():
            case constants.TINY_VGG_MODEL_NAME:
                self.models_factory.register_model(
                    name=model_name.lower(), model=TinyVGG
                )
            case constants.EFFICIENT_NET_B0_MODEL_NAME:
                self.models_factory.register_model(
                    name=model_name.lower(),
                    model=efficientnet.efficient_net_b0,
                )
            case constants.EFFICIENT_NET_B2_MODEL_NAME:
                self.models_factory.register_model(
                    name=model_name.lower(),
                    model=efficientnet.efficient_net_b2,
                )
            case constants.VISION_TRANSFORMER_MODEL_NAME:
                self.models_factory.register_model(name=model_name.lower(), model=ViT)
            case _:
                logger.error(f"{error_messages.unsupported_model_name} `{model_name}`.")
                raise custom_exceptions.UnsupportedModelNameError(
                    f"{error_messages.unsupported_model_name} `{model_name}`."
                )

        model = self.models_factory.get_model(
            name=model_name.lower(), num_classes=num_classes
        )
        return model

    def optimizers_client(
        self,
        optimizer_name: str,
        model_params: Iterator[torch.nn.parameter.Parameter],
        learning_rate: float,
    ) -> torch.optim.Optimizer:
        """
        Returns an optimizer based on the provided optimizer name.

        This function serves as a client to obtain different pre-defined
        optimizers for training neural network models. The optimizer_name
        argument determines which specific optimizer will be instantiated.

        Supported optimizer names:
            - 'sgd': Stochastic Gradient Descent (SGD) optimizer.
            - 'adam': Adam optimizer.

        Args:
            optimizer_name (str): The name of the optimizer to be instantiated.
            model_params (Iterator[torch.nn.parameter.Parameter]):
                An iterator of model parameters.
            learning_rate (float): The learning rate for the optimizer.

        Returns:
            torch.optim.Optimizer: An instance of the specified optimizer.

        Raises:
            UnsupportedOptimizerNameError:
                If the specified optimizer name is not supported.

        Example:
            >>> # Initialize a simple PyTorch model
            >>> import torch
            >>> from torch import nn
            >>> class SimpleModel(nn.Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.fc = nn.Linear(10, 2)
            ...
            ...     def forward(self, x):
            ...         return self.fc(x)
            >>> simple_model = SimpleModel()
            >>>
            >>> # Get optimizer
            >>> from image_multiclass_classification.factories import client
            >>> client = client.Client()
            >>> optimizer_instance = client.optimizers_client(
            ...     name='sgd', model_params=simple_model.parameters(),
            ...     learning_rate=0.01
            ... )
            >>> print(optimizer_instance)
            SGD (
            Parameter Group 0
                lr: 0.01
                momentum: 0
                dampening: 0
                weight_decay: 0
                nesterov: False
            )
        """
        match optimizer_name.lower():
            case constants.SGD_OPTIMIZER_NAME:
                self.optimizers_factory.register_optimizer(
                    name=optimizer_name.lower(), optimizer=torch.optim.SGD
                )
            case constants.ADAM_OPTIMIZER_NAME:
                self.optimizers_factory.register_optimizer(
                    name=optimizer_name.lower(), optimizer=torch.optim.Adam
                )
            case _:
                logger.error(
                    f"{error_messages.unsupported_optimizer_name}"
                    f"`{optimizer_name}`."
                )
                raise custom_exceptions.UnsupportedOptimizerNameError(
                    f"{error_messages.unsupported_optimizer_name}"
                    f"`{optimizer_name}`."
                )
        optimizer = self.optimizers_factory.get_optimizer(
            name=optimizer_name.lower(),
            model_params=model_params,
            learning_rate=learning_rate,
        )
        return optimizer

    def transforms_client(self, model_name: str) -> torchvision.transforms.Compose:
        """
        Creates data transformations based on the provided `model_name`.

        Args:
            model_name (str):
                The name of the model to choose transformations for.

        Returns:
            torchvision.transforms.Compose:
                The model specific data transformations.

        Raises:
            UnsupportedModelNameError: If `model_name` is not supported.

        Example:
            >>> from image_multiclass_classification.factories import client
            >>> client = client.Client()
            >>>
            >>> # Obtain transformations for the EfficientNet-B0 model
            >>> transformations = client.transforms_client(
            ...     model_name='efficient_net_b0'
            ... )
            >>> print(transformations)
            Compose(
                ...
            )
        """
        match model_name.lower():
            case constants.TINY_VGG_MODEL_NAME:
                self.transforms_factory.register_transforms(
                    name=model_name.lower(),
                    transforms=create_tinyvgg_transforms,
                )
            case constants.EFFICIENT_NET_B0_MODEL_NAME:
                self.transforms_factory.register_transforms(
                    name=model_name.lower(),
                    transforms=create_efficient_net_b0_transforms,
                )
            case constants.EFFICIENT_NET_B2_MODEL_NAME:
                self.transforms_factory.register_transforms(
                    name=model_name.lower(),
                    transforms=create_efficient_net_b2_transforms,
                )
            case constants.VISION_TRANSFORMER_MODEL_NAME:
                self.transforms_factory.register_transforms(
                    name=model_name.lower(), transforms=create_vit_transforms
                )
            case _:
                logger.error(f"{error_messages.unsupported_model_name}`{model_name}`.")
                raise custom_exceptions.UnsupportedModelNameError(
                    f"{error_messages.unsupported_model_name}`{model_name}`."
                )

        transforms = self.transforms_factory.get_transforms(name=model_name.lower())

        return transforms
