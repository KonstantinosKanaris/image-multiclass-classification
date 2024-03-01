from typing import Any, Callable, Dict, Iterator

import torch
import torchvision


class ModelsFactory:
    """
    A factory class for registering and instantiating PyTorch models.

    Attributes:
        _models (Dict[str, Any]):
            A dictionary to store registered models.
    """

    def __init__(self) -> None:
        self._models: Dict[str, Any] = {}

    def register_model(self, name: str, model: Any) -> None:
        """
        Registers a PyTorch model with the given name.

        Args:
            name (str): The name of the model to register.
            model (Any):
                The PyTorch model class or function.
        """
        self._models[name] = model

    def get_model(self, name: str, num_classes: int) -> Any:
        """
        Instantiates and returns a PyTorch model by name.

        Args:
            name (str): The name of the model to instantiate.
            num_classes (int): The number of output classes for the model.

        Returns:
            torch.nn.Module: An instance of the specified PyTorch model.
        """
        model = self._models[name]
        return model(num_classes)


class OptimizersFactory:
    """
    A factory class for registering and instantiating PyTorch optimizers.

    Attributes:
        _optimizers (Dict[str, Any]):
            A dictionary to store registered optimizers.
    """

    def __init__(self) -> None:
        self._optimizers: Dict[str, Any] = {}

    def register_optimizer(self, name: str, optimizer: Any) -> None:
        """
        Registers a PyTorch optimizer with the given name.

        Args:
            name (str): The name of the optimizer to register.
            optimizer (Any):
                The PyTorch optimizer class.
        """
        self._optimizers[name] = optimizer

    def get_optimizer(
        self,
        name: str,
        model_params: Iterator[torch.nn.parameter.Parameter],
        hyperparameters: Dict[str, Any],
    ) -> Any:
        """Instantiates and returns a PyTorch optimizer by name.

        Args:
            name (str): The name of the optimizer to instantiate.
            model_params (Iterator[torch.nn.parameter.Parameter]):
                Iterator of the model parameters.
            hyperparameters (Dict[str, Any]):
                Hyperparameters to be passed to the optimizer.

        Returns:
            torch.optim.Optimizer:
                An instance of the specified PyTorch optimizer.
        """
        optimizer = self._optimizers[name]
        return optimizer(model_params, **hyperparameters)


class TransformsFactory:
    """
    A factory class for registering and obtaining torchvision transforms.

    Attributes:
        _transforms (Dict[str, Callable[[], torchvision.transforms.Compose]]):
            A dictionary to store registered transform functions.
    """

    def __init__(self) -> None:
        self._transforms: Dict[str, Callable[[], torchvision.transforms.Compose]] = {}

    def register_transforms(
        self, name: str, transforms: Callable[[], torchvision.transforms.Compose]
    ) -> None:
        """
        Registers a transform function with the given name.

        Args:
            name (str): The name of the transform to register.
            transforms (Callable[[], torchvision.transforms.Compose]):
                The transform function that applies multiple transforms.
        """
        self._transforms[name] = transforms

    def get_transforms(
        self,
        name: str,
    ) -> torchvision.transforms.Compose:
        """
        Retrieves a set of torchvision transforms by name.

        Args:
            name (str): The name of the transforms to retrieve.

        Returns:
            torchvision.transforms.Compose:
                An instance of the specified torchvision transforms.
        """
        transforms = self._transforms[name]
        return transforms()
