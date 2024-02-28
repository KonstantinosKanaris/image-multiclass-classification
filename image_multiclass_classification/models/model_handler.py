import os
from typing import Any, Dict

import torch

from image_multiclass_classification import logger


def load_general_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, filepath: str
) -> Dict[str, Any]:
    """
    Loads a general checkpoint.

    Loads the saved `state_dict` for the model and optimizer, as well
    as the values of the last epoch and loss from the most recent
    general checkpoint created during training.

    Args:
        model (torch.nn.Module):
            The model to be updated with its saved `state_dict`.
        optimizer (torch.optim.Optimizer):
            The optimizer to be updated with its saved `state_dict`.
        filepath (str): The file path of the general checkpoint.

    Returns:
        A dictionary containing the following keys:
            - 'model': The updated model with its saved `state_dict`.
            - 'optimizer': The updated optimizer with its saved `state_dict`.
            - 'epoch': The epoch value from the last checkpoint.
            - 'loss': The loss value from the last checkpoint.
    """
    checkpoint = torch.load(f=filepath)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return {
        "model": model,
        "optimizer": optimizer,
        "epoch": checkpoint["epoch"],
        "val_loss": checkpoint["loss"],
    }


def save_model(
    model: torch.nn.Module,
    model_path: str,
) -> None:
    """Saves a PyTorch model to a target filepath.

    Args:
      model (torch.nn.Module): A target PyTorch model to save.
      model_path (str): A filepath to save the model. Should include
        either `.pth` or `.pt` as the file extension.

    Example::

      save_model(
        model=model,
        model_path=model_path
      )
    """
    # Create a target directory
    if not os.path.isdir(s=os.path.dirname(model_path)):
        os.makedirs(name=os.path.dirname(model_path), exist_ok=True)

    # Create model save path
    assert os.path.basename(model_path).endswith(
        (".pth", ".pt")
    ), "model_name should end with '.pt' or '.pth'"

    # Save the model state_dict()
    logger.info(f"Saving model to: {model_path}")
    torch.save(obj=model.state_dict(), f=model_path)


def load_model(
    model: torch.nn.Module,
    model_path: str,
) -> torch.nn.Module:
    """
    Loads a trained PyTorch model.

    A fresh instance of the saved model is given as input in order
    to be updated with the `state_dict()` of the saved model.

    Args:
        model (torch.nn.Module): A fresh instance of the same model to be
            updated with the `state_dict()` of the saved model.
        model_path (str): The path of the trained PyTorch model.

    Returns:
        torch.nn.Module:
            The stored PyTorch model.
    """
    logger.info(
        f"Loading `{model.__class__.__name__}`'s trained weights "
        f"from {model_path}..."
    )
    model.load_state_dict(state_dict=torch.load(f=model_path))

    return model
