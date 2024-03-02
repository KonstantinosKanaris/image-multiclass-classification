"""
Contains various utility functions for PyTorch model training and saving.
"""

from __future__ import annotations

import errno
import os
import time
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import torch
import yaml
from torch.utils.tensorboard.writer import SummaryWriter

from image_multiclass_classification import logger


def create_writer(
    experiment_name: str,
    model_name: str,
    start_dir: str = "runs",
    extra: Optional[str] = None,
) -> torch.utils.tensorboard.writer.SummaryWriter:
    """
    Creates a `torch.utils.tensorboard.writer.SummaryWriter()`
    instance for saving experiments to a specific `log_dir`.

    All experiments on certain day live in the same folder.

    `log_dir` is combination of
    `<start_dir>/<timestamp>/<experiment_name>/<model_name>/<extra>`
    where timestamp is the current date in `YYYY-MM-DD` format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        start_dir (str): The first directory where all the
            experiments will be saved. Defaults to `runs`.
        extra (str, optional):
            Anything extra to add to the directory.
            Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter():
            Instance of a writer saving to `log_dir`.

    Example:

        >>> # Create a writer to
        >>> # "runs/2022-06-05/data_10_percent/effnetb2/5_epochs/":
        >>> writer = create_writer(
        ...    start_dir="runs",
        ...    experiment_name="data_10_percent",
        ...    model_name="effnetb2",
        ...    extra="5_epochs"
        ... )
        >>> print(writer)
        <torch.utils.tensorboard.writer.SummaryWriter at 0x7c107effea10>
    """
    timestamp = datetime.now().strftime("%Y-%m-%d")

    log_dir = os.path.join(start_dir, timestamp, experiment_name, model_name)
    if extra:
        log_dir = os.path.join(log_dir, extra)

    logger.info(f"Created SummaryWriter, saving to {log_dir}\n")
    return SummaryWriter(log_dir=log_dir)


def load_yaml_file(filepath: str) -> Any:
    """Loads a `yaml` configuration file into a dictionary.

    Args:
        filepath (str): The path to the `yaml` file.

    Returns:
        Any:
            The configuration parameters.
    """
    if not os.path.isfile(path=filepath):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filepath)

    with open(file=filepath, mode="r", encoding="utf-8") as f:
        config = yaml.safe_load(stream=f)
    logger.info("Configuration file loaded successfully.")

    return config


def set_seeds(seed: int = 42) -> None:
    """Sets random seeds for torch operations.

    Args:
      seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)

    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)


class Timer:
    """Context manager to count elapsed time.

    Example::

      with Timer() as t:
          y = f(x)
      print(f"Invocation of f took {t.elapsed}s!")
    """

    def __enter__(self) -> Timer:
        """
        Starts the time counting.

        Returns:
          Timer: An instance of the `Timer` class.
        """
        self._start = time.time()
        return self

    def __exit__(self, *args: int | str) -> None:
        """
        Stops the time counting.

        Args:
          args (int | str)
        """
        self._end = time.time()
        self._elapsed = self._end - self._start
        self.elapsed = str(timedelta(seconds=self._elapsed))


class EarlyStopping:
    """Implements early stopping during training.

    Args:
        patience (int, optional):
            Number of epochs to wait before early stopping.
            (default=5).
        delta (float, optional):
            Minimum change in monitored quantity to qualify
            as an improvement (default=0).
        verbose (bool, optional):
            If `True`, prints a message for each improvement.
            Defaults to `False`.
        path (str, optional):
            Path to save the checkpoint. Should include either
            `.pth` or `.pt` as the file extension. Defaults to
            `'./checkpoint.pt'`.
    """

    def __init__(
        self,
        patience: int = 5,
        delta: float = 0,
        verbose: bool = False,
        path: str = "./checkpoint.pt",
    ) -> None:
        assert os.path.basename(path).endswith(
            (".pth", ".pt")
        ), "model_name should end with '.pt' or '.pth'"

        self.patience: int = patience
        self.delta: float = delta
        self.verbose: bool = verbose
        self.path: str = path

        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False
        self.val_loss_min = np.Inf

    def __call__(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        val_loss: float,
    ) -> None:
        """
        Call method to check if the model's performance has improved.

        Args:
            epoch (int): Current epoch.
            model (torch.nn.Module):
               Model to be saved if the performance improves.
            optimizer (torch.optim.Optimizer): Optimizer used for training.
            val_loss (float): Validation loss to be monitored.
        """
        score = -val_loss

        if not self.best_score:
            self.best_score = score
            self.save_general_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                val_loss=val_loss,
            )
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_general_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                val_loss=val_loss,
            )
            self.counter = 0

    def save_general_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        val_loss: float,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """
        Saves a general checkpoint during training.

        In addition to the model's `state_dict`, a general checkpoint
        also includes the optimizer's `state_dict`, the current epoch,
        and the validation loss value.

        Args:
            epoch (int): Current epoch.
            model (torch.nn.Module): Model to be saved.
            val_loss (float):
                Validation loss at the time of saving the checkpoint.
            optimizer (torch.optim.Optimizer): Optimizer used for training.
        """
        if not os.path.isdir(s=os.path.dirname(self.path)):
            os.makedirs(name=os.path.dirname(self.path), exist_ok=True)

        if self.verbose:
            logger.info(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> "
                f"{val_loss:.6f}). Saving model to {self.path}..."
            )

        torch.save(
            obj={
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            f=self.path,
        )
        self.val_loss_min = val_loss
