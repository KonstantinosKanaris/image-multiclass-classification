"""
Contains functionality for creating PyTorch DataLoader's for
image classification data.
"""

from typing import Callable, List, Optional, Tuple

import torchvision.transforms
from torch.utils.data import DataLoader
from torchvision import datasets


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    batch_size: int = 1,
    num_workers: int = 0,
    transform: Optional[Callable[[], torchvision.transforms.Compose]] = None,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Creates training and testing DataLoaders.

    Takes in a training directory and testing directory path and
    turns them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
      train_dir (str):
        Path to training directory.
      test_dir (str):
        Path to testing directory.
      transform: (callable, optional):
        Torchvision transforms to perform on training and testing data.
        Defaults to None.
      batch_size (int, optional):
        How many samples per batch to load (default: ``1``).
      num_workers (int, optional):
        How many subprocesses to use for data loading. ``0`` means that the
        data will be loaded in the main process.(default: ``0``)

    Returns:
      Tuple[DataLoader, DataLoader, List[str]]:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class names is a list of the target classes.

    Example::

      train_dataloader, test_dataloader, class_names = create_dataloaders(
          train_dir=path/to/train_dir,
          test_dir=path/to/test_dir,
          transform=some_transform,
          batch_size=32,
          num_workers=4
      )
    """
    # Use PyTorch ImageFolder to create datasets
    train_data = datasets.ImageFolder(
        root=str(train_dir),
        transform=transform,
    )
    test_data = datasets.ImageFolder(root=str(test_dir), transform=transform)

    # Turn datasets into DataLoaders
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Get class names
    class_names = train_data.classes

    return train_dataloader, test_dataloader, class_names
