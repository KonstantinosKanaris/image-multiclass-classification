"""
Contains a class for training and testing a PyTorch model.
"""

from typing import Dict, List, Tuple

import mlflow
import torch
import torch.utils.data
import torchmetrics
from tqdm.auto import tqdm

from image_multiclass_classification import logger
from image_multiclass_classification.models.model_handler import load_general_checkpoint
from image_multiclass_classification.utils.aux import EarlyStopping


class TrainingExperiment:
    """
    Class to conduct a training experiment for a PyTorch model
    on custom data.

    Args:
        checkpoint_path (str):
            The file path to save or load the model checkpoint.
        model (torch.nn.Module): The PyTorch model to be trained.
        train_dataloader (torch.utils.data.DataLoader):
            A `DataLoader` instance for providing batches of
            training data.
        test_dataloader (torch.utils.data.DataLoader):
            A `DataLoader` instance for providing batches of
            testing data.
        loss_fn (torch.nn.Module):
            The loss function used for optimization.
        accuracy_fn (torchmetrics.Metric):
            The accuracy function.
        optimizer (torch.optim.Optimizer):
            The optimizer used for updating model parameters.
        epochs (int, optional):
            The number of training epochs. (default=5).
        patience (int, optional):
            Number of epochs to wait before early stopping.
            (default=5)
        delta (float, optional):
            Early stopping specific. Minimum change in monitored
            quantity to qualify as an improvement. (default=0).
        resume (bool): If True, resumes training from the specified checkpoint.
            Defaults to `False`.

    Example:
        >>> # Set device-agnostic code
        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>>
        >>> # Generate random data for multi-class classification
        >>> from torch.utils.data import DataLoader, TensorDataset
        >>> train_data = torch.randn(1000, 10)
        >>> train_labels = torch.randint(0, 3, (1000,))
        >>> test_data = torch.randn(200, 10)
        >>> test_labels = torch.randint(0, 3, (200,))
        >>>
        >>> # Create TensorDataset for train and test data
        >>> train_dataset = TensorDataset(train_data, train_labels)
        >>> test_dataset = TensorDataset(test_data, test_labels)
        >>>
        >>> # Create DataLoader for train and test data
        >>> train_dataloader_example = DataLoader(
        ...     train_dataset, batch_size=32, shuffle=True
        ... )
        >>> test_dataloader_example = DataLoader(test_dataset, batch_size=32)
        >>>
        >>> # Initialize a simple CNN PyTorch model
        >>> import torch
        >>> from torch import nn
        >>> class SimpleCNN(nn.Module):
        >>>     def __init__(self, num_classes):
        ...         super(SimpleCNN, self).__init__()
        ...         self.features = nn.Sequential(
        ...             nn.Conv2d(
        ...                 in_channels=3,
        ...                 out_channels=16,
        ...                 kernel_size=3,
        ...                 stride=1,
        ...                 padding=1
        ...             ),
        ...             nn.ReLU(),
        ...             nn.MaxPool2d(kernel_size=2, stride=2),
        ...             nn.Conv2d(
        ...                 in_channels=16,
        ...                 out_channels=32,
        ...                 kernel_size=3,
        ...                 stride=1,
        ...                 padding=1
        ...             ),
        ...             nn.ReLU(),
        ...             nn.MaxPool2d(kernel_size=2, stride=2)
        ...         )
        ...         self.classifier = nn.Sequential(
        ...             nn.Linear(32 * 56 * 56, 512),
        ...             nn.ReLU(),
        ...             nn.Linear(512, num_classes)
        ...         )
        ...
        ...     def forward(self, x):
        ...         return self.fc(x)
        >>>
        >>> simple_cnn = SimpleCNN().to(device)
        >>>
        >>> # Setup loss function and optimizer
        >>> my_loss_function = nn.CrossEntropyLoss()
        >>> my_accuracy_function = torchmetrics.Accuracy(
        ...     task="multiclass", num_classes=3
        ... ).to(device)
        >>> my_optimizer = torch.optim.Adam(
        ...     params=simple_cnn.parameters(), lr=0.001)
        >>>
        >>> experiment = TrainingExperiment(
        ...     model=simple_cnn,
        ...     optimizer=my_optimizer,
        ...     loss_fn=my_loss_function,
        ...     train_dataloader=train_dataloader_example,
        ...     test_dataloader=test_dataloader_example,
        ...     epochs=10,
        ...     patience=3,
        ...     delta=0.05,
        ...     checkpoint_path='checkpoint.pth',
        ...     resume_training=False
        ... )
        >>>
        >>> results = experiment.train()
        >>> print(results)  # doctest: +ELLIPSIS
            {
                'train_loss': [...],
                'train_acc': [...],
                'test_loss': [...],
                'test_acc': [...]
            }

        Example of results dictionary for 2 epochs::

                {
                    "train_loss": [2.0616, 1.0537],
                    "train_acc": [0.3945, 0.4257],
                    "test_loss": [1.2641, 1.5706],
                    "test_acc": [0.3400, 0.3573]
                }
    """

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        accuracy_fn: torchmetrics.Metric,
        optimizer: torch.optim.Optimizer,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        checkpoint_path: str,
        epochs: int = 5,
        patience: int = 5,
        delta: float = 0,
        resume: bool = False,
    ) -> None:
        self.model: torch.nn.Module = model
        self.loss_fn: torch.nn.Module = loss_fn
        self.accuracy_fn: torchmetrics.Metric = accuracy_fn.to(self.__class__.DEVICE)
        self.optimizer: torch.optim.Optimizer = optimizer
        self.train_dataloader: torch.utils.data.DataLoader = train_dataloader
        self.test_dataloader: torch.utils.data.DataLoader = test_dataloader
        self.checkpoint_path: str = checkpoint_path
        self.epochs: int = epochs
        self.patience: int = patience
        self.delta: float = delta
        self.resume: bool = resume
        self.early_stopping: EarlyStopping = EarlyStopping(
            patience=self.patience,
            delta=self.delta,
            path=self.checkpoint_path,
            verbose=True,
        )

    def train(self) -> Dict[str, List[float]]:
        """Trains a PyTorch model for a number of epochs.

        Performs the training of a PyTorch model using the provided data
        loaders, loss and accuracy functions, and optimizer. It also evaluates
        the model on the test data at the end of each epoch. Checkpointing is supported,
        optionally allowing for the resumption of training from a saved checkpoint.

        The training process includes early stopping to prevent over-fitting,
        where training is stopped if the validation loss does not improve for a
        certain number of epochs.

        Calculates, prints and stores evaluation metrics throughout.

        Returns:
            Dict[str, List[float]]:
                A dictionary containing the training and testing metrics including
                'train_loss', 'train_acc', 'test_loss', and 'test_acc'.
        """
        logger.info("-------------------------- Training --------------------------")
        logger.info(
            f"Training on {len(self.train_dataloader)} batches of "
            f"{self.train_dataloader.batch_size} samples."
        )
        logger.info(
            f"Evaluating on {len(self.test_dataloader)} batches of "
            f"{self.test_dataloader.batch_size} samples."
        )
        logger.info(f"Training model: {self.model.__class__.__name__}")
        logger.info(f"Optimizer: {self.optimizer.__class__.__name__}")
        logger.info(f"Loss function: {self.loss_fn.__class__.__name__}")
        logger.info("Early Stopping: Yes")
        logger.info(f"Target device: {self.__class__.DEVICE}")
        logger.info(f"Epochs: {self.epochs}\n")

        self.model.to(self.__class__.DEVICE)

        results: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
        }

        start_epoch = 0
        if self.resume:
            checkpoint = load_general_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                filepath=self.checkpoint_path,
            )
            self.model = checkpoint["model"].to(self.__class__.DEVICE)
            self.optimizer = checkpoint["optimizer"]
            loss_value = checkpoint["val_loss"]
            start_epoch = checkpoint["epoch"] + 1

            logger.info(
                f"Resume training from general checkpoint: {self.checkpoint_path}."
            )
            logger.info(f"Last training loss value: {loss_value:.4f}")
            logger.info(f"Resuming from {start_epoch + 1} epoch...")

        for epoch in tqdm(range(start_epoch, self.epochs), position=0, leave=True):
            train_loss, train_acc = self.train_step()
            test_loss, test_acc = self.test_step()

            # Print out what's happening
            logger.info(
                f"===>>> epoch: {epoch + 1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f}"
            )

            # Update results dictionary
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

            mlflow.log_metrics(
                metrics={
                    "train_loss": train_loss,
                    "val_loss": test_loss,
                    "train_acc": train_acc,
                    "val_acc": test_acc,
                },
                step=epoch,
            )

            self.early_stopping(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                val_loss=test_loss,
            )
            if self.early_stopping.early_stop:
                logger.info("Training stopped due to early stopping.")
                break
            else:
                continue

        return results

    def train_step(self) -> Tuple[float, float]:
        """Trains a PyTorch model for a single epoch.

        Turns a target PyTorch model to `train` mode and then
        runs through all the required training steps (forward
        pass, loss calculation, optimizer step).

        Returns:
          Tuple[float, float]:
            The training loss and training accuracy metrics in the form
            (train_loss, train_accuracy). For example: (0.1112, 0.8743).
        """
        # Put the model in train mode
        self.model.train()

        # Setup train loss and train accuracy values
        train_loss, train_acc = 0.0, 0.0

        # Use prefetch_generator for iterating through data
        # pbar = enumerate(BackgroundGenerator(dataloader))

        # Loop through data loader data batches
        for batch, (X, y) in enumerate(self.train_dataloader):
            # Send the data to the target device
            X, y = X.to(self.__class__.DEVICE), y.to(self.__class__.DEVICE)

            # 1. Forward pass (returns logits)
            y_pred = self.model(X)

            # 2. Calculate and accumulate loss
            loss = self.loss_fn(y_pred, y)
            train_loss += loss.item()

            # 3. Optimize zero grad
            self.optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            self.optimizer.step()

            # Calculate and accumulate accuracy metric across all batches
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += self.accuracy_fn(y_pred_class, y).item()

        # Divide total train loss and accuracy by length of dataloader
        train_loss /= len(self.train_dataloader)
        train_acc /= len(self.train_dataloader)

        return train_loss, train_acc

    def test_step(self) -> Tuple[float, float]:
        """Tests a PyTorch model for a single epoch.

        Turns a target PyTorch model to `eval` model and then performs
        a forward pass on a testing dataset.

        Returns:
          Tuple[float, float]:
            The testing loss and testing accuracy metrics in the form
            (test_loss, test_accuracy). For example: (0.0223, 0.8985).
        """
        # Put model in eval mode
        self.model.eval()

        # Use prefetch_generator and tqdm for iterating through data
        # pbar = enumerate(BackgroundGenerator(dataloader))

        # Setup test loss and test accuracy values
        test_loss, test_acc = 0.0, 0.0

        # Turn on inference context manager
        with torch.inference_mode():
            # Loop through data loader data batches
            for batch, (X, y) in enumerate(self.test_dataloader):
                # Send data to the target device
                X, y = X.to(self.__class__.DEVICE), y.to(self.__class__.DEVICE)

                # 1. Forward pass
                test_y_pred = self.model(X)

                # 2. Calculate and accumulate loss
                loss = self.loss_fn(test_y_pred, y)
                test_loss += loss.item()

                # Calculate and accumulate accuracy
                test_pred_labels = test_y_pred.argmax(dim=1)
                test_acc += self.accuracy_fn(test_pred_labels, y).item()

        # Divide total test loss and accuracy by length of dataloader
        test_loss /= len(self.test_dataloader)
        test_acc /= len(self.test_dataloader)

        return test_loss, test_acc
