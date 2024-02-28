"""
Contains a class for training and testing a PyTorch model.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.utils.tensorboard
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
        optimizer (torch.optim.Optimizer):
            The optimizer used for updating model parameters.
        epochs (int, optional):
            The number of training epochs. Defaults to 5.
        writer (torch.utils.tensorboard.writer.SummaryWriter, optional):
            Optional 'SummaryWriter' instance for logging training metrics
            to TensorBoard. Defaults to None.
        resume (bool): If True, resumes training from the specified checkpoint.
            Defaults to False.

    Example:
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
        >>> simple_cnn = SimpleCNN()
        >>>
        >>> # Setup loss function and optimizer
        >>> my_loss_function = nn.CrossEntropyLoss()
        >>> my_optimizer = torch.optim.Adam(
        ...     params=simple_cnn.parameters(), lr=0.001)
        >>>
        >>> # Create summary writer
        >>> from image_multiclass_classification.utils.aux import create_writer
        >>> summary_writer = create_writer(
        ...     experiment_name="experiment_name",
        ...     model_name="model_name"
        ... )
        >>> experiment = TrainingExperiment(
        ...     model=simple_cnn,
        ...     optimizer=my_optimizer,
        ...     loss_fn=my_loss_function,
        ...     train_dataloader=train_dataloader_example,
        ...     test_dataloader=test_dataloader_example,
        ...     epochs=10,
        ...     checkpoint_path='checkpoint.pth',
        ...     resume_training=False
        ...     writer=summary_writer,
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
        optimizer: torch.optim.Optimizer,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        checkpoint_path: str,
        epochs: int = 5,
        resume: bool = False,
        writer: Optional[torch.utils.tensorboard.writer.SummaryWriter] = None,
    ) -> None:
        self.model: torch.nn.Module = model
        self.loss_fn: torch.nn.Module = loss_fn
        self.optimizer: torch.optim.Optimizer = optimizer
        self.train_dataloader: torch.utils.data.DataLoader = train_dataloader
        self.test_dataloader: torch.utils.data.DataLoader = test_dataloader
        self.checkpoint_path: str = checkpoint_path
        self.epochs: int = epochs
        self.resume: bool = resume
        self.writer: Optional[torch.utils.tensorboard.writer.SummaryWriter] = writer

    def train(self) -> Dict[str, List[float]]:
        """Trains a PyTorch model on custom data.

        Performs the training of a PyTorch model using the provided data
        loaders, loss function, and optimizer. It also evaluates the model
        on the test data at the end of each epoch. Checkpointing is supported,
        optionally allowing for the resumption of training from a saved checkpoint.

        The training process includes early stopping to prevent over-fitting,
        where training is stopped if the validation loss does not improve for a
        certain number of epochs.

        Calculates, prints and stores evaluation metrics throughout.

        Stores metrics to specified writer `log_dir` if present. Refer
        to `image_multiclass_classification.utils.aux.create_writer`
        function for more.

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

        early_stopping = EarlyStopping(path=self.checkpoint_path, verbose=True)

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

            # Track experiments with SummaryWriter
            if self.writer:
                self.writer.add_scalars(
                    main_tag="Loss",
                    tag_scalar_dict={
                        "train_loss": train_loss,
                        "test_loss": test_loss,
                    },
                    global_step=epoch,
                )
                self.writer.add_scalars(
                    main_tag="Accuracy",
                    tag_scalar_dict={"train_acc": train_acc, "test_acc": test_acc},
                    global_step=epoch,
                )
                self.writer.close()

            early_stopping(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                val_loss=test_loss,
            )
            if early_stopping.early_stop:
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
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)

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
                test_pred_logits = self.model(X)

                # 2. Calculate and accumulate loss
                loss = self.loss_fn(test_pred_logits, y)
                test_loss += loss.item()

                # Calculate and accumulate accuracy
                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)
        # Divide total test loss and accuracy by length of dataloader
        test_loss /= len(self.test_dataloader)
        test_acc /= len(self.test_dataloader)

        return test_loss, test_acc


# def train(
#     checkpoint_path: str,
#     model: torch.nn.Module,
#     train_dataloader: torch.utils.data.DataLoader,
#     test_dataloader: torch.utils.data.DataLoader,
#     loss_fn: torch.nn.Module,
#     optimizer: torch.optim.Optimizer,
#     epochs: int = 5,
#     device: str = "cpu",
#     writer: Optional[torch.utils.tensorboard.writer.SummaryWriter] = None,
#     resume: bool = False,
# ) -> Dict[str, List[float]]:
#     """Trains a PyTorch model on custom data.
#
#     This function performs the training of a PyTorch model using the provided
#     data loaders, loss function, and optimizer. It also evaluates the model
#     on the test data at the end of each epoch. Checkpointing is supported,
#     optionally allowing for the resumption of training from a saved checkpoint.
#
#     The training process includes early stopping to prevent overfitting,
#     where training is halted if the validation loss does not improve for a
#     certain number of epochs.
#
#     Calculates, prints and stores evaluation metrics throughout.
#
#     Stores metrics to specified writer `log_dir` if present. Refer
#     to `image_multiclass_classification.utils.aux.create_writer`
#     function for more.
#
#     Args:
#         checkpoint_path (str):
#             The file path to save or load the model checkpoint.
#         model (torch.nn.Module): The PyTorch model to be trained.
#         train_dataloader (torch.utils.data.DataLoader):
#             A `DataLoader` instance for providing batches of
#             training data.
#         test_dataloader (torch.utils.data.DataLoader):
#             A `DataLoader` instance for providing batches of
#             testing data.
#         loss_fn (torch.nn.Module):
#             The loss function used for optimization.
#         optimizer (torch.optim.Optimizer):
#             The optimizer used for updating model parameters.
#         epochs (int, optional):
#             The number of training epochs. Defaults to 5.
#         device (str, optional):
#             The device used for training, either 'cpu' or 'cuda'.
#             Defaults to 'cpu'.
#         writer (torch.utils.tensorboard.writer.SummaryWriter, optional):
#             Optional 'SummaryWriter' instance for logging training metrics
#             to TensorBoard. Defaults to None.
#         resume (bool): If True, resumes training from the specified checkpoint.
#             Defaults to False.
#
#     Returns:
#         Dict[str, List[float]]:
#             A dictionary containing the training and testing metrics including
#             'train_loss', 'train_acc', 'test_loss', and 'test_acc'.
#
#     Example:
#         >>> # Generate random data for multi-class classification
#         >>> from torch.utils.data import DataLoader, TensorDataset
#         >>> train_data = torch.randn(1000, 10)
#         >>> train_labels = torch.randint(0, 3, (1000,))
#         >>> test_data = torch.randn(200, 10)
#         >>> test_labels = torch.randint(0, 3, (200,))
#         >>>
#         >>> # Create TensorDataset for train and test data
#         >>> train_dataset = TensorDataset(train_data, train_labels)
#         >>> test_dataset = TensorDataset(test_data, test_labels)
#         >>>
#         >>> # Create DataLoader for train and test data
#         >>> train_dataloader_example = DataLoader(
#         ...     train_dataset, batch_size=32, shuffle=True
#         ... )
#         >>> test_dataloader_example = DataLoader(test_dataset, batch_size=32)
#         >>>
#         >>> # Initialize a simple CNN PyTorch model
#         >>> import torch
#         >>> from torch import nn
#         >>> class SimpleCNN(nn.Module):
#         >>>     def __init__(self, num_classes):
#         ...         super(SimpleCNN, self).__init__()
#         ...         self.features = nn.Sequential(
#         ...             nn.Conv2d(
#         ...                 in_channels=3,
#         ...                 out_channels=16,
#         ...                 kernel_size=3,
#         ...                 stride=1,
#         ...                 padding=1
#         ...             ),
#         ...             nn.ReLU(),
#         ...             nn.MaxPool2d(kernel_size=2, stride=2),
#         ...             nn.Conv2d(
#         ...                 in_channels=16,
#         ...                 out_channels=32,
#         ...                 kernel_size=3,
#         ...                 stride=1,
#         ...                 padding=1
#         ...             ),
#         ...             nn.ReLU(),
#         ...             nn.MaxPool2d(kernel_size=2, stride=2)
#         ...         )
#         ...         self.classifier = nn.Sequential(
#         ...             nn.Linear(32 * 56 * 56, 512),
#         ...             nn.ReLU(),
#         ...             nn.Linear(512, num_classes)
#         ...         )
#         ...
#         ...     def forward(self, x):
#         ...         return self.fc(x)
#         >>>
#         >>> simple_cnn = SimpleCNN()
#         >>>
#         >>> # Setup loss function and optimizer
#         >>> my_loss_function = nn.CrossEntropyLoss()
#         >>> my_optimizer = torch.optim.Adam(
#         ...     params=simple_cnn.parameters(), lr=0.001)
#         >>>
#         >>> # Create summary writer
#         >>> from image_multiclass_classification.utils.aux import create_writer
#         >>> summary_writer = create_writer(
#         ...     experiment_name="experiment_name",
#         ...     model_name="model_name"
#         ... )
#         >>> training_results = train(
#         ...     checkpoint_path='checkpoint.pth',
#         ...     model=simple_cnn,
#         ...     train_dataloader=train_dataloader_example,
#         ...     test_dataloader=test_dataloader_example,
#         ...     loss_fn=my_loss_function,
#         ...     optimizer=my_optimizer,
#         ...     epochs=10,
#         ...     device='cuda',
#         ...     writer=summary_writer,
#         ...     resume_training=False
#         ... )
#         >>> print(results)  # doctest: +ELLIPSIS
#             {
#                 'train_loss': [...],
#                 'train_acc': [...],
#                 'test_loss': [...],
#                 'test_acc': [...]
#             }
#
#         Example of results dictionary for 2 epochs::
#
#                 {
#                     "train_loss": [2.0616, 1.0537],
#                     "train_acc": [0.3945, 0.4257],
#                     "test_loss": [1.2641, 1.5706],
#                     "test_acc": [0.3400, 0.3573]
#                 }
#     """
#     logger.info("-------------------------- Training --------------------------")
#     logger.info(
#         f"Training on {len(train_dataloader)} batches of "
#         f"{train_dataloader.batch_size} samples."
#     )
#     logger.info(
#         f"Evaluating on {len(test_dataloader)} batches of "
#         f"{test_dataloader.batch_size} samples."
#     )
#     logger.info(f"Training model: {model.__class__.__name__}")
#     logger.info(f"Optimizer: {optimizer.__class__.__name__}")
#     logger.info(f"Loss function: {loss_fn.__class__.__name__}")
#     logger.info("Early Stopping: Yes")
#     logger.info(f"Target device: {device}")
#     logger.info(f"Epochs: {epochs}\n")
#
#     model.to(device)
#
#     results: Dict[str, List[float]] = {
#         "train_loss": [],
#         "train_acc": [],
#         "test_loss": [],
#         "test_acc": [],
#     }
#
#     early_stopping = EarlyStopping(path=checkpoint_path, verbose=True)
#
#     start_epoch = 0
#     if resume:
#         checkpoint = load_general_checkpoint(
#             model=model, optimizer=optimizer, filepath=checkpoint_path
#         )
#         model = checkpoint["model"].to(device)
#         optimizer = checkpoint["optimizer"]
#         loss_value = checkpoint["val_loss"]
#         start_epoch = checkpoint["epoch"] + 1
#
#         logger.info(f"Resume training from general checkpoint: {checkpoint_path}.")
#         logger.info(f"Last training loss value: {loss_value:.4f}")
#         logger.info(f"Resuming from {start_epoch + 1} epoch...")
#
#     for epoch in tqdm(range(start_epoch, epochs), position=0, leave=True):
#         train_loss, train_acc = train_step(
#             model=model,
#             dataloader=train_dataloader,
#             loss_fn=loss_fn,
#             optimizer=optimizer,
#             device=device,
#         )
#         test_loss, test_acc = test_step(
#             model=model,
#             dataloader=test_dataloader,
#             loss_fn=loss_fn,
#             device=device,
#         )
#
#         # Print out what's happening
#         logger.info(
#             f"===>>> epoch: {epoch + 1} | "
#             f"train_loss: {train_loss:.4f} | "
#             f"train_acc: {train_acc:.4f} | "
#             f"test_loss: {test_loss:.4f} | "
#             f"test_acc: {test_acc:.4f}"
#         )
#
#         # Update results dictionary
#         results["train_loss"].append(train_loss)
#         results["train_acc"].append(train_acc)
#         results["test_loss"].append(test_loss)
#         results["test_acc"].append(test_acc)
#
#         # Track experiments with SummaryWriter
#         if writer:
#             writer.add_scalars(
#                 main_tag="Loss",
#                 tag_scalar_dict={
#                     "train_loss": train_loss,
#                     "test_loss": test_loss,
#                 },
#                 global_step=epoch,
#             )
#             writer.add_scalars(
#                 main_tag="Accuracy",
#                 tag_scalar_dict={"train_acc": train_acc, "test_acc": test_acc},
#                 global_step=epoch,
#             )
#             writer.close()
#
#         early_stopping(
#             epoch=epoch,
#             model=model,
#             optimizer=optimizer,
#             val_loss=test_loss,
#         )
#         if early_stopping.early_stop:
#             logger.info("Training stopped due to early stopping.")
#             break
#         else:
#             continue
#
#     return results
#
#
# def train_step(
#     model: torch.nn.Module,
#     dataloader: torch.utils.data.DataLoader,
#     loss_fn: torch.nn.Module,
#     optimizer: torch.optim.Optimizer,
#     device: str = "cpu",
# ) -> Tuple[float, float]:
#     """Trains a PyTorch model for a single epoch.
#
#     Turns a target PyTorch model to `train` mode and then
#     runs through all the required training steps (forward
#     pass, loss calculation, optimizer step).
#
#     Args:
#       model (torch.nn.Module):
#         A PyTorch model to be trained.
#       dataloader (torch.utils.data.DataLoader):
#         A `DataLoader` instance for the model to be trained on.
#       loss_fn (torch.nn.Module):
#         A PyTorch loss function to minimize.
#       optimizer (torch.optim.Optimizer):
#         A PyTorch optimizer to help minimize the loss function.
#       device (str, optional):
#         A target device to compute on, i.e., ``cuda`` or ``cpu``
#         (default: ``cpu``).
#
#     Returns:
#       Tuple[float, float]:
#         The training loss and training accuracy metrics in the form
#         (train_loss, train_accuracy). For example: (0.1112, 0.8743).
#     """
#     # Put the model in train mode
#     model.train()
#
#     # Setup train loss and train accuracy values
#     train_loss, train_acc = 0.0, 0.0
#
#     # Use prefetch_generator for iterating through data
#     # pbar = enumerate(BackgroundGenerator(dataloader))
#
#     # Loop through data loader data batches
#     for batch, (X, y) in enumerate(dataloader):
#         # Send the data to the target device
#         X, y = X.to(device), y.to(device)
#
#         # 1. Forward pass (returns logits)
#         y_pred = model(X)
#
#         # 2. Calculate and accumulate loss
#         loss = loss_fn(y_pred, y)
#         train_loss += loss.item()
#
#         # 3. Optimize zero grad
#         optimizer.zero_grad()
#
#         # 4. Loss backward
#         loss.backward()
#
#         # 5. Optimizer step
#         optimizer.step()
#
#         # Calculate and accumulate accuracy metric across all batches
#         y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
#         train_acc += (y_pred_class == y).sum().item() / len(y_pred)
#
#     # Divide total train loss and accuracy by length of dataloader
#     train_loss /= len(dataloader)
#     train_acc /= len(dataloader)
#
#     return train_loss, train_acc
#
#
# def test_step(
#     model: torch.nn.Module,
#     dataloader: torch.utils.data.DataLoader,
#     loss_fn: torch.nn.Module,
#     device: str = "cpu",
# ) -> Tuple[float, float]:
#     """Tests a PyTorch model for a single epoch.
#
#     Turns a target PyTorch model to `eval` model and then performs
#     a forward pass on a testing dataset.
#
#     Args:
#       model (torch.nn.Module): A PyTorch model to be tested.
#       dataloader (torch.utils.data.DataLoader):
#         A `DataLoader` instance for the model to be tested on.
#       loss_fn (torch.nn.Module):
#         A PyTorch loss function to calculate loss on the test data.
#       device (str, optional):
#         A target device to compute on, i.e., ``cuda`` or ``cpu``
#         (default: ``cpu``).
#
#     Returns:
#       Tuple[float, float]:
#         The testing loss and testing accuracy metrics in the form
#         (test_loss, test_accuracy). For example: (0.0223, 0.8985).
#     """
#     # Put model in eval mode
#     model.eval()
#
#     # Use prefetch_generator and tqdm for iterating through data
#     # pbar = enumerate(BackgroundGenerator(dataloader))
#
#     # Setup test loss and test accuracy values
#     test_loss, test_acc = 0.0, 0.0
#
#     # Turn on inference context manager
#     with torch.inference_mode():
#         # Loop through data loader data batches
#         for batch, (X, y) in enumerate(dataloader):
#             # Send data to the target device
#             X, y = X.to(device), y.to(device)
#
#             # 1. Forward pass
#             test_pred_logits = model(X)
#
#             # 2. Calculate and accumulate loss
#             loss = loss_fn(test_pred_logits, y)
#             test_loss += loss.item()
#
#             # Calculate and accumulate accuracy
#             test_pred_labels = test_pred_logits.argmax(dim=1)
#             test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)
#     # Divide total test loss and accuracy by length of dataloader
#     test_loss /= len(dataloader)
#     test_acc /= len(dataloader)
#
#     return test_loss, test_acc
