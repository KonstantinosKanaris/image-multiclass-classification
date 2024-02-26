import os
from typing import Any, Dict

import torch
import torch.utils.tensorboard
from torch import nn

from image_multiclass_classification import logger
from image_multiclass_classification.data_setup import create_dataloaders
from image_multiclass_classification.engine import trainer
from image_multiclass_classification.factories.client import Client
from image_multiclass_classification.utils.aux import Timer, create_writer


class ExperimentManager:
    """A class to manage and run experiments with PyTorch models
    on custom data.

    This class provides methods to run multiple experiments specified
    in a configuration file.Each experiment consists of parameters such
    as the experiment name, directories for tracking and saving models,
    data paths for training and testing, and hyperparameters for training.

    Args:
        config (Dict[str, Any]):
            A dictionary containing experiment configurations.
        resume_from_checkpoint (bool):
            If 'True' the selected model will resume training
            from the last selected checkpoint. Defaults to `False.

    Attributes:
        config (Dict[str, Any]):
            The experiment configurations loaded from the
            configuration file.
        resume_from_checkpoint (bool):
            If 'True' the selected model will resume training
            from the last selected checkpoint. Defaults to `False.
        device (str):
            The device (`cuda` or `cpu`) used for training,
            determined based on availability.

    Example:

        To use this class, instantiate an `ExperimentManager` object with
        the experiment configurations, and then call the `run_experiments()`
        method to execute the experiments.

        >>> experiment_config = {
        ...     'experiments': [
        ...         {
        ...             'name': 'experiment1',
        ...             'tracking_dir': './runs',
        ...             'data': {
        ...                 'paths': {
        ...                     'train_dir': './data/train',
        ...                     'test_dir': './data/test'
        ...                 }
        ...             },
        ...             'hyperparameters': {
        ...                 'model_name': 'ResNet',
        ...                 'batch_size': 32,
        ...                 'learning_rate': 0.001,
        ...                 'num_epochs': 10
        ...             }
        ...         },
        ...         ...
        ...     ]
        ... }
        >>> experiment_manager = ExperimentManager(config=experiment_config)
        >>> experiment_manager.run_experiments()
    """

    def __init__(
        self, config: Dict[str, Any], resume_from_checkpoint: bool = False
    ) -> None:
        self.config: Dict[str, Any] = config
        self.resume_from_checkpoint: bool = resume_from_checkpoint
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        self.client: Client = Client()

    def run_experiments(self) -> None:
        """Runs multiple experiments with PyTorch models on custom data.

        This method iterates over each experiment specified in the
        configuration file and runs it using the `run_experiment()`
        method.
        """
        for i, experiment in enumerate(self.config["experiments"]):
            logger.info(f"Experiment {i+1}")
            self.run_experiment(experiment=experiment)

    def run_experiment(self, experiment: Dict[str, Any]) -> None:
        """Runs a training experiment with a PyTorch model on custom data.

        This method performs an end-to-end training experiment using the
        provided experiment parameters.It includes the following steps:
            1. Data preparation: Creates data transforms and data loaders
               for training and testing data.
            2. Model setup: Initializes the model architecture.
            3. Loss and optimizer setup: Defines the loss function and
               optimizer.
            4. Experiment tracking: Sets up experiment tracking using
               `TensorBoard`.
            5. Model training: Trains the model for the specified number
               of epochs.
            6. Logging: Logs the training duration.

        Args:
            experiment (Dict[str, Any]): A dictionary containing experiment
            parameters including:
                - 'name' (str): Name of the experiment.
                - 'tracking_dir' (str): Directory for experiment tracking.
                - 'data' (Dict[str, Any]): Dictionary containing data paths.
                    - 'train_dir' (str): Directory path for training data.
                    - 'test_dir' (str): Directory path for testing data.
                - 'hyperparameters' (Dict[str, Any]): Dictionary containing
                  hyperparameters for the experiment.
                    - 'model_name' (str): Name of the model architecture.
                    - 'batch_size' (int): Batch size for training.
                    - 'learning_rate' (float): Learning rate for optimization.
                    - 'num_epochs' (int): Number of epochs for training.

        Example:
            To run an experiment, provide the experiment details in
            the following format:

            >>> experiment_example = {
            ...     'name': '<name_of_the_experiment>',
            ...     'tracking_dir': '<dir_to_save_experiments>',
            ...     'data': {
            ...         'train_dir': '<path_to_train_data_dir>',
            ...         'test_dir': '<path_to_test_data_dir>'
            ...     },
            ...     'hyperparameters': {
            ...         'model_name': '<model_to_train>',
            ...         'batch_size': 32,
            ...         'learning_rate': 0.001,
            ...         'num_epochs': 1
            ...     }
            ... }
        """
        data_transforms = self.client.transforms_client(
            model_name=experiment["hyperparameters"]["model_name"]
        )

        train_dataloader, test_dataloader, class_names = create_dataloaders(
            train_dir=experiment["data"]["train_dir"],
            test_dir=experiment["data"]["test_dir"],
            batch_size=experiment["hyperparameters"]["batch_size"],
            transform=data_transforms,
        )

        model = self.client.models_client(
            model_name=experiment["hyperparameters"]["model_name"],
            num_classes=len(class_names),
        )
        loss_fn = nn.CrossEntropyLoss()
        optimizer = self.client.optimizers_client(
            optimizer_name=experiment["hyperparameters"]["optimizer_name"],
            model_params=model.parameters(),
            learning_rate=experiment["hyperparameters"]["learning_rate"],
        )
        writer = create_writer(
            start_dir=experiment["tracking_dir"],
            experiment_name=experiment["name"],
            model_name=experiment["hyperparameters"]["model_name"],
        )

        model_name = (
            f"{experiment['hyperparameters']['model_name']}_"
            f"{experiment['name']}_checkpoint.pth"
        )
        checkpoint_path = os.path.join(
            experiment["models"]["checkpoints_dir"], model_name
        )
        # Train the model
        with Timer() as t:
            trainer.train(
                model=model.to(self.device),
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=experiment["hyperparameters"]["num_epochs"],
                device=self.device,
                writer=writer,
                checkpoint_path=checkpoint_path,
                resume=self.resume_from_checkpoint,
            )
        logger.info(f"Training took {t.elapsed} seconds.")