# Multi-class Image Classification with PyTorch

## Table of Contents

* [Overview](#Overview)
* [Key Features](#Key--Features)
* [Development](#Development)
* [Data Preparation](#Data--Preparation)
* [Experiment Configuration](#Experiment--Configuration)
* [Training](#Training)
* [Experiment Tracking](#Experiment--Tracking)

## Overview 🔍

A simple but efficient framework for image multi-class classification
using PyTorch, empowering users to efficiently train and evaluate deep learning
models on custom datasets. With support for various pre-defined PyTorch models,
and the ability to easily integrate custom models, and experiment tracking capabilities,
this framework streamlines the process of conducting image classification tasks.

### Project Structure 🌲
```
image-multiclass-classification
├── .gitignore
├── .pre-commit-config.yaml                | Pre-commit hooks
├── LICENSE
├── Makefile                               | Development commands for code formatting, linting, etc.
├── Pipfile                                | Project's dependencies and their versions using the `pipenv` format
├── Pipfile.lock                           | Auto-generated file that locks the dependencies to specific versions for reproducibility.
├── checkpoints                            | Directory path to save checkpoints during training.
├── README.md
├── colours.mk                             | A Makefile fragment containing color codes for terminal output styling.
├── configs
│   └── experiments.yaml             | Experiment configuration file. Define data paths, hyperparameters for each experiemnt.
├── image_multiclass_classification        | The main Python package containing the project's source code.
│   ├── __about__.py                 | Metadata about the project, i.e., version number, author information, etc.
│   ├── __init__.py
│   ├── __main__.py                  | The entry point for running the package as a script
│   ├── core.py                      | Contains core functionalities for the image multi-class classification project.
│   ├── data_setup.py
│   ├── engine
│   │   ├── __init__.py
│   │   └── trainer.py         | Contains the training loop.
│   ├── factories
│   │   ├── __init__.py
│   │   ├── client.py          | Interacts with the factories to return different instances of models, optimizers, and transforms.
│   │   └── factories.py       | Contains factory classes for creating different types of models, optimizers, and transforms.
│   ├── logger                       | A sub-package containing logging configurations.
│   │   └── logging.ini        | Configuration file for Python's logging module.
│   ├── models                       | A sub-package containing definitions for different model architectures.
│   │   ├── __init__.py
│   │   ├── efficientnet.py
│   │   ├── model_handler.py
│   │   └── tinyvgg.py
│   ├── transforms
│   │   ├── __init__.py
│   │   └── custom_transforms.py  | Contains model-specific transformations for data preprocessing
│   └── utils
│       ├── __init__.py
│       ├── aux.py                   | Auxiliary functions and classes used across the project
│       ├── constants.py             | Defines constants
│       ├── custom_exceptions.py     | Implements custom exceptions
│       └── error_messages.py        | Defines custom error messages
└── mypy.ini                               | Configuration file for the MyPy static type checker
```


## Key Features 🔑

* **Model Variety**: Choose from a wide range of pre-defined PyTorch models, including
TinyVGG, EfficientNet (B0 and B2), for your classification tasks.
* **Customizable Models**: Easily integrate custom PyTorch models into your classification
tasks, allowing for seamless experimentation with novel architectures and configurations
* **Customizable Experiments**: Define multiple experiments easily by configuring model
architecture, optimizer, learning rate, batch size, and other hyperparameters in a YAML
configuration file
* **Experiment Tracking**: Utilize TensorBoard for real-time visualization of training
metrics and performance evaluation, enabling easy monitoring of experiment progress
* **Checkpointing**: Ensure training progress is saved with checkpointing functionality,
allowing for easy resumption of training from the last saved state
* **EarlyStopping**: Automatically stop training when the model's performance stops
improving on a validation set

## Development 🐍
Clone the repository:
  ```bash
  $ git clone https://github.com/KonstantinosKanaris/image-multiclass-classification.git
  ```

### Set up the environment

#### Create environment
Python 3.10 is required.

- Create the environment and install the dependencies:
    ```bash
    $ pipenv --python 3.10
    $ pipenv install --dev
    ```
- Enable the newly-created virtual environment, with:
    ```bash
    $ pipenv shell
    ```
## Data Preparation 📂
Bellow is an illustration of the storage format for the training and test images.
The image numbers are arbitrary. Define the train and test directories in the configuration file.

```
cat_dog_horse_dataset
├── test
│   ├── cat
│   │   ├── image01.jpg
│   │   └── image02.jpg
│   │   └── ...
│   ├── dog
│   │   ├── image45.jpg
│   │   └── image46.jpg
│   │   └── ...
│   └── horse
│       ├── image92.jpg
│       └── image93.jpg
│       └── ...
└── train
    ├── cat
    │   ├── image101.jpg
    │   └── image102.jpeg
    │   └── ...
    ├── dog
    │   ├── image154.jpg
    │   └── image155.jpg
    │   └── ...
    └── horse
        ├── image191.jpg
        └── image192.jpg
        └── ...
```

## Experiment Configuration 🧪
To conduct training experiments, define the configuration parameters for each experiment, including data paths and
hyperparameters, in the configuration (YAML) file. Here's how you can define experiments:

```yaml
experiments:
  -
    name: experiment_1
    data:
        train_dir: ./data/cat_dog_horse_dataset_20_percent/train
        test_dir: ./data/cat_dog_horse_dataset_20_percent/test
    hyperparameters:
      num_epochs: 10
      batch_size: 32
      learning_rate: 0.001
      optimizer_name: adam
      model_name: tiny_vgg
  -
    name: experiment_2
    data:
        train_dir: ./data/cat_dog_horse_dataset_40_percent/train
        test_dir: ./data/cat_dog_horse_dataset_40_percent/test
    hyperparameters:
      num_epochs: 2
      batch_size: 32
      learning_rate: 0.002
      optimizer_name: adam
      model_name: efficient_net_b0
  ...
```
Define each experiment within the `experiments` list, providing a unique name, data directories for the
training and test images, and hyperparameters such as the number of epochs, batch size, learning rate,
optimizer, and model name.

## Training 🚀
*Command Line*
>
>From the root directory of the project execute:
>```bash
>$ python -m image_multiclass_classification train --config ./configs/experiments.yaml
>```
>To resume training from a saved checkpoint execute:
>```bash
>$ python -m image_multiclass_classification train --config ./configs/experiments.yaml --resume_from_checkpoint yes
>```
>where the checkpoint directory path is defined in the configuration file.

## Experiment Tracking 📉
>Track your experiments with tensorboard by executing:
>```bash
>$ tensorboard --logdir <tracking_dir>
>```
>where the `tracking_dir` is defined in the configuration file.
