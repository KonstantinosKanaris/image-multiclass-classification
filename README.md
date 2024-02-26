# Multi-class Image Classification with PyTorch

## Table of Contents

* [Overview](#Overview)
* [Key Features](#Key Features)
* [Development](#Development)

## Overview 🔍

This project provides a robust framework for image multi-class classification
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
│   └── experiments.yaml
├── image_multiclass_classification        | The main Python package containing the project's source code.
│   ├── __about__.py                 | Metadata about the project, i.e., version number, author information, etc.
│   ├── __init__.py
│   ├── __main__.py                  | The entry point for running the package as a script
│   ├── core.py                      | Contains core functionalities for the image multi-class classification project.
│   ├── data_setup.py
│   ├── engine
│   │   ├── __init__.py
│   │   ├── predictor.py
│   │   └── trainer.py         | Contains the training loop.
│   ├── factories
│   │   ├── __init__.py
│   │   ├── client.py          | Interacts with the factories to return different instances of models, optimizers, transforms.
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
