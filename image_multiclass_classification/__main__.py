import argparse

from image_multiclass_classification import __title__, logger
from image_multiclass_classification.core import ExperimentManager
from image_multiclass_classification.utils.aux import load_yaml_file

CLASS_NAMES = ["pizza", "steak", "sushi"]
MODEL_DIR_PATH = "./models"
MODEL_NAME = "05_going_modular_script_mode_tinyvgg_model.pth"


def parse_arguments() -> argparse.Namespace:
    """
    Constructs parsers and subparsers.

    Returns:
        argparse.Namespace:
            The parser/subparser with its arguments.
    """
    parser = argparse.ArgumentParser(
        description=f"Command Line Interface for {__title__}"
    )

    subparsers = parser.add_subparsers(
        description="Project functionalities", dest="mode"
    )

    train = subparsers.add_parser(
        name="train", help="This is the subparser for training."
    )

    train.add_argument(
        "--config",
        type=str,
        required=True,
        help="The config file (yaml) required for training."
        "Contains data paths, hyperparameters, etc.",
    )

    train.add_argument(
        "--resume_from_checkpoint",
        type=str,
        choices={"yes", "no"},
        required=False,
        default="no",
        help="If `yes` the training will resume from the last " "saved checkpoint.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    if arguments.resume_from_checkpoint == "yes":
        resume = True
    else:
        resume = False

    config = load_yaml_file(filepath=arguments.config)

    brain = ExperimentManager(config=config, resume_from_checkpoint=resume)
    if arguments.mode == "train":
        brain.run_experiments()
    else:
        logger.error("Not supported mode.")
