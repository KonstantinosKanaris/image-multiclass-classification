"""
The project implements a global logger which is initialized in
the main package.

The logger redirects the log messages to both the console and
in a .log file.

Example of using logger
-----------------------
Import:
    .. highlight:: python
    .. code-block:: python

        from python_research_template import logger
Usage:
    .. highlight:: python
    .. code-block:: python

        logger.info("info message")
        logger.debug("debug message")
        logger.error("error message")

For all the logging messages look at:
    https://docs.python.org/3.8/howto/logging.html
"""

import logging.config
import os

from image_multiclass_classification.__about__ import (
    __MAJOR__,
    __MINOR__,
    __PATCH__,
    __author__,
    __copyright__,
    __email__,
    __summary__,
    __title__,
    __version__,
)

__all__ = [
    "__MAJOR__",
    "__MINOR__",
    "__PATCH__",
    "__author__",
    "__copyright__",
    "__email__",
    "__summary__",
    "__title__",
    "__version__",
    "logger",
]

basename = os.path.basename(os.getcwd())
log_config_fpath = os.path.join(os.getcwd(), __title__, "logger/logging.ini")


for name in [
    "matplotlib",
    "PIL",
]:
    logging.getLogger(name).setLevel(logging.ERROR)

logging.config.fileConfig(
    fname=log_config_fpath,
    disable_existing_loggers=False,
    encoding="utf-8",
)

logger = logging.getLogger(__name__)
