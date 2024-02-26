""" Contains the project metadata,
    e.g., *title*, *version*, *summary* etc.
"""

import os
from datetime import datetime

__MAJOR__ = 0
__MINOR__ = 1
__PATCH__ = 0

__title__ = os.path.basename(os.getcwd()).replace("-", "_")
__version__ = ".".join([str(__MAJOR__), str(__MINOR__), str(__PATCH__)])
__summary__ = "A multi-class image classification project with PyTorch."
__author__ = "Konstantinos Kanaris"
__copyright__ = f"Copyright (C) {datetime.now().date().year}  {__author__}"
__email__ = "konskan95@outlook.com.gr"
