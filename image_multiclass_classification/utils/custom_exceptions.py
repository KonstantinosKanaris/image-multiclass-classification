class ImageClassificationBaseException(Exception):
    """Base exception class for the image multi-class classification project.

    Args:
        message (str, optional):
            The error message associated with the exception.

    Attributes:
        message (str, optional):
            The error message associated with the exception.
            Defaults to `An error occured.`

    Example::

        >>> raise ImageClassificationBaseException("error message")
    """

    def __init__(self, message: str = "An error occurred.") -> None:
        super().__init__(message)
        self.message: str = message


class UnsupportedModelNameError(ImageClassificationBaseException):
    """
    Exception raised when the provided model name is not supported.

    This exception is raised when attempting to create an instance
    of a PyTorch model with a model name that is not recognized or
    supported.

    Example::

        >>> raise UnsupportedModelNameError(
        ...     message="This model name is not supported."
        ... )
    """

    pass


class UnsupportedOptimizerNameError(ImageClassificationBaseException):
    """
    Exception raised when the provided optimizer name is not supported.

    This exception is raised when attempting to create PyTorch optimizer
    with an optimizer name that is not recognized or supported.

    Example::

        >>> raise UnsupportedOptimizerNameError(
        ...        message="This optimizer name is not supported."
        ... )
    """

    pass
