class MethodError(Exception):
    """Base exception for method-related errors."""


class ParameterValidationError(MethodError):
    """Raised when parameters are invalid."""


class EarlyStopRequested(MethodError):
    """Internal signal when early stopping triggers."""