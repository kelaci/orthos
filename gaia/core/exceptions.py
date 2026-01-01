"""
Custom exceptions for GAIA.

Defines the exception hierarchy used throughout the framework.
"""

class GAIAError(Exception):
    """Base exception for all GAIA-related errors."""
    pass

class InputShapeError(GAIAError):
    """Raised when input tensor shape does not match expected dimensions."""
    def __init__(self, message: str, expected: tuple = None, actual: tuple = None):
        if expected and actual:
            message = f"{message} (Expected: {expected}, Got: {actual})"
        super().__init__(message)

class ConfigurationError(GAIAError):
    """Raised when configuration parameters are invalid or missing."""
    pass

class PlasticityError(GAIAError):
    """Raised when plasticity operations fail."""
    pass

class HierarchyError(GAIAError):
    """Raised when hierarchy integrity is violated (e.g. duplicate levels)."""
    pass
