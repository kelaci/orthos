"""
Custom exceptions for ORTHOS.

Defines the exception hierarchy used throughout the framework.
"""

class ORTHOSError(Exception):
    """Base exception for all ORTHOS-related errors."""
    pass

class InputShapeError(ORTHOSError):
    """Raised when input tensor shape does not match expected dimensions."""
    def __init__(self, message: str, expected: tuple = None, actual: tuple = None):
        if expected and actual:
            message = f"{message} (Expected: {expected}, Got: {actual})"
        super().__init__(message)

class ConfigurationError(ORTHOSError):
    """Raised when configuration parameters are invalid or missing."""
    pass

class PlasticityError(ORTHOSError):
    """Raised when plasticity operations fail."""
    pass

class HierarchyError(ORTHOSError):
    """Raised when hierarchy integrity is violated (e.g. duplicate levels)."""
    pass
