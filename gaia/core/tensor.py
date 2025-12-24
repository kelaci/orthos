"""
Tensor operations and utilities.

This module provides fundamental tensor operations used throughout GAIA,
including weight initialization, activation functions, and normalization.

Supports both NumPy (CPU) and CuPy (GPU) backends automatically.
"""

from typing import Tuple, Optional, Union, Any

# Import GPU utilities first
try:
    from .gpu_utils import get_array_module, get_device, to_cpu, to_gpu
    from .gpu_utils import CUPY_AVAILABLE
    USING_GPU = get_device() == 'cuda'
except ImportError:
    import numpy as np
    get_array_module = lambda: np
    get_device = lambda: 'cpu'
    to_cpu = lambda x: x
    to_gpu = lambda x: x
    CUPY_AVAILABLE = False
    USING_GPU = False


def initialize_weights(shape: Tuple[int, ...], init_type: str = 'he') -> Any:
    """
    Initialize weights with specified initialization on current device.

    Args:
        shape: Shape of the weight matrix
        init_type: Initialization type ('he', 'xavier', 'normal', 'uniform')

    Returns:
        Initialized weight matrix
    """
    xp = get_array_module()
    
    if init_type == 'he':
        return xp.random.randn(*shape) * xp.sqrt(2.0 / shape[0])
    elif init_type == 'xavier':
        return xp.random.randn(*shape) * xp.sqrt(1.0 / shape[0])
    elif init_type == 'normal':
        return xp.random.randn(*shape) * 0.01
    elif init_type == 'uniform':
        return xp.random.uniform(-0.01, 0.01, shape)
    else:
        raise ValueError(f"Unknown initialization type: {init_type}")

def apply_activation(x: Any, activation: str) -> Any:
    """
    Apply activation function on current device.

    Args:
        x: Input tensor
        activation: Activation function name ('relu', 'sigmoid', 'tanh', 'linear')

    Returns:
        Activated tensor
    """
    xp = get_array_module()
    
    if activation == 'relu':
        return xp.maximum(0, x)
    elif activation == 'sigmoid':
        return 1 / (1 + xp.exp(-x))
    elif activation == 'tanh':
        return xp.tanh(x)
    elif activation == 'linear':
        return x
    else:
        raise ValueError(f"Unknown activation: {activation}")

def apply_activation_derivative(x: Any, activation: str) -> Any:
    """
    Apply activation function derivative.

    Args:
        x: Input tensor
        activation: Activation function name

    Returns:
        Derivative tensor
    """
    xp = get_array_module()
    
    if activation == 'relu':
        return (x > 0).astype(float)
    elif activation == 'sigmoid':
        s = 1 / (1 + xp.exp(-x))
        return s * (1 - s)
    elif activation == 'tanh':
        return 1.0 - xp.tanh(x)**2
    elif activation == 'linear':
        return xp.ones_like(x)
    else:
        raise ValueError(f"Unknown activation: {activation}")

def normalize_tensor(x: Any, axis: Optional[int] = None, eps: float = 1e-8) -> Any:
    """
    Normalize tensor along specified axis.

    Args:
        x: Input tensor
        axis: Axis to normalize along
        eps: Small constant to prevent division by zero

    Returns:
        Normalized tensor
    """
    xp = get_array_module()
    norm = xp.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)

def temporal_convolution(x: Any, kernel: Any, mode: str = 'same') -> Any:
    """
    Apply temporal convolution.

    Args:
        x: Input tensor (time, features)
        kernel: Convolution kernel
        mode: Convolution mode ('valid', 'same', 'full')

    Returns:
        Convolved tensor (time, features)
    """
    from .gpu_utils import convolve
    xp = get_array_module()
    
    if len(x.shape) != 2:
        raise ValueError(f"Input x must be 2D (time, features), got {x.shape}")

    time_steps, features = x.shape
    
    # Delegate to gpu_utils.convolve which handles dispatch
    if len(kernel.shape) == 1:
        result = xp.zeros_like(x)
        for f in range(features):
            result[:, f] = convolve(x[:, f], kernel, mode=mode)
        return result
    elif len(kernel.shape) == 2:
        result = xp.zeros_like(x)
        for f in range(features):
            result[:, f] = convolve(x[:, f], kernel[:, f], mode=mode)
        return result
    else:
        raise NotImplementedError("Multi-dimensional temporal convolution not yet implemented")

def correlation_matrix(x: Any) -> Any:
    """
    Compute correlation matrix of input tensor.

    Args:
        x: Input tensor (batch/time, features)

    Returns:
        Correlation matrix (features, features)
    """
    from .gpu_utils import corrcoef
    return corrcoef(x)