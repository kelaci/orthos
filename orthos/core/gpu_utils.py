"""
GPU support utilities for ORTHOS v4.x (NumPy version).

This module provides CuPy-based GPU acceleration with automatic
CPU fallback for NumPy-based operations.
"""

from typing import Optional, Tuple, Any
import warnings

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    import cupyx
    from cupyx.scipy.ndimage import map_coordinates
except ImportError:
    import numpy as cp  # Use numpy as fallback
    CUPY_AVAILABLE = False
    warnings.warn(
        "CuPy not available. Using CPU (NumPy). "
        "Install with: pip install cupy-cuda11x  # or cupy-cuda12x"
    )

# Import numpy for CPU operations
import numpy as np

# Global device preference
_device_preference: Optional[str] = None  # 'cpu', 'cuda', 'auto'

def set_device(device: str) -> None:
    """
    Set device preference for operations.
    
    Args:
        device: Device preference ('cpu', 'cuda', 'auto')
    """
    global _device_preference
    _device_preference = device.lower()

def get_device() -> str:
    """
    Get current device being used.
    
    Returns:
        'cuda' if using GPU, 'cpu' otherwise
    """
    if _device_preference == 'cpu':
        return 'cpu'
    elif _device_preference == 'cuda' and not CUPY_AVAILABLE:
        warnings.warn("CUDA requested but CuPy not available. Using CPU.")
        return 'cpu'
    elif CUPY_AVAILABLE:
        return 'cuda'
    else:
        return 'cpu'

def get_array_module() -> Any:
    """
    Get appropriate array module (CuPy or NumPy) based on device.
    
    Returns:
        cupy if GPU available and requested, numpy otherwise
    """
    if get_device() == 'cuda' and CUPY_AVAILABLE:
        return cp
    else:
        return np

def ensure_device(arr: Any) -> Any:
    """
    Ensure array is on the correct device.
    
    Args:
        arr: Input array (can be numpy or cupy array)
    
    Returns:
        Array on correct device
    """
    if isinstance(arr, np.ndarray):
        # NumPy array
        if get_device() == 'cuda' and CUPY_AVAILABLE:
            # Move to GPU
            return cp.asarray(arr.get())
        return arr
    elif CUPY_AVAILABLE and hasattr(arr, 'get'):
        # CuPy array, ensure on correct device
        return arr
    else:
        return arr

def to_cpu(arr: Any) -> Any:
    """
    Move array to CPU.
    
    Args:
        arr: Input array (can be numpy or cupy array)
    
    Returns:
        NumPy array on CPU
    """
    if CUPY_AVAILABLE and hasattr(arr, 'get'):
        return cp.asnumpy(arr)
    elif isinstance(arr, np.ndarray):
        return arr
    else:
        return np.array(arr)

def to_gpu(arr: Any) -> Any:
    """
    Move array to GPU if available.
    
    Args:
        arr: Input array (can be numpy or cupy array)
    
    Returns:
        CuPy array on GPU, or NumPy array if GPU unavailable
    """
    if CUPY_AVAILABLE:
        return cp.asarray(arr)
    else:
        return np.asarray(arr)

def zeros(shape: Tuple[int, ...], dtype: type = np.float32) -> Any:
    """
    Create zeros array on current device.
    
    Args:
        shape: Shape of the array
        dtype: Data type
    
    Returns:
        Array of zeros
    """
    xp = get_array_module()
    return xp.zeros(shape, dtype=dtype)

def ones(shape: Tuple[int, ...], dtype: type = np.float32) -> Any:
    """
    Create ones array on current device.
    
    Args:
        shape: Shape of the array
        dtype: Data type
    
    Returns:
        Array of ones
    """
    xp = get_array_module()
    return xp.ones(shape, dtype=dtype)

def randn(shape: Tuple[int, ...]) -> Any:
    """
    Create random normal array on current device.
    
    Args:
        shape: Shape of the array
    
    Returns:
        Random normal array
    """
    xp = get_array_module()
    return xp.random.randn(*shape)

def uniform(low: float = 0.0, high: float = 1.0, 
              size: Optional[Tuple[int, ...]] = None) -> Any:
    """
    Create uniform random array on current device.
    
    Args:
        low: Lower bound
        high: Upper bound
        size: Shape of output array
    
    Returns:
        Uniform random array
    """
    xp = get_array_module()
    return xp.random.uniform(low, high, size)

def dot(a: Any, b: Any) -> Any:
    """
    Compute dot product on current device.
    
    Args:
        a: First array
        b: Second array
    
    Returns:
        Dot product
    """
    xp = get_array_module()
    return xp.dot(a, b)

def matmul(a: Any, b: Any) -> Any:
    """
    Compute matrix multiplication on current device.
    
    Args:
        a: First matrix
        b: Second matrix
    
    Returns:
        Matrix multiplication result
    """
    xp = get_array_module()
    return xp.matmul(a, b)

def outer(a: Any, b: Any) -> Any:
    """
    Compute outer product on current device.
    
    Args:
        a: First array
        b: Second array
    
    Returns:
        Outer product
    """
    xp = get_array_module()
    return xp.outer(a, b)

def sum(arr: Any, axis: Optional[int] = None, keepdims: bool = False) -> Any:
    """
    Compute sum on current device.
    
    Args:
        arr: Input array
        axis: Axis to sum along
        keepdims: Whether to keep dimensions
    
    Returns:
        Sum
    """
    xp = get_array_module()
    return xp.sum(arr, axis=axis, keepdims=keepdims)

def mean(arr: Any, axis: Optional[int] = None, keepdims: bool = False) -> Any:
    """
    Compute mean on current device.
    
    Args:
        arr: Input array
        axis: Axis to compute mean along
        keepdims: Whether to keep dimensions
    
    Returns:
        Mean
    """
    xp = get_array_module()
    return xp.mean(arr, axis=axis, keepdims=keepdims)

def std(arr: Any, axis: Optional[int] = None, keepdims: bool = False) -> Any:
    """
    Compute standard deviation on current device.
    
    Args:
        arr: Input array
        axis: Axis to compute std along
        keepdims: Whether to keep dimensions
    
    Returns:
        Standard deviation
    """
    xp = get_array_module()
    return xp.std(arr, axis=axis, keepdims=keepdims)

def norm(arr: Any, axis: Optional[int] = None, keepdims: bool = False, 
          ord: Optional[int] = None) -> Any:
    """
    Compute norm on current device.
    
    Args:
        arr: Input array
        axis: Axis to compute norm along
        keepdims: Whether to keep dimensions
        ord: Order of the norm (1=L1, 2=L2, inf=Linf)
    
    Returns:
        Norm
    """
    xp = get_array_module()
    return xp.linalg.norm(arr, axis=axis, keepdims=keepdims, ord=ord)

def exp(arr: Any) -> Any:
    """
    Compute exponential on current device.
    
    Args:
        arr: Input array
    
    Returns:
        Exponential
    """
    xp = get_array_module()
    return xp.exp(arr)

def tanh(arr: Any) -> Any:
    """
    Compute hyperbolic tangent on current device.
    
    Args:
        arr: Input array
    
    Returns:
        tanh(arr)
    """
    xp = get_array_module()
    return xp.tanh(arr)

def sigmoid(arr: Any) -> Any:
    """
    Compute sigmoid on current device.
    
    Args:
        arr: Input array
    
    Returns:
        sigmoid(arr)
    """
    xp = get_array_module()
    return 1 / (1 + xp.exp(-arr))

def relu(arr: Any) -> Any:
    """
    Compute ReLU on current device.
    
    Args:
        arr: Input array
    
    Returns:
        max(0, arr)
    """
    xp = get_array_module()
    return xp.maximum(0, arr)

def maximum(a: Any, b: Any) -> Any:
    """
    Compute element-wise maximum on current device.
    
    Args:
        a: First array
        b: Second array
    
    Returns:
        Element-wise maximum
    """
    xp = get_array_module()
    return xp.maximum(a, b)

def clip(arr: Any, a_min: float, a_max: float) -> Any:
    """
    Clip array values on current device.
    
    Args:
        arr: Input array
        a_min: Minimum value
        a_max: Maximum value
    
    Returns:
        Clipped array
    """
    xp = get_array_module()
    return xp.clip(arr, a_min, a_max)

def abs(arr: Any) -> Any:
    """
    Compute absolute value on current device.
    
    Args:
        arr: Input array
    
    Returns:
        Absolute value
    """
    xp = get_array_module()
    return xp.abs(arr)

def sqrt(arr: Any) -> Any:
    """
    Compute square root on current device.
    
    Args:
        arr: Input array
    
    Returns:
        Square root
    """
    xp = get_array_module()
    return xp.sqrt(arr)

def square(arr: Any) -> Any:
    """
    Compute square on current device.
    
    Args:
        arr: Input array
    
    Returns:
        arr^2
    """
    xp = get_array_module()
    return xp.square(arr)

def convolve(arr: Any, kernel: Any, mode: str = 'same') -> Any:
    """
    Compute 1D convolution on current device.
    
    Args:
        arr: Input array
        kernel: Convolution kernel
        mode: Convolution mode ('valid', 'same', 'full')
    
    Returns:
        Convolved array
    """
    xp = get_array_module()
    
    if not CUPY_AVAILABLE:
        # Use scipy for CPU
        from scipy import signal
        return signal.convolve(arr, kernel, mode=mode)
    
    # CuPy version
    from cupyx.scipy.ndimage import convolve1d
    return convolve1d(arr, kernel, mode=mode)

def corrcoef(arr: Any) -> Any:
    """
    Compute correlation coefficient matrix on current device.
    
    Args:
        arr: Input array (time, features)
    
    Returns:
        Correlation matrix (features, features)
    """
    xp = get_array_module()
    
    # Standardize
    arr_mean = mean(arr, axis=0, keepdims=True)
    arr_std = std(arr, axis=0, keepdims=True) + 1e-8
    arr_standardized = (arr - arr_mean) / arr_std
    
    # Compute correlation
    n = arr.shape[0]
    if n > 1:
        corr = matmul(arr_standardized.T, arr_standardized) / (n - 1)
    else:
        corr = xp.eye(arr.shape[1])
    
    return corr

def eye(n: int, dtype: type = np.float32) -> Any:
    """
    Create identity matrix on current device.
    
    Args:
        n: Size of identity matrix
        dtype: Data type
    
    Returns:
        Identity matrix
    """
    xp = get_array_module()
    return xp.eye(n, dtype=dtype)

class GPUContext:
    """
    Context manager for GPU operations.
    
    Automatically moves arrays to GPU and back to CPU.
    
    Example:
        >>> with GPUContext():
        >>>     result = heavy_computation(data)
        >>> # result is back on CPU
    """
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize GPU context.
        
        Args:
            device: Device preference ('cpu', 'cuda', 'auto')
        """
        self.old_device = _device_preference
        set_device(device)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous device preference
        global _device_preference
        _device_preference = self.old_device
        return False

def benchmark_device(operation: callable, warmup: int = 10, 
                   repeats: int = 100) -> dict:
    """
    Benchmark operation on current device.
    
    Args:
        operation: Function to benchmark
        warmup: Number of warmup iterations
        repeats: Number of benchmark iterations
    
    Returns:
        Dictionary with timing statistics
    """
    import time
    
    xp = get_array_module()
    device = get_device()
    
    # Warmup
    for _ in range(warmup):
        operation()
    
    # Synchronize device (for GPU)
    if CUPY_AVAILABLE and device == 'cuda':
        cp.cuda.Stream.null.synchronize()
    
    # Benchmark
    times = []
    for _ in range(repeats):
        start = time.time()
        result = operation()
        
        # Synchronize for accurate timing
        if CUPY_AVAILABLE and device == 'cuda':
            cp.cuda.Stream.null.synchronize()
        
        elapsed = time.time() - start
        times.append(elapsed)
    
    times = np.array(times)
    
    return {
        'device': device,
        'mean_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'min_time': float(np.min(times)),
        'max_time': float(np.max(times)),
        'total_time': float(np.sum(times)),
    }

def get_device_info() -> dict:
    """
    Get information about available devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'cuda_available': CUPY_AVAILABLE,
        'preferred_device': get_device(),
    }
    
    if CUPY_AVAILABLE:
        try:
            info['cuda_version'] = cp.cuda.runtime.get_version()
            info['device_count'] = cp.cuda.runtime.getDeviceCount()
            info['device_name'] = cp.cuda.Device().name
            info['device_memory'] = f"{cp.cuda.Device().mem_info[0].total / 1e9:.2f} GB"
        except:
            info['cuda_version'] = 'unknown'
            info['device_count'] = 1
            info['device_name'] = 'unknown'
            info['device_memory'] = 'unknown'
    
    return info

# Export key functions for convenience
__all__ = [
    'CUPY_AVAILABLE',
    'set_device',
    'get_device',
    'get_array_module',
    'ensure_device',
    'to_cpu',
    'to_gpu',
    'zeros',
    'ones',
    'randn',
    'uniform',
    'dot',
    'matmul',
    'outer',
    'sum',
    'mean',
    'std',
    'norm',
    'exp',
    'tanh',
    'sigmoid',
    'relu',
    'maximum',
    'clip',
    'abs',
    'sqrt',
    'square',
    'convolve',
    'corrcoef',
    'eye',
    'GPUContext',
    'benchmark_device',
    'get_device_info',
]