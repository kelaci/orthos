"""
Core module providing base classes, type definitions, and utility functions.
"""

from .base import Module, Layer, PlasticComponent, HierarchicalLevel
from .types import *
from .tensor import *
from .gpu_utils import (
    CUPY_AVAILABLE, set_device, get_device, get_array_module,
    ensure_device, to_cpu, to_gpu, GPUContext, benchmark_device, get_device_info
)

__all__ = [
    # Base classes
    'Module', 'Layer', 'PlasticComponent', 'HierarchicalLevel',
    
    # Types
    'Tensor', 'Shape', 'PlasticityParams', 'LearningRate', 'TimeStep',
    'WeightMatrix', 'ActivationFunction', 'ConfigDict',
    'HierarchyConfig', 'PlasticityConfig', 'ESConfig',
    
    # Tensor operations
    'initialize_weights', 'apply_activation', 'normalize_tensor',
    'temporal_convolution',
    
    # GPU utilities
    'CUPY_AVAILABLE', 'set_device', 'get_device', 'get_array_module',
    'ensure_device', 'to_cpu', 'to_gpu', 'GPUContext', 
    'benchmark_device', 'get_device_info',
]