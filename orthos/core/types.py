"""
Type definitions for ORTHOS.

This module provides type aliases and custom types for better code clarity
and type checking throughout the ORTHOS codebase.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Callable, Optional

# Type aliases for better code clarity
Tensor = np.ndarray
Shape = Tuple[int, ...]
PlasticityParams = Dict[str, float]
LearningRate = float
TimeStep = int
WeightMatrix = np.ndarray
ActivationFunction = Callable[[np.ndarray], np.ndarray]

# Configuration types
ConfigDict = Dict[str, Any]
HierarchyConfig = Dict[str, Any]
PlasticityConfig = Dict[str, float]
ESConfig = Dict[str, float]
LayerConfig = Dict[str, Any]

# Performance metrics types
PerformanceMetrics = Dict[str, float]
LearningHistory = List[float]
ParamHistory = List[Dict[str, float]]