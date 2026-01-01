"""
Logging utilities for ORTHOS.

This module provides comprehensive logging functionality for debugging,
monitoring, and performance tracking.
"""

import logging
from typing import Optional, Dict, Any
import numpy as np
import time
from orthos.core.types import Tensor

def setup_logging(name: str = 'orthos', level: str = 'INFO',
                 log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging for ORTHOS.

    Args:
        name: Logger name
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional log file path

    Returns:
        Configured logger instance

    Raises:
        ValueError: If invalid logging level is specified
    """
    # Validate logging level
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if level.upper() not in valid_levels:
        raise ValueError(f"Invalid logging level: {level}. Must be one of {valid_levels}")

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Create formatter with timestamp
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Create file handler if specified
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def log_tensor_stats(logger: logging.Logger, tensor: Tensor,
                    name: str, level: str = 'DEBUG') -> None:
    """
    Log statistics about a tensor.

    Args:
        logger: Logger instance
        tensor: Tensor to log
        name: Tensor name for identification
        level: Logging level

    Raises:
        ValueError: If invalid logging level is specified
    """
    stats = {
        'shape': tensor.shape,
        'dtype': str(tensor.dtype),
        'mean': float(np.mean(tensor)),
        'std': float(np.std(tensor)),
        'min': float(np.min(tensor)),
        'max': float(np.max(tensor)),
        'norm': float(np.linalg.norm(tensor))
    }

    # Log at specified level
    getattr(logger, level.lower())(f"Tensor {name} stats: {stats}")

def log_plasticity_update(logger: logging.Logger, params: Dict[str, float],
                         performance: float, step: int) -> None:
    """
    Log plasticity parameter update.

    Args:
        logger: Logger instance
        params: Plasticity parameters
        performance: Current performance metric
        step: Training step or iteration
    """
    param_str = ", ".join(f"{k}={v:.4f}" for k, v in params.items())
    logger.info(f"Step {step}: Performance = {performance:.4f}, Params = [{param_str}]")

def log_hierarchy_processing(logger: logging.Logger, level_id: int,
                           input_shape: tuple, output_shape: tuple,
                           processing_time: float) -> None:
    """
    Log hierarchy level processing information.

    Args:
        logger: Logger instance
        level_id: Hierarchy level ID
        input_shape: Input tensor shape
        output_shape: Output tensor shape
        processing_time: Processing time in seconds
    """
    logger.debug(f"Level {level_id}: {input_shape} → {output_shape} in {processing_time:.4f}s")

def log_learning_progress(logger: logging.Logger, episode: int,
                         performance: float, params: Dict[str, float],
                         duration: float) -> None:
    """
    Log learning progress information.

    Args:
        logger: Logger instance
        episode: Episode or iteration number
        performance: Current performance metric
        params: Current parameters
        duration: Duration of episode in seconds
    """
    param_str = ", ".join(f"{k}={v:.4f}" for k, v in params.items())
    logger.info(f"Episode {episode}: Performance = {performance:.4f}, Params = [{param_str}], Duration = {duration:.2f}s")

def time_function(logger: logging.Logger, func: callable, *args, **kwargs) -> Any:
    """
    Time a function execution and log the duration.

    Args:
        logger: Logger instance
        func: Function to time
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function call
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    duration = end_time - start_time
    logger.debug(f"Function {func.__name__} executed in {duration:.4f}s")

    return result

def log_system_info(logger: logging.Logger) -> None:
    """
    Log system information for debugging.

    Args:
        logger: Logger instance
    """
    import platform
    import sys

    system_info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'system': platform.system(),
        'release': platform.release(),
        'machine': platform.machine()
    }

    logger.info(f"System Info: {system_info}")

def create_performance_logger(name: str = 'orthos.performance') -> logging.Logger:
    """
    Create a specialized logger for performance metrics.

    Args:
        name: Logger name

    Returns:
        Configured performance logger
    """
    logger = setup_logging(name, 'INFO')
    return logger

def create_debug_logger(name: str = 'orthos.debug') -> logging.Logger:
    """
    Create a specialized logger for debug information.

    Args:
        name: Logger name

    Returns:
        Configured debug logger
    """
    logger = setup_logging(name, 'DEBUG')
    return logger