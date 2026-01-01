"""
JIT compilation utilities for ORTHOS.

This module provides conditional Numba/JAX decorators for performance optimization.
If libraries are not available, it handles graceful fallback to standard Python.

Key Features:
- Conditional Numba JIT compilation
- Support for customizable JIT parameters
- Transparent fallback for systems without JIT libraries
"""

import functools
from typing import Any, Callable, TypeVar

try:
    from numba import jit as numba_jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

F = TypeVar('F', bound=Callable[..., Any])

def jit(
    nopython: bool = True, 
    cache: bool = True, 
    **kwargs: Any
) -> Callable[[F], F]:
    """
    Conditional Numba JIT decorator.
    
    If Numba is installed, it applies @numba.jit with the specified parameters.
    Otherwise, it returns the original function unchanged.

    Args:
        nopython: Whether to use Numba nopython mode.
        cache: Whether to cache the compiled function.
        **kwargs: Additional parameters passed to numba.jit.

    Returns:
        A decorator that optionally compiles the function.
    """
    def decorator(func: F) -> F:
        if HAS_NUMBA:
            return numba_jit(nopython=nopython, cache=cache, **kwargs)(func)
        return func
    return decorator
