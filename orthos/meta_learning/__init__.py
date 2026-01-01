"""
Meta-learning module for ORTHOS.

This module provides meta-optimization capabilities for learning
optimal plasticity parameters and adaptation strategies.
"""

from .optimizer import MetaOptimizer
from .metrics import *

__all__ = [
    'MetaOptimizer',
    'measure_plasticity_efficiency',
    'measure_adaptation_speed',
    'measure_stability_plasticity_tradeoff',
    'evaluate_meta_learning_curve'
]