"""
Plasticity module for ORTHOS.

This module provides plasticity control and meta-learning capabilities
using evolutionary strategies and various learning rules.
"""

from .controller import PlasticityController
from .es_optimizer import EvolutionaryStrategy
from .rules import PlasticityRule, HebbianRule, OjasRule, BCMRule
from .structural import (
    StructuralOptimizer, SynapticConsolidation, 
    SparsityScheduler, StructuralGuardrails, SparsityMonitor
)

__all__ = [
    'PlasticityController', 
    'EvolutionaryStrategy',
    'PlasticityRule', 
    'HebbianRule', 
    'OjasRule', 
    'BCMRule',
    'StructuralOptimizer',
    'SynapticConsolidation',
    'SparsityScheduler',
    'StructuralGuardrails',
    'SparsityMonitor'
]