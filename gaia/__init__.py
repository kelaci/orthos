"""
GAIA - Hierarchical Neural Architecture with Temporal Abstraction and Meta-Learning

GAIA v4.2 introduces Sparse Attention and Structural Plasticity (SAS).
"""

from .core import *
from .layers import *
from .hierarchy import *
from .plasticity import *
from .meta_learning import *
from .utils import *
from .config import *

__version__ = "4.2.0"
__author__ = "GAIA Development Team"
__license__ = "MIT"
__all__ = [
    # Core modules
    'Module', 'Layer', 'PlasticComponent', 'HierarchicalLevel',
    'Tensor', 'Shape', 'PlasticityParams', 'LearningRate', 'TimeStep',

    # Layers
    'ReactiveLayer', 'HebbianCore', 'TemporalLayer', 'SparseAttentionLayer', 'MaskedLinear',

    # Hierarchy
    'HierarchicalLevel', 'HierarchyManager',

    # Plasticity
    'PlasticityController', 'EvolutionaryStrategy',
    'PlasticityRule', 'HebbianRule', 'OjasRule', 'BCMRule',
    'StructuralOptimizer', 'SynapticConsolidation', 
    'SparsityScheduler', 'StructuralGuardrails', 'SparsityMonitor',

    # Meta Learning
    'MetaOptimizer',

    # Utilities
    'setup_logging', 'log_tensor_stats', 'log_plasticity_update',
    'plot_hierarchy_representations', 'plot_learning_curve',
    'plot_weight_matrix', 'plot_plasticity_parameters',

    # Configuration
    'DEFAULT_HIERARCHY_CONFIG', 'DEFAULT_PLASTICITY_CONFIG',
    'DEFAULT_ES_CONFIG', 'DEFAULT_LAYER_CONFIGS'
]