"""
Default configurations for ORTHOS.

This module provides sensible default configurations for all ORTHOS components,
making it easy to get started while allowing full customization.
"""

from typing import Dict, Any, List

# Default hierarchy configuration
DEFAULT_HIERARCHY_CONFIG: Dict[str, Any] = {
    "num_levels": 4,
    "temporal_compression": 2,  # Each level compresses time by this factor
    "base_resolution": 1,       # Base temporal resolution
    "level_sizes": [64, 128, 256, 512],  # Feature sizes at each level
    "communication_interval": 5,  # Steps between inter-level communication
    "max_levels": 8,             # Maximum number of levels allowed
    "min_levels": 2              # Minimum number of levels required
}

# Default plasticity configuration
DEFAULT_PLASTICITY_CONFIG: Dict[str, float] = {
    "learning_rate": 0.01,
    "ltp_coefficient": 1.0,      # Long-Term Potentiation strength
    "ltd_coefficient": 0.8,      # Long-Term Depression strength
    "decay_rate": 0.001,         # Weight decay rate
    "homeostatic_strength": 0.1, # Homeostatic regulation strength
    "bcm_theta": 1.0,            # BCM rule threshold
    "stdp_tau": 20.0             # STDP time constant
}

# Default Sparse Attention and Structural Plasticity configuration
DEFAULT_SPARSE_CONFIG: Dict[str, Any] = {
    "sparsity_target": 0.1,         # Target connectivity density (10%)
    "activation_sparsity": 0.2,     # kWTA fraction (20% active)
    "prune_threshold": 0.01,        # Weight magnitude threshold for pruning
    "regrow_fraction": 0.05,        # Fraction of synapses to regrow per cycle
    "consolidation_strength": 0.5,  # Importance-based protection strength
    "maturity_age": 50,             # Steps before a synapse can be pruned
    "block_size": 1,                # Size of sparsity blocks
    "rewire_interval": 100          # Steps between rewiring cycles
}

# Default ES (Evolutionary Strategy) configuration
DEFAULT_ES_CONFIG: Dict[str, Any] = {
    "population_size": 50,
    "sigma": 0.1,                # Mutation strength
    "learning_rate": 0.01,       # Mean update rate
    "elite_fraction": 0.2,      # Fraction of top performers to select
    "max_sigma": 1.0,            # Maximum mutation strength
    "min_sigma": 0.001,          # Minimum mutation strength
    "adaptation_rate": 0.01      # Sigma adaptation rate
}

# Default layer configurations
DEFAULT_LAYER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "reactive": {
        "activation": "relu",
        "init_type": "he",
        "use_bias": True,
        "bias_init": 0.0
    },
    "hebbian": {
        "plasticity_rule": "hebbian",
        "params": DEFAULT_PLASTICITY_CONFIG,
        "activity_trace_length": 100,
        "normalization": "l2"
    },
    "temporal": {
        "activation": "tanh",
        "time_window": 10,
        "recurrent_init": "he",
        "state_normalization": True
    },
    "sparse_attention": {
        "sparsity_target": 0.1,
        "activation_sparsity": 0.2,
        "n_heads": 4,
        "block_size": 1
    }
}

# Default meta-learning configuration
DEFAULT_META_CONFIG: Dict[str, Any] = {
    "num_episodes": 100,
    "task_switch_frequency": 10,
    "performance_threshold": 0.8,
    "outer_optimizer": "adam",
    "adaptation_strategy": "uniform",
    "evaluation_interval": 5
}

# Default visualization configuration
DEFAULT_VIS_CONFIG: Dict[str, Any] = {
    "figsize": (12, 8),
    "dpi": 100,
    "colormap": "viridis",
    "linewidth": 2,
    "alpha": 0.7,
    "grid_alpha": 0.3
}

# Default logging configuration
DEFAULT_LOG_CONFIG: Dict[str, Any] = {
    "level": "INFO",
    "file_logging": False,
    "log_file": "orthos.log",
    "console_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}

def get_default_config(component: str) -> Dict[str, Any]:
    """
    Get default configuration for a specific component.

    Args:
        component: Component name ('hierarchy', 'plasticity', 'es', 'layers', 'meta', 'vis', 'log')

    Returns:
        Default configuration dictionary

    Raises:
        ValueError: If unknown component is specified
    """
    component = component.lower()

    if component == 'hierarchy':
        return DEFAULT_HIERARCHY_CONFIG.copy()
    elif component == 'plasticity':
        return DEFAULT_PLASTICITY_CONFIG.copy()
    elif component == 'es':
        return DEFAULT_ES_CONFIG.copy()
    elif component == 'layers':
        return DEFAULT_LAYER_CONFIGS.copy()
    elif component == 'meta':
        return DEFAULT_META_CONFIG.copy()
    elif component == 'vis':
        return DEFAULT_VIS_CONFIG.copy()
    elif component == 'log':
        return DEFAULT_LOG_CONFIG.copy()
    elif component == 'sparse':
        return DEFAULT_SPARSE_CONFIG.copy()
    else:
        raise ValueError(f"Unknown component: {component}")

def validate_config(config: Dict[str, Any], config_type: str) -> bool:
    """
    Validate a configuration dictionary.

    Args:
        config: Configuration dictionary to validate
        config_type: Type of configuration to validate

    Returns:
        True if configuration is valid, False otherwise
    """
    if config_type not in ['hierarchy', 'plasticity', 'es', 'layers', 'meta', 'vis', 'log', 'sparse']:
        return False
        
    # Get reference default to check types against
    try:
        if config_type == 'layers':
            # Layers usually have nested structure, validate against generic layer config
            # or check for required keys like 'reactive', 'hebbian', etc.
            return all(isinstance(v, dict) for v in config.values())
        
        default_ref = get_default_config(config_type)
        
        for key, value in config.items():
            if key not in default_ref:
                # Unknown key
                continue
            
            expected_type = type(default_ref[key])
            if not isinstance(value, expected_type):
                # Allow int for float and vice versa if they are numbers
                if isinstance(value, (int, float)) and isinstance(default_ref[key], (int, float)):
                    continue
                return False
                
        return True
    except Exception:
        return False

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration with recursive nested dict support
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursive merge for nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
            
    return merged

# Configuration parameter descriptions
CONFIG_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    'hierarchy': {
        'num_levels': 'Number of hierarchical levels in the architecture',
        'temporal_compression': 'Time compression factor between levels',
        'base_resolution': 'Temporal resolution at the base level',
        'level_sizes': 'Feature dimensions at each level',
        'communication_interval': 'Steps between inter-level communication',
        'max_levels': 'Maximum allowed number of levels',
        'min_levels': 'Minimum required number of levels'
    },
    'plasticity': {
        'learning_rate': 'Base learning rate for weight updates',
        'ltp_coefficient': 'Long-Term Potentiation strength multiplier',
        'ltd_coefficient': 'Long-Term Depression strength multiplier',
        'decay_rate': 'Weight decay rate for regularization',
        'homeostatic_strength': 'Strength of homeostatic regulation',
        'bcm_theta': 'BCM rule sliding threshold',
        'stdp_tau': 'STDP time constant (ms)'
    },
    'es': {
        'population_size': 'Number of individuals in ES population',
        'sigma': 'Mutation strength (standard deviation)',
        'learning_rate': 'Learning rate for mean parameter updates',
        'elite_fraction': 'Fraction of top performers to select',
        'max_sigma': 'Maximum allowed mutation strength',
        'min_sigma': 'Minimum allowed mutation strength',
        'adaptation_rate': 'Rate of sigma adaptation'
    },
    'meta': {
        'num_episodes': 'Number of meta-training episodes',
        'task_switch_frequency': 'Steps before switching tasks',
        'performance_threshold': 'Target performance threshold',
        'outer_optimizer': 'Optimizer for outer loop (adam, sgd, rmsprop)',
        'adaptation_strategy': 'Task sampling strategy (uniform, curriculum)',
        'evaluation_interval': 'Steps between performance evaluations'
    },
    'sparse': {
        'sparsity_target': 'Target connectivity density (fraction of active synapses)',
        'activation_sparsity': 'kWTA fraction (fraction of active neurons/attention scores)',
        'prune_threshold': 'Weight magnitude threshold for pruning',
        'regrow_fraction': 'Fraction of synapses to regrow per cycle',
        'consolidation_strength': 'Strength of importance-based protection',
        'maturity_age': 'Minimum steps before a synapse becomes eligible for pruning',
        'block_size': 'Size of structural sparsity blocks',
        'rewire_interval': 'Steps between structural rewiring cycles'
    }
}

def get_config_template(component: str) -> Dict[str, Any]:
    """
    Get a configuration template with descriptions.

    Args:
        component: Component name

    Returns:
        Configuration template with value and description for each parameter
    """
    defaults = get_default_config(component)
    descriptions = CONFIG_DESCRIPTIONS.get(component, {})
    
    template = {}
    for key, value in defaults.items():
        template[key] = {
            'value': value,
            'description': descriptions.get(key, 'No description available'),
            'type': type(value).__name__
        }
    
    return template