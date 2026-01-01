# Architecture Overview

## 🎯 System Vision

ORTHOS v5.0 is a hierarchical neural architecture designed for:


- **Temporal abstraction**: Processing information at multiple time scales (1x, 2x, 4x, 8x)
- **Probabilistic Spine**: Sequential Bayesian estimation using Kalman and Particle filters
- **Consensus Engine**: "Wisdom of Crowds" aggregation across hierarchical levels
- **Meta-learning of plasticity**: Learning how to learn through evolutionary strategies
- **Biological plausibility**: Hebbian learning, top-down feedback, and dual-timescale memory

## 🏗️ Core Architecture

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                                ORTHOS System (v5.0)                            │
├───────────────────────────────────────────────────────────────────────────────┤

│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                            Consensus Layer                              │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │              ConsensusHierarchyManager (Auto-Projection)             │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────┬──────────────────────────────────────▲──────────────┘  │
│                      │                                      │                 │
│                      │ (Top-Down Feedback)                  │ (Bottom-Up)     │
│                      ▼                                      │                 │
│  ┌──────────────────────────────────────────────────────────┴──────────────┐  │
│  │                        Hierarchical & Probabilistic                     │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │           Level 3: Concepts + Kalman Filter (8x res)                │  │  │
│  │  │           - Joseph Form Stability, Diagonal Covariance               │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  │                                     ▲                                   │  │
│  │  ┌──────────────────────────────────┴──────────────────────────────────┐  │  │
│  │  │           Level 2: Sequences + Particle Filter (4x res)              │  │  │
│  │  │           - Multi-modal state estimation                            │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  │                                     ▲                                   │  │
│  │  ┌──────────────────────────────────┴──────────────────────────────────┐  │  │
│  │  │           Level 1: Features + EKF (2x res)                          │  │  │
│  │  │           - Non-linear sensory mapping                              │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  │                                     ▲                                   │  │
│  │  ┌──────────────────────────────────┴──────────────────────────────────┐  │  │
│  │  │           Level 0: Raw Input Processing (1x res)                   │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                          Learning & Plasticity                          │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │           PlasticityController (Meta-Learning/ES)                   │  │  │
│  │  ├─────────────────────────────────────────────────────────────────────┤  │  │
│  │  │           HebbianCore (Classic, Oja, BCM, STDP)                     │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow

1. **Input Processing**: Raw data enters at Level 0
2. **Hierarchical Abstraction**: Each level compresses temporal information and extracts higher-level features
3. **Temporal Processing**: TemporalLayer maintains context across time steps
4. **Plasticity Control**: PlasticityController adapts learning parameters via evolutionary strategies
5. **Meta-Learning**: System learns optimal plasticity parameters for different tasks

## 🎯 Key Design Principles

### Modularity
- Each component has a single responsibility
- Clear interfaces between modules
- Easy to replace or extend individual components

### Hierarchical Organization
- Multiple levels of temporal abstraction
- Progressive information compression
- Bidirectional communication between levels

### Plasticity Control
- Evolutionary Strategy for parameter optimization
- Multiple plasticity rules (Hebbian, Oja, BCM, STDP)
- Dual-timescale traces (Hippocampal/Neocortical)
- Dynamic adaptation to task requirements

### Probabilistic Spine (v5.0)
- **Filters**: Kalman (Linear), EKF (Non-linear), Particle (General)
- **Numerical Stability**: Joseph form covariance updates
- **Performance**: Diagonal covariance approximation for high-dim states
- **Consensus**: Multi-level aggregation with outlier rejection and auto-projection


### Extensibility
- Abstract base classes for all major components
- Plugin architecture for new learning rules
- Configuration-driven behavior

## 📋 Component Relationships

```
[Input Data]
    ↓
[Level 0] → [ReactiveLayer] → [HebbianCore]
    ↓
[Level 1] → [TemporalLayer] → [PlasticityController]
    ↓
[Level 2] → [HierarchyManager]
    ↓
[Level 3]
    ↓
[Output/Representation]
```

## 🔮 Future Directions

- **v5.0 ✅**: Major Rebrand, Probabilistic Spine, Consensus Engine, Numerical Stability
- **v5.1**: Advanced ES variants (CMA-ES, NES) and Attention mechanisms
- **v5.2**: Neuroevolution and Direct Topology Optimization
- **v5.3**: Multi-modal sensory integration (Cross-modal attention)


See [Roadmap](ROADMAP.md) for detailed development plans.# Core Components

## 📦 Core Module Structure

The core module provides the foundation for all ORTHOS components, including base classes, type definitions, and utility functions.

### `core/base.py` - Abstract Base Classes

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import numpy as np

class Module(ABC):
    """Base class for all ORTHOS modules."""

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the module."""
        pass

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass for gradient computation."""
        pass

    @abstractmethod
    def update(self, lr: float) -> None:
        """Update module parameters."""
        pass

    @abstractmethod
    def reset_state(self) -> None:
        """Reset internal state."""
        pass

class Layer(Module):
    """Base class for all layer implementations."""

    @abstractmethod
    def activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        pass

    @abstractmethod
    def get_weights(self) -> np.ndarray:
        """Get current weights."""
        pass

    @abstractmethod
    def set_weights(self, weights: np.ndarray) -> None:
        """Set weights."""
        pass

class PlasticComponent(ABC):
    """Base class for components with plasticity."""

    @abstractmethod
    def get_plasticity_params(self) -> Dict[str, float]:
        """Get plasticity parameters."""
        pass

    @abstractmethod
    def set_plasticity_params(self, params: Dict[str, float]) -> None:
        """Set plasticity parameters."""
        pass

class HierarchicalLevel(ABC):
    """Base class for hierarchical levels."""

    @abstractmethod
    def process_time_step(self, input_data: np.ndarray, t: int) -> np.ndarray:
        """Process a single time step."""
        pass

    @abstractmethod
    def get_representation(self) -> np.ndarray:
        """Get current representation."""
        pass
```

### `core/types.py` - Type Definitions

```python
import numpy as np
from typing import Dict, List, Tuple, Any

# Type aliases for better code clarity
Tensor = np.ndarray
Shape = Tuple[int, ...]
PlasticityParams = Dict[str, float]
LearningRate = float
TimeStep = int
WeightMatrix = np.ndarray
ActivationFunction = callable

# Configuration types
ConfigDict = Dict[str, Any]
HierarchyConfig = Dict[str, Any]
PlasticityConfig = Dict[str, float]
ESConfig = Dict[str, float]
```

### `core/tensor.py` - Tensor Operations

```python
import numpy as np
from typing import Tuple, Optional

def initialize_weights(shape: Tuple[int, ...], init_type: str = 'he') -> np.ndarray:
    """
    Initialize weights with specified initialization.

    Args:
        shape: Shape of the weight matrix
        init_type: Initialization type ('he', 'xavier', 'normal', 'uniform')

    Returns:
        Initialized weight matrix
    """
    if init_type == 'he':
        return np.random.randn(*shape) * np.sqrt(2.0 / shape[0])
    elif init_type == 'xavier':
        return np.random.randn(*shape) * np.sqrt(1.0 / shape[0])
    elif init_type == 'normal':
        return np.random.randn(*shape) * 0.01
    elif init_type == 'uniform':
        return np.random.uniform(-0.01, 0.01, shape)
    else:
        raise ValueError(f"Unknown initialization type: {init_type}")

def apply_activation(x: np.ndarray, activation: str) -> np.ndarray:
    """
    Apply activation function.

    Args:
        x: Input tensor
        activation: Activation function name ('relu', 'sigmoid', 'tanh', 'linear')

    Returns:
        Activated tensor
    """
    if activation == 'relu':
        return np.maximum(0, x)
    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif activation == 'tanh':
        return np.tanh(x)
    elif activation == 'linear':
        return x
    else:
        raise ValueError(f"Unknown activation: {activation}")

def normalize_tensor(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Normalize tensor along specified axis.

    Args:
        x: Input tensor
        axis: Axis to normalize along

    Returns:
        Normalized tensor
    """
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + 1e-8)

def temporal_convolution(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply temporal convolution.

    Args:
        x: Input tensor (time, features)
        kernel: Convolution kernel (time,)

    Returns:
        Convolved tensor
    """
    # TODO: Implement efficient temporal convolution
    pass
```

## 📦 Layer Implementations

### `layers/reactive.py` - ReactiveLayer

```python
import numpy as np
from typing import Optional, Dict
from orthos.core.base import Layer
from orthos.core.types import Tensor, ActivationFunction

class ReactiveLayer(Layer):
    """
    Fast, reactive processing layer with fixed weights.

    Attributes:
        input_size: Size of input features
        output_size: Size of output features
        weights: Weight matrix (output_size, input_size)
        biases: Bias vector (output_size,)
        activation_fn: Activation function
    """

    def __init__(self, input_size: int, output_size: int,
                 activation: str = 'relu', init_type: str = 'he'):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = None
        self.biases = None
        self.activation_fn = activation
        self.init_type = init_type

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize weights and biases."""
        from orthos.core.tensor import initialize_weights

        self.weights = initialize_weights((self.output_size, self.input_size), self.init_type)
        self.biases = np.zeros(self.output_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the layer.

        Args:
            x: Input tensor (batch_size, input_size)

        Returns:
            Output tensor (batch_size, output_size)
        """
        from orthos.core.tensor import apply_activation

        # Linear transformation
        output = np.dot(x, self.weights.T) + self.biases

        # Apply activation
        return apply_activation(output, self.activation_fn)

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backward pass for gradient computation.

        Args:
            grad: Gradient from next layer (batch_size, output_size)

        Returns:
            Gradient for previous layer (batch_size, input_size)
        """
        # TODO: Implement backward pass
        pass

    def update(self, lr: float) -> None:
        """
        Update layer parameters.

        Args:
            lr: Learning rate
        """
        # ReactiveLayer has fixed weights - no update
        pass

    def reset_state(self) -> None:
        """Reset internal state."""
        # No internal state to reset
        pass

    def activation(self, x: Tensor) -> Tensor:
        """
        Apply activation function.

        Args:
            x: Input tensor

        Returns:
            Activated tensor
        """
        from orthos.core.tensor import apply_activation
        return apply_activation(x, self.activation_fn)

    def get_weights(self) -> Tensor:
        """Get current weights."""
        return self.weights

    def set_weights(self, weights: Tensor) -> None:
        """Set weights."""
        if weights.shape != self.weights.shape:
            raise ValueError(f"Weight shape mismatch: expected {self.weights.shape}, got {weights.shape}")
        self.weights = weights
```

### `layers/hebbian.py` - HebbianCore

```python
import numpy as np
from typing import Dict, Optional
from orthos.core.base import PlasticComponent
from orthos.core.types import Tensor, PlasticityParams

class HebbianCore(PlasticComponent):
    """
    Hebbian learning implementation with multiple plasticity rules.

    Attributes:
        input_size: Size of input features
        output_size: Size of output features
        weights: Weight matrix (output_size, input_size)
        pre_synaptic: Pre-synaptic activity trace
        post_synaptic: Post-synaptic activity trace
        plasticity_params: Plasticity parameters
    """

    def __init__(self, input_size: int, output_size: int,
                 plasticity_rule: str = 'hebbian',
                 params: Optional[Dict[str, float]] = None):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = None
        self.pre_synaptic = None
        self.post_synaptic = None
        self.plasticity_rule = plasticity_rule

        # Default plasticity parameters
        self.plasticity_params = params or {
            'learning_rate': 0.01,
            'decay_rate': 0.001,
            'ltp_coefficient': 1.0,
            'ltd_coefficient': 0.8,
            'homeostatic_strength': 0.1
        }

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize weights and activity traces."""
        from orthos.core.tensor import initialize_weights

        self.weights = initialize_weights((self.output_size, self.input_size), 'he')
        self.pre_synaptic = np.zeros(self.input_size)
        self.post_synaptic = np.zeros(self.output_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the Hebbian core.

        Args:
            x: Input tensor (batch_size, input_size)

        Returns:
            Output tensor (batch_size, output_size)
        """
        # Update activity traces
        self.pre_synaptic = x.mean(axis=0)  # Average over batch
        output = np.dot(x, self.weights.T)
        self.post_synaptic = output.mean(axis=0)  # Average over batch

        return output

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backward pass for gradient computation.

        Args:
            grad: Gradient from next layer

        Returns:
            Gradient for previous layer
        """
        # TODO: Implement backward pass
        pass

    def update(self, lr: float) -> None:
        """
        Update weights using Hebbian learning.

        Args:
            lr: Learning rate (overrides plasticity_params if provided)
        """
        effective_lr = lr if lr is not None else self.plasticity_params['learning_rate']

        if self.plasticity_rule == 'hebbian':
            # Classic Hebbian: Δw = η * pre * post
            weight_update = effective_lr * np.outer(self.post_synaptic, self.pre_synaptic)
        elif self.plasticity_rule == 'oja':
            # Oja's rule: Δw = η * post * (pre - post * w)
            weight_update = effective_lr * np.outer(
                self.post_synaptic,
                self.pre_synaptic - np.dot(self.weights.T, self.post_synaptic)
            )
        elif self.plasticity_rule == 'bcm':
            # BCM rule with sliding threshold
            theta = np.mean(self.post_synaptic)
            weight_update = effective_lr * np.outer(
                self.post_synaptic * (self.post_synaptic - theta),
                self.pre_synaptic
            )
        else:
            raise ValueError(f"Unknown plasticity rule: {self.plasticity_rule}")

        # Apply weight update
        self.weights += weight_update

        # Apply weight decay
        self.weights *= (1.0 - self.plasticity_params['decay_rate'])

        # Homeostatic regulation
        self._homeostatic_regulation()

    def _homeostatic_regulation(self) -> None:
        """Apply homeostatic regulation to maintain stable activity."""
        # Simple weight normalization
        norm = np.linalg.norm(self.weights, axis=1, keepdims=True)
        self.weights = self.weights / (norm + 1e-8)

    def reset_state(self) -> None:
        """Reset internal state."""
        self.pre_synaptic = np.zeros(self.input_size)
        self.post_synaptic = np.zeros(self.output_size)

    def get_plasticity_params(self) -> PlasticityParams:
        """Get plasticity parameters."""
        return self.plasticity_params

    def set_plasticity_params(self, params: PlasticityParams) -> None:
        """Set plasticity parameters."""
        self.plasticity_params.update(params)
```

### `layers/temporal.py` - TemporalLayer

```python
import numpy as np
from typing import Optional, Dict
from orthos.core.base import Layer
from orthos.core.types import Tensor

class TemporalLayer(Layer):
    """
    Layer with temporal context processing.

    Attributes:
        input_size: Size of input features
        hidden_size: Size of hidden state
        time_window: Number of time steps to maintain
        hidden_state: Current hidden state
        recurrent_weights: Recurrent weight matrix
    """

    def __init__(self, input_size: int, hidden_size: int,
                 time_window: int = 10, activation: str = 'tanh'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.time_window = time_window
        self.hidden_state = None
        self.activation = activation
        self.recurrent_weights = None

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize parameters."""
        from orthos.core.tensor import initialize_weights

        # Input to hidden weights
        self.weights = initialize_weights((self.hidden_size, self.input_size), 'he')

        # Recurrent weights
        self.recurrent_weights = initialize_weights((self.hidden_size, self.hidden_size), 'he')

        # Hidden state initialization
        self.hidden_state = np.zeros(self.hidden_size)

    def forward(self, x: Tensor, t: Optional[int] = None) -> Tensor:
        """
        Forward pass with temporal processing.

        Args:
            x: Input tensor (batch_size, input_size)
            t: Optional time step

        Returns:
            Output tensor (batch_size, hidden_size)
        """
        from orthos.core.tensor import apply_activation

        # Linear transformation
        linear_output = np.dot(x, self.weights.T) + np.dot(self.hidden_state, self.recurrent_weights.T)

        # Apply activation
        output = apply_activation(linear_output, self.activation)

        # Update hidden state
        self.hidden_state = output.mean(axis=0)  # Average over batch

        return output

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backward pass for gradient computation.

        Args:
            grad: Gradient from next layer

        Returns:
            Gradient for previous layer
        """
        # TODO: Implement backward pass
        pass

    def update(self, lr: float) -> None:
        """
        Update layer parameters.

        Args:
            lr: Learning rate
        """
        # TODO: Implement parameter update
        pass

    def reset_state(self) -> None:
        """Reset internal state."""
        self.hidden_state = np.zeros(self.hidden_size)

    def get_temporal_context(self) -> Tensor:
        """
        Get current temporal context.

        Returns:
            Temporal context tensor
        """
        return self.hidden_state.copy()
```

## 📦 Hierarchy Components

### `hierarchy/level.py` - HierarchicalLevel

```python
import numpy as np
from typing import Optional, List, Dict
from orthos.core.base import HierarchicalLevel
from orthos.core.types import Tensor

class HierarchicalLevel(HierarchicalLevel):
    """
    Implementation of a hierarchical level.

    Attributes:
        level_id: Level identifier
        input_size: Size of input features
        output_size: Size of output features
        temporal_resolution: Time compression factor
        parent_level: Reference to parent level
        child_levels: List of child levels
        processing_layers: List of processing layers
    """

    def __init__(self, level_id: int, input_size: int, output_size: int,
                 temporal_resolution: int = 1):
        self.level_id = level_id
        self.input_size = input_size
        self.output_size = output_size
        self.temporal_resolution = temporal_resolution
        self.parent_level = None
        self.child_levels: List['HierarchicalLevel'] = []
        self.processing_layers = []
        self.current_representation = None
        self.time_step = 0

    def add_layer(self, layer: Layer) -> None:
        """Add a processing layer to this level."""
        self.processing_layers.append(layer)

    def process_time_step(self, input_data: Tensor, t: int) -> Tensor:
        """
        Process a single time step.

        Args:
            input_data: Input data for this time step
            t: Global time step

        Returns:
            Processed output
        """
        # Only process every temporal_resolution time steps
        if t % self.temporal_resolution != 0:
            return self.current_representation

        # Process through all layers
        output = input_data
        for layer in self.processing_layers:
            output = layer.forward(output)

        # Store current representation
        self.current_representation = output
        self.time_step = t

        return output

    def get_representation(self) -> Tensor:
        """Get current representation."""
        return self.current_representation

    def communicate_with_parent(self) -> Optional[Tensor]:
        """
        Communicate with parent level.

        Returns:
            Data to send to parent, or None
        """
        if self.parent_level is None:
            return None

        # TODO: Implement communication protocol
        return self.current_representation

    def communicate_with_children(self, data: Tensor) -> None:
        """
        Communicate with child levels.

        Args:
            data: Data received from parent
        """
        # TODO: Implement communication protocol
        pass

    def reset_state(self) -> None:
        """Reset internal state."""
        for layer in self.processing_layers:
            layer.reset_state()
        self.current_representation = None
        self.time_step = 0
```

### `hierarchy/manager.py` - HierarchyManager

```python
from typing import List, Dict, Optional
import numpy as np
from orthos.core.types import Tensor
from orthos.hierarchy.level import HierarchicalLevel

class HierarchyManager:
    """
    Manages multiple hierarchical levels.

    Attributes:
        levels: List of hierarchical levels
        communication_schedule: Communication timing
    """

    def __init__(self):
        self.levels: List[HierarchicalLevel] = []
        self.communication_schedule = {}

    def add_level(self, level: HierarchicalLevel) -> None:
        """Add a level to the hierarchy."""
        self.levels.append(level)

        # Sort levels by level_id
        self.levels.sort(key=lambda x: x.level_id)

        # Update parent/child relationships
        self._update_hierarchy_relationships()

    def _update_hierarchy_relationships(self) -> None:
        """Update parent/child relationships between levels."""
        for i, level in enumerate(self.levels):
            # Set parent (level above)
            if i > 0:
                level.parent_level = self.levels[i-1]

            # Set children (levels below)
            level.child_levels = [self.levels[j] for j in range(i+1, len(self.levels))
                                if self.levels[j].level_id == level.level_id + 1]

    def process_hierarchy(self, input_data: Tensor, time_steps: int) -> Dict[int, List[Tensor]]:
        """
        Process input through the entire hierarchy.

        Args:
            input_data: Input data sequence (time_steps, input_size)
            time_steps: Number of time steps to process

        Returns:
            Dictionary of representations at each level
        """
        representations = {level.level_id: [] for level in self.levels}

        for t in range(time_steps):
            current_input = input_data[t]

            # Process through each level
            for level in self.levels:
                if level.level_id == 0:
                    # Input level
                    output = level.process_time_step(current_input, t)
                else:
                    # Higher levels
                    output = level.process_time_step(output, t)

                representations[level.level_id].append(output)

            # Hierarchical communication
            self._hierarchical_communication(t)

        return representations

    def _hierarchical_communication(self, t: int) -> None:
        """Handle communication between hierarchical levels."""
        # TODO: Implement communication protocol
        pass

    def get_all_representations(self) -> Dict[int, Tensor]:
        """
        Get current representations from all levels.

        Returns:
            Dictionary of current representations
        """
        return {level.level_id: level.get_representation() for level in self.levels}

    def reset_state(self) -> None:
        """Reset state of all levels."""
        for level in self.levels:
            level.reset_state()
```

## 📦 Plasticity Components

### `plasticity/controller.py` - PlasticityController

```python
import numpy as np
from typing import List, Dict, Optional
from orthos.core.base import PlasticComponent
from orthos.core.types import Tensor, PlasticityParams

class PlasticityController:
    """
    Controls plasticity parameters using Evolutionary Strategy.

    Attributes:
        target_modules: List of modules to control
        es_optimizer: Evolutionary Strategy optimizer
        plasticity_params: Current plasticity parameters
        adaptation_rate: Rate of parameter adaptation
        exploration_noise: Noise for exploration
    """

    def __init__(self, target_modules: List[PlasticComponent],
                 adaptation_rate: float = 0.01, exploration_noise: float = 0.1):
        self.target_modules = target_modules
        self.adaptation_rate = adaptation_rate
        self.exploration_noise = exploration_noise

        # Initialize ES optimizer
        from orthos.plasticity.es_optimizer import EvolutionaryStrategy
        self.es_optimizer = EvolutionaryStrategy()

        # Initialize plasticity parameters
        self.plasticity_params = self._initialize_params()

    def _initialize_params(self) -> Tensor:
        """Initialize plasticity parameters."""
        # Get parameter dimensions from target modules
        param_dims = sum(len(module.get_plasticity_params()) for module in self.target_modules)

        # Initialize with reasonable defaults
        initial_params = np.ones(param_dims) * 0.01

        return initial_params

    def adapt_plasticity(self, performance_metric: float) -> None:
        """
        Adapt plasticity parameters based on performance.

        Args:
            performance_metric: Current performance metric
        """
        # Sample perturbed parameters
        perturbed_params = self.es_optimizer.generate_population(self.plasticity_params)

        # Evaluate fitness of perturbed parameters
        fitness_scores = []
        for params in perturbed_params:
            # Apply parameters temporarily
            self._apply_params_temporarily(params)

            # Evaluate performance (simplified for now)
            fitness = self._evaluate_performance()
            fitness_scores.append(fitness)

        # Update parameters based on fitness
        self.es_optimizer.update_mean(self.plasticity_params, perturbed_params, fitness_scores)
        self.plasticity_params = self.es_optimizer.get_mean()

        # Apply updated parameters
        self._apply_params()

    def _apply_params_temporarily(self, params: Tensor) -> None:
        """Temporarily apply parameters for evaluation."""
        # TODO: Implement temporary parameter application
        pass

    def _apply_params(self) -> None:
        """Apply current parameters to target modules."""
        param_index = 0
        for module in self.target_modules:
            module_params = module.get_plasticity_params()
            num_params = len(module_params)

            # Update each parameter
            for i, param_name in enumerate(module_params.keys()):
                module_params[param_name] = self.plasticity_params[param_index]
                param_index += 1

            module.set_plasticity_params(module_params)

    def _evaluate_performance(self) -> float:
        """Evaluate current performance."""
        # TODO: Implement proper performance evaluation
        return np.random.random()  # Placeholder

    def get_current_params(self) -> Tensor:
        """Get current plasticity parameters."""
        return self.plasticity_params.copy()
```

### `plasticity/es_optimizer.py` - EvolutionaryStrategy

```python
import numpy as np
from typing import List, Tuple

class EvolutionaryStrategy:
    """
    Evolutionary Strategy optimizer for plasticity parameters.

    Attributes:
        population_size: Number of individuals in population
        sigma: Mutation strength
        learning_rate: Learning rate for mean update
        elite_fraction: Fraction of elites to select
    """

    def __init__(self, population_size: int = 50, sigma: float = 0.1,
                 learning_rate: float = 0.01, elite_fraction: float = 0.2):
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.elite_fraction = elite_fraction
        self.mean = None

    def generate_population(self, initial_mean: np.ndarray) -> List[np.ndarray]:
        """
        Generate population of perturbed parameters.

        Args:
            initial_mean: Initial parameter vector

        Returns:
            List of perturbed parameter vectors
        """
        self.mean = initial_mean.copy()
        population = []

        for _ in range(self.population_size):
            # Sample from multivariate normal distribution
            perturbation = np.random.randn(*self.mean.shape) * self.sigma
            perturbed_params = self.mean + perturbation
            population.append(perturbed_params)

        return population

    def update_mean(self, current_mean: np.ndarray,
                   population: List[np.ndarray],
                   fitness_scores: List[float]) -> None:
        """
        Update mean based on fitness scores.

        Args:
            current_mean: Current mean parameters
            population: List of parameter vectors
            fitness_scores: Corresponding fitness scores
        """
        # Select elites
        elite_indices = self._select_elites(fitness_scores)
        elites = [population[i] for i in elite_indices]

        # Update mean toward elites
        self.mean = current_mean + self.learning_rate * np.mean(elites - current_mean, axis=0)

    def _select_elites(self, fitness_scores: List[float]) -> List[int]:
        """Select elite individuals based on fitness."""
        # Get indices of top performers
        num_elites = int(self.population_size * self.elite_fraction)
        elite_indices = np.argsort(fitness_scores)[-num_elites:]

        return list(elite_indices)

    def get_mean(self) -> np.ndarray:
        """Get current mean parameters."""
        return self.mean.copy()

    def adapt_sigma(self, fitness_improvement: float) -> None:
        """
        Adapt mutation strength based on fitness improvement.

        Args:
            fitness_improvement: Improvement in fitness
        """
        # Simple adaptation rule
        if fitness_improvement > 0:
            self.sigma *= 1.1  # Increase exploration
        else:
            self.sigma *= 0.9  # Decrease exploration
```

### `plasticity/rules.py` - Plasticity Rules

```python
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

class PlasticityRule(ABC):
    """Base class for plasticity rules."""

    @abstractmethod
    def apply(self, weights: np.ndarray, pre_activity: np.ndarray,
              post_activity: np.ndarray) -> np.ndarray:
        """Apply plasticity rule to weights."""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, float]:
        """Get rule parameters."""
        pass

class HebbianRule(PlasticityRule):
    """Classic Hebbian learning rule: Δw = η * pre * post"""

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def apply(self, weights: np.ndarray, pre_activity: np.ndarray,
              post_activity: np.ndarray) -> np.ndarray:
        """
        Apply Hebbian learning rule.

        Args:
            weights: Current weight matrix
            pre_activity: Pre-synaptic activity
            post_activity: Post-synaptic activity

        Returns:
            Updated weights
        """
        weight_update = self.learning_rate * np.outer(post_activity, pre_activity)
        return weights + weight_update

    def get_parameters(self) -> Dict[str, float]:
        """Get rule parameters."""
        return {'learning_rate': self.learning_rate}

class OjasRule(PlasticityRule):
    """Oja's learning rule: Δw = η * post * (pre - post * w)"""

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def apply(self, weights: np.ndarray, pre_activity: np.ndarray,
              post_activity: np.ndarray) -> np.ndarray:
        """
        Apply Oja's learning rule.

        Args:
            weights: Current weight matrix
            pre_activity: Pre-synaptic activity
            post_activity: Post-synaptic activity

        Returns:
            Updated weights
        """
        weight_update = self.learning_rate * np.outer(
            post_activity,
            pre_activity - np.dot(weights.T, post_activity)
        )
        return weights + weight_update

    def get_parameters(self) -> Dict[str, float]:
        """Get rule parameters."""
        return {'learning_rate': self.learning_rate}

class BCMRule(PlasticityRule):
    """Bienenstock-Cooper-Munro rule with sliding threshold."""

    def __init__(self, learning_rate: float = 0.01, theta: float = 1.0):
        self.learning_rate = learning_rate
        self.theta = theta

    def apply(self, weights: np.ndarray, pre_activity: np.ndarray,
              post_activity: np.ndarray) -> np.ndarray:
        """
        Apply BCM learning rule.

        Args:
            weights: Current weight matrix
            pre_activity: Pre-synaptic activity
            post_activity: Post-synaptic activity

        Returns:
            Updated weights
        """
        # Update threshold based on post-synaptic activity
        self.theta = 0.9 * self.theta + 0.1 * np.mean(post_activity)

        weight_update = self.learning_rate * np.outer(
            post_activity * (post_activity - self.theta),
            pre_activity
        )
        return weights + weight_update

    def get_parameters(self) -> Dict[str, float]:
        """Get rule parameters."""
        return {'learning_rate': self.learning_rate, 'theta': self.theta}
```

## 📦 Meta-Learning Components

### `meta_learning/optimizer.py` - MetaOptimizer

```python
import numpy as np
from typing import List, Dict, Callable
from orthos.core.types import Tensor

class MetaOptimizer:
    """
    Meta-optimizer for plasticity learning.

    Attributes:
        inner_loop: Plasticity controller for inner optimization
        outer_loop: Outer optimization algorithm
        task_distribution: Distribution of tasks
        learning_history: History of learning performance
    """

    def __init__(self, plasticity_controller, outer_optimizer='adam'):
        self.inner_loop = plasticity_controller
        self.outer_optimizer = outer_optimizer
        self.task_distribution = None
        self.learning_history = []

    def meta_train(self, num_episodes: int, tasks: List[Callable]) -> None:
        """
        Perform meta-training.

        Args:
            num_episodes: Number of training episodes
            tasks: List of task functions
        """
        for episode in range(num_episodes):
            # Sample a task
            task = np.random.choice(tasks)

            # Inner loop: adapt to task
            task_performance = self.inner_loop_adaptation(task)

            # Outer loop: update meta-parameters
            self.outer_update(task_performance)

            # Record learning history
            self.learning_history.append(task_performance)

    def inner_loop_adaptation(self, task: Callable) -> float:
        """
        Inner loop adaptation to a specific task.

        Args:
            task: Task function

        Returns:
            Task performance
        """
        # Reset plasticity controller
        self.inner_loop.reset_state()

        # Adapt to task
        performance = 0.0
        for step in range(10):  # Fixed number of adaptation steps
            task_data = task(step)
            performance += self.inner_loop.adapt_plasticity(task_data)

        return performance / 10  # Average performance

    def outer_update(self, performance: float) -> None:
        """
        Outer loop update of meta-parameters.

        Args:
            performance: Task performance
        """
        # TODO: Implement meta-parameter update
        pass

    def evaluate_meta_performance(self) -> Dict[str, float]:
        """
        Evaluate meta-learning performance.

        Returns:
            Dictionary of performance metrics
        """
        if not self.learning_history:
            return {'average_performance': 0.0, 'improvement': 0.0}

        metrics = {
            'average_performance': np.mean(self.learning_history),
            'improvement': self.learning_history[-1] - self.learning_history[0],
            'stability': np.std(self.learning_history[-10:]) if len(self.learning_history) >= 10 else 0.0
        }

        return metrics
```

### `meta_learning/metrics.py` - Performance Metrics

```python
import numpy as np
from typing import Dict, List
from orthos.core.base import Module

def measure_plasticity_efficiency(module: Module) -> float:
    """
    Measure plasticity efficiency of a module.

    Args:
        module: Module to evaluate

    Returns:
        Plasticity efficiency score (0-1)
    """
    # TODO: Implement proper efficiency measurement
    return np.random.random()

def measure_adaptation_speed(module: Module) -> float:
    """
    Measure adaptation speed of a module.

    Args:
        module: Module to evaluate

    Returns:
        Adaptation speed score (0-1)
    """
    # TODO: Implement proper speed measurement
    return np.random.random()

def measure_stability_plasticity_tradeoff(module: Module) -> float:
    """
    Measure stability-plasticity tradeoff.

    Args:
        module: Module to evaluate

    Returns:
        Tradeoff score (0-1, higher is better)
    """
    # TODO: Implement proper tradeoff measurement
    return np.random.random()

def evaluate_meta_learning_curve(learning_history: List[float]) -> Dict[str, float]:
    """
    Evaluate meta-learning curve.

    Args:
        learning_history: History of learning performance

    Returns:
        Dictionary of evaluation metrics
    """
    if len(learning_history) < 2:
        return {
            'convergence_rate': 0.0,
            'final_performance': 0.0,
            'improvement': 0.0,
            'stability': 0.0
        }

    metrics = {
        'convergence_rate': (learning_history[-1] - learning_history[0]) / len(learning_history),
        'final_performance': learning_history[-1],
        'improvement': learning_history[-1] - learning_history[0],
        'stability': np.std(learning_history[-10:]) if len(learning_history) >= 10 else 0.0
    }

    return metrics
```

## 📦 Configuration & Utilities

### `config/defaults.py` - Default Configurations

```python
# Default hierarchy configuration
DEFAULT_HIERARCHY_CONFIG = {
    "num_levels": 4,
    "temporal_compression": 2,
    "base_resolution": 1,
    "level_sizes": [64, 128, 256, 512]
}

# Default plasticity configuration
DEFAULT_PLASTICITY_CONFIG = {
    "learning_rate": 0.01,
    "ltp_coefficient": 1.0,
    "ltd_coefficient": 0.8,
    "decay_rate": 0.001,
    "homeostatic_strength": 0.1
}

# Default ES configuration
DEFAULT_ES_CONFIG = {
    "population_size": 50,
    "sigma": 0.1,
    "learning_rate": 0.01,
    "elite_fraction": 0.2
}

# Default layer configurations
DEFAULT_LAYER_CONFIGS = {
    "reactive": {
        "activation": "relu",
        "init_type": "he"
    },
    "hebbian": {
        "plasticity_rule": "hebbian",
        "params": DEFAULT_PLASTICITY_CONFIG
    },
    "temporal": {
        "activation": "tanh",
        "time_window": 10
    }
}
```

### `utils/logging.py` - Logging Utilities

```python
import logging
from typing import Optional, Dict
import numpy as np

def setup_logging(name: str = 'orthos', level: str = 'INFO',
                 log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging for ORTHOS.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Create formatter
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

def log_tensor_stats(logger: logging.Logger, tensor: np.ndarray,
                    name: str, level: str = 'DEBUG') -> None:
    """
    Log statistics about a tensor.

    Args:
        logger: Logger instance
        tensor: Tensor to log
        name: Tensor name
        level: Logging level
    """
    stats = {
        'shape': tensor.shape,
        'mean': np.mean(tensor),
        'std': np.std(tensor),
        'min': np.min(tensor),
        'max': np.max(tensor)
    }

    getattr(logger, level.lower())(f"Tensor {name} stats: {stats}")

def log_plasticity_update(logger: logging.Logger, params: Dict[str, float],
                         performance: float, step: int) -> None:
    """
    Log plasticity parameter update.

    Args:
        logger: Logger instance
        params: Plasticity parameters
        performance: Current performance
        step: Training step
    """
    logger.info(f"Step {step}: Performance = {performance:.4f}, Params = {params}")
```

### `utils/visualization.py` - Visualization Tools

```python
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional

def plot_hierarchy_representations(representations: Dict[int, List[np.ndarray]],
                                 title: str = "Hierarchical Representations") -> None:
    """
    Plot representations from different hierarchical levels.

    Args:
        representations: Dictionary of representations by level
        title: Plot title
    """
    plt.figure(figsize=(12, 8))

    for level, reps in representations.items():
        # Average representation over time
        avg_rep = np.mean(reps, axis=0)

        plt.subplot(2, 2, level + 1)
        plt.imshow(avg_rep.reshape(1, -1), aspect='auto')
        plt.title(f"Level {level}")
        plt.colorbar()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_learning_curve(learning_history: List[float],
                       title: str = "Learning Curve") -> None:
    """
    Plot learning curve.

    Args:
        learning_history: History of learning performance
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(learning_history)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Performance")
    plt.grid(True)
    plt.show()

def plot_weight_matrix(weights: np.ndarray, title: str = "Weight Matrix") -> None:
    """
    Plot weight matrix.

    Args:
        weights: Weight matrix
        title: Plot title
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(weights, cmap='viridis', aspect='auto')
    plt.title(title)
    plt.colorbar()
    plt.show()

def plot_plasticity_parameters(params_history: List[Dict[str, float]],
                             title: str = "Plasticity Parameters") -> None:
    """
    Plot plasticity parameter evolution.

    Args:
        params_history: History of plasticity parameters
        title: Plot title
    """
    if not params_history:
        return

    param_names = list(params_history[0].keys())
    num_params = len(param_names)

    plt.figure(figsize=(12, 6))
    for i, param_name in enumerate(param_names):
        values = [params[param_name] for params in params_history]
        plt.plot(values, label=param_name)

    plt.title(title)
    plt.xlabel("Update Step")
    plt.ylabel("Parameter Value")
    plt.legend()
    plt.grid(True)
    plt.show()
```

## 📦 Examples

### `examples/basic_demo.py` - Basic Usage Example

```python
import numpy as np
from orthos.core.base import Module
from orthos.layers.reactive import ReactiveLayer
from orthos.layers.hebbian import HebbianCore
from orthos.layers.temporal import TemporalLayer
from orthos.hierarchy.level import HierarchicalLevel
from orthos.hierarchy.manager import HierarchyManager

def basic_demo():
    """Basic demonstration of ORTHOS components."""
    print("ORTHOS Basic Demo")
    print("=" * 50)

    # Create a simple hierarchy
    manager = HierarchyManager()

    # Level 0: Input level
    level0 = HierarchicalLevel(0, input_size=10, output_size=20, temporal_resolution=1)
    level0.add_layer(ReactiveLayer(10, 20, activation='relu'))
    manager.add_level(level0)

    # Level 1: Intermediate level
    level1 = HierarchicalLevel(1, input_size=20, output_size=40, temporal_resolution=2)
    level1.add_layer(HebbianCore(20, 40, plasticity_rule='hebbian'))
    manager.add_level(level1)

    # Level 2: High-level
    level2 = HierarchicalLevel(2, input_size=40, output_size=80, temporal_resolution=4)
    level2.add_layer(TemporalLayer(40, 80, time_window=5))
    manager.add_level(level2)

    # Generate some random input data
    time_steps = 20
    input_data = np.random.randn(time_steps, 10)

    # Process through hierarchy
    print("Processing input through hierarchy...")
    representations = manager.process_hierarchy(input_data, time_steps)

    # Display results
    print("\nRepresentations:")
    for level_id, reps in representations.items():
        print(f"Level {level_id}: {len(reps)} representations, shape {reps[0].shape}")

    print("\nDemo completed successfully!")

if __name__ == "__main__":
    basic_demo()
```

## 📦 Main Package Structure

### `__init__.py` Files

```python
# orthos/__init__.py
from .core import *
from .layers import *
from .hierarchy import *
from .plasticity import *
from .meta_learning import *
from .utils import *
from .config import *

# orthos/core/__init__.py
from .base import Module, Layer, PlasticComponent, HierarchicalLevel
from .types import *
from .tensor import *

# orthos/layers/__init__.py
from .reactive import ReactiveLayer
from .hebbian import HebbianCore
from .temporal import TemporalLayer

# orthos/hierarchy/__init__.py
from .level import HierarchicalLevel
from .manager import HierarchyManager

# orthos/plasticity/__init__.py
from .controller import PlasticityController
from .es_optimizer import EvolutionaryStrategy
from .rules import *

# orthos/meta_learning/__init__.py
from .optimizer import MetaOptimizer
from .metrics import *

# orthos/utils/__init__.py
from .logging import *
from .visualization import *

# orthos/config/__init__.py
from .defaults import *
```

## 🎯 Implementation Notes

### Type Hints
- All functions and methods include comprehensive type hints
- Custom type aliases for better code clarity
- Type checking can be enabled with mypy

### Error Handling
- Input validation for all public methods
- Clear error messages for debugging
- Graceful fallbacks where appropriate

### Testing
- Each component has clear interfaces for unit testing
- Empty methods marked with TODO for implementation
- Example usage provided for each major component

### Performance
- Vectorized operations using numpy
- Efficient memory usage patterns
- Minimal computational overhead

## 🔮 Next Steps

The v4.2 architecture is now the stable production baseline. Future development will focus on:

1. **Advanced Optimization**: Integrating CMA-ES for more robust meta-learning
2. **Structural Plasticity**: Sparse attention and dynamic topology
3. **Multi-Modal Hubs**: Cross-level integration of visual/auditory/tactile streams

This architecture provides a solid foundation for ORTHOS's journey toward artificial general intelligence!

# Plasticity System

## 🎯 Plasticity Overview

The plasticity system is the heart of ORTHOS's learning capability, enabling adaptive behavior through meta-learning of plasticity parameters using Evolutionary Strategies.

## 🏗️ Plasticity Architecture

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                            PlasticityController                             │
├───────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                        EvolutionaryStrategy                            │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                     Population Generation                           │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                     Fitness Evaluation                              │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                     Parameter Update                                │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                        Plasticity Rules                                │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                     HebbianRule                                     │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                     OjasRule                                        │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                     BCMRule                                         │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                        Target Modules                                  │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                     HebbianCore                                     │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                     TemporalLayer                                   │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Plasticity Control Flow

1. **Performance Evaluation**: Measure current system performance
2. **Parameter Perturbation**: Generate population of perturbed parameters
3. **Fitness Assessment**: Evaluate each perturbation's impact
4. **Elite Selection**: Select top-performing parameter sets
5. **Parameter Update**: Update mean parameters toward elites
6. **Application**: Apply new parameters to target modules

## 📦 PlasticityController

### Core Functionality

```python
class PlasticityController:
    def __init__(self, target_modules, adaptation_rate=0.01, exploration_noise=0.1):
        self.target_modules = target_modules  # Modules to control
        self.es_optimizer = EvolutionaryStrategy()  # ES optimizer
        self.plasticity_params = self._initialize_params()  # Current parameters

    def adapt_plasticity(self, performance_metric):
        """Main adaptation loop."""
        # 1. Generate perturbed parameters
        # 2. Evaluate fitness
        # 3. Update parameters
        # 4. Apply to modules

    def _apply_params(self):
        """Apply parameters to target modules."""
```

### Key Features

1. **Meta-Learning**: Learns optimal plasticity parameters
2. **Evolutionary Optimization**: Uses ES for parameter search
3. **Dynamic Adaptation**: Continuously adapts to task requirements
4. **Multi-Module Control**: Manages multiple target modules

## 📦 EvolutionaryStrategy

### Core Algorithm

```python
class EvolutionaryStrategy:
    def __init__(self, population_size=50, sigma=0.1,
                 learning_rate=0.01, elite_fraction=0.2):
        self.population_size = population_size
        self.sigma = sigma  # Mutation strength
        self.learning_rate = learning_rate
        self.elite_fraction = elite_fraction

    def generate_population(self, initial_mean):
        """Generate population through Gaussian perturbation."""
        population = []
        for _ in range(self.population_size):
            perturbation = np.random.randn(*initial_mean.shape) * self.sigma
            population.append(initial_mean + perturbation)
        return population

    def update_mean(self, current_mean, population, fitness_scores):
        """Update mean toward elite performers."""
        elites = self._select_elites(fitness_scores)
        self.mean = current_mean + self.learning_rate * np.mean(elites - current_mean, axis=0)
```

### Evolutionary Parameters

- **Population Size**: Number of parameter perturbations (50)
- **Sigma (σ)**: Mutation strength (0.1)
- **Learning Rate**: Mean update rate (0.01)
- **Elite Fraction**: Fraction of top performers to select (0.2)

## 📦 Plasticity Rules

### HebbianRule

```python
class HebbianRule(PlasticityRule):
    """Δw = η * pre * post"""

    def apply(self, weights, pre_activity, post_activity):
        weight_update = self.learning_rate * np.outer(post_activity, pre_activity)
        return weights + weight_update
```

### OjasRule

```python
class OjasRule(PlasticityRule):
    """Δw = η * post * (pre - post * w)"""

    def apply(self, weights, pre_activity, post_activity):
        weight_update = self.learning_rate * np.outer(
            post_activity,
            pre_activity - np.dot(weights.T, post_activity)
        )
        return weights + weight_update
```

### BCMRule

```python
class BCMRule(PlasticityRule):
    """Bienenstock-Cooper-Munro rule with sliding threshold."""

    def apply(self, weights, pre_activity, post_activity):
        self.theta = 0.9 * self.theta + 0.1 * np.mean(post_activity)
        weight_update = self.learning_rate * np.outer(
            post_activity * (post_activity - self.theta),
            pre_activity
        )
        return weights + weight_update
```

## 🔧 Configuration

### Default Plasticity Configuration

```python
DEFAULT_PLASTICITY_CONFIG = {
    "learning_rate": 0.01,
    "ltp_coefficient": 1.0,      # Long-Term Potentiation
    "ltd_coefficient": 0.8,      # Long-Term Depression
    "decay_rate": 0.001,         # Weight decay
    "homeostatic_strength": 0.1  # Homeostatic regulation
}

DEFAULT_ES_CONFIG = {
    "population_size": 50,
    "sigma": 0.1,
    "learning_rate": 0.01,
    "elite_fraction": 0.2
}
```

### Parameter Ranges

| Parameter | Range | Description |
|-----------|-------|-------------|
| learning_rate | [0.001, 0.1] | Base learning rate |
| ltp_coefficient | [0.5, 2.0] | LTP strength |
| ltd_coefficient | [0.1, 1.0] | LTD strength |
| decay_rate | [0.0001, 0.01] | Weight decay rate |
| homeostatic_strength | [0.01, 0.5] | Homeostatic regulation |

## 📊 Adaptation Process

### Parameter Evolution

```
Initial Parameters → Perturbation → Evaluation → Selection → Update → New Parameters
```

### Fitness Landscape

- **Performance Metrics**: Task accuracy, adaptation speed, stability
- **Multi-Objective Optimization**: Balance multiple performance aspects
- **Dynamic Fitness**: Adaptive fitness functions based on task requirements

## 🎯 Design Principles

### Meta-Learning
- Learn optimal plasticity parameters
- Adapt to different task distributions
- Continuous improvement over time

### Evolutionary Optimization
- Population-based search
- Parallel evaluation
- Robust to local optima

### Modular Control
- Control multiple modules simultaneously
- Module-specific parameter sets
- Dynamic module registration

## 🔮 Future Enhancements

### v4.1 Features
- **Advanced ES Variants**: CMA-ES, Natural Evolution Strategies
- **Multi-Objective Optimization**: Pareto front optimization
- **Adaptive Population Sizing**: Dynamic population adjustment

### v4.2 Features
- **Neuroevolution**: Direct neural architecture evolution
- **Plasticity Rule Discovery**: Automatic rule generation
- **Transfer Learning**: Cross-task plasticity adaptation

## 📋 Implementation Checklist

- [x] PlasticityController base class
- [x] EvolutionaryStrategy implementation
- [x] Basic plasticity rules (Hebbian, Oja, BCM)
- [x] Parameter application mechanism
- [ ] Advanced ES variants
- [ ] Multi-objective optimization
- [ ] Neuroevolution capabilities

## 🎯 Usage Example

```python
# Create target modules
hebbian_core = HebbianCore(input_size=20, output_size=40)
temporal_layer = TemporalLayer(input_size=40, output_size=80)

# Create plasticity controller
controller = PlasticityController(
    target_modules=[hebbian_core, temporal_layer],
    adaptation_rate=0.01,
    exploration_noise=0.1
)

# Adaptation loop
for episode in range(100):
    # Run task and get performance
    performance = run_task()

    # Adapt plasticity parameters
    controller.adapt_plasticity(performance)

    # Log progress
    print(f"Episode {episode}: Performance = {performance:.4f}")
```

## 📊 Performance Monitoring

### Key Metrics

1. **Adaptation Speed**: Time to reach target performance
2. **Stability**: Variance in performance over time
3. **Plasticity Efficiency**: Learning rate vs. forgetting rate
4. **Parameter Convergence**: Stability of plasticity parameters

### Visualization

```python
# Plot parameter evolution
params_history = controller.get_parameter_history()
plot_plasticity_parameters(params_history)

# Plot performance curve
plot_learning_curve(performance_history)
```

This plasticity system provides ORTHOS with powerful meta-learning capabilities for adaptive behavior!# Hierarchy System

## 🎯 Hierarchical Processing Overview

The hierarchy system is the core of ORTHOS's temporal abstraction capability, enabling processing at multiple time scales and levels of abstraction.

## 🏗️ Hierarchy Architecture

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                            HierarchyManager                                │
├───────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                         Level 3 (High-level)                         │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                     TemporalLayer (8x)                         │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                         Level 2 (Intermediate)                       │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                     TemporalLayer (4x)                         │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                         Level 1 (Low-level)                          │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                     HebbianCore (2x)                           │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                         Level 0 (Input)                              │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                     ReactiveLayer (1x)                          │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Temporal Abstraction Mechanism

### Time Compression
- Each level processes input at a different temporal resolution
- Level 0: 1x (every time step)
- Level 1: 2x (every 2nd time step)
- Level 2: 4x (every 4th time step)
- Level 3: 8x (every 8th time step)

### Information Flow
```
Input → Level 0 → Level 1 → Level 2 → Level 3
  ↓       ↓        ↓         ↓
Raw    Features  Sequences  Concepts
```

## 📦 HierarchyManager

### Core Functionality

```python
class HierarchyManager:
    def __init__(self):
        self.levels = []  # List of HierarchicalLevel instances
        self.communication_schedule = {}

    def add_level(self, level: HierarchicalLevel):
        """Add a level to the hierarchy with automatic relationship management."""

    def process_hierarchy(self, input_data, time_steps):
        """Process input through all levels with temporal abstraction."""

    def _hierarchical_communication(self, t):
        """Handle bidirectional communication between levels."""
```

### Key Features

1. **Automatic Relationship Management**: Automatically sets parent/child relationships
2. **Temporal Processing**: Handles different temporal resolutions
3. **Communication Protocol**: Manages inter-level communication
4. **State Management**: Coordinates state across all levels

## 📦 HierarchicalLevel

### Core Functionality

```python
class HierarchicalLevel:
    def __init__(self, level_id, input_size, output_size, temporal_resolution=1):
        self.level_id = level_id
        self.input_size = input_size
        self.output_size = output_size
        self.temporal_resolution = temporal_resolution
        self.parent_level = None
        self.child_levels = []
        self.processing_layers = []

    def process_time_step(self, input_data, t):
        """Process input only at appropriate temporal resolution."""

    def communicate_with_parent(self):
        """Send representation to parent level."""

    def communicate_with_children(self, data):
        """Receive data from parent and distribute to children."""
```

### Processing Flow

1. **Temporal Filtering**: Only process every `temporal_resolution` time steps
2. **Layer Processing**: Sequential processing through all layers
3. **Representation Storage**: Maintain current representation
4. **Communication**: Bidirectional communication with adjacent levels

## 🔧 Configuration

### Default Hierarchy Configuration

```python
DEFAULT_HIERARCHY_CONFIG = {
    "num_levels": 4,
    "temporal_compression": 2,  # Each level compresses time by this factor
    "base_resolution": 1,       # Base temporal resolution
    "level_sizes": [64, 128, 256, 512]  # Feature sizes at each level
}
```

### Custom Configuration Example

```python
custom_config = {
    "num_levels": 3,
    "temporal_compression": 3,  # More aggressive compression
    "base_resolution": 1,
    "level_sizes": [128, 256, 512],
    "communication_interval": 5  # Communicate every 5 time steps
}
```

## 📊 Communication Protocols

### Bidirectional Communication

1. **Bottom-Up**: Raw data → Features → Sequences → Concepts
2. **Top-Down**: Contextual information → Expectations → Attention

### Communication Timing

- **Synchronous**: All levels communicate at each time step
- **Asynchronous**: Levels communicate based on temporal resolution
- **Scheduled**: Communication at specific intervals

## 🎯 Design Principles

### Modularity
- Each level operates independently
- Clear interfaces between levels
- Easy to add/remove levels

### Scalability
- Linear scaling with number of levels
- Efficient memory usage
- Parallel processing capabilities

### Flexibility
- Configurable temporal resolutions
- Customizable layer compositions
- Adaptable communication protocols

## 🔮 Future Enhancements

### v4.1 Features
- **Attention Mechanisms**: Selective focus at different levels
- **Dynamic Hierarchies**: Adaptive level creation/destruction
- **Cross-Level Learning**: Shared learning across levels

### v4.2 Features
- **Multi-Modal Hierarchies**: Separate hierarchies for different modalities
- **Hierarchy Optimization**: Automatic level configuration
- **Memory Systems**: Long-term memory integration

## 📋 Implementation Checklist

- [x] HierarchyManager base class
- [x] HierarchicalLevel implementation
- [x] Temporal abstraction mechanism
- [x] Basic communication protocol
- [ ] Advanced communication strategies
- [ ] Attention mechanisms
- [ ] Dynamic hierarchy management

## 🎯 Usage Example

```python
# Create hierarchy manager
manager = HierarchyManager()

# Add levels with different temporal resolutions
level0 = HierarchicalLevel(0, input_size=10, output_size=20, temporal_resolution=1)
level0.add_layer(ReactiveLayer(10, 20))
manager.add_level(level0)

level1 = HierarchicalLevel(1, input_size=20, output_size=40, temporal_resolution=2)
level1.add_layer(HebbianCore(20, 40))
manager.add_level(level1)

# Process input data
input_data = np.random.randn(100, 10)  # 100 time steps, 10 features
representations = manager.process_hierarchy(input_data, 100)

# Access results
level0_reps = representations[0]  # 100 representations
level1_reps = representations[1]  # 50 representations (every 2nd step)
```

This hierarchy system provides the foundation for ORTHOS's temporal abstraction and multi-scale processing capabilities!# Advanced Plasticity System

## 🧬 Overview

ORTHOS's advanced plasticity system implements **dual-timescale Hebbian learning** with **BitNet quantization** and **diagnostic tracking**. This document details the PyTorch implementation (v3.1) which extends the conceptual framework of the NumPy-based v4.x architecture.

---

## 1. DiagnosticPlasticLinear

### 1.1 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DiagnosticPlasticLinear                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Input x ──────────┬────────────────────────────────┐             │
│                     │                                 │             │
│                     ▼                                 ▼             │
│              ┌─────────────┐                   ┌───────────┐        │
│              │  W_static   │                   │  Hebbian  │        │
│              │ (learnable) │                   │  Update   │        │
│              └──────┬──────┘                   └─────┬─────┘        │
│                     │                                 │             │
│                     ▼                                 ▼             │
│              ┌─────────────┐                   ┌───────────┐        │
│              │   BitNet    │                   │ fast_trace │        │
│              │  Quantize   │                   │  (τ=0.95)  │        │
│              └──────┬──────┘                   └─────┬─────┘        │
│                     │                                 │             │
│                     │                                 ▼             │
│                     │                          ┌───────────┐        │
│                     │                          │ slow_trace │        │
│                     │                          │  (τ=0.99)  │        │
│                     │                          └─────┬─────┘        │
│                     │                                 │             │
│                     └────────────┬────────────────────┘             │
│                                  ▼                                  │
│                     ┌───────────────────────┐                       │
│                     │  W_eff = Q(W_static)  │                       │
│                     │  + 0.1*fast + 0.05*slow │                     │
│                     └───────────┬───────────┘                       │
│                                 │                                   │
│                                 ▼                                   │
│                     ┌───────────────────────┐                       │
│                     │   y = W_eff · x       │                       │
│                     └───────────────────────┘                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Implementation

```python
class DiagnosticPlasticLinear(nn.Module):
    """
    Enhanced Plastic Linear layer with:
    - BitNet 1.58-bit quantization
    - Dual-timescale Hebbian traces
    - Homeostatic regulation
    - Comprehensive diagnostics
    """
    
    def __init__(self, in_features: int, out_features: int, cfg: OrthosConfig):
        super().__init__()
        self.cfg = cfg
        
        # Statikus súlyok (Static weights)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        
        # Plasztikus nyomok (Plastic traces) - registered as buffers
        self.register_buffer("fast_trace", torch.zeros(out_features, in_features))
        self.register_buffer("slow_trace", torch.zeros(out_features, in_features))
        
        # Diagnostic tracking
        self.plasticity_enabled = True
        self.register_buffer("trace_norm_history", torch.zeros(cfg.trace_history_len))
        self.register_buffer("update_magnitude_history", torch.zeros(cfg.trace_history_len))
        self.step_counter = 0
```

### 1.3 BitNet Quantization

```python
def bitnet_quantize(self, w: torch.Tensor) -> torch.Tensor:
    """
    1.58-bit quantization: weights → {-1, 0, +1}
    
    Process:
    1. Compute per-row scaling factor (mean absolute value)
    2. Normalize weights by scale
    3. Round to nearest integer and clamp to [-1, 1]
    4. Re-scale for effective weight magnitudes
    
    Memory: 32 bits → 1.58 bits per weight
    Compute: Multiplications → Additions (hardware efficient)
    """
    scale = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
    w_normalized = w / scale
    w_quantized = w_normalized.round().clamp(-1, 1)
    return w_quantized * scale
```

### 1.4 Trace Update Mechanism

```python
def update_traces(self, x: torch.Tensor, y: torch.Tensor):
    """
    Hebbian trace update with homeostatic regulation
    
    Hungarian Comments Preserved:
    - Hebbian update: csak aktív neuronok (only active neurons)
    - Gyors nyom frissítése (Fast trace update)
    - Lassú nyom - konszolidáció (Slow trace - consolidation)
    """
    if not self.plasticity_enabled:
        return
    
    with torch.no_grad():
        # Hebbian update: csak aktív neuronok
        y_active = F.relu(y)
        delta = torch.matmul(y_active.t(), x) / x.shape[0]
        
        # Gyors nyom frissítése
        self.fast_trace.mul_(self.cfg.fast_trace_decay)      # τ_fast = 0.95
        self.fast_trace.add_(delta, alpha=self.cfg.fast_trace_lr)  # η_fast = 0.05
        
        # Homeostatic normalization
        fast_norm = self.fast_trace.norm()
        if fast_norm > self.cfg.homeostatic_target:  # target = 5.0
            self.fast_trace.mul_(self.cfg.homeostatic_target / (fast_norm + 1e-6))
        
        # Lassú nyom (konszolidáció)
        self.slow_trace.mul_(self.cfg.slow_trace_decay)      # τ_slow = 0.99
        self.slow_trace.add_(self.fast_trace, alpha=self.cfg.slow_trace_lr)  # η_slow = 0.01
```

### 1.5 Forward Pass

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Step 1: Quantize static weights (kvantált váz)
    w_static = self.bitnet_quantize(self.weight)
    
    # Step 2: Compute plastic modulation (plasztikus moduláció)
    w_plastic = 0.1 * self.fast_trace + 0.05 * self.slow_trace
    
    # Step 3: Combine for effective weight (effektív súly)
    w_effective = w_static + w_plastic
    
    # Step 4: Linear transformation
    y = F.linear(x, w_effective)
    
    # Step 5: Update traces (online learning)
    self.update_traces(x, y)
    
    return y
```

---

## 2. Configuration System

### 2.1 OrthosConfigEnhanced

```python
@dataclass
class OrthosConfigEnhanced:
    # === Core Dimensions ===
    state_dim: int = 1          # State space dimensionality
    action_dim: int = 1         # Action space dimensionality
    hidden_dim: int = 64        # Hidden layer size
    n_ensemble: int = 5         # Number of ensemble members
    
    # === Hebbian Plasticity ===
    fast_trace_decay: float = 0.95      # τ_fast: Fast trace decay
    fast_trace_lr: float = 0.05         # η_fast: Fast trace learning rate
    slow_trace_decay: float = 0.99      # τ_slow: Slow trace decay
    slow_trace_lr: float = 0.01         # η_slow: Slow trace learning rate
    homeostatic_target: float = 5.0     # H: Maximum trace norm
    
    # === Active Inference ===
    planning_samples: int = 30          # Number of action samples
    exploration_weight: float = 0.5     # β: Epistemic bonus weight
    temperature: float = 0.1            # τ: Action selection temperature
    
    # === Learning ===
    weight_scale: float = 5.0           # Policy update scaling
    wm_lr: float = 1e-3                 # World model learning rate
    policy_lr: float = 1e-3             # Policy learning rate
    
    # === Diagnostics ===
    track_metrics: bool = True          # Enable metric tracking
    trace_history_len: int = 1000       # History buffer size
```

### 2.2 Parameter Sensitivity

| Parameter | Range | Effect | Sensitivity |
|-----------|-------|--------|-------------|
| `fast_trace_decay` | 0.9-0.99 | Memory duration | High |
| `fast_trace_lr` | 0.01-0.1 | Adaptation speed | Medium |
| `homeostatic_target` | 1.0-10.0 | Stability vs. capacity | High |
| `exploration_weight` | 0.0-1.0 | Exploit vs. explore | Medium |
| `temperature` | 0.01-1.0 | Action selection sharpness | Low |

---

## 3. EnhancedDeepPlasticMember

### 3.1 Architecture

```python
class EnhancedDeepPlasticMember(nn.Module):
    """
    Deep network with plastic layers and layer normalization
    
    Structure:
        Input → PlasticLinear → LayerNorm → ReLU
              → PlasticLinear → LayerNorm → ReLU
              → Linear (output)
    """
    
    def __init__(self, cfg: OrthosConfigEnhanced):
        super().__init__()
        inp = cfg.state_dim + cfg.action_dim
        h = cfg.hidden_dim
        
        # Plastic layers (with traces)
        self.l1 = DiagnosticPlasticLinear(inp, h, cfg)
        self.l2 = DiagnosticPlasticLinear(h, h, cfg)
        
        # Output layer (non-plastic)
        self.l3 = nn.Linear(h, cfg.state_dim)
        
        # Layer normalization for stability
        self.ln1 = nn.LayerNorm(h)
        self.ln2 = nn.LayerNorm(h)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        h = self.ln1(F.relu(self.l1(x)))
        h = self.ln2(F.relu(self.l2(h)))
        return self.l3(h)
```

### 3.2 Design Rationale

**Why Layer Normalization?**
- Stabilizes activations during online learning
- Reduces internal covariate shift from trace updates
- Complements homeostatic regulation

**Why Non-Plastic Output Layer?**
- Predictions need to be stable for EFE computation
- Plasticity in hidden layers captures features
- Output layer learned via gradient descent

---

## 4. Ensemble World Model

### 4.1 Implementation

```python
class EnhancedEnsembleWorldModel(nn.Module):
    """
    Ensemble of plastic world models for uncertainty estimation
    
    Properties:
    - n_ensemble independent models
    - Each model has its own plastic traces
    - Mean prediction + disagreement-based uncertainty
    """
    
    def __init__(self, cfg: OrthosConfigEnhanced):
        super().__init__()
        self.models = nn.ModuleList(
            [EnhancedDeepPlasticMember(cfg) for _ in range(cfg.n_ensemble)]
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Collect predictions from all ensemble members
        preds = torch.stack([m(state, action) for m in self.models])
        
        # Mean: expected next state
        # Std: epistemic uncertainty (model disagreement)
        return preds.mean(dim=0), preds.std(dim=0)
```

### 4.2 Uncertainty Interpretation

```
┌────────────────────────────────────────────────────────────────┐
│                  Ensemble Predictions                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   Model 1: ─────•──────                                       │
│   Model 2: ────────•───                  High Uncertainty     │
│   Model 3: ──•─────────    ←─ Large spread                    │
│   Model 4: ───────────•                                       │
│   Model 5: ────•───────                                       │
│                                                                │
│   vs.                                                          │
│                                                                │
│   Model 1: ──────•─────                                       │
│   Model 2: ──────•─────                  Low Uncertainty      │
│   Model 3: ─────•──────    ←─ Small spread                    │
│   Model 4: ──────•─────                                       │
│   Model 5: ──────•─────                                       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 5. Diagnostics System

### 5.1 Tracked Metrics

```python
def get_diagnostics(self) -> Dict[str, float]:
    """
    Comprehensive diagnostic metrics for research analysis
    """
    return {
        # Trace state
        "fast_trace_norm": self.fast_trace.norm().item(),
        "slow_trace_norm": self.slow_trace.norm().item(),
        
        # Historical statistics
        "mean_trace_norm": self.trace_norm_history[:valid_steps].mean().item(),
        "mean_update_mag": self.update_magnitude_history[:valid_steps].mean().item(),
        
        # Weight state
        "weight_static_norm": self.weight.norm().item(),
    }
```

### 5.2 Metric Interpretation

| Metric | Healthy Range | Warning Signs |
|--------|---------------|---------------|
| `fast_trace_norm` | 0.5 - 5.0 | > 5.5 (unstable), < 0.1 (not learning) |
| `slow_trace_norm` | 0.1 - 2.0 | > 3.0 (accumulating noise) |
| `mean_update_mag` | 0.01 - 0.5 | > 1.0 (volatile), < 0.001 (stagnant) |
| `weight_static_norm` | 0.1 - 5.0 | Growing unboundedly |

### 5.3 Validation Protocol

```python
def run_comprehensive_validation():
    """
    200-step validation protocol
    
    Checkpoints at steps: 50, 100, 150, 200
    Metrics: WM Loss, Epistemic Uncertainty, Trace Norms, State
    """
    agent = OrthosAgentEnhanced(cfg)
    state = torch.tensor([[0.0]], device=device)
    
    for step in range(200):
        action = agent.select_action(state)
        next_state = state * 0.9 + action + 0.01 * torch.randn_like(state)
        wm_loss, uncertainty = agent.learn(state, action, next_state)
        state = next_state
        
        if (step + 1) % 50 == 0:
            summary = agent.get_summary()
            # Log metrics...
    
    # Final stability check
    final_trace_norm = diag.get("l1_fast_trace_norm", 0)
    stable = 0.0 < final_trace_norm < cfg.homeostatic_target * 1.1
```

---

## 6. Integration with Active Inference

### 6.1 Action Selection Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Active Inference Loop                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌───────────┐     ┌──────────────┐     ┌─────────────────┐   │
│   │   State   │────▶│ Action Model │────▶│ Sample Actions  │   │
│   └───────────┘     │   (μ, σ)     │     │ (30 samples)    │   │
│                     └──────────────┘     └────────┬────────┘   │
│                                                    │            │
│                                                    ▼            │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │            Ensemble World Model                          │  │
│   │    (state, action) → (mean_next, std_next)              │  │
│   └────────────────────────────┬────────────────────────────┘  │
│                                │                               │
│                                ▼                               │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │            Expected Free Energy                          │  │
│   │    EFE = pragmatic - β * epistemic                      │  │
│   └────────────────────────────┬────────────────────────────┘  │
│                                │                               │
│                                ▼                               │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │            Softmax Selection                             │  │
│   │    probs = softmax(-EFE / τ)                            │  │
│   │    action = multinomial(probs)                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Learning Loop

```python
def learn(self, state, action, next_state):
    """
    Combined world model and policy learning
    
    World Model: Supervised learning on (s, a) → s'
    Policy: Reinforcement via performance improvement signal
    """
    # === World Model Update ===
    preds = torch.stack([m(state, action) for m in self.wm.models])
    target = next_state.unsqueeze(0).expand_as(preds)
    wm_loss = F.mse_loss(preds, target)
    
    self.wm_opt.zero_grad()
    wm_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)  # Stability
    self.wm_opt.step()
    
    # === Policy Update ===
    with torch.no_grad():
        before = F.mse_loss(state, self.preferred_state)
        after = F.mse_loss(next_state, self.preferred_state)
        improvement = (before - after).item()
    
    if improvement > 0:  # Only update if action was beneficial
        logp = self.am.log_prob(state, action)
        weight = torch.sigmoid(torch.tensor(improvement * self.cfg.weight_scale))
        loss = -(logp * weight)
        
        self.act_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.am.parameters(), 1.0)
        self.act_opt.step()
```

---

## 7. Performance Characteristics

### 7.1 Computational Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Forward pass | O(n² × h) | O(n × h) |
| Trace update | O(n² × h) | O(n × h) |
| BitNet quantize | O(n × h) | O(n × h) |
| Ensemble forward | O(E × n² × h) | O(E × n × h) |

Where: n = input dim, h = hidden dim, E = ensemble size

### 7.2 Memory Usage

```
Static weights:     4 bytes/weight (float32)
Quantized weights:  ~0.2 bytes/weight (1.58-bit)
Traces (fast+slow): 8 bytes/weight (2 × float32)
Diagnostics:        ~8KB per layer (1000 × 2 floats)
```

### 7.3 Stability Guarantees

✅ **Bounded traces** via homeostatic normalization  
✅ **Gradient clipping** prevents exploding gradients  
✅ **Layer normalization** stabilizes activations  
✅ **Sigmoid-weighted policy updates** prevent overcorrection  

---

## 8. Usage Examples

### 8.1 Basic Usage

```python
# Configuration
cfg = OrthosConfigEnhanced(
    state_dim=4,
    action_dim=2,
    hidden_dim=128,
    n_ensemble=5
)

# Create agent
agent = OrthosAgentEnhanced(cfg)

# Training loop
for episode in range(1000):
    state = env.reset()
    for step in range(100):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, next_state)
        state = next_state
        if done:
            break
```

### 8.2 Accessing Diagnostics

```python
# Get ensemble diagnostics
diag = agent.wm.get_ensemble_diagnostics()
print(f"Fast trace norm: {diag['l1_fast_trace_norm']:.4f}")
print(f"Slow trace norm: {diag['l1_slow_trace_norm']:.4f}")

# Get learning summary
summary = agent.get_summary()
print(f"Mean WM loss: {summary['wm_loss_mean']:.4f}")
print(f"Mean uncertainty: {summary['epistemic_uncertainty_mean']:.4f}")
```

### 8.3 Disabling Plasticity

```python
# For inference-only mode
for model in agent.wm.models:
    model.l1.plasticity_enabled = False
    model.l2.plasticity_enabled = False
```

---

*For the theoretical foundations, see [Theoretical Foundations](SCIENCE.md). For research extensions, see [Future Directions](SCIENCE.md).*