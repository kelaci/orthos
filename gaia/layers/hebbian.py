"""
HebbianCore implementation.

Hebbian learning layer with multiple plasticity rules for adaptive feature learning.
"""

import numpy as np
from typing import Dict, Optional, List, Tuple, Any
from gaia.core.base import Layer, PlasticComponent
from gaia.core.types import Tensor, PlasticityParams
from gaia.core.tensor import initialize_weights
from gaia.core.gpu_utils import get_array_module

class HebbianCore(Layer, PlasticComponent):
    """
    Hebbian learning implementation with multiple plasticity rules.

    This layer implements various Hebbian learning rules for adaptive
    weight updates based on pre- and post-synaptic activity.

    Attributes:
        input_size: Size of input features
        output_size: Size of output features
        weights: Weight matrix (output_size, input_size)
        pre_synaptic: Pre-synaptic activity trace
        post_synaptic: Post-synaptic activity trace
        plasticity_rule: Current plasticity rule
        plasticity_params: Plasticity parameters
        activity_history: History of pre/post-synaptic activities
    """

    def __init__(self, input_size: int, output_size: int,
                 plasticity_rule: str = 'hebbian',
                 params: Optional[Dict[str, float]] = None):
        """
        Initialize HebbianCore.

        Args:
            input_size: Size of input features
            output_size: Size of output features
            plasticity_rule: Plasticity rule ('hebbian', 'oja', 'bcm')
            params: Plasticity parameters
        """
        self.input_size = input_size
        self.output_size = output_size
        self.weights = None
        self.pre_synaptic = None
        self.post_synaptic = None
        self.plasticity_rule = plasticity_rule
        self.activity_history: List[Tuple[np.ndarray, np.ndarray]] = []

        # Default plasticity parameters
        self.plasticity_params = params or {
            'learning_rate': 0.01,
            'decay_rate': 0.001,
            'ltp_coefficient': 1.0,
            'ltd_coefficient': 0.8,
            'homeostatic_strength': 0.1,
            'bcm_theta': 1.0
        }

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize weights and activity traces."""
        from gaia.core.gpu_utils import zeros, get_array_module
        self.weights = initialize_weights((self.output_size, self.input_size), 'he')
        self.pre_synaptic = zeros((self.input_size,))
        self.post_synaptic = zeros((self.output_size,))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the Hebbian core.

        Args:
            x: Input tensor (batch_size, input_size)

        Returns:
            Output tensor (batch_size, output_size)

        Raises:
            ValueError: If input shape doesn't match expected dimensions
        """
        xp = get_array_module()
        
        if x.shape[1] != self.input_size:
            raise ValueError(f"Input size mismatch: expected {self.input_size}, got {x.shape[1]}")

        # Update activity traces (average over batch)
        # Handle both numpy and cupy
        self.pre_synaptic = xp.mean(x, axis=0)
        output = xp.dot(x, self.weights.T)
        self.post_synaptic = xp.mean(output, axis=0)

        # Store activity history (keep on same device or move to CPU if needed for logging?)
        # For performance, keep on device. But copy() is needed.
        self.activity_history.append((self.pre_synaptic.copy(), self.post_synaptic.copy()))
        if len(self.activity_history) > 100:  # Limit history size
            self.activity_history.pop(0)

        return output

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backward pass for gradient computation.

        Args:
            grad: Gradient from next layer

        Returns:
            Gradient for previous layer
        """
        xp = get_array_module()
        return xp.dot(grad, self.weights)

    def update(self, lr: Optional[float] = None) -> None:
        """
        Update weights using Hebbian learning.

        Args:
            lr: Optional learning rate override
        """
        xp = get_array_module()
        effective_lr = lr if lr is not None else self.plasticity_params['learning_rate']

        if self.plasticity_rule == 'hebbian':
            # Classic Hebbian: Δw = η * pre * post
            weight_update = effective_lr * xp.outer(self.post_synaptic, self.pre_synaptic)
        elif self.plasticity_rule == 'oja':
            # Oja's rule: Δw = η * post * (pre - post * w)
            weight_update = effective_lr * xp.outer(
                self.post_synaptic,
                self.pre_synaptic - xp.dot(self.weights.T, self.post_synaptic)
            )
        elif self.plasticity_rule == 'bcm':
            # BCM rule with sliding threshold
            theta = self.plasticity_params['bcm_theta']
            weight_update = effective_lr * xp.outer(
                self.post_synaptic * (self.post_synaptic - theta),
                self.pre_synaptic
            )
            # Update threshold based on post-synaptic activity
            self.plasticity_params['bcm_theta'] = 0.9 * theta + 0.1 * float(xp.mean(self.post_synaptic))
        else:
            raise ValueError(f"Unknown plasticity rule: {self.plasticity_rule}")

        # Apply weight update
        self.weights += weight_update

        # Apply weight decay
        self.weights *= (1.0 - self.plasticity_params['decay_rate'])

        # Homeostatic regulation
        self._homeostatic_regulation()

    def _homeostatic_regulation(self) -> None:
        """
        Apply homeostatic regulation to maintain stable activity.

        Implements synaptic scaling and weight normalization to prevent
        runaway excitation and maintain stability.
        """
        xp = get_array_module()
        strength = self.plasticity_params.get('homeostatic_strength', 0.1)
        
        # 1. Synaptic scaling toward target mean activity
        target_activity = 0.5
        current_activity = xp.mean(self.post_synaptic)
        scaling_factor = 1.0 + strength * (target_activity - current_activity)
        self.weights *= scaling_factor

        # 2. Weight normalization (L2 norm per neuron)
        norm = xp.linalg.norm(self.weights, axis=1, keepdims=True)
        self.weights = self.weights / (norm + 1e-8)

    def activation(self, x: Tensor) -> Tensor:
        """
        Apply activation function (identity for HebbianCore).

        Args:
            x: Input tensor

        Returns:
            Output tensor (unchanged)
        """
        return x

    def get_weights(self) -> Tensor:
        """
        Get current weight matrix.

        Returns:
            Weight matrix
        """
        return self.weights

    def set_weights(self, weights: Tensor) -> None:
        """
        Set weight matrix.

        Args:
            weights: New weight matrix
        """
        self.weights = weights

    def reset_state(self) -> None:
        """Reset internal state."""
        from gaia.core.gpu_utils import zeros
        self.pre_synaptic = zeros((self.input_size,))
        self.post_synaptic = zeros((self.output_size,))
        self.activity_history = []

    def get_plasticity_params(self) -> PlasticityParams:
        """
        Get plasticity parameters.

        Returns:
            Dictionary of plasticity parameters
        """
        return self.plasticity_params.copy()

    def set_plasticity_params(self, params: PlasticityParams) -> None:
        """
        Set plasticity parameters.

        Args:
            params: Dictionary of new plasticity parameters
        """
        self.plasticity_params.update(params)

    def set_plasticity_rule(self, rule: str) -> None:
        """
        Set plasticity rule.

        Args:
            rule: Plasticity rule name ('hebbian', 'oja', 'bcm')

        Raises:
            ValueError: If unknown rule is specified
        """
        if rule not in ['hebbian', 'oja', 'bcm']:
            raise ValueError(f"Unknown plasticity rule: {rule}")
        self.plasticity_rule = rule

    def get_activity_history(self, steps: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get recent activity history.

        Args:
            steps: Number of recent steps to return

        Returns:
            List of (pre_synaptic, post_synaptic) activity pairs
        """
        return self.activity_history[-steps:]

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration.

        Returns:
            Dictionary containing layer configuration
        """
        return {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'plasticity_rule': self.plasticity_rule,
            'plasticity_params': self.plasticity_params.copy()
        }

    def __str__(self) -> str:
        """String representation of the layer."""
        return f"HebbianCore({self.input_size}→{self.output_size}, {self.plasticity_rule})"