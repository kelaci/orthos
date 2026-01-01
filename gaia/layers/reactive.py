"""
ReactiveLayer implementation.

Fast, reactive processing layer with fixed weights for initial feature extraction.

This layer supports GPU acceleration through CuPy when available.
"""

import numpy as np
from typing import Optional, Dict, Tuple, Any
from gaia.core.base import Layer
from gaia.core.types import Tensor, ActivationFunction
from gaia.core.exceptions import InputShapeError, ConfigurationError
from gaia.core.gpu_utils import (
    get_array_module, get_device, zeros, dot, matmul,
    CUPY_AVAILABLE
)

USING_GPU = get_device() == 'cuda'
xp = get_array_module()

class ReactiveLayer(Layer):
    """
    Fast, reactive processing layer with fixed weights for initial feature extraction.

    Supports GPU acceleration when CuPy is available.

    Attributes:
        input_size: Size of input features
        output_size: Size of output features
        weights: Weight matrix (output_size, input_size)
        biases: Bias vector (output_size,)
        activation_fn: Activation function name
        init_type: Weight initialization type
    """

    def __init__(self, input_size: int, output_size: int,
                 activation: str = 'relu', init_type: str = 'he'):
        """
        Initialize ReactiveLayer.

        Args:
            input_size: Size of input features
            output_size: Size of output features
            activation: Activation function ('relu', 'sigmoid', 'tanh', 'linear')
            init_type: Weight initialization type ('he', 'xavier', 'normal', 'uniform')
        """
        self.input_size = input_size
        self.output_size = output_size
        self.weights = None
        self.biases = None
        self.activation_fn = activation
        self.init_type = init_type

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize weights and biases using GPU if available."""
        # Use GPU utilities if available, otherwise NumPy
        self.weights = zeros((self.output_size, self.input_size))
        self.biases = zeros((self.output_size,))

        # Different initialization based on type
        if self.init_type == 'he':
            # He initialization: N(0, 1/sqrt(n))
            self.weights = xp.random.randn(self.output_size, self.input_size)
            self.weights *= xp.sqrt(2.0 / self.input_size)
        elif self.init_type == 'xavier':
            # Xavier initialization: N(0, 1/sqrt(fan_in))
            self.weights = xp.random.randn(self.output_size, self.input_size)
            self.weights *= xp.sqrt(1.0 / self.input_size)
        elif self.init_type == 'normal':
            # Normal initialization: N(0, 0.01)
            self.weights = xp.random.randn(self.output_size, self.input_size) * 0.01
        elif self.init_type == 'uniform':
            # Uniform initialization: U(-0.01, 0.01)
            self.weights = xp.random.uniform(-0.01, 0.01, 
                                            (self.output_size, self.input_size))
        else:
            raise ConfigurationError(f"Unknown init_type: {self.init_type}")

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the layer.

        Args:
            x: Input tensor (batch_size, input_size)

        Returns:
            Output tensor (batch_size, output_size)

        Raises:
            InputShapeError: If input shape doesn't match expected dimensions
        """
        # Use GPU-accelerated dot operation
        if x.shape[1] != self.input_size:
            raise InputShapeError(f"Input size mismatch", expected=(None, self.input_size), actual=x.shape)

        # Linear transformation: y = x @ W + b
        output = dot(x, self.weights.T) + self.biases

        # Apply activation
        return self.activation(output)

    def activation(self, x: Tensor) -> Tensor:
        """
        Apply activation function.

        Args:
            x: Input tensor

        Returns:
            Activated tensor

        Raises:
            ValueError: If unknown activation function
        """
        if self.activation_fn == 'relu':
            return xp.maximum(0, x)
        elif self.activation_fn == 'sigmoid':
            return 1.0 / (1.0 + xp.exp(-x))
        elif self.activation_fn == 'tanh':
            return xp.tanh(x)
        elif self.activation_fn == 'linear':
            return x
        else:
            raise ValueError(f"Unknown activation: {self.activation_fn}")

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backward pass for gradient computation.

        Args:
            grad: Gradient tensor from next layer (batch_size, output_size)

        Returns:
            Gradient tensor for previous layer (batch_size, input_size)

        Raises:
            ValueError: If gradient shape doesn't match expected dimensions
        """
        if grad.shape[1] != self.output_size:
            raise ValueError(f"Gradient size mismatch: expected {self.output_size}, got {grad.shape[1]}")

        # Compute gradient with respect to input: grad @ W
        # Using GPU-accelerated matmul
        input_grad = matmul(grad, self.weights)

        return input_grad

    def update(self, lr: float) -> None:
        """
        Update layer parameters.

        Note: ReactiveLayer has fixed weights - no update implemented.

        Args:
            lr: Learning rate (ignored for this layer type)
        """
        # ReactiveLayer has fixed weights - no update needed
        pass

    def reset_state(self) -> None:
        """
        Reset internal state.

        ReactiveLayer has no internal state - this is a no-op.
        """
        # No internal state to reset
        pass

    def get_weights(self) -> Tensor:
        """
        Get current weights.

        Returns:
            Weight matrix (output_size, input_size)
        """
        return self.weights

    def set_weights(self, weights: Tensor) -> None:
        """
        Set weights.

        Args:
            weights: New weight matrix (output_size, input_size)

        Raises:
            ValueError: If weight shape doesn't match expected dimensions
        """
        if weights.shape != (self.output_size, self.input_size):
            raise ValueError(f"Weight shape mismatch: expected {(self.output_size, self.input_size)}, got {weights.shape}")
        self.weights = weights.copy()

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration.

        Returns:
            Dictionary containing layer configuration
        """
        return {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'activation': self.activation_fn,
            'init_type': self.init_type,
            'using_gpu': USING_GPU,
            'device': get_device() if CUPY_AVAILABLE else 'cpu'
        }

    def __str__(self) -> str:
        """String representation of the layer."""
        device = "GPU" if USING_GPU else "CPU"
        return f"ReactiveLayer({self.input_size}→{self.output_size}, {self.activation_fn}, {device})"
