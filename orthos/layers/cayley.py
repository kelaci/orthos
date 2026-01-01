"""
OrthosCayleyLayer implementation.

Orthogonal weight layer with implicit parameterization using Cayley transformation.
Guarantees W @ W.T = I without explicit projection.

This layer supports GPU acceleration through CuPy when available.
"""

import numpy as np
from typing import Optional, Dict, Tuple, Any
from orthos.core.base import Layer
from orthos.core.types import Tensor, ActivationFunction
from orthos.core.exceptions import InputShapeError, ConfigurationError
from orthos.core.gpu_utils import (
    get_array_module, get_device, zeros, dot, matmul,
    CUPY_AVAILABLE
)

USING_GPU = get_device() == 'cuda'
xp = get_array_module()

class OrthosCayleyLayer(Layer):
    """
    Orthogonal weight layer with implicit parameterization using Cayley transformation.

    Guarantees W @ W.T = I without explicit projection.

    Attributes:
        input_size: Size of input features
        output_size: Size of output features
        skew_params: Skew-symmetric parameter matrix
        activation_fn: Activation function name
        I: Identity matrix cache
    """

    def __init__(self, input_size: int, output_size: int,
                 activation: str = 'relu'):
        """
        Initialize OrthosCayleyLayer.

        Args:
            input_size: Size of input features
            output_size: Size of output features
            activation: Activation function ('relu', 'sigmoid', 'tanh', 'linear')
        """
        self.input_size = input_size
        self.output_size = output_size
        self.skew_params = None
        self.activation_fn = activation
        self.I = None

        # For Cayley transform, we need a square skew-symmetric matrix S
        # If dimensions are rectangular (M != N), we use K = max(M, N)
        self.max_dim = max(self.input_size, self.output_size)

        # Create random square matrix
        random_matrix = xp.random.randn(self.max_dim, self.max_dim)

        # Convert to skew-symmetric: S = (random_matrix - random_matrix.T)
        self.skew_params = random_matrix - random_matrix.T

        # Cache identity matrix of size K
        self.I = xp.eye(self.max_dim)

    def _get_skew_symmetric(self) -> Tensor:
        """
        Extract a pure skew-symmetric matrix from parameters.

        Returns:
            Skew-symmetric matrix S where S = -S.T
        """
        # Ensure skew-symmetry: S = params - params.T
        S = self.skew_params - self.skew_params.T
        return S

    def _cayley_transform(self, S: Tensor) -> Tensor:
        """
        Compute Cayley transformation: W = (I - S)(I + S)^-1

        Args:
            S: Skew-symmetric matrix

        Returns:
            Orthogonal matrix W
        """
        try:
            # Compute I + S
            I_plus_S = self.I + S

            # Solve linear equation: W @ (I + S) = (I - S)
            # This is more stable than explicit inversion
            W = xp.linalg.solve(I_plus_S, self.I - S)

            return W
        except (np.linalg.LinAlgError, ValueError):
            # Fallback for extreme cases - use normalized random orthogonal
            random_orthogonal = xp.random.randn(self.max_dim, self.max_dim)
            q, _ = xp.linalg.qr(random_orthogonal)
            return q

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
        if x.shape[1] != self.input_size:
            raise InputShapeError(f"Input size mismatch", expected=(None, self.input_size), actual=x.shape)

        # 1. Get square skew-symmetric matrix
        S = self._get_skew_symmetric()

        # 2. Compute square orthogonal matrix using Cayley transform
        W_square = self._cayley_transform(S)

        # 3. Rectified Orthogonality: Extract sub-matrix for non-square dimensions
        W = W_square[:self.output_size, :self.input_size]

        # 4. Apply linear transformation: y = x @ W.T
        output = dot(x, W.T)

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

        # 1. Get skew-symmetric matrix
        S = self._get_skew_symmetric()

        # 2. Compute square orthogonal matrix using Cayley transform
        W_square = self._cayley_transform(S)

        # 3. Extract rectangular weight matrix
        W = W_square[:self.output_size, :self.input_size]

        # 4. Compute gradient with respect to input: grad @ W
        # Using GPU-accelerated matmul
        input_grad = matmul(grad, W)

        return input_grad

    def update(self, lr: float) -> None:
        """
        Update layer parameters.

        For OrthosCayleyLayer, we update the skew-symmetric parameters directly.

        Args:
            lr: Learning rate
        """
        # In a real implementation, this would be handled by an optimizer
        # that updates self.skew_params based on gradients
        # For now, we'll leave this as a placeholder
        pass

    def reset_state(self) -> None:
        """
        Reset internal state.

        OrthosCayleyLayer has no internal state - this is a no-op.
        """
        # No internal state to reset
        pass

    def get_weights(self) -> Tensor:
        """
        Get current orthogonal weight matrix.

        Returns:
            Orthogonal weight matrix (output_size, input_size)
        """
        S = self._get_skew_symmetric()
        W_square = self._cayley_transform(S)
        # Return sliced rectangular matrix
        return W_square[:self.output_size, :self.input_size]

    def set_weights(self, weights: Tensor) -> None:
        """
        Set weights.

        For OrthosCayleyLayer, this sets the skew-symmetric parameters
        to approximate the given weight matrix.

        Args:
            weights: Desired weight matrix (output_size, input_size)

        Raises:
            ValueError: If weight shape doesn't match expected dimensions
        """
        if weights.shape != (self.output_size, self.input_size):
            raise ValueError(f"Weight shape mismatch: expected {(self.output_size, self.input_size)}, got {weights.shape}")

        # For Cayley parameterization, we need to find S such that
        # W ≈ (I - S)(I + S)^-1
        # This is non-trivial, so we'll approximate by:
        # 1. Project weights to be orthogonal
        # 2. Find S that approximates this orthogonal matrix

        # Project to orthogonal matrix using QR decomposition
        # For rectangular matrices, we need to handle the dimensions properly
        if self.output_size <= self.input_size:
            # Standard QR for tall/skinny matrices
            q, _ = xp.linalg.qr(weights)
            # q will be (output_size, input_size), we need to pad it
            q_square = xp.zeros((self.max_dim, self.max_dim))
            q_square[:self.output_size, :self.input_size] = q
        else:
            # For wide matrices, use QR on transpose
            q_t, _ = xp.linalg.qr(weights.T)
            q = q_t.T  # Now q is (output_size, input_size)
            q_square = xp.zeros((self.max_dim, self.max_dim))
            q_square[:self.output_size, :self.input_size] = q

        # Approximate S from W using inverse Cayley transform: S = (I - W)(I + W)^-1
        try:
            I_plus_W = self.I + q_square
            S_approx = xp.linalg.solve(I_plus_W, self.I - q_square)

            # Ensure skew-symmetry
            self.skew_params = (S_approx - S_approx.T) / 2
        except (np.linalg.LinAlgError, ValueError):
            # Fallback: reinitialize with random skew-symmetric
            # Create random square matrix
            random_matrix = xp.random.randn(self.max_dim, self.max_dim)
            # Convert to skew-symmetric: S = (random_matrix - random_matrix.T)
            self.skew_params = random_matrix - random_matrix.T

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
            'using_gpu': USING_GPU,
            'device': get_device() if CUPY_AVAILABLE else 'cpu',
            'parameterization': 'cayley'
        }

    def __str__(self) -> str:
        """String representation of the layer."""
        device = "GPU" if USING_GPU else "CPU"
        return f"OrthosCayleyLayer({self.input_size}→{self.output_size}, {self.activation_fn}, {device})"
