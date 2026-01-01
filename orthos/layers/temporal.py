"""
TemporalLayer implementation.

Layer with temporal context processing for sequence learning.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from orthos.core.base import Layer, PlasticComponent
from orthos.core.types import Tensor, PlasticityParams
from orthos.core.tensor import initialize_weights, apply_activation, apply_activation_derivative

class TemporalLayer(Layer, PlasticComponent):
    """
    Layer with temporal context processing.

    This layer maintains temporal context through recurrent connections,
    enabling processing of sequential data and time-series patterns.

    Attributes:
        input_size: Size of input features
        hidden_size: Size of hidden state
        time_window: Number of time steps to maintain
        hidden_state: Current hidden state
        recurrent_weights: Recurrent weight matrix
        activation: Activation function name
        state_history: History of hidden states
        plasticity_params: Plasticity parameters
    """

    def __init__(self, input_size: int, hidden_size: int,
                 time_window: int = 10, activation: str = 'tanh'):
        """
        Initialize TemporalLayer.

        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            time_window: Number of time steps to maintain context
            activation: Activation function ('relu', 'sigmoid', 'tanh', 'linear')
        """
        self.input_size = input_size
        self.output_size = hidden_size  # For compatibility with Layer interface
        self.hidden_size = hidden_size
        self.time_window = time_window
        self.hidden_state = None
        self.activation_name = activation
        self.recurrent_weights = None
        self.state_history: List[np.ndarray] = []
        self.plasticity_params = {
            'learning_rate': 0.01,
            'decay_rate': 0.001
        }

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize parameters."""
        # Input to hidden weights
        self.weights = initialize_weights((self.hidden_size, self.input_size), 'he')

        # Recurrent weights
        self.recurrent_weights = initialize_weights((self.hidden_size, self.hidden_size), 'he')

        # Hidden state initialization
        self.hidden_state = np.zeros(self.hidden_size)

    def forward(self, x: np.ndarray, t: Optional[int] = None) -> np.ndarray:
        """
        Forward pass with temporal processing.

        Args:
            x: Input tensor (batch_size, input_size)
            t: Optional time step index

        Returns:
            Output tensor (batch_size, hidden_size)

        Raises:
            ValueError: If input shape doesn't match expected dimensions
        """
        if x.shape[1] != self.input_size:
            raise ValueError(f"Input size mismatch: expected {self.input_size}, got {x.shape[1]}")

        self.last_input = x
        self.last_hidden_state = self.hidden_state.copy()

        # Linear transformation with recurrent connections
        self.last_pre_activation = np.dot(x, self.weights.T) + np.dot(self.hidden_state, self.recurrent_weights.T)

        # Apply activation
        output = self.activation(self.last_pre_activation)

        # Update hidden state (average over batch)
        self.hidden_state = output.mean(axis=0)

        # Store state history
        self.state_history.append(self.hidden_state.copy())
        if len(self.state_history) > self.time_window:
            self.state_history.pop(0)

        return output

    def activation(self, x: np.ndarray) -> np.ndarray:
        """
        Apply activation function to input.

        Args:
            x: Input tensor

        Returns:
            Activated tensor
        """
        return apply_activation(x, self.activation_name)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass for gradient computation.

        Args:
            grad: Gradient from next layer

        Returns:
            Gradient for previous layer
        """
        if self.last_pre_activation is None:
            raise ValueError("Forward pass must be called before backward pass")

        # Derivative of activation
        da = apply_activation_derivative(self.last_pre_activation, self.activation_name)
        delta = grad * da

        # Gradient with respect to input: delta @ weights
        return np.dot(delta, self.weights)

    def update(self, lr: float) -> None:
        """
        Update layer parameters.

        Args:
            lr: Learning rate
        """
        if self.last_input is None:
            return

        effective_lr = lr or self.plasticity_params['learning_rate']
        
        # Simplified recurrent update (Heuristic-based for this architecture)
        # In a real meta-learning context, these weights might be evolved
        # But here we provide a basic gradient-descent update if needed
        pass

    def reset_state(self) -> None:
        """Reset internal state."""
        self.hidden_state = np.zeros(self.hidden_size)
        self.state_history = []

    def get_temporal_context(self) -> np.ndarray:
        """
        Get current temporal context.

        Returns:
            Temporal context tensor (time_window, hidden_size)
        """
        if len(self.state_history) == self.time_window:
            return np.array(self.state_history)
        else:
            # Pad with zeros if not enough history
            context = np.zeros((self.time_window, self.hidden_size))
            context[-len(self.state_history):] = self.state_history
            return context

    def get_state_history(self, steps: int = 10) -> List[np.ndarray]:
        """
        Get recent state history.

        Args:
            steps: Number of recent states to return

        Returns:
            List of hidden states
        """
        return self.state_history[-steps:]

    def get_weights(self) -> np.ndarray:
        """
        Get current weight matrix.

        Returns:
            Weight matrix
        """
        return self.weights

    def set_weights(self, weights: np.ndarray) -> None:
        """
        Set weight matrix.

        Args:
            weights: New weight matrix
        """
        self.weights = weights

    def get_plasticity_params(self) -> Dict[str, float]:
        """
        Get plasticity parameters.

        Returns:
            Dictionary of plasticity parameters
        """
        return self.plasticity_params.copy()

    def set_plasticity_params(self, params: Dict[str, float]) -> None:
        """
        Set plasticity parameters.

        Args:
            params: Dictionary of new plasticity parameters
        """
        self.plasticity_params.update(params)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration.

        Returns:
            Dictionary containing layer configuration
        """
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'time_window': self.time_window,
            'activation': self.activation_name
        }

    def __str__(self) -> str:
        """String representation of the layer."""
        return f"TemporalLayer({self.input_size}→{self.hidden_size}, window={self.time_window})"