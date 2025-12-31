"""
Sparse Attention layers with structural plasticity support.
"""

from typing import Dict, Optional, Tuple, Any, List, Union
import numpy as np
from gaia.core.base import Layer, PlasticComponent
from gaia.core.tensor import initialize_weights
from gaia.core.gpu_utils import get_array_module, dot, matmul, zeros, sqrt, exp, sum, mean, to_cpu, to_gpu
from gaia.plasticity.structural import StructuralOptimizer

def kwta_mask(scores: Any, k: int) -> Any:
    """
    k-Winners-Take-All mask.
    
    Args:
        scores: Input scores (..., N)
        k: Number of winners to keep
        
    Returns:
        Binary mask of same shape as scores
    """
    xp = get_array_module()
    if k <= 0:
        return xp.zeros_like(scores)
    if k >= scores.shape[-1]:
        return xp.ones_like(scores)
        
    # Get threshold for top-k
    # Note: sort is expensive but robust. For larger N, partition is better.
    sorted_scores = xp.sort(scores, axis=-1)
    threshold = sorted_scores[..., -k].reshape(scores.shape[:-1] + (1,))
    
    return (scores >= threshold).astype(xp.float32)

class MaskedLinear(Layer):
    """
    Linear layer with enforced sparse topology.
    """
    def __init__(self, in_features: int, out_features: int, 
                 density: float = 0.1, block_size: int = 1):
        self.in_features = in_features
        self.out_features = out_features
        self.density = density
        self.block_size = block_size
        
        # Parameters
        self.weights = initialize_weights((out_features, in_features))
        self.mask = self._init_sparse_mask(density, block_size)
        
        xp = get_array_module()
        self.synaptic_age = xp.zeros_like(self.mask)
        self.input_cache = None
        self.weight_grads = None
        
    def _init_sparse_mask(self, density: float, block_size: int) -> Any:
        xp = get_array_module()
        if block_size <= 1:
            mask = xp.random.random((self.out_features, self.in_features)) < density
        else:
            # Block sparsity
            bh, bw = self.out_features // block_size, self.in_features // block_size
            if bh == 0 or bw == 0:
                mask = xp.random.random((self.out_features, self.in_features)) < density
            else:
                block_mask = xp.random.random((bh, bw)) < density
                mask = xp.kron(block_mask, xp.ones((block_size, block_size)))
                # Pad if dimensions weren't perfectly divisible
                if mask.shape != (self.out_features, self.in_features):
                    full_mask = xp.zeros((self.out_features, self.in_features))
                    h, w = mask.shape
                    full_mask[:h, :w] = mask
                    mask = full_mask
        return mask.astype(xp.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: (batch, in_features)
        self.input_cache = x
        effective_weights = self.weights * self.mask
        return dot(x, effective_weights.T)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # grad: (batch, out_features)
        if self.input_cache is None:
            raise RuntimeError("Forward pass must be called before backward")
            
        effective_weights = self.weights * self.mask
        
        # dL/dX = grad @ W_eff
        grad_input = dot(grad, effective_weights)
        
        # dL/dW = grad.T @ X
        self.weight_grads = dot(grad.T, self.input_cache)
        self.weight_grads *= self.mask # Gradients only for active synapses
        
        return grad_input

    def update(self, lr: float) -> None:
        if self.weight_grads is not None:
            self.weights -= lr * self.weight_grads
            self.weights *= self.mask # Keep inactive strictly zero

    def activation(self, x: np.ndarray) -> np.ndarray:
        return x

    def get_weights(self) -> np.ndarray:
        return self.weights

    def set_weights(self, weights: np.ndarray) -> None:
        self.weights = weights

    def reset_state(self) -> None:
        self.input_cache = None
        self.weight_grads = None

    def get_density(self) -> float:
        xp = get_array_module()
        return float(xp.mean(self.mask))

class SparseAttentionLayer(Layer, PlasticComponent):
    """
    Sparse attention with structural plasticity.
    """
    def __init__(self, d_model: int, n_heads: int, 
                 sparsity_target: float = 0.1,
                 activation_sparsity: float = 0.2,
                 block_size: int = 1):
        self.d_model = d_model
        self.n_heads = n_heads
        self.sparsity_target = sparsity_target
        self.activation_sparsity = activation_sparsity
        
        # Projections
        self.q_proj = MaskedLinear(d_model, d_model, sparsity_target, block_size)
        self.k_proj = MaskedLinear(d_model, d_model, sparsity_target, block_size)
        self.v_proj = MaskedLinear(d_model, d_model, sparsity_target, block_size)
        
        # Structural optimizers (one per projection to maintain separate state)
        # We assume standard defaults if not provided in kwargs (TODO: Inject global defaults)
        optimizer_kwargs = {
            'prune_threshold': 0.01,
            'regrow_fraction': 0.05,
            'consolidation_strength': 0.5
        }
        
        self.q_optimizer = StructuralOptimizer(**optimizer_kwargs)
        self.k_optimizer = StructuralOptimizer(**optimizer_kwargs)
        self.v_optimizer = StructuralOptimizer(**optimizer_kwargs)
        
        self.context_bias = None
        self.scores_cache = None
        self.weights_cache = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: (batch, seq_len, d_model)
        xp = get_array_module()
        batch, seq, dim = x.shape
        
        # Reshape for multi-head? Document says simplified head logic for now.
        # Following spec's simplified (batch, seq, d_model) approach.
        
        Q = self.q_proj.forward(x.reshape(-1, dim)).reshape(batch, seq, dim)
        K = self.k_proj.forward(x.reshape(-1, dim)).reshape(batch, seq, dim)
        V = self.v_proj.forward(x.reshape(-1, dim)).reshape(batch, seq, dim)
        
        if self.context_bias is not None:
            Q = Q + self.context_bias
            
        # Attention scores: (batch, seq, seq)
        scores = matmul(Q, K.transpose(0, 2, 1)) / sqrt(float(dim))
        
        # k-WTA activation
        k = max(1, int(self.activation_sparsity * seq))
        mask = kwta_mask(scores, k)
        
        # Masked Softmax
        masked_scores = scores.copy()
        masked_scores[mask == 0] = -1e9
        
        exp_scores = exp(masked_scores - xp.max(masked_scores, axis=-1, keepdims=True))
        exp_scores *= mask
        attn_weights = exp_scores / (sum(exp_scores, axis=-1, keepdims=True) + 1e-8)
        
        self.scores_cache = scores
        self.weights_cache = attn_weights
        
        return matmul(attn_weights, V)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # grad: (batch, seq, d_model)
        # Full backprop for attention is complex, providing structured approximation
        xp = get_array_module()
        if self.weights_cache is None:
            raise RuntimeError("Forward pass must be called before backward")
            
        # dL/dV = Attn.T @ Grad
        grad_v = matmul(self.weights_cache.transpose(0, 2, 1), grad)
        
        # Propagate through projections (MaskedLinear handles its own param grads)
        # Reshape to 2D for linear layers
        batch, seq, dim = grad.shape
        grad_input_v = self.v_proj.backward(grad_v.reshape(-1, dim)).reshape(batch, seq, dim)
        
        # Note: Contribution from Q and K through Attn weights requires more complex Jacobian.
        # Simplified for now: only V path for heavy lifting, Q/K path placeholder.
        # Consistent with doc's "TODO: Implement full attention gradient chain"
        return grad_input_v 

    def update(self, lr: float) -> None:
        self.q_proj.update(lr)
        self.k_proj.update(lr)
        self.v_proj.update(lr)

    def activation(self, x: np.ndarray) -> np.ndarray:
        return x

    def reset_state(self) -> None:
        self.context_bias = None
        self.scores_cache = None
        self.weights_cache = None
        self.q_proj.reset_state()
        self.k_proj.reset_state()
        self.v_proj.reset_state()

    def get_weights(self) -> np.ndarray:
        return self.q_proj.get_weights()

    def set_weights(self, weights: np.ndarray) -> None:
        # Setting weights for multiprojection layer is complex
        pass

    def get_plasticity_params(self) -> Dict[str, float]:
        return {
            'sparsity_target': self.sparsity_target,
            'activation_sparsity': self.activation_sparsity
        }

    def set_plasticity_params(self, params: Dict[str, float]) -> None:
        if 'sparsity_target' in params:
            self.sparsity_target = params['sparsity_target']
        if 'activation_sparsity' in params:
            self.activation_sparsity = params['activation_sparsity']

    def top_down_modulate(self, context: np.ndarray) -> None:
        self.context_bias = context

    def rewire(self) -> Dict[str, Any]:
        """Execute rewiring for all projections."""
        metrics = {}
        metrics['q'] = self.q_optimizer.rewire(self.q_proj)
        metrics['k'] = self.k_optimizer.rewire(self.k_proj)
        metrics['v'] = self.v_optimizer.rewire(self.v_proj)
        return metrics
