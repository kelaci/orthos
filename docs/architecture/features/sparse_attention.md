# Technical Specification: Sparse Attention & Structural Plasticity (SAS)

**Target Version**: v4.2  
**Status**: Planning (Enterprise-Ready Specification)  
**Module**: `gaia.layers.attention`  
**Last Updated**: 2024-12-31

---

## 1. Executive Summary

The SAS system transitions GAIA from a **Passive/Dense** processing model to an **Active/Sparse** model, targeting:
- **Computational Efficiency**: $O(N^2) \rightarrow O(N \log N)$ complexity reduction
- **Adaptive Topology**: Self-organizing network structure via Hebbian structural plasticity
- **Business Value**: 60-80% reduction in compute costs, edge deployment capability

---

## 2. Architectural Philosophy

### 2.1 Core Principles
| Principle | Dense (v4.1) | Sparse (v4.2) |
| :--- | :--- | :--- |
| **Connectivity** | All-to-all | Selective (target: 10-30%) |
| **Processing** | Passive (all inputs) | Active (query relevant inputs) |
| **Topology** | Fixed after init | Dynamic (rewiring) |
| **Complexity** | $O(N^2)$ | $O(N \log N)$ or $O(N)$ |

### 2.2 Biological Inspiration
The human brain maintains ~100 trillion synapses but only ~10% are active at any moment. SAS mimics this through:
1. **Activation Sparsity**: k-Winners-Take-All (kWTA) attention
2. **Structural Sparsity**: Pruning/regrowth cycles (synaptic turnover)

---

## 3. Component Design

### 3.1 The Sparse Tensor Core

#### Class: `MaskedLinear`
Wraps a standard weight matrix but enforces a binary topology mask. Inherits from `gaia.core.base.Layer` and supports GPU acceleration.

```python
from typing import Optional, Tuple
import numpy as np
from gaia.core.base import Layer
from gaia.core.tensor import initialize_weights
from gaia.core.gpu_utils import get_array_module, dot, matmul, zeros

class MaskedLinear(Layer):
    """
    Linear layer with enforced sparsity topology.
    
    Attributes:
        weights: Weight matrix (out_features, in_features)
        mask: Binary topology mask (1=active, 0=inactive)
        synaptic_age: Age of each synapse (for consolidation)
        density: Target connectivity density
        input_cache: Cache of input for backward pass
    """
    def __init__(self, in_features: int, out_features: int, 
                 density: float = 0.1, block_size: int = 1):
        self.in_features = in_features
        self.out_features = out_features
        self.density = density
        self.block_size = block_size
        
        # Initialize parameters
        self.weights = initialize_weights((out_features, in_features))
        self.mask = self._init_sparse_mask(density, block_size)
        
        xp = get_array_module()
        self.synaptic_age = xp.zeros_like(self.mask)
        self.input_cache = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with masked weights.
        
        Args:
            x: Input tensor (batch_size, in_features)
        """
        xp = get_array_module()
        
        # Cache input for backward pass
        self.input_cache = x
        
        # W_eff = W * Mask (zero-out inactive connections)
        effective_weights = self.weights * self.mask
        
        # Linear projection: x @ W_eff.T
        return dot(x, effective_weights.T)
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass computing gradients for inputs and weights.
        
        Args:
            grad: Gradient from next layer (batch_size, out_features)
        """
        xp = get_array_module()
        
        if self.input_cache is None:
            raise RuntimeError("Forward pass must be called before backward")
            
        # 1. Gradient w.r.t Input: grad @ W_eff
        effective_weights = self.weights * self.mask
        grad_input = dot(grad, effective_weights)
        
        # 2. Gradient w.r.t Weights: grad.T @ input
        # Accumulate gradients (to be used in update)
        self.weight_grads = dot(grad.T, self.input_cache)
        
        # Enforce sparsity on gradients immediately (optional, or in update)
        self.weight_grads *= self.mask
        
        return grad_input

    def update(self, lr: float) -> None:
        """Update weights using accumulated gradients."""
        if hasattr(self, 'weight_grads'):
            self.weights -= lr * self.weight_grads
            # Re-apply mask to ensure zero weights stay zero (redundant but safe)
            self.weights *= self.mask
    
    def activation(self, x: np.ndarray) -> np.ndarray:
        """Identity activation for linear layer."""
        return x

    def get_weights(self) -> np.ndarray:
        return self.weights

    def set_weights(self, weights: np.ndarray) -> None:
        self.weights = weights
        
    def reset_state(self) -> None:
        self.input_cache = None

    def _init_sparse_mask(self, density: float, block_size: int) -> np.ndarray:
        xp = get_array_module()
        if block_size == 1:
            # Unstructured sparsity
            mask = xp.random.random(self.weights.shape) < density
        else:
            # Structured (block) sparsity for GPU efficiency
            h, w = self.weights.shape
            block_h, block_w = h // block_size, w // block_size
            # Create block mask on CPU first then move if needed (easier for kron)
            # Or use pure xp if available
            block_mask = xp.random.random((block_h, block_w)) < density
            mask = xp.kron(block_mask, xp.ones((block_size, block_size)))
        return mask.astype(np.float32)
```

### 3.2 Sparse Hebbian Attention

#### Mathematical Formulation

**Standard Attention**:
$$ A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V $$

**Sparse Hebbian Attention**:
$$ A_{sparse} = \text{k-WTA}\left(\frac{QK^T}{\sqrt{d_k}} \odot M_{structural}\right) V $$

Where:
- $M_{structural}$: The binary connectivity mask (learned topology)
- $\text{k-WTA}$: k-Winners-Take-All activation (top $k$ attention scores per query)
- $k = \lfloor \alpha \cdot N \rfloor$ where $\alpha \in [0.1, 0.3]$ is the activation sparsity target

#### Class: `SparseAttentionLayer`

```python
from gaia.core.base import PlasticComponent
from gaia.core.gpu_utils import get_array_module, sqrt, matmul, softmax

class SparseAttentionLayer(Layer, PlasticComponent):
    """
    Sparse attention with structural plasticity.
    """
    def __init__(self, d_model: int, n_heads: int, 
                 sparsity_target: float = 0.1,
                 activation_sparsity: float = 0.2,
                 block_size: int = 8):
        self.d_model = d_model
        self.n_heads = n_heads
        self.sparsity_target = sparsity_target
        self.activation_sparsity = activation_sparsity
        
        # Projections with masked connectivity
        self.q_proj = MaskedLinear(d_model, d_model, sparsity_target, block_size)
        self.k_proj = MaskedLinear(d_model, d_model, sparsity_target, block_size)
        self.v_proj = MaskedLinear(d_model, d_model, sparsity_target, block_size)
        
        # Structural optimizer (rewiring engine)
        self.topology_optimizer = StructuralOptimizer(
            prune_threshold=0.01,
            regrow_fraction=0.1,
            consolidation_strength=0.5
        )
        
        # For top-down modulation
        self.context_bias = None
        self.scores_cache = None
        self.attn_weights_cache = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        xp = get_array_module()
        
        # Linear projections
        Q = self.q_proj.forward(x)
        K = self.k_proj.forward(x)
        V = self.v_proj.forward(x)
        
        # Apply context bias from higher level (if available)
        if self.context_bias is not None:
            Q = Q + self.context_bias
        
        # Compute attention scores
        # scores: (batch, seq, seq) or (batch, heads, seq, seq) 
        # Simplified here as (batch, d_model, d_model) for illustration
        scores = matmul(Q, K.T) / sqrt(self.d_model)
        self.scores_cache = scores
        
        # Apply k-WTA activation sparsity
        k = int(self.activation_sparsity * scores.shape[-1])
        topk_mask = self._kwta_mask(scores, k)
        sparse_scores = scores * topk_mask
        
        # Softmax over non-zero entries (masked softmax)
        # Using numeric stability trick: exp(x - max)
        attn_weights = self._masked_softmax(sparse_scores, topk_mask)
        self.attn_weights_cache = attn_weights
        
        return matmul(attn_weights, V)
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass for sparse attention.
        
        Simplified flow:
        d_Out -> d_V, d_Attn
        d_Attn -> d_Q, d_K (approximate through k-WTA)
        """
        xp = get_array_module()
        # TODO: Implement full attention gradient chain
        # 1. Propagate through V projection
        # d_V = attn_weights.T @ grad
        # grad_v_in = self.v_proj.backward(d_V)
        
        # 2. Propagate through Attention weights
        # d_Attn = grad @ V.T
        
        # 3. Propagate through Softmax & k-WTA
        # Note: k-WTA is non-differentiable effectively, but we use straight-through 
        # or masked gradient approximation for active elements.
        
        # Placeholder for full implementation
        return xp.zeros_like(grad) 

    def update(self, lr: float) -> None:
        self.q_proj.update(lr)
        self.k_proj.update(lr)
        self.v_proj.update(lr)

    def activation(self, x: np.ndarray) -> np.ndarray:
        return x
        
    def get_weights(self) -> np.ndarray:
        # Aggregate weights for inspection
        return self.q_proj.get_weights()
        
    def set_weights(self, weights: np.ndarray) -> None:
        pass # Complex for multi-component layer

    def reset_state(self) -> None:
        self.context_bias = None
        self.scores_cache = None
        self.q_proj.reset_state()
        self.k_proj.reset_state()
        self.v_proj.reset_state()

    def top_down_modulate(self, context_vector: np.ndarray) -> None:
        """Receives signal from higher hierarchical level."""
        self.context_bias = context_vector
        
    def get_plasticity_params(self) -> dict:
        return {'sparsity_target': self.sparsity_target, 
                'activation_sparsity': self.activation_sparsity}

    def set_plasticity_params(self, params: dict) -> None:
        if 'sparsity_target' in params:
            self.sparsity_target = params['sparsity_target']
        if 'activation_sparsity' in params:
            self.activation_sparsity = params['activation_sparsity']

    def _kwta_mask(self, scores: np.ndarray, k: int) -> np.ndarray:
        xp = get_array_module()
        # Get k-th largest value per row
        # argsort is expensive, can use argpartition if available or approx
        indices = xp.argsort(scores, axis=-1)
        # Create mask
        mask = xp.zeros_like(scores)
        # This implementation is simplified; production should use faster top-k
        top_k_indices = indices[..., -k:]
        # Scatter 1s (requires advanced indexing depending on backend)
        # Placeholder logic:
        threshold = xp.min(xp.take_along_axis(scores, top_k_indices, axis=-1), axis=-1, keepdims=True)
        return (scores >= threshold).astype(np.float32)

    def _masked_softmax(self, scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
        xp = get_array_module()
        # Exp of masked scores (treat 0 as -inf for softmax purposes)
        # Warning: scores * mask preserves 0s. 
        # Proper way: scores masked with very negative number
        safe_scores = scores.copy()
        safe_scores[mask == 0] = -1e9
        
        exp_scores = xp.exp(safe_scores - xp.max(safe_scores, axis=-1, keepdims=True))
        exp_scores *= mask # Ensure zeroed are strictly zero
        return exp_scores / (xp.sum(exp_scores, axis=-1, keepdims=True) + 1e-8)
```

### 3.3 Structural Plasticity (Rewiring Engine)

#### The Rewiring Cycle

Runs every $T_{epoch}$ steps.

```python
class StructuralOptimizer:
    """
    Manages synaptic pruning and regrowth.
    """
    def __init__(self, prune_threshold: float = 0.01,
                 regrow_fraction: float = 0.1,
                 consolidation_strength: float = 0.5,
                 maturity_age: int = 50):
        self.prune_threshold = prune_threshold
        self.regrow_fraction = regrow_fraction
        self.consolidation_strength = consolidation_strength
        self.maturity_age = maturity_age
        
    def rewire(self, layer: MaskedLinear) -> dict:
        """
        Execute one pruning/regrowth cycle.
        """
        xp = get_array_module()
        metrics = {}
        
        # 1. PRUNING: Remove weak, mature synapses
        prune_candidates = (
            (xp.abs(layer.weights) < self.prune_threshold) &
            (layer.synaptic_age > self.maturity_age) &
            (layer.mask == 1)
        )
        
        # Apply consolidation: older synapses are harder to prune
        consolidation_factor = 1.0 / (1.0 + self.consolidation_strength * layer.synaptic_age)
        prune_prob = prune_candidates * consolidation_factor
        actual_prunes = prune_prob > xp.random.random(prune_prob.shape)
        
        layer.mask[actual_prunes] = 0
        layer.weights[actual_prunes] = 0
        layer.synaptic_age[actual_prunes] = 0
        metrics['pruned'] = int(xp.sum(actual_prunes))
        
        # 2. REGROWTH: Add new synapses where connectivity is low
        current_density = xp.mean(layer.mask)
        target_new = int(self.regrow_fraction * layer.mask.size)
        
        if current_density < layer.density:
            # Find inactive positions
            # Note: argwhere on GPU can be slow, might want to do this part on CPU
            # For specification purposes, we show the logic:
            
            # Move mask to CPU for complex index manipulation if needed
            mask_cpu = to_cpu(layer.mask)
            inactive_indices = np.argwhere(mask_cpu == 0)
            
            if len(inactive_indices) > 0:
                n_grow = min(target_new, len(inactive_indices))
                grow_idx = np.random.choice(len(inactive_indices), n_grow, replace=False)
                
                # Apply updates
                for idx in grow_idx:
                    i, j = inactive_indices[idx]
                    layer.mask[i, j] = 1
                    # Initialize with small random weight
                    layer.weights[i, j] = float(np.random.randn() * 0.01)
                    layer.synaptic_age[i, j] = 0
                    
                metrics['grown'] = n_grow
        
        # 3. INCREMENT AGE for all active synapses
        layer.synaptic_age += layer.mask
        
        metrics['density'] = float(xp.mean(layer.mask))
        return metrics
```

---

## 4. Robustness & Enterprise Features

### 4.1 Sparsity Scheduler (Gradual Sparsification)

Networks should start dense and gradually become sparse as they learn the important pathways.

```python
import numpy as np

class SparsityScheduler:
    """
    Controls the sparsification rate over training.
    
    Supports:
    - Linear decay
    - Cosine annealing
    - Step-wise reduction
    """
    def __init__(self, initial_density: float = 1.0,
                 target_density: float = 0.1,
                 warmup_steps: int = 1000,
                 total_steps: int = 10000,
                 schedule: str = 'cosine'):
        self.initial_density = initial_density
        self.target_density = target_density
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.schedule = schedule
        self.current_step = 0
        
    def get_density(self) -> float:
        """Get current target density based on schedule."""
        if self.current_step < self.warmup_steps:
            # Warmup: stay dense
            return self.initial_density
        
        progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        
        if self.schedule == 'linear':
            density = self.initial_density - progress * (self.initial_density - self.target_density)
        elif self.schedule == 'cosine':
            density = self.target_density + 0.5 * (self.initial_density - self.target_density) * (1 + np.cos(np.pi * progress))
        elif self.schedule == 'step':
            # 3 discrete steps
            if progress < 0.33:
                density = self.initial_density
            elif progress < 0.66:
                density = (self.initial_density + self.target_density) / 2
            else:
                density = self.target_density
        else:
            density = self.target_density
            
        return density
    
    def step(self) -> None:
        self.current_step += 1
```

### 4.2 Sparsity Monitor (Observability)

```python
from gaia.core.gpu_utils import get_array_module, to_cpu

class SparsityMonitor:
    """
    Tracks and logs sparsity-related metrics for observability.
    Handles GPU arrays gracefully by converting to CPU for statistics.
    """
    def __init__(self):
        self.history = {
            'density': [],
            'turnover': [],
            'entropy': [],
            'pruned': [],
            'grown': []
        }
        self._last_mask = None
        
    def log(self, layer: MaskedLinear, rewire_metrics: dict = None) -> dict:
        """Log metrics for a layer."""
        xp = get_array_module()
        metrics = {}
        
        # Connectivity Density
        density = float(xp.mean(layer.mask))
        metrics['density'] = density
        self.history['density'].append(density)
        
        # Synaptic Turnover (change rate)
        if self._last_mask is not None:
            # Compute turnover: fraction of connections that changed state
            diff = layer.mask != self._last_mask
            turnover = float(xp.mean(diff))
            metrics['turnover'] = turnover
            self.history['turnover'].append(turnover)
            
        # Store copy of mask (ensure it persists if layer updates in place)
        self._last_mask = layer.mask.copy()
        
        # Mask Entropy (structural organization)
        p = density
        if 0 < p < 1:
            entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)
        else:
            entropy = 0.0
        metrics['entropy'] = entropy
        self.history['entropy'].append(entropy)
        
        # Rewiring metrics
        if rewire_metrics:
            self.history['pruned'].append(rewire_metrics.get('pruned', 0))
            self.history['grown'].append(rewire_metrics.get('grown', 0))
            
        return metrics
    
    def get_summary(self) -> dict:
        """Get summary statistics."""
        return {
            'avg_density': np.mean(self.history['density'][-100:]),
            'avg_turnover': np.mean(self.history['turnover'][-100:]) if self.history['turnover'] else 0,
            'final_entropy': self.history['entropy'][-1] if self.history['entropy'] else 0,
            'total_pruned': sum(self.history['pruned']),
            'total_grown': sum(self.history['grown'])
        }
```

### 4.3 Stability Guardrails

```python
from gaia.core.gpu_utils import get_array_module, to_cpu

class StructuralGuardrails:
    """
    Prevents catastrophic topology changes.
    """
    def __init__(self, min_density: float = 0.05,
                 max_turnover: float = 0.1,
                 warmup_steps: int = 1000):
        self.min_density = min_density
        self.max_turnover = max_turnover
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self._frozen = True
        
    def should_rewire(self) -> bool:
        """Check if rewiring is allowed."""
        if self.current_step < self.warmup_steps:
            return False  # Structural warmup: no rewiring
        return True
    
    def validate_rewire(self, layer: MaskedLinear, proposed_prunes: np.ndarray) -> np.ndarray:
        """Validate and potentially limit pruning."""
        xp = get_array_module()
        
        # Ensure minimum connectivity per neuron
        # Using GPU reduction
        post_prune_density = float(xp.mean(layer.mask) - xp.mean(proposed_prunes))
        
        if post_prune_density < self.min_density:
            # Reduce pruning to meet minimum
            # Perform indices selection on CPU for ease
            proposed_prunes_cpu = to_cpu(proposed_prunes)
            
            n_to_keep = int((self.min_density - post_prune_density) * layer.mask.size)
            # Ensure n_to_keep is positive and valid
            n_to_keep = max(0, n_to_keep)
            
            prune_indices = np.argwhere(proposed_prunes_cpu)
            
            if len(prune_indices) > 0 and n_to_keep > 0:
                keep_idx = np.random.choice(len(prune_indices), min(n_to_keep, len(prune_indices)), replace=False)
                
                # Turn off pruning for kept indices
                limited_prunes = proposed_prunes.copy() # GPU copy
                if hasattr(limited_prunes, 'get'):
                     # If limited_prunes is GPU, might be complex to index with CPU numpy array
                     # Best to just manipulate on CPU and move back to GPU
                     limited_prunes_cpu = to_cpu(limited_prunes)
                     for idx in keep_idx:
                         limited_prunes_cpu[tuple(prune_indices[idx])] = 0
                     
                     # Move back
                     xp = get_array_module()
                     if xp != np:
                        limited_prunes = xp.asarray(limited_prunes_cpu)
                     else:
                        limited_prunes = limited_prunes_cpu
                else:
                    # CPU path
                     for idx in keep_idx:
                         limited_prunes[tuple(prune_indices[idx])] = 0

                return limited_prunes
                
        return proposed_prunes
    
    def step(self) -> None:
        self.current_step += 1
        if self.current_step >= self.warmup_steps:
            self._frozen = False
```

### 4.4 Synaptic Consolidation (Memory Protection)

Implements **Elastic Weight Consolidation (EWC)**-like protection:

```python
from gaia.core.gpu_utils import get_array_module

class SynapticConsolidation:
    """
    Protects important synapses from pruning.
    
    Synapses with:
    - High weight magnitude
    - High age (maturity)
    - High "importance" (Fisher information proxy)
    
    ...are protected from pruning.
    """
    def __init__(self, importance_decay: float = 0.99,
                 protection_threshold: float = 0.5):
        self.importance_decay = importance_decay
        self.protection_threshold = protection_threshold
        self.importance_scores = None
        
    def update_importance(self, layer: MaskedLinear, loss_grad: np.ndarray = None) -> None:
        """Update importance scores based on gradient magnitude."""
        xp = get_array_module()
        
        if self.importance_scores is None:
            self.importance_scores = xp.zeros_like(layer.weights)
        
        # Decay old importance
        self.importance_scores *= self.importance_decay
        
        # Add new importance from gradients (Fisher Information proxy)
        if loss_grad is not None:
            weight_grad = xp.abs(loss_grad)
            self.importance_scores += weight_grad ** 2
            
    def get_protection_mask(self, layer: MaskedLinear) -> np.ndarray:
        """Return mask of synapses that should NOT be pruned."""
        xp = get_array_module()
        
        if self.importance_scores is None:
            return xp.zeros_like(layer.mask, dtype=bool)
        
        # Normalize importance
        norm_importance = self.importance_scores / (xp.max(self.importance_scores) + 1e-8)
        
        # Combine with age-based protection
        age_factor = layer.synaptic_age / (xp.max(layer.synaptic_age) + 1)
        
        protection_score = 0.5 * norm_importance + 0.5 * age_factor
        
        return protection_score > self.protection_threshold
```

---

## 5. Meta-Learning Integration (ES Loop)

The sparsity parameters should be **learned**, not hardcoded.

### 5.1 Sparse Meta-Parameters

Add to ES optimization target:
```python
SPARSE_META_PARAMS = {
    'sparsity_target': (0.05, 0.5),      # Range: 5% to 50% density
    'prune_threshold': (0.001, 0.1),     # Weight magnitude threshold
    'activation_sparsity': (0.1, 0.5),   # kWTA fraction
    'regrow_fraction': (0.01, 0.2),      # Regrowth rate
    'consolidation_strength': (0.1, 1.0) # Memory protection
}
```

### 5.2 Integration with `PlasticityController`

```python
# In PlasticityController.__init__
self.sparse_params = {
    'sparsity_target': 0.1,
    'prune_threshold': 0.01,
    'activation_sparsity': 0.2
}

# In ES optimization loop
def _fitness_function(self, params):
    # Apply sparse params to layers
    for layer in self.sparse_layers:
        layer.sparsity_target = params['sparsity_target']
        layer.topology_optimizer.prune_threshold = params['prune_threshold']
    
    # Evaluate on task
    performance = self._evaluate_task()
    
    # Penalize high density (encourage efficiency)
    density_penalty = 0.1 * np.mean([l.get_density() for l in self.sparse_layers])
    
    return performance - density_penalty
```

---

## 6. Benchmark Methodology

### 6.1 Lottery Ticket Validation

To prove structural plasticity finds good sparse networks:

```python
def lottery_ticket_benchmark(model, dataset, sparsity_target=0.1):
    """
    Compare:
    1. Random sparse (baseline)
    2. Magnitude pruned (post-hoc)
    3. SAS-evolved (structural plasticity)
    """
    results = {}
    
    # 1. Random Sparse Baseline
    random_model = create_random_sparse_model(sparsity_target)
    results['random'] = evaluate(random_model, dataset)
    
    # 2. Magnitude Pruning (train dense, then prune)
    dense_model = train_dense(dataset)
    pruned_model = magnitude_prune(dense_model, sparsity_target)
    results['magnitude_pruned'] = evaluate(pruned_model, dataset)
    
    # 3. SAS (train with structural plasticity)
    sas_model = train_with_sas(dataset, sparsity_target)
    results['sas'] = evaluate(sas_model, dataset)
    
    # SAS should outperform random and approach magnitude pruned
    assert results['sas'] > results['random'] * 1.2, "SAS not better than random!"
    
    return results
```

### 6.2 Efficiency Benchmarks

| Metric | Target |
| :--- | :--- |
| Memory reduction | ≥70% vs dense |
| FLOPs reduction | ≥60% vs dense |
| Performance retention | ≥95% of dense |
| Steps to converge | ≤1.2x dense |

---

## 7. Implementation 

- [ ] Implement `MaskedLinear` with block sparsity support
- [ ] Unit tests for mask operations
- [ ] Gradient verification (masked weights have zero grad)
- [ ] Implement `SparseAttentionLayer`
- [ ] k-WTA activation function
- [ ] Integration with existing `Layer` interface
- [ ] Implement `StructuralOptimizer`
- [ ] Implement `SynapticConsolidation`
- [ ] Implement `SparsityScheduler`
- [ ] Implement `StructuralGuardrails`
- [ ] Implement `SparsityMonitor`
- [ ] Dashboard/visualization script
- [ ] Logging integration
- [ ] Add sparse params to ES loop
- [ ] Implement fitness with density penalty
- [ ] End-to-end meta-learning test
- [ ] Top-down modulation in `HierarchyManager`
- [ ] Level-specific sparsity targets
- [ ] Full v4.2 integration test

---

## 8. Risk Matrix & Mitigations

| Risk | Impact | Probability | Mitigation |
| :--- | :--- | :--- | :--- |
| **Death by Sparsity** | High | Medium | `StructuralGuardrails` + min density |
| **Memory Loss** | High | Medium | `SynapticConsolidation` + EWC |
| **Performance Overhead** | Medium | High | Block sparsity + CuPy kernels |
| **Unstable Rewiring** | High | Low | Warmup phase + turnover limit |
| **Lottery Ticket Failure** | Medium | Medium | Benchmark validation gate |

---

## 9. Success Criteria

**v4.2 is considered complete when:**

1. ✅ All tests pass with sparse attention layers
2. ✅ Lottery ticket benchmark shows SAS > Random × 1.2
3. ✅ Memory usage reduced by ≥70% at 10% density
4. ✅ SparsityMonitor shows stable turnover < 5% after warmup
5. ✅ ES successfully optimizes sparsity parameters
6. ✅ Hierarchical integration passes integration tests

---

## 10. References

- [Lottery Ticket Hypothesis (Frankle & Carlin, 2019)](https://arxiv.org/abs/1803.03635)
- [Rigging the Lottery (Evci et al., 2020)](https://arxiv.org/abs/1911.11134)
- [Elastic Weight Consolidation (Kirkpatrick et al., 2017)](https://arxiv.org/abs/1612.00796)
- [Block Sparsity in Neural Networks (NVIDIA)](https://developer.nvidia.com/blog/accelerating-sparse-deep-neural-networks/)
- [Structural Plasticity in Spiking Neural Networks (Zenke et al., 2017)](https://www.nature.com/articles/ncomms15624)
