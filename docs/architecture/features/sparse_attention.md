# Technical Specification: Sparse Attention & Structural Plasticity (SAS)

**Target Version**: v4.2  
**Status**: Planning  
**Module**: `gaia.layers.attention`

## 1. Architectural Philosophy
The goal of the SAS system is to transition GAIA from a **Passive/Dense** processing model to an **Active/Sparse** model. 
- **Passive vs Active**: Instead of processing all inputs, the system actively queries relevant features (Attention).
- **Dense vs Sparse**: Instead of maintaining full connectivity ($O(N^2)$), the system maintains a dynamic set of active synapses ($O(N \log N)$ or $O(N)$).

## 2. Component Design

### 2.1 The Sparse Tensor Core
Since we are aiming for biological plausibility and efficiency, we cannot rely on standard dense matrix multiplication. However, for the python/numpy implementation, we will use **Masked Dense Matrices** as a functionally equivalent prototype before moving to true sparse CSR/COO formats in the GPU/CuPy backend.

#### Class: `MaskedLinear`
Wraps a standard weight matrix but enforces a binary topology mask.
```python
class MaskedLinear(Layer):
    """
    Linear layer with enforced sparsity topology.
    """
    def __init__(self, in_features, out_features, density=0.1):
        self.weights = ... # initialized sparsely
        self.mask = ...    # binary mask (1=active, 0=inactive)
        
    def forward(self, x):
        # W_eff = W * Mask
        effective_weights = self.weights * self.mask
        return x @ effective_weights.T
```

### 2.2 Sparse Hebbian Attention
The core innovation is combining Dot-Product Attention with Hebbian learning rules.

#### Mathematical Formulation
Standard Attention:
$$ A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V $$

**Sparse Hebbian Attention**:
$$ A_{sparse} = \text{k-WTA}\left(\frac{QK^T}{\sqrt{d_k}} \odot M_{structural}\right) V $$

Where:
- $M_{structural}$: The binary connectivity mask.
- $\text{k-WTA}$: "k-Winners-Take-All" activation, which zeros out all but the top $k$ attention scores. This enforces **Activation Sparsity**.

### 2.3 Structural Plasticity (Rewiring)
Deep learning typically uses fixed graphs. SAS introduces dynamic graph topology.

**The Rewiring Cycle (runs every $T_{epoch}$):**
1.  **Synaptic Pruning**: 
    If $|w_{ij}| < \theta_{prune}$ AND $\text{age}_{ij} > T_{mature}$:
    $$ M_{ij} \leftarrow 0, \quad w_{ij} \leftarrow 0 $$
    
2.  **Synaptic Growth (Genesis)**:
    For neurons with low total connectivity ($ \sum_j M_{ij} < K_{target} $):
    - Sample random potential connections.
    - Evaluate potential Hebbian correlation (Gradient alignment).
    - Enable best candidates: $ M_{new} \leftarrow 1 $.

## 3. Data Structures & Interfaces

### 3.1 New Layer Interface within `gaia.layers`
File: `gaia/layers/attention.py`

```python
class SparseAttentionLayer(Layer, PlasticComponent):
    def __init__(self, d_model, n_heads, sparsity_target=0.1):
        self.q_proj = MaskedLinear(d_model, d_model)
        self.k_proj = MaskedLinear(d_model, d_model)
        self.v_proj = MaskedLinear(d_model, d_model)
        self.topology_optimizer = StructuralOptimizer(...)

    def top_down_modulate(self, context_vector):
        """
        Receives signal from higher hierarchical level to bias 
        the query generation.
        """
        pass
```

### 3.2 Feedback Integration in `HierarchyManager`
The Hierarchy currently flows strictly Bottom-Up. We must introduce a Top-Down feedback phase.

**Execution Loop Update**:
1.  **Forward (Bottom-Up)**: Sensory -> Level 1 -> Level 2
2.  **Feedback (Top-Down)**: Level 2 -> Level 1 (Attention Biasing)
    - *Note*: This might require a "predictive coding" cycle where feedback serves as a prediction.

## 4. Implementation Phase Plan

### Phase 1: The Masked Primitive (Week 1)
- **Goal**: Create `MaskedLinear` that passes all gradient checks.
- **Verification**: Ensure that gradients for masked-out weights are exactly zero.

### Phase 2: The Attention Mechanism (Week 2)
- **Goal**: Implement `SparseAttentionLayer` with fixed random sparsity.
- **Verification**: Compare performance against a dense Attention layer on a simple "Needle in a Haystack" synthetic benchmark.

### Phase 3: Structural Plasticity (Week 3-4)
- **Goal**: Implement the Pruning/Regrowth logic (`StructuralOptimizer`).
- **Verification**: Visualize the adjacency matrix evolving from random -> structured.

### Phase 4: Hierarchical Integration (Week 5)
- **Goal**: Hook into `HierarchyManager`.
- **Verification**: Demonstrate Level 2 successfully attending to specific features in Level 1 based on a high-level task directive.

## 5. Potential Pitfalls & Mitigations
| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| **"Death by Sparsity"** | Network disconnects and gradients die. | Enforce minimum connectivity per neuron. |
| **Performance Overhead** | Masking in Python is slower than dense. | Accept this for the prototype; move to custom CuPy kernels for prod. |
| **Unstable Rewiring** | Topology changes too fast, erasing memory. | Use "Synaptic Consolidation" (weights that survive long get harder to prune). |
