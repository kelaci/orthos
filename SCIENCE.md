# Theoretical Foundations of ORTHOS

## 🧠 Overview

ORTHOS (Orthogonal Recursive Hierarchical Optimization System) is built upon a convergence of computational neuroscience principles, information theory, and modern machine learning techniques. This document provides the scientific foundation underlying ORTHOS's design.

---

## 1. The Free Energy Principle

### 1.1 Core Concept

The **Free Energy Principle** (FEP), developed by Karl Friston, proposes that all adaptive systems minimize a quantity called *variational free energy*. In the context of ORTHOS:

```
F = E_q[log q(s) - log p(o,s)]
```

Where:
- `F` = Variational Free Energy
- `q(s)` = Recognition density (beliefs about hidden states)
- `p(o,s)` = Generative model (joint probability of observations and states)

### 1.2 Expected Free Energy (EFE)

ORTHOS uses **Expected Free Energy** for action selection:

```python
def approximate_efe(mean: Tensor, std: Tensor) -> Tensor:
    """
    Expected Free Energy: pragmatic + epistemic value
    
    EFE = E[log p(o|π) - log q(o|π)] + E[log q(s|o,π) - log q(s|π)]
           ↑ Pragmatic value           ↑ Epistemic value (information gain)
    """
    pragmatic = F.mse_loss(mean, preferred_state, reduction="none").mean(-1)
    epistemic = std.mean(-1)  # Uncertainty as proxy for information gain
    return pragmatic - exploration_weight * epistemic
```

### 1.3 Active Inference

Active Inference is a corollary of FEP where agents:
1. **Perceive** by updating beliefs to minimize prediction error
2. **Act** by selecting actions that minimize expected free energy

In ORTHOS's action selection:

```python
# Sample actions from policy
actions = tanh(mean + std * randn_like(mean))

# Evaluate EFE for each action
mean_next, std_next = world_model(state, actions)
efe = approximate_efe(mean_next, std_next)

# Select action with lowest EFE (softmax selection)
probs = softmax(-efe / temperature, dim=0)
```

---

## 2. Hebbian Plasticity

### 2.1 Classical Hebbian Learning

> "Neurons that fire together, wire together" — Donald Hebb (1949)

The basic Hebbian rule:

```
Δw_ij = η * x_i * y_j
```

Where:
- `Δw_ij` = Weight change
- `η` = Learning rate
- `x_i` = Pre-synaptic activity
- `y_j` = Post-synaptic activity

### 2.2 ORTHOS's Plasticity Rules

ORTHOS implements multiple biologically-inspired rules:

| Rule | Formula | Purpose |
|------|---------|---------|
| **Hebbian** | `Δw = η * pre * post` | Correlation learning |
| **Oja's** | `Δw = η * post * (pre - post * w)` | PCA-like normalization |
| **BCM** | `Δw = η * post * (post - θ) * pre` | Sliding threshold |
| **STDP** | Timing-dependent | Temporal precision |

### 2.3 Dual-Timescale Plasticity

ORTHOS's innovation: **separate fast and slow traces** mimicking biological memory consolidation:

```python
# Fast trace: rapid adaptation (hippocampus-like)
fast_trace = fast_trace * τ_fast + η_fast * hebbian_update
# τ_fast = 0.95, η_fast = 0.05

# Slow trace: consolidation (neocortex-like)
slow_trace = slow_trace * τ_slow + η_slow * fast_trace
# τ_slow = 0.99, η_slow = 0.01
```

**Biological Analogy:**
- **Fast trace**: Hippocampal rapid encoding
- **Slow trace**: Neocortical consolidation during replay/sleep

---

## 3. Memory Consolidation Theory

### 3.1 Complementary Learning Systems

ORTHOS's dual-timescale design aligns with the **Complementary Learning Systems** (CLS) theory:

```
┌─────────────────────────────────────────────────────────────┐
│                    ORTHOS Memory System                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Fast Trace (τ=0.95)          Slow Trace (τ=0.99)          │
│  ┌───────────────┐            ┌───────────────┐            │
│  │  Hippocampus  │───────────▶│   Neocortex   │            │
│  │  Analogue     │            │   Analogue    │            │
│  └───────────────┘            └───────────────┘            │
│        │                            │                       │
│        ▼                            ▼                       │
│  • Rapid encoding            • Gradual consolidation       │
│  • Pattern separation        • Pattern completion          │
│  • High learning rate        • Low learning rate           │
│  • Volatile storage          • Stable storage              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Synaptic Tagging and Capture

Future extension: **Synaptic Tagging** for selective consolidation:

```python
# Theoretical extension
if hebbian_update.norm() > tag_threshold:
    tag = create_synaptic_tag(hebbian_update)
    if plasticity_related_proteins_available():
        consolidate(tag, slow_trace)
```

---

## 4. Homeostatic Regulation

### 4.1 The Stability Problem

Hebbian learning is inherently **unstable** — positive feedback can cause unbounded weight growth:

```
w → w + η*x*y → larger y → larger Δw → even larger y → explosion
```

### 4.2 ORTHOS's Homeostatic Mechanisms

**Mechanism 1: Trace Normalization**
```python
fast_norm = fast_trace.norm()
if fast_norm > homeostatic_target:  # target = 5.0
    fast_trace *= homeostatic_target / (fast_norm + 1e-6)
```

**Mechanism 2: Weight Decay**
```python
weights *= (1.0 - decay_rate)  # decay_rate = 0.001
```

**Mechanism 3: BCM Sliding Threshold**
```python
θ = 0.9 * θ + 0.1 * mean(post_activity)
# θ adapts to maintain balanced LTP/LTD
```

### 4.3 Biological Basis

These mechanisms mirror biological homeostasis:
- **Synaptic scaling**: Global multiplicative adjustment
- **Intrinsic plasticity**: Adjustment of neuronal excitability
- **Metaplasticity**: Plasticity of plasticity

---

## 5. Quantization Theory

### 5.1 BitNet Architecture

ORTHOS incorporates **1.58-bit quantization** for deployment efficiency:

```python
def bitnet_quantize(w: Tensor) -> Tensor:
    """
    Quantize weights to {-1, 0, +1} with per-row scaling
    
    Benefits:
    - 10x memory reduction (32-bit → 1.58-bit)
    - Hardware-friendly (additions replace multiplications)
    - Maintains expressivity through scaling factors
    """
    scale = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
    w_normalized = w / scale
    w_quantized = w_normalized.round().clamp(-1, 1)
    return w_quantized * scale
```

### 5.2 Hybrid Digital-Analog Computing

ORTHOS's architecture is **hybrid**:

```
┌──────────────────────────────────────────────────────────────┐
│                     Effective Weight                         │
│                                                              │
│   w_effective = w_static + 0.1 * fast_trace + 0.05 * slow   │
│                    ↑              ↑                ↑         │
│               Quantized      Continuous       Continuous     │
│               (digital)      (analog)         (analog)       │
│                                                              │
│   This mimics biological systems where:                      │
│   - Dendritic structure = fixed (digital-like)               │
│   - Synaptic strengths = plastic (analog)                    │
└──────────────────────────────────────────────────────────────┘
```

---

## 6. Ensemble Methods & Uncertainty

### 6.1 Epistemic vs Aleatoric Uncertainty

ORTHOS distinguishes two types of uncertainty:

| Type | Source | Reducible? | ORTHOS's Measure |
|------|--------|------------|----------------|
| **Epistemic** | Model uncertainty | Yes (more data) | Ensemble disagreement |
| **Aleatoric** | Data noise | No | Intrinsic variance |

### 6.2 Ensemble Uncertainty Estimation

```python
class EnsembleWorldModel:
    def forward(self, state, action):
        # Each ensemble member makes prediction
        preds = stack([m(state, action) for m in self.models])
        
        # Mean = expected prediction
        # Std = epistemic uncertainty
        return preds.mean(dim=0), preds.std(dim=0)
```

### 6.3 Uncertainty-Driven Exploration

Epistemic uncertainty drives exploration (Active Inference):

```python
# Higher uncertainty → more information gain → lower EFE → preferred action
efe = pragmatic_value - exploration_weight * epistemic_uncertainty
```

---

## 7. Mathematical Framework

### 7.1 Formal Definitions

**Definition 1 (Plastic Linear Layer)**:
```
y = W_eff · x
W_eff = Q(W_static) + α_fast · T_fast + α_slow · T_slow

where:
- Q: Quantization function
- T_fast, T_slow: Trace matrices
- α_fast = 0.1, α_slow = 0.05
```

**Definition 2 (Trace Dynamics)**:
```
T_fast(t+1) = τ_fast · T_fast(t) + η_fast · H(x, y)
T_slow(t+1) = τ_slow · T_slow(t) + η_slow · T_fast(t)

where H(x, y) = ReLU(y)^T · x / batch_size
```

**Definition 3 (Homeostatic Constraint)**:
```
||T_fast|| ≤ H_target

Enforced via: T_fast ← T_fast · min(1, H_target / ||T_fast||)
```

### 7.2 Stability Analysis

**Theorem (Bounded Trace Norms)**:
Under homeostatic regulation with target H, the trace norms satisfy:
```
lim sup ||T_fast(t)|| ≤ H
       t→∞
```

**Proof sketch**:
The normalization step ensures ||T_fast|| ≤ H immediately after any update where the norm exceeds H. Combined with the decay factor τ < 1, this guarantees bounded dynamics.

---

## 8. References

1. Friston, K. (2010). The free-energy principle: a unified brain theory?
2. Hebb, D.O. (1949). The Organization of Behavior
3. McClelland, J.L., et al. (1995). Why there are complementary learning systems
4. Oja, E. (1982). Simplified neuron model as a principal component analyzer
5. Bienenstock, E.L., et al. (1982). Theory for the development of neuron selectivity (BCM)
6. Ma, S., et al. (2024). The Era of 1-bit LLMs (BitNet)
7. Turrigiano, G.G. (2008). The self-tuning neuron: synaptic scaling of excitatory synapses

---

*This document provides the theoretical foundation for ORTHOS's design choices. For implementation details, see [Advanced Plasticity](../architecture/advanced-plasticity.md).*# Research Extensions & Future Directions

## 🔮 Overview

This document outlines promising research directions for extending ORTHOS's capabilities. These extensions are grounded in neuroscience literature and aim to enhance the system's learning efficiency, adaptability, and biological plausibility.

---

## 1. Meta-Plasticity: Learning to Learn

### 1.1 Concept

**Meta-plasticity** is the "plasticity of plasticity" — the ability to dynamically adjust learning rules themselves based on experience.

### 1.2 Current ORTHOS State

ORTHOS currently uses fixed trace decay rates:
```python
fast_trace_decay = 0.95  # Fixed
slow_trace_decay = 0.99  # Fixed
```

### 1.3 Proposed Extension

**Learnable Decay Rates:**

```python
class MetaPlasticLinear(DiagnosticPlasticLinear):
    """
    Extension with learnable plasticity parameters
    """
    
    def __init__(self, in_features: int, out_features: int, cfg: OrthosConfig):
        super().__init__(in_features, out_features, cfg)
        
        # Meta-parameters (learnable)
        self.log_fast_decay = nn.Parameter(torch.tensor(np.log(0.95)))
        self.log_slow_decay = nn.Parameter(torch.tensor(np.log(0.99)))
        self.log_fast_lr = nn.Parameter(torch.tensor(np.log(0.05)))
        
    @property
    def fast_decay(self):
        return torch.sigmoid(self.log_fast_decay)  # Ensures (0, 1)
    
    @property
    def slow_decay(self):
        return torch.sigmoid(self.log_slow_decay)
```

**Meta-Learning Objective:**

```python
def meta_loss(agent, task_batch):
    """
    MAML-style meta-learning for plasticity parameters
    
    Objective: Maximize adaptation speed across tasks
    """
    meta_grad = 0
    
    for task in task_batch:
        # Inner loop: adapt with current plasticity params
        adapted_agent = inner_adapt(agent, task, steps=10)
        
        # Outer loop: evaluate adaptation quality
        performance = evaluate(adapted_agent, task.test_set)
        meta_grad += compute_grad(performance, agent.meta_params)
    
    return meta_grad / len(task_batch)
```

### 1.4 Expected Benefits

| Benefit | Description |
|---------|-------------|
| Task-Adaptive | Decay rates adjust per-task |
| Sample Efficiency | Faster adaptation to new domains |
| Biological Plausibility | Mirrors BCM-style metaplasticity |

### 1.5 Research Questions

1. What is the optimal timescale for meta-parameter updates?
2. Should meta-parameters be global or per-layer?
3. How to prevent meta-parameter collapse?

---

## 2. Attention over Trace History

### 2.1 Concept

Instead of simple exponential decay, use **attention mechanisms** to selectively retrieve relevant past experiences from trace history.

### 2.2 Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Attention-Based Trace                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Trace History: [T₁, T₂, T₃, ..., Tₙ]                        │
│                      │                                          │
│                      ▼                                          │
│              ┌───────────────┐                                  │
│              │    Query      │                                  │
│   Current ──▶│   Attention   │──▶ Weighted Trace               │
│   Context    │   (Q, K, V)   │                                  │
│              └───────────────┘                                  │
│                                                                 │
│   Output: ∑ᵢ αᵢ · Tᵢ  where αᵢ = softmax(Q · Kᵢ)              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Implementation Sketch

```python
class AttentiveTraceMemory(nn.Module):
    """
    Attention-based trace retrieval
    
    Instead of exponential decay, selectively attend to
    relevant historical traces based on current context.
    """
    
    def __init__(self, trace_dim: int, history_len: int = 100, n_heads: int = 4):
        super().__init__()
        self.history_len = history_len
        
        # Trace history buffer
        self.register_buffer("trace_history", 
                           torch.zeros(history_len, trace_dim))
        self.write_ptr = 0
        
        # Attention components
        self.query_proj = nn.Linear(trace_dim, trace_dim)
        self.key_proj = nn.Linear(trace_dim, trace_dim)
        self.value_proj = nn.Linear(trace_dim, trace_dim)
        self.n_heads = n_heads
    
    def write(self, trace: torch.Tensor):
        """Store new trace in history"""
        self.trace_history[self.write_ptr] = trace.detach()
        self.write_ptr = (self.write_ptr + 1) % self.history_len
    
    def read(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve weighted combination of historical traces"""
        Q = self.query_proj(query)
        K = self.key_proj(self.trace_history)
        V = self.value_proj(self.trace_history)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.T) / np.sqrt(Q.shape[-1])
        weights = F.softmax(scores, dim=-1)
        
        return torch.matmul(weights, V)
```

### 2.4 Biological Inspiration

This mirrors **hippocampal memory indexing**:
- Hippocampus stores episodic traces
- Retrieval is cue-dependent (attention-like)
- Consolidation is selective (not all traces equal)

### 2.5 Research Questions

1. What should the query be? (Current input? Prediction error?)
2. How to balance recency vs. relevance?
3. Computational cost vs. benefit tradeoff?

---

## 3. Multi-Scale Temporal Hierarchies

### 3.1 Concept

Extend from dual-timescale (fast/slow) to **multi-scale** temporal hierarchies with ultrafast, fast, medium, slow, and ultraslow traces.

### 3.2 Biological Motivation

Different brain regions operate at different timescales:

| Timescale | Duration | Brain Region | Function |
|-----------|----------|--------------|----------|
| Ultrafast | ~10ms | Sensory cortex | Immediate perception |
| Fast | ~100ms | PFC | Working memory |
| Medium | ~1s | Hippocampus | Episode encoding |
| Slow | ~minutes | Striatum | Skill learning |
| Ultraslow | ~hours/days | Neocortex | Semantic memory |

### 3.3 Proposed Extension

```python
class MultiScaleTraceLayer(nn.Module):
    """
    Multi-scale temporal trace system
    
    Implements 5 timescales: ultrafast → ultraslow
    """
    
    TIMESCALES = {
        'ultrafast': {'decay': 0.8, 'lr': 0.1, 'weight': 0.2},
        'fast':      {'decay': 0.95, 'lr': 0.05, 'weight': 0.15},
        'medium':    {'decay': 0.99, 'lr': 0.02, 'weight': 0.1},
        'slow':      {'decay': 0.999, 'lr': 0.005, 'weight': 0.05},
        'ultraslow': {'decay': 0.9999, 'lr': 0.001, 'weight': 0.02},
    }
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        
        # Static weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        
        # Multi-scale traces
        for name, params in self.TIMESCALES.items():
            self.register_buffer(f"{name}_trace", 
                               torch.zeros(out_features, in_features))
    
    def update_all_traces(self, hebbian_delta: torch.Tensor):
        """Update all timescale traces"""
        with torch.no_grad():
            for name, params in self.TIMESCALES.items():
                trace = getattr(self, f"{name}_trace")
                trace.mul_(params['decay'])
                trace.add_(hebbian_delta, alpha=params['lr'])
                
                # Homeostatic normalization per scale
                norm = trace.norm()
                if norm > 5.0 * params['weight']:
                    trace.mul_(5.0 * params['weight'] / (norm + 1e-6))
    
    def get_effective_modulation(self) -> torch.Tensor:
        """Combine all traces weighted by timescale importance"""
        modulation = torch.zeros_like(self.weight)
        for name, params in self.TIMESCALES.items():
            trace = getattr(self, f"{name}_trace")
            modulation += params['weight'] * trace
        return modulation
```

### 3.4 Hierarchy Visualization

```
┌─────────────────────────────────────────────────────────────────┐
│              Multi-Scale Temporal Hierarchy                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Ultraslow (τ=0.9999) ─────────────────────────────────────── │
│   Slow (τ=0.999) ─────────────────────────────────             │
│   Medium (τ=0.99) ─────────────────────────                    │
│   Fast (τ=0.95) ─────────────────                              │
│   Ultrafast (τ=0.8) ─────────                                  │
│                                                                 │
│   Time →                                                        │
│   ════════════════════════════════════════════════════════════ │
│                                                                 │
│   Effective = Σᵢ wᵢ · Traceᵢ                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.5 Research Questions

1. Optimal number of timescales?
2. How should weights be distributed across scales?
3. Should timescales be task-dependent?

---

## 4. Synaptic Tagging & Capture

### 4.1 Concept

**Synaptic Tagging and Capture (STC)** is a biological mechanism for selective memory consolidation:
- Strong activation creates a "tag" at synapses
- Tags capture plasticity-related proteins (PRPs)
- Only tagged synapses undergo long-term consolidation

### 4.2 Implementation Sketch

```python
class TaggingPlasticLinear(DiagnosticPlasticLinear):
    """
    Synaptic Tagging and Capture implementation
    
    Key mechanisms:
    1. Tags set by strong Hebbian updates
    2. PRPs (proteins) enable consolidation
    3. Only tagged synapses consolidate to slow trace
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Synaptic tags (binary mask)
        self.register_buffer("synaptic_tags", 
                           torch.zeros_like(self.fast_trace))
        
        # Tag decay (tags fade without capture)
        self.tag_decay = 0.95
        
        # Tag threshold
        self.tag_threshold = 0.1
        
        # PRP availability (global signal)
        self.prp_level = 0.0
    
    def update_traces(self, x: torch.Tensor, y: torch.Tensor):
        """Enhanced update with tagging mechanism"""
        with torch.no_grad():
            # Standard Hebbian update
            y_active = F.relu(y)
            delta = torch.matmul(y_active.t(), x) / x.shape[0]
            
            # === Tagging ===
            # Strong updates create tags
            strong_update_mask = delta.abs() > self.tag_threshold
            self.synaptic_tags[strong_update_mask] = 1.0
            
            # Decay existing tags
            self.synaptic_tags *= self.tag_decay
            
            # === Fast trace (always updated) ===
            self.fast_trace.mul_(self.cfg.fast_trace_decay)
            self.fast_trace.add_(delta, alpha=self.cfg.fast_trace_lr)
            
            # === Slow trace (only tagged synapses) ===
            if self.prp_level > 0.5:  # PRPs available
                tagged_fast = self.fast_trace * self.synaptic_tags
                self.slow_trace.mul_(self.cfg.slow_trace_decay)
                self.slow_trace.add_(tagged_fast, alpha=self.cfg.slow_trace_lr)
                
                # Clear captured tags
                self.synaptic_tags *= (1 - self.synaptic_tags * 0.5)
    
    def inject_prp(self, amount: float = 1.0):
        """Simulate PRP release (e.g., from reward signal)"""
        self.prp_level = min(1.0, self.prp_level + amount)
    
    def decay_prp(self):
        """PRPs decay over time"""
        self.prp_level *= 0.99
```

### 4.3 Biological Relevance

```
┌─────────────────────────────────────────────────────────────────┐
│              Synaptic Tagging & Capture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Weak Input ───▶ No Tag ───▶ Fast Trace Only (decays)         │
│                                                                 │
│   Strong Input ──▶ TAG SET ──┬─▶ Fast Trace                    │
│                              │                                  │
│                              └─▶ If PRPs available:            │
│                                  TAG CAPTURED → Slow Trace     │
│                                                                 │
│   PRPs released by:                                             │
│   • Reward signals                                              │
│   • Novelty detection                                           │
│   • Emotional arousal                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 Research Questions

1. What triggers PRP release in artificial systems?
2. Optimal tag threshold?
3. Can tagging be learned end-to-end?

---

## 5. Neuromodulation

### 5.1 Concept

**Neuromodulation** refers to the global regulation of neural activity by diffuse neurotransmitter systems (dopamine, serotonin, norepinephrine, acetylcholine).

### 5.2 Current ORTHOS State

Fixed plasticity parameters:
```python
fast_trace_lr = 0.05  # Always the same
```

### 5.3 Proposed Extension

```python
class NeuromodulatedPlasticLinear(DiagnosticPlasticLinear):
    """
    Context-dependent plasticity via neuromodulation
    
    Neuromodulators:
    - Dopamine (DA): Reward prediction, motivation
    - Norepinephrine (NE): Arousal, attention
    - Acetylcholine (ACh): Learning, memory encoding
    - Serotonin (5-HT): Mood, behavioral flexibility
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Neuromodulator levels (normalized 0-1)
        self.modulators = {
            'dopamine': 0.5,
            'norepinephrine': 0.5,
            'acetylcholine': 0.5,
            'serotonin': 0.5,
        }
    
    def compute_effective_lr(self) -> float:
        """
        Compute effective learning rate based on neuromodulator state
        
        DA high → higher LR (reward-driven learning)
        ACh high → higher LR (attention/encoding)
        NE high → broader exploration
        5-HT high → more stable, less reactive
        """
        da = self.modulators['dopamine']
        ach = self.modulators['acetylcholine']
        ne = self.modulators['norepinephrine']
        sht = self.modulators['serotonin']
        
        # Effective learning rate
        base_lr = self.cfg.fast_trace_lr
        modulation = (da * 0.4 + ach * 0.4 + ne * 0.1 + (1 - sht) * 0.1)
        
        return base_lr * (0.5 + modulation)
    
    def compute_exploration_noise(self) -> float:
        """NE increases exploration"""
        return 0.1 * self.modulators['norepinephrine']
    
    def set_modulator(self, name: str, level: float):
        """Set neuromodulator level"""
        self.modulators[name] = np.clip(level, 0, 1)
    
    def process_reward(self, reward: float):
        """
        Update neuromodulators based on reward
        
        Positive reward → DA spike
        Negative reward → NE spike (arousal)
        Neutral → ACh boost (attention)
        """
        if reward > 0.5:
            self.modulators['dopamine'] = min(1.0, self.modulators['dopamine'] + 0.2)
        elif reward < -0.5:
            self.modulators['norepinephrine'] = min(1.0, self.modulators['norepinephrine'] + 0.2)
        else:
            self.modulators['acetylcholine'] = min(1.0, self.modulators['acetylcholine'] + 0.1)
        
        # Decay all modulators toward baseline
        for key in self.modulators:
            self.modulators[key] = 0.9 * self.modulators[key] + 0.1 * 0.5
```

### 5.4 Modulator Effects

| Modulator | High Level | Low Level |
|-----------|------------|-----------|
| **Dopamine** | Increased LR, reinforcement | Reduced motivation |
| **Norepinephrine** | Increased exploration | Focused exploitation |
| **Acetylcholine** | Enhanced encoding | Retrieval mode |
| **Serotonin** | Stability, patience | Reactivity, impulsivity |

### 5.5 Research Questions

1. How to learn neuromodulator dynamics?
2. Should different layers have different modulation?
3. Integration with Active Inference framework?

---

## 6. Additional Research Directions

### 6.1 Sparse Coding

```python
# Top-k sparsity in hidden activations
def sparse_activation(x, k=10):
    """Keep only top-k activations per sample"""
    topk_vals, topk_idx = torch.topk(x, k, dim=-1)
    sparse_x = torch.zeros_like(x)
    sparse_x.scatter_(-1, topk_idx, topk_vals)
    return sparse_x
```

**Benefits**: Reduced interference, improved capacity

### 6.2 Predictive Coding

```python
# Prediction error drives learning
def predictive_coding_update(predicted, actual, trace):
    """Update based on prediction error, not raw activity"""
    error = actual - predicted
    return trace + lr * outer(error, input)
```

**Benefits**: More efficient information encoding

### 6.3 Sleep-Like Consolidation

```python
# Offline replay for consolidation
def sleep_consolidation(agent, steps=100):
    """Replay experiences without new input"""
    agent.plasticity_enabled = False  # No new encoding
    
    for _ in range(steps):
        # Replay from memory buffer
        state, action, next_state = sample_memory()
        
        # Strengthen relevant traces
        agent.consolidate_replay(state, action, next_state)
    
    agent.plasticity_enabled = True
```

**Benefits**: Memory consolidation, catastrophic forgetting prevention

### 6.4 Continual Learning

```python
# EWC-style parameter importance
def elastic_weight_consolidation(agent, task_data, lambda_ewc=0.4):
    """Protect important weights for previous tasks"""
    fisher = compute_fisher_information(agent, task_data)
    
    # During future learning:
    # loss += lambda_ewc * sum((param - old_param)^2 * fisher)
```

**Benefits**: Multi-task learning without forgetting

---

## 7. Implementation Priorities

### 7.1 Short-Term (1-3 months)

| Priority | Extension | Complexity | Impact |
|----------|-----------|------------|--------|
| 1 | Multi-scale traces | Low | High |
| 2 | Basic neuromodulation | Medium | High |
| 3 | Sleep consolidation | Low | Medium |

### 7.2 Medium-Term (3-6 months)

| Priority | Extension | Complexity | Impact |
|----------|-----------|------------|--------|
| 1 | Meta-plasticity | High | Very High |
| 2 | Synaptic tagging | Medium | High |
| 3 | Attention over traces | High | High |

### 7.3 Long-Term (6-12 months)

| Priority | Extension | Complexity | Impact |
|----------|-----------|------------|--------|
| 1 | Full neuromodulation | Very High | Very High |
| 2 | Predictive coding | High | High |
| 3 | Continual learning | High | High |

---

## 8. Collaboration Opportunities

### 8.1 Neuroscience

- Memory consolidation mechanisms
- Neuromodulator dynamics
- Sleep and offline processing

### 8.2 Machine Learning

- Meta-learning algorithms
- Continual learning benchmarks
- Attention mechanisms

### 8.3 Neuromorphic Computing

- Hardware implementation
- Spiking neural networks
- Low-power deployment

---

## 9. References

1. Frey, U., & Morris, R. G. (1997). Synaptic tagging and long-term potentiation
2. Abraham, W. C., & Bear, M. F. (1996). Metaplasticity: the plasticity of synaptic plasticity
3. Doya, K. (2002). Metalearning and neuromodulation
4. Vaswani, A., et al. (2017). Attention is all you need
5. Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks
6. Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex

---

*This document is a living research guide. For implementation details, see [Advanced Plasticity](../architecture/advanced-plasticity.md). For theoretical background, see [Theoretical Foundations](SCIENCE.md).*