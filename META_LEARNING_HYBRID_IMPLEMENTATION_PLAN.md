# ORTHOS Hybrid Meta-Learning (HML) Phase II - Deep Dive Implementation Plan

## 🧠 Philosophy: "What should Meta-Learning actually optimize?"

In ORTHOS, we reject the notion of meta-learning the entire weight matrix. Instead, we target the **low-dimensional control knobs** that govern systemic behavior. This ensures fast convergence, better generalization, and alignment with the adaptive nature of ORTHOS.

### 🎯 Target Meta-Parameters (The "Knobs")

| Parameter | Symbol | Role |
| :--- | :--- | :--- |
| **Plasticity Rates** | $\eta_{fast}, \eta_{slow}$ | Hebbian/Neo-Hebbian learning speed |
| **Decay Constants** | $\tau_{fast}, \tau_{slow}$ | Integration/forgetting time scales |
| **Sparsity Ratio** | $k_{WTA}$ | Top-k activation threshold for SAS |
| **Consensus Threshold** | $\sigma_{outlier}$ | Robust Z-score for rejecting level predictions |
| **Noise Covariance Scale** | $s_R, s_Q$ | Dynamic scaling for KF/PF measurement and process noise |
| **Plasticity Gating** | $\theta_{gate}$ | Uncertainty threshold to enable/disable structural rewiring |

---

## 🏗️ Architecture: The Hybrid Strategy (HML)

We implement a hierarchical meta-control loop:

1.  **Offline / Global (NES)**: Natural Evolution Strategy maps the global landscape and finds robust base parameters across multiple datasets or simulated environments.
2.  **Online / Contextual (Bandit)**: A Contextual Multi-Armed Bandit (MAB) performs real-time adaptation of the "knobs" based on current performance metrics.
3.  **Local / Micro (Hebbian)**: Direct synaptic learning (SAS) handles feature extraction.
4.  **Safety (Clamp Layer)**: Hard physical limits to ensure system stability.

---

## 🥇 Component 1: NES - Natural Evolution Strategy (Global)

**Objective**: Replace standard ES with NES for better convergence and natural gradient support.

### Mathematical Formulation
Instead of simple elite selection, NES estimates the gradient of the expected fitness:
$$\nabla_{\theta} J \approx \frac{1}{N \sigma} \sum_{i=1}^{N} F_i \epsilon_i$$
Where:
- $F_i$ is the fitness (rank-normalized for robustness).
- $\epsilon_i$ is the perturbation vector.
- $\sigma$ is the mutation strength.

### Implementation Steps
1.  **Refactor `EvolutionaryStrategy`** to `NaturalEvolutionStrategy`.
2.  **Add Rank-Normalization**: Map fitness scores to [-0.5, 0.5] to handle outliers and non-linear reward scales.
3.  **Natural Gradient Update**: Use the Fisher Information Matrix (identity approx) for the parameter update.

---

## 🥈 Component 2: Online Bandit Meta-Controller (Contextual)

**Objective**: Real-time adaptation of $\eta$, $\tau$, and $k$ based on the "state of the system".

### State Representation ($s_t$)
- `prediction_error`: Moving average of reconstruction loss.
- `uncertainty`: Cumulative uncertainty from KF/PF levels.
- `sparsity`: Current activity levels in SAS layers.
- `drift`: Rate of change in the state estimate.

### Optimization (Thompson Sampling / UCB)
We treat the meta-parameters as "arms".
- **Action**: $a_t = \text{adjust}(\eta, \tau, k)$.
- **Reward**: $r_t = -(\text{Error} + \lambda \cdot \text{Uncertainty} + \gamma \cdot \text{Instability})$.

### Implementation Steps
1.  **Create `BanditMetaController` layer**:
    - Manage a small internal neural network or Gaussian Process for context-arm mapping.
    - Provide `get_optimal_parameters(context)` method.
2.  **Integrate with `FilteredHierarchicalLevel`**:
    - The level reports its state to the Bandit controller.
    - The Bandit controller returns modified meta-params for the next time step.

---

## 🥉 Component 3: Policy Gating & Safety

**Objective**: Prevent meta-learning from "exploding" the system.

- **Damping Layer**: Use a momentum-based update for meta-parameters (Slow Meta-Learning).
- **Stability Clamps**:
    - $\eta \in [0.0001, 0.1]$
    - $k_{WTA} \in [0.01, 0.2]$
    - $s_R, s_Q \in [0.1, 10.0]$

---

## 🧬 Hybrid Work-Flow (The "Winner")

1.  **Initialization**: Load baseline parameters from pre-trained NES runs.
2.  **Environment Step**:
    - `FilteredHierarchicalLevel` processes input.
    - System calculates `reconstruction_error` and `uncertainty`.
3.  **Online Adaptation (Bandit)**:
    - Bandit controller observes context $[error, unc, sparsity]$.
    - Modulates $\eta, \tau, R$ for the *next* step.
4.  **Local Learning (Hebbian)**:
    - SAS updates synapses using the modulated $\eta$.
5.  **Offline Consolidation (NES)**:
    - Periodic "sleep cycles" where the global NES evaluates the performance and updates the base distribution.

---

## 🛠️ Code Implementation Roadmap

### 1. `orthos/meta_learning/nes.py`
- Implementation of `NaturalEvolutionStrategy` class.
- Support for Rank-normalization.
- GPU-accelerated perturbation generation.

### 2. `orthos/meta_learning/bandit.py`
- `MetaBandit` class with Thompson Sampling.
- State-Action buffer for online training of the bandit.

### 3. `orthos/meta_learning/hybrid_manager.py`
- Orchestrator that connects NES, Bandit, and Filtered Levels.
- Implements the "Gated Meta-Parameter" application logic.

### 4. Integration with `FilteredHierarchicalLevel`
- Add `meta_control_hook` to the forward pass.
- Export internal metrics (Uncertainty, Sparsity) to the hybrid manager.

---

## 📌 Conclusion: Why this works?
Unlike MAML (too heavy) or pure ES (too slow), this **Hybrid NES+Bandit** approach gives ORTHOS both **global stability** (NES) and **local agility** (Bandit). It respects the sparsity constraints and leverages probabilistic uncertainty as a first-class citizen in the meta-optimization loop.
