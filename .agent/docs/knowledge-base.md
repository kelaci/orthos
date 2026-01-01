# 🧠 Agent Knowledge Base & Traps

This document catalogs "tribal knowledge," common bugs, and mathematical nuances discovered during the development of ORTHOS. Consult this before refactoring core logic.

## 🪤 Common Traps

### 1. The Temporal Scale Mismatch
**Issue**: Layers at Level 1 expecting $N$ inputs but receiving $N/2$ because of temporal downsampling.
**Fix**: Always check `self.temporal_resolution` in the `HierarchicalLevel`. Ensure the `HierarchyManager` logic for buffer collection matches your expectations.

### 2. Hebbian Explosions
**Issue**: Pure Hebbian learning ($ \Delta w = \eta \cdot x \cdot y $) is unstable and leads to infinite weight growth.
**Fix**: Always use a normalization rule (like Oja's) or weight decay. Check `orthos/plasticity/rules.py` for "Homeostatic" variants.

### 3. State Leakage in Tests
**Issue**: Tests passing individually but failing when run in a suite because a layer's internal state (traces) wasn't reset.
**Fix**: Ensure `reset()` is called between different input sequences.

### 4. NumPy vs PyTorch Dtypes
**Issue**: ORTHOS supports both. Using `np.float64` in a PyTorch-heavy environment can cause unnecessary copying and slowdowns.
**Fix**: Use `orthos.core.types.FLOAT_TYPE` for all tensor initializations.

## 💡 Pro Tips for Agents

- **Vectorization**: Avoid explicit loops over the batch dimension. ORTHOS assumes `(batch_size, ...)` for all core operations.
- **Trace Decay**: A common value for `gamma` (trace decay) is `0.95`. If a layer is "forgetting" too fast, check the `PlasticityController` defaults.
- **ES Optimization**: Evolutionary Strategies are sensitive to population size. If `ESOptimizer` isn't converging, try increasing `pop_size` in `config/defaults.py` rather than changing the math.

## 🔬 Scientific Nuances

- **Active Inference**: The `FreeEnergyLayer` minimizes *Expected Free Energy*. This is NOT the same as traditional cross-entropy. It includes an "epistemic value" (exploration) term.
- **Complementary Learning**: The dual-trace system (`fast` and `slow`) is intended to mimic the Hippocampus and Neocortex. Fast traces should update every step, while slow traces should update every $N$ steps or with a very low $\eta$.

---

## 📅 Version History (Agent Notes)
- **v4.0**: Initial modularization.
- **v4.1**: Introduction of `PlasticityController`.
- **v4.2 (Planned)**: Integration of Attention mechanisms. Focus on sparse connectivity.
