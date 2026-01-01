# 🛠️ Development Guide for Agents

This guide defines the standards and protocols for AI agents contributing to the ORTHOS codebase.

## 📜 Coding Standards

### 1. Type Hinting
Every function and method **must** have full type hints.
```python
# GOOD
def compute_delta(pre: np.ndarray, post: np.ndarray, eta: float) -> np.ndarray:
    ...

# BAD
def compute_delta(pre, post, eta):
    ...
```

### 2. Docstrings (Google Style)
Include a clear description, Args, Returns, and an Example if the logic is complex.
```python
def normalize_weights(weights: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Standardizes weight magnitudes along a specific axis.
    
    Args:
        weights: The weight matrix to normalize.
        axis: The axis along which to compute the norm.
        
    Returns:
        The normalized weight matrix.
    """
```

### 3. Modularity First
If a logic block is used in two places, move it to `orthos/utils/` or `orthos/core/`.

## 🧪 Testing Protocol (Strict)

Before any non-trivial change, follow this "Agent Test Loop":

1.  **Baseline**: Run `python test_gaia.py` to ensure current state is green.
2.  **Implementation**: Make your changes.
3.  **Local Test**: Create a temporary test script or add a case to `test_gaia.py`.
4.  **Verification**: Run the full suite again.
5.  **Benchmark**: If you changed a core loop, run `python benchmark.py` to ensure no performance regression.

## 🚀 Common Workflow Templates

### Adding a New Layer
1.  Inherit from `BaseLayer` in `orthos/core/base.py`.
2.  Implement `forward`, `update`, and `reset`.
3.  Add the layer to `orthos/layers/__init__.py`.
4.  Add a test case in `test_gaia.py` under the `Layers` section.

### Modifying Plasticity Rules
1.  Locate the rule in `orthos/plasticity/rules.py`.
2.  Update the mathematical implementation.
3.  Check if `PlasticityController` needs to pass new parameters.
4.  Update documentation in `docs/architecture/plasticity-system.md`.

## 📎 Commit & PR Guidelines

As an agent, when you finish a task:
- **Summarize Changes**: Provide a bulleted list of what was changed and why.
- **Evidence of Success**: Paste the output of the relevant tests.
- **Impact Assessment**: Note if any breaking changes were made to the internal API.

---

## 🚫 What NOT to do
- **Do NOT** use `print()` for debugging in production code. Use `orthos.utils.logging`.
- **Do NOT** hardcode hyperparameters. Use `orthos.config.defaults`.
- **Do NOT** ignore Lint errors.
- **Do NOT** modify `orthos/core/` without explicit permission or thorough architectural review.
