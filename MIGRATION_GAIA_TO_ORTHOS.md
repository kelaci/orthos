# Migration Guide: ORTHOS v4.2 → ORTHOS v5.0

## ⚡ Overview

ORTHOS v5.0 is a major rebranding release from ORTHOS. This guide helps you migrate your existing codebase to use the new ORTHOS framework.

## ⚠️ Breaking Changes

### 📦 Package Identity
The package has been renamed from `orthos-neuro` to `orthos-neuro`.

### 🧩 Import Path Changes
All top-level imports have changed from `orthos` to `orthos`.

```python
# ❌ Old (ORTHOS)
from orthos.layers.hebbian import HebbianCore
from orthos.hierarchy.manager import HierarchyManager

# ✅ New (ORTHOS)
from orthos.layers.hebbian import HebbianCore
from orthos.hierarchy.manager import HierarchyManager
```

## 🛠️ How to Migrate

### 1. Update Dependencies
If you have `orthos-neuro` in your `requirements.txt` or `pyproject.toml`, update it:

```bash
# Uninstall old package
pip uninstall orthos-neuro

# Install new package
pip install orthos-neuro
```

### 2. Batch Replace Imports
Use `sed` or your IDE's search and replace to update your code:

**Using sed (Linux/macOS):**
```bash
find . -name "*.py" -exec sed -i 's/\borthos\b/orthos/g' {} +
find . -name "*.py" -exec sed -i 's/\bORTHOS\b/ORTHOS/g' {} +
```

### 3. Update File References
If you were using scripts like `run_orthos_v42.py`, they have been renamed:

- `run_orthos_v42.py` → `run_orthos_v42.py`
- `test_orthos.py` → `test_orthos.py`
- `verify_orthos_fixes.py` → `verify_orthos_fixes.py`

## ✨ What's New in ORTHOS?

- **Branding**: Full transition to **Orthogonal Recursive Hierarchical Optimization System**.
- **Mathematical Focus**: Refined focus on orthogonal processing and Kalman filtering.
- **Consistency**: Unified naming across code, documentation, and infrastructure.
- **Support**: New documentation structure for better agentic support.

## 🆘 Need Help?

If you encounter issues during migration:
- Check the [New README.md](README.md)
- Open an issue on [GitHub](https://github.com/kelaci/orthos/issues)
- Join our [Discussions](https://github.com/kelaci/orthos/discussions)
