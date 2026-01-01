<p align="center">
  <h1 align="center">🔧 ORTHOS v5.0</h1>
  <p align="center">
    <strong>The Architecture Candidate</strong>
  </p>
  <p align="center">
    A biologically-inspired neural architecture integrating <em>Sparse Attention</em>,<br/>
    <em>Hierarchical Bayesian Filters</em>, and <em>Structural Plasticity</em>.
  </p>
</p>


<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-documentation">Documentation</a> •
  <a href="#-contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-5.0.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/python-3.8+-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-purple.svg" alt="License">
  <img src="https://img.shields.io/badge/status-research--grade-orange.svg" alt="Status">
</p>


---

## 🌟 What is ORTHOS?

ORTHOS is an **open-source research project** exploring biologically-inspired neural architectures that can:

- **Learn how to learn** through meta-learning of plasticity parameters
- **Process information hierarchically** at multiple temporal scales
- **Adapt online** using Hebbian learning rules
- **Make decisions** using Active Inference and the Free Energy Principle

Whether you're a **neuroscience researcher**, **ML engineer**, or **curious student**, ORTHOS provides a playground for exploring cutting-edge concepts in adaptive learning systems.

```
┌─────────────────────────────────────────────────────────────────┐
│                 ORTHOS v5.0 ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   🎛️ Consensus Engine (Uncertainty-Weighted)                    │
│   └─ Bayesian Fusion & Auto-Projection                          │
│                                                                 │
│   ⚡ Sparse Attention (SAS Framework)                           │
│   └─ Structural Plasticity & k-WTA                              │
│                                                                 │
│   🎯 Meta-Learning Layer (v5.1 Hybrid)                          │
│   └─ Hybrid NES + Online Contextual Bandit                      │
│                                                                 │
│   📊 Probabilistic Spine (Hierarchical Filters)                 │
│   ├─ Level 3: SR-KF / Block-Diagonal (8x res)                   │
│   ├─ Level 2: Particle Filter (4x res)                          │
│   ├─ Level 1: EKF / Diagonal-KF (2x res)                        │
│   └─ Level 0: Raw Temporal input (1x res)                       │
│                                                                 │
│   🧬 Core Foundations                                           │
│   ├─ HebbianCore: Adaptive Plasticity Rules                     │
│   ├─ ReactiveLayer: Fast Feedforward Transition                 │
│   └─ TemporalLayer: Recurrent Context Traces                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 30-Second Setup

```bash
# Clone the repository
git clone https://github.com/kelaci/orthos.git
cd orthos

# Install dependencies
pip install -r requirements.txt

# Run tests to verify installation
python test_orthos.py

```

### Your First ORTHOS Program

```python
import numpy as np
from orthos.layers.hebbian import HebbianCore
from orthos.hierarchy.manager import HierarchyManager
from orthos.hierarchy.level import HierarchicalLevel
from orthos.layers.reactive import ReactiveLayer

# Create a simple 2-level hierarchy
manager = HierarchyManager()

# Level 0: Input processing
level0 = HierarchicalLevel(0, input_size=10, output_size=20, temporal_resolution=1)
level0.add_layer(ReactiveLayer(10, 20, activation='relu'))
manager.add_level(level0)

# Level 1: Feature extraction with Hebbian learning
level1 = HierarchicalLevel(1, input_size=20, output_size=40, temporal_resolution=2)
level1.add_layer(HebbianCore(20, 40, plasticity_rule='hebbian'))
manager.add_level(level1)

# Process a sequence
input_data = np.random.randn(100, 10)  # 100 time steps, 10 features
representations = manager.process_hierarchy(input_data, time_steps=100)

print(f"✅ Processed {len(representations)} levels!")
print(f"   Level 0: {len(representations[0])} representations")
print(f"   Level 1: {len(representations[1])} representations")
```

**Expected output:**
```
✅ Processed 2 levels!
   Level 0: 100 representations
   Level 1: 100 representations
```

---

### ⚡ Sparse Attention (SAS Framework)

- **Structural Plasticity** - Enforces 10-30% selective connectivity
- **k-WTA (k-Winners-Take-All)** - Active economy of neural triggers
- **Dynamic Rewiring** - Synaptic turnover for optimal topology

### 📊 Probabilistic Spine (v5.0 Optimized)

- **Square Root Kalman Filters** - Doubles numerical precision for stability
- **Block-Diagonal Updates** - O(N·B²) efficiency for high-dim scaling
- **Joseph Form Updates** - Guaranteed positive semi-definite covariance
- **Uncertainty-Weighted Consensus** - Optimized Bayesian aggregation

### 🎯 Meta-Learning & Plasticity

- **Hybrid Meta-Learning (HML)** - Combined **NES** (Global) and **Contextual Bandit** (Online) optimization
- **Dual-Timescale Memory** - Fast (hippocampal) + slow (neocortical) traces
- **Homeostatic Regulation** - Stable weight normalization and decay
- **Active Inference** - Decision-making via Free Energy Principle (FEP)

### 🛡️ Robustness & Safety (New in v4.2)

ORTHOS isn't just theory—it's built to survive. We benchmark against critical failure modes:

- **Drone Saviour Protocol** 🚁: Prevents crashes during **GPS denial** by switching to optical flow/IMU fusion via the Probabilistic Spine.
- **Chaos Resilience**: Maintains O(1) stability even when SNR drops below 5dB.
- **SAS Economy**: Reduces energy/memory footprint by 70% via structural sparsity.

👉 [Read the Research Utility Test Plan](docs/research/TEST_PLAN_UTILITY.md)

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Basic Installation

```bash
pip install -r requirements.txt
```

### Dependencies

```
numpy>=1.20.0          # Core numerical operations
matplotlib>=3.4.0      # Visualization
scipy>=1.7.0           # Scientific computing
```

### Development Installation

```bash
# Clone with full history
git clone https://github.com/kelaci/orthos.git
cd orthos

# Install all dependencies (including dev tools)
pip install -r requirements.txt

# Verify installation
python test_orthos.py

```

### Optional: PyTorch Support

For GPU acceleration and the advanced v3.1 features:

```bash
pip install torch torchvision
```

---

## 📖 Documentation

The documentation has been consolidated for efficiency:

| Document | Description |
|----------|-------------|
| [🏗️ ARCHITECTURE.md](ARCHITECTURE.md) | System design, core components, and SAS architecture |
| [🔬 SCIENCE.md](SCIENCE.md) | Theoretical foundations (FEP, Active Inference, Plasticity) |
| [📘 GUIDES.md](GUIDES.md) | Quickstart, GPU usage, and validation |
| [🗺️ ROADMAP.md](ROADMAP.md) | Version milestones and future development |
| [📐 MATHEMATICAL_METHODS_DEEP_DIVE.md](ORTHOS_MATHEMATICAL_METHODS_DEEP_DIVE_EN.md) | **Deep dive into filters, consensus, and SAS math** |
| [🤖 .agent/docs/README.md](.agent/docs/README.md) | **Onboarding for Agentic AI Coding** |


---

## 🤖 Agentic Support

This repository is **Agent-Ready**. If you are an AI agent:
1. Start with [.agent/docs/README.md](.agent/docs/README.md).
2. Follow the instructions in [.agent/instructions.md](.agent/instructions.md).
3. Use the workflows in [.agent/workflows/](.agent/workflows/).

### 🎓 Key Concepts

<details>
<summary><strong>What is Hebbian Learning?</strong></summary>

> "Neurons that fire together, wire together" — Donald Hebb (1949)

Hebbian learning is a biologically-inspired learning rule where connection strengths increase when neurons are simultaneously active. ORTHOS implements multiple variants:

```python
# Classic Hebbian
Δw = η * pre * post

# Oja's rule (with normalization)
Δw = η * post * (pre - post * w)

# BCM rule (with sliding threshold)
Δw = η * post * (post - θ) * pre
```

</details>

<details>
<summary><strong>What is the Free Energy Principle?</strong></summary>

The Free Energy Principle (FEP), developed by Karl Friston, proposes that adaptive systems minimize "free energy" — a measure of surprise or prediction error.

ORTHOS uses **Expected Free Energy (EFE)** for action selection:
- **Pragmatic value**: How well does action achieve goals?
- **Epistemic value**: How much information does action provide?

```python
EFE = pragmatic_value - exploration_weight * epistemic_uncertainty
```

</details>

<details>
<summary><strong>What is Meta-Learning?</strong></summary>

Meta-learning is "learning to learn" — optimizing the learning process itself.

ORTHOS uses a **Hybrid Meta-Learning** strategy to optimize plasticity parameters:
1. **Natural Evolution Strategies (NES)**: Global scale optimization using natural gradients and rank-normalization.
2. **Contextual Bandit Meta-Control**: Real-time modulation of learning rates and noise scales based on prediction error and uncertainty.

This allows ORTHOS to discover optimal learning rates, decay rates, and other hyperparameters automatically.

</details>

---

## 🗂️ Project Structure

```
orthos/
├── 📄 README.md                 ← You are here
├── 📄 LICENSE                   ← MIT License
├── 📄 requirements.txt          ← Dependencies
├── 📄 test_orthos.py            ← Test suite
│
├── 📁 orthos/                     ← Main package

│   ├── core/                    ← Base classes & types
│   │   ├── base.py              ← Abstract base classes
│   │   ├── tensor.py            ← Tensor operations
│   │   └── types.py             ← Type definitions
│   │
│   ├── layers/                  ← Neural layers
│   │   ├── hebbian.py           ← HebbianCore implementation
│   │   ├── reactive.py          ← ReactiveLayer (feedforward)
│   │   └── temporal.py          ← TemporalLayer (recurrent)
│   │
│   ├── consensus/               ← Consensus Layer (v4.2)
│   │   └── engine.py            ← Aggregation logic
│   │
│   ├── filters/                 ← Probabilistic Spine (v4.2)
│   │   ├── kalman.py            ← KalmanFilter & EKF
│   │   └── particle.py          ← ParticleFilter
│   │
│   ├── hierarchy/               ← Hierarchical processing
│   │   ├── level.py             ← HierarchicalLevel
│   │   ├── filtered_level.py    ← FilteredHierarchicalLevel (v4.2)
│   │   ├── consensus_manager.py ← ConsensusHierarchyManager (v4.2)
│   │   └── manager.py           ← Base HierarchyManager
│   │
│   ├── plasticity/              ← Plasticity control
│   │   ├── controller.py        ← PlasticityController
│   │   ├── es_optimizer.py      ← Evolutionary Strategy
│   │   └── rules.py             ← Plasticity rules
│   │
│   ├── meta_learning/           ← Meta-learning
│   │   ├── optimizer.py         ← MetaOptimizer
│   │   └── metrics.py           ← Performance metrics
│   │
│   ├── config/                  ← Configuration
│   │   └── defaults.py          ← Default configs
│   │
│   ├── utils/                   ← Utilities
│   │   ├── logging.py           ← Logging helpers
│   │   └── visualization.py     ← Plotting functions
│   │
│   └── examples/                ← Example scripts
│       ├── basic_demo.py
│       ├── plasticity_demo.py
│       └── meta_learning_demo.py
│
├── 📁 .agent/                   ← Agent configurations
│   ├── 📁 docs/                 ← Agent-specific documentation
│   └── 📁 workflows/            ← Agent high-efficiency paths
│
└── 📁 docs/                     ← General documentation
    ├── architecture/            ← System design docs
    ├── science/                 ← Theoretical foundations
    ├── guides/                  ← How-to guides
    ├── research/                ← Research directions
    └── development/             ← Dev roadmap
```

---

## 🧪 Running Tests

```bash
# Run all tests
python test_orthos.py

# Expected output:
# 🚀 Running ORTHOS v5.0.0 Tests

# ==================================================
# 🧪 Testing Layers...
# ✅ ReactiveLayer test passed
# ✅ HebbianCore test passed
# ✅ TemporalLayer test passed
# ...
# 🧪 Testing Probabilistic Spine (v5.0)...
# ✅ KalmanFilter test passed
# ✅ ConsensusEngine test passed
# ...
# 🎉 All tests passed successfully!
```

### Test Coverage

| Component | Tests |
|-----------|-------|
| Layers | ReactiveLayer, HebbianCore, TemporalLayer |
| Hierarchy | Level creation, Manager processing |
| Plasticity | Rules, Controller, ES Optimizer |
| Meta-Learning | MetaOptimizer training |
| Configuration | Default configs |
| Integration | Full pipeline |
| Probabilistic Spine (v5.0) | Kalman, Particle, Consensus |
| Research Utility | Drone Saviour, Noise Resilience |

---

## 🤝 Contributing

We welcome contributions from researchers, engineers, and enthusiasts!

### Ways to Contribute

| Contribution | Description |
|--------------|-------------|
| 🐛 **Bug Reports** | Found a bug? [Open an issue](https://github.com/kelaci/orthos/issues) |
| 💡 **Feature Requests** | Have an idea? Share it in [Discussions](https://github.com/kelaci/orthos/discussions) |
| 📝 **Documentation** | Improve docs, fix typos, add examples |
| 🧪 **Tests** | Add test coverage, edge cases |
| 🔬 **Research** | Implement new plasticity rules, architectures |

### Getting Started as a Contributor

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/orthos.git
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add docstrings to new functions
   - Include type hints
   - Write tests for new functionality

4. **Run tests**
   ```bash
   python test_orthos.py

   ```

5. **Submit a pull request**
   - Describe your changes
   - Reference any related issues

### Code Style

```python
def example_function(input_data: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
    """
    Brief description of function.
    
    Args:
        input_data: Input tensor (batch_size, features)
        learning_rate: Learning rate parameter
    
    Returns:
        Processed output tensor
    
    Example:
        >>> result = example_function(data, lr=0.05)
    """
    # Implementation here
    pass
```

---

## 📚 Research Background

ORTHOS is built on foundational work from neuroscience and machine learning:

### Key References

| Paper | Author(s) | Year | Relevance |
|-------|-----------|------|-----------|
| The Organization of Behavior | D.O. Hebb | 1949 | Hebbian learning |
| Free Energy Principle | K. Friston | 2010 | Active Inference |
| Complementary Learning Systems | McClelland et al. | 1995 | Memory consolidation |
| BCM Theory | Bienenstock et al. | 1982 | Sliding threshold |
| BitNet | Ma et al. | 2024 | Weight quantization |

### Implemented Concepts

- **Hebbian Plasticity** - Local learning rules based on neural correlation
- **Homeostatic Regulation** - Maintaining stable activity through synaptic scaling
- **Temporal Abstraction** - Processing at multiple time scales
- **Active Inference** - Decision-making via free energy minimization
- **Meta-Learning** - Learning to learn through evolutionary optimization

---

## 🗺️ Roadmap

### Current: v5.1.0 (Hybrid Intelligence) ✅

- [x] **Hybrid Meta-Learning (HML)**
  - [x] Natural Evolution Strategies (NES)
  - [x] Contextual Bandit Meta-Controller
  - [x] HybridMetaManager Orchestration
- [x] **Advanced Hierarchical Probabilistic Spine**
  - [x] Square Root & Block-Diagonal Kalman Filters
  - [x] Outlier-robust, Uncertainty-weighted Consensus
  - [x] Joseph Form stability & Auto-Projection
- [x] **Sparse Attention (SAS Framework)**
- [x] **Full Rebrand & Consolidation**

### Coming: v5.2.0 (Active Adaptation) 🚧

- [ ] Multi-objective reward structures for Active Inference
- [ ] Enhanced GPU acceleration kernels for masked operations

### Future: v6.0.0+
- [ ] Neuroevolution of hierarchical topologies
- [ ] Cross-modal sensory feedback integration (e.g., Audio-Visual)
- [ ] Real-time embedded deployment (Quantized BitNet)

See [Development Roadmap](ROADMAP.md) for details.

---

## ❓ FAQ

<details>
<summary><strong>Is ORTHOS suitable for production use?</strong></summary>

ORTHOS is a **research project** focused on exploring novel learning architectures. While the code is well-tested and stable, it's designed for research and experimentation rather than production deployment. That said, the modular architecture makes it easy to extract and use specific components.

</details>

<details>
<summary><strong>How does ORTHOS compare to traditional neural networks?</strong></summary>

| Aspect | Traditional NN | ORTHOS |
|--------|----------------|------|
| Learning | Backpropagation | Hebbian + ES |
| Adaptation | Offline training | Online learning |
| Hierarchy | Feedforward | Multi-scale temporal |
| Inspiration | Mathematical | Biological |

</details>

<details>
<summary><strong>Can I use ORTHOS with PyTorch/TensorFlow?</strong></summary>

Yes! ORTHOS v5.0 is designed for hybrid performance:
- **NumPy Backend**: Default pure Python research mode.
- **CuPy/PyTorch Backend**: High-performance GPU acceleration for high-dim SAS architectures.

See [GPU Integration Guide](docs/guides/gpu-integration.md).

</details>

<details>
<summary><strong>What's the difference between fast and slow traces?</strong></summary>

Inspired by hippocampal-neocortical memory systems:
- **Fast trace** (τ=0.95): Rapid adaptation, like hippocampal encoding
- **Slow trace** (τ=0.99): Gradual consolidation, like neocortical storage

This dual-timescale design prevents catastrophic forgetting while enabling quick adaptation.

</details>

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 ORTHOS Development Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

---

## 🙏 Acknowledgments

- **Karl Friston** - Free Energy Principle inspiration
- **Donald Hebb** - Foundational learning rule
- **The open-source community** - Tools and libraries

---

## 📬 Contact

- **GitHub Issues**: [Bug reports & feature requests](https://github.com/kelaci/orthos/issues)
- **GitHub Discussions**: [Questions & ideas](https://github.com/kelaci/orthos/discussions)
- **Repository**: [github.com/kelaci/orthos](https://github.com/kelaci/orthos)

---

<p align="center">
  <strong>⭐ Star us on GitHub if you find ORTHOS interesting! ⭐</strong>
</p>

<p align="center">
  Made with 🧠 by researchers, for researchers
</p>
