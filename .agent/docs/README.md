# 🤖 AI Agent Onboarding & Documentation

Welcome, Agent. This repository, **ORTHOS (Orthogonal Recursive Hierarchical Optimization System)**, is designed with a high degree of modularity and scientific rigor. This document serves as your primary guide to understanding, maintaining, and extending the codebase efficiently.

## 🎯 Your Mission

Your goal is to assist in the research and development of ORTHOS while maintaining:
1.  **Mathematical Integrity**: Ensure all learning rules and temporal transformations are mathematically sound.
2.  **Code Quality**: Follow strict typing, docstring standards, and modular design.
3.  **Test Coverage**: Never commit code without corresponding tests in `test_gaia.py`.
4.  **Documentation Sync**: Update architectural docs whenever a core component changes.

## 🗺️ Quick Navigation for Agents

| File/Directory | Purpose | Why you should care |
| :--- | :--- | :--- |
| `orthos/core/` | Base classes & types | Start here to understand the interfaces you must implement. |
| `orthos/layers/` | Core neural blocks | Where the "work" happens (Hebbian, Temporal, etc). |
| `orthos/hierarchy/` | Level management | How layers are stacked and time-scaled. |
| `orthos/plasticity/` | Learning logic | The brain of the adaptation system. |
| `test_gaia.py` | Central test suite | Your safety net. Run this before every PR. |
| `.agent/docs/` | Agent-specific docs | Detailed guides on how to work as an AI in this repo. |

## 🛠️ Essential Commands for Agents

Run these to get your bearings:

```bash
# Verify the current state of the project
python test_gaia.py

# Run benchmarks to check performance
python benchmark.py

# Exploratory: Check the configuration defaults
cat orthos/config/defaults.py
```

## 🧠 Brain-Friendly Documentation

We have prepared specific deep-dives for your LLM context:

- [**System Architecture for LLMs**](./system-architecture.md): A structural breakdown of how modules interact.
- [**Development Guide for Agents**](./development-guide.md): Coding standards, testing protocols, and workflow steps.
- [**Knowledge Base & Traps**](./knowledge-base.md): Known edge cases in Hebbian learning and temporal scaling.

---

## ⚡ Agent Workflows

We use `.agent/workflows/` to define high-efficiency paths for common tasks. If your environment supports it, use these templates:

- `/refactor-layer`: Steps for modifying a core layer in `orthos/layers/`.
- `/add-learning-rule`: Protocol for implementing a new plasticity rule.
- `/fix-test`: Diagnostic path for debugging test failures.

> **Note**: As an AI agent, you should be proactive but cautious. Always validate your understanding of the Free Energy Principle or Hebbian dynamics before modifying the core math.
