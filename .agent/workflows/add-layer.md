---
description: Protocol for implementing a new layer in ORTHOS
---
# Add Layer Workflow

Follow these steps to add a new neural layer.

1. Create the new layer file in `orthos/layers/` (e.g., `orthos/layers/new_layer.py`).
2. Ensure it inherits from `orthos.core.base.BaseLayer`.
3. Implement `forward(self, x)`, `update(self)`, and `reset(self)`.
4. Register the layer in `orthos/layers/__init__.py`.

// turbo
5. Add a test case to `test_gaia.py`. Use the existing `ReactiveLayer` test as a template.

// turbo
6. Run the test suite:
   `python test_gaia.py`

7. Update `docs/architecture/overview.md` if the layer introduces a new architectural concept.
