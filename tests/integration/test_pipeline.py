"""
Integration and full pipeline tests.
"""
import pytest
import numpy as np
from orthos.hierarchy.manager import HierarchyManager
from orthos.hierarchy.level import HierarchicalLevel
from orthos.layers.reactive import ReactiveLayer
from orthos.layers.hebbian import HebbianCore
from orthos.plasticity.controller import PlasticityController
from orthos.meta_learning.optimizer import MetaOptimizer

def test_full_pipeline():
    """Test full integration of ORTHOS components."""
    manager = HierarchyManager()

    # Build hierarchy
    level0 = HierarchicalLevel(0, 10, 20, temporal_resolution=1)
    level0.add_layer(ReactiveLayer(10, 20))
    manager.add_level(level0)

    level1 = HierarchicalLevel(1, 20, 40, temporal_resolution=2)
    level1.add_layer(HebbianCore(20, 40))
    manager.add_level(level1)

    # Create plasticity controller
    hebbian_core = level1.processing_layers[0] # Get the actual layer instance
    plasticity_controller = PlasticityController([hebbian_core])

    # Create meta-optimizer
    meta_optimizer = MetaOptimizer(plasticity_controller)

    # Test full pipeline
    input_data = np.random.randn(20, 10)
    representations = manager.process_hierarchy(input_data, 20)

    # Verify representations
    assert len(representations) == 2
    assert len(representations[0]) == 20
    assert len(representations[1]) == 20

    # Test plasticity adaptation
    for _ in range(5):
        performance = np.random.random()
        plasticity_controller.adapt_plasticity(performance)

    # Test meta-learning
    def test_task(step):
        return 0.6 + 0.01 * step

    tasks = [test_task]
    meta_optimizer.meta_train(3, tasks)
