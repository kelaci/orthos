"""
Basic test script to verify ORTHOS v5.0.0 implementation.

This script tests the core functionality of ORTHOS's hierarchical architecture,

plasticity control, and meta-learning capabilities.
"""

import numpy as np
from orthos.layers.reactive import ReactiveLayer
from orthos.layers.hebbian import HebbianCore
from orthos.layers.temporal import TemporalLayer
from orthos.hierarchy.level import HierarchicalLevel
from orthos.hierarchy.manager import HierarchyManager
from orthos.plasticity.controller import PlasticityController
from orthos.plasticity.es_optimizer import EvolutionaryStrategy
from orthos.plasticity.rules import HebbianRule, OjasRule, BCMRule
from orthos.meta_learning.optimizer import MetaOptimizer
from orthos.config.defaults import get_default_config

def test_layers():
    """Test layer implementations."""
    print("🧪 Testing Layers...")

    # Test ReactiveLayer
    reactive = ReactiveLayer(10, 20, activation='relu')
    input_data = np.random.randn(5, 10)
    output = reactive.forward(input_data)
    assert output.shape == (5, 20), f"ReactiveLayer output shape mismatch: {output.shape}"
    print("✅ ReactiveLayer test passed")

    # Test HebbianCore
    hebbian = HebbianCore(20, 40, plasticity_rule='hebbian')
    input_data_20 = np.random.randn(5, 20)  # Correct size for HebbianCore
    output = hebbian.forward(input_data_20)
    assert output.shape == (5, 40), f"HebbianCore output shape mismatch: {output.shape}"
    hebbian.update(0.01)
    print("✅ HebbianCore test passed")

    # Test TemporalLayer
    temporal = TemporalLayer(40, 80, time_window=5)
    input_data_40 = np.random.randn(5, 40)  # Correct size for TemporalLayer
    output = temporal.forward(input_data_40, t=0)
    assert output.shape == (5, 80), f"TemporalLayer output shape mismatch: {output.shape}"
    print("✅ TemporalLayer test passed")

def test_hierarchy():
    """Test hierarchy implementation."""
    print("\n🧪 Testing Hierarchy...")

    # Create hierarchy manager
    manager = HierarchyManager()

    # Create levels
    level0 = HierarchicalLevel(0, 10, 20, temporal_resolution=1)
    level0.add_layer(ReactiveLayer(10, 20))
    manager.add_level(level0)

    level1 = HierarchicalLevel(1, 20, 40, temporal_resolution=2)
    level1.add_layer(HebbianCore(20, 40))
    manager.add_level(level1)

    # Test hierarchy processing - input_data should be (time_steps, input_size)
    input_data = np.random.randn(10, 10)  # 10 time steps, 10 features
    representations = manager.process_hierarchy(input_data, 10)

    assert len(representations) == 2, f"Expected 2 levels, got {len(representations)}"
    assert len(representations[0]) == 10, f"Level 0 should have 10 representations"
    assert len(representations[1]) == 10, f"Level 1 should have 10 representations (ZOH outputs)"
    # Note: Level 1 updates its internal state every 2 steps, but returns continuous output

    print("✅ Hierarchy test passed")

def test_plasticity():
    """Test plasticity control."""
    print("\n🧪 Testing Plasticity...")

    # Create target modules
    hebbian = HebbianCore(20, 40)
    temporal = TemporalLayer(40, 80)

    # Create plasticity controller
    controller = PlasticityController([hebbian, temporal])

    # Test parameter adaptation
    initial_params = controller.get_current_params()
    controller.adapt_plasticity(0.7)  # Good performance
    final_params = controller.get_current_params()

    assert len(initial_params) > 0, "No parameters found"
    assert len(initial_params) == len(final_params), "Parameter count mismatch"

    print("✅ Plasticity test passed")

def test_plasticity_rules():
    """Test plasticity rules."""
    print("\n🧪 Testing Plasticity Rules...")

    # Test data
    weights = np.random.randn(5, 10)
    pre_activity = np.random.randn(10)
    post_activity = np.random.randn(5)

    # Test HebbianRule
    hebbian_rule = HebbianRule(0.01)
    updated_weights = hebbian_rule.apply(weights, pre_activity, post_activity)
    assert updated_weights.shape == weights.shape, "HebbianRule shape mismatch"

    # Test OjasRule
    oja_rule = OjasRule(0.01)
    updated_weights = oja_rule.apply(weights, pre_activity, post_activity)
    assert updated_weights.shape == weights.shape, "OjasRule shape mismatch"

    # Test BCMRule
    bcm_rule = BCMRule(0.01)
    updated_weights = bcm_rule.apply(weights, pre_activity, post_activity)
    assert updated_weights.shape == weights.shape, "BCMRule shape mismatch"

    print("✅ Plasticity rules test passed")

def test_es_optimizer():
    """Test Evolutionary Strategy optimizer."""
    print("\n🧪 Testing ES Optimizer...")

    # Create ES optimizer
    es = EvolutionaryStrategy(population_size=10, sigma=0.1)

    # Test population generation
    initial_mean = np.array([0.1, 0.2, 0.3])
    population = es.generate_population(initial_mean)

    assert len(population) == 10, f"Expected 10 individuals, got {len(population)}"
    assert all(len(ind) == 3 for ind in population), "Individual dimension mismatch"

    # Test mean update
    fitness_scores = np.random.random(10)
    es.update_mean(initial_mean, population, fitness_scores)
    final_mean = es.get_mean()

    assert len(final_mean) == 3, "Mean dimension mismatch"

    print("✅ ES optimizer test passed")

def test_meta_learning():
    """Test meta-learning optimizer."""
    print("\n🧪 Testing Meta-Learning...")

    # Create target modules
    hebbian = HebbianCore(20, 40)
    temporal = TemporalLayer(40, 80)

    # Create plasticity controller
    plasticity_controller = PlasticityController([hebbian, temporal])

    # Create meta-optimizer
    meta_optimizer = MetaOptimizer(plasticity_controller)

    # Test meta-learning with simple tasks
    def simple_task(step):
        return 0.5 + 0.01 * step + 0.05 * np.random.randn()

    tasks = [simple_task for _ in range(3)]
    meta_optimizer.meta_train(5, tasks)  # Short training for test

    # Check learning history
    learning_history = meta_optimizer.learning_history
    assert len(learning_history) == 5, f"Expected 5 episodes, got {len(learning_history)}"

    print("✅ Meta-learning test passed")

def test_configurations():
    """Test configuration system."""
    print("\n🧪 Testing Configurations...")

    # Test default config retrieval
    hierarchy_config = get_default_config('hierarchy')
    assert 'num_levels' in hierarchy_config, "Missing num_levels in hierarchy config"

    plasticity_config = get_default_config('plasticity')
    assert 'learning_rate' in plasticity_config, "Missing learning_rate in plasticity config"

    es_config = get_default_config('es')
    assert 'population_size' in es_config, "Missing population_size in ES config"

    print("✅ Configuration test passed")

def test_integration():
    """Test full integration of ORTHOS components."""
    print("\n🧪 Testing Integration...")

    # Create complete ORTHOS system
    manager = HierarchyManager()

    # Build hierarchy
    level0 = HierarchicalLevel(0, 10, 20, temporal_resolution=1)
    level0.add_layer(ReactiveLayer(10, 20))
    manager.add_level(level0)

    level1 = HierarchicalLevel(1, 20, 40, temporal_resolution=2)
    level1.add_layer(HebbianCore(20, 40))
    manager.add_level(level1)

    # Create plasticity controller
    hebbian_core = HebbianCore(20, 40)
    plasticity_controller = PlasticityController([hebbian_core])

    # Create meta-optimizer
    meta_optimizer = MetaOptimizer(plasticity_controller)

    # Test full pipeline
    input_data = np.random.randn(20, 10)
    representations = manager.process_hierarchy(input_data, 20)

    # Verify representations
    assert len(representations) == 2, "Hierarchy processing failed"

    # Test plasticity adaptation
    for _ in range(5):
        performance = np.random.random()
        plasticity_controller.adapt_plasticity(performance)

    # Test meta-learning
    def test_task(step):
        return 0.6 + 0.01 * step

    tasks = [test_task]
    meta_optimizer.meta_train(3, tasks)

    print("✅ Integration test passed")

def run_all_tests():
    """Run all tests."""
    print("🚀 Running ORTHOS v5.0.0 Tests")
    print("=" * 50)


    try:
        test_layers()
        test_hierarchy()
        test_plasticity()
        test_plasticity_rules()
        test_es_optimizer()
        test_meta_learning()
        test_configurations()
        test_integration()

        print("\n🎉 All tests passed successfully!")
        print("ORTHOS v5.0.0 implementation is working correctly.")


    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    run_all_tests()