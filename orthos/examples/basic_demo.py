"""
Basic demonstration of ORTHOS's hierarchical processing capabilities.

This example shows how to create a simple hierarchy with multiple levels
and process input data through the hierarchical architecture.
"""

import numpy as np
from orthos.layers.reactive import ReactiveLayer
from orthos.layers.hebbian import HebbianCore
from orthos.layers.temporal import TemporalLayer
from orthos.hierarchy.level import HierarchicalLevel
from orthos.hierarchy.manager import HierarchyManager
from orthos.utils.logging import setup_logging
from orthos.utils.visualization import plot_hierarchy_representations

def basic_demo():
    """Basic demonstration of ORTHOS components."""
    print("🚀 ORTHOS Basic Demo")
    print("=" * 50)

    # Setup logging
    logger = setup_logging('gaia_demo', 'INFO')
    logger.info("Starting ORTHOS basic demo")

    # Create a simple hierarchy
    manager = HierarchyManager()

    # Level 0: Input processing with ReactiveLayer
    logger.info("Creating Level 0 (Input Processing)")
    level0 = HierarchicalLevel(0, input_size=10, output_size=20, temporal_resolution=1)
    level0.add_layer(ReactiveLayer(10, 20, activation='relu'))
    manager.add_level(level0)
    logger.info(f"Level 0: {level0}")

    # Level 1: Feature extraction with HebbianCore
    logger.info("Creating Level 1 (Feature Extraction)")
    level1 = HierarchicalLevel(1, input_size=20, output_size=40, temporal_resolution=2)
    level1.add_layer(HebbianCore(20, 40, plasticity_rule='hebbian'))
    manager.add_level(level1)
    logger.info(f"Level 1: {level1}")

    # Level 2: Sequence processing with TemporalLayer
    logger.info("Creating Level 2 (Sequence Processing)")
    level2 = HierarchicalLevel(2, input_size=40, output_size=80, temporal_resolution=4)
    level2.add_layer(TemporalLayer(40, 80, time_window=5))
    manager.add_level(level2)
    logger.info(f"Level 2: {level2}")

    print("\nHierarchy Configuration:")
    print(manager)

    # Generate some random input data (100 time steps, 10 features)
    time_steps = 100
    input_data = np.random.randn(time_steps, 10)

    logger.info(f"Processing {time_steps} time steps through hierarchy...")
    print(f"\nProcessing {time_steps} time steps through hierarchy...")

    # Process through hierarchy
    representations = manager.process_hierarchy(input_data, time_steps)

    # Display results
    print("\n📊 Hierarchical Representations:")
    for level_id, reps in representations.items():
        print(f"Level {level_id}: {len(reps)} representations, shape {reps[0].shape}")

    # Visualize representations (use subset for clarity)
    sample_reps = {level: reps[:10] for level, reps in representations.items()}
    plot_hierarchy_representations(sample_reps, title="ORTHOS Hierarchical Processing")

    # Get current representations
    current_reps = manager.get_all_representations()
    print("\n📋 Current Representations:")
    for level_id, rep in current_reps.items():
        if rep is not None:
            print(f"Level {level_id}: shape {rep.shape}, mean={np.mean(rep):.4f}, std={np.std(rep):.4f}")
        else:
            print(f"Level {level_id}: No representation (not processed yet)")

    print("\n✅ Demo completed successfully!")
    logger.info("ORTHOS basic demo completed successfully")

if __name__ == "__main__":
    basic_demo()