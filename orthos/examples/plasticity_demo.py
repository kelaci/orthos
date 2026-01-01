"""
Plasticity demonstration showing ORTHOS's adaptive learning capabilities.

This example demonstrates how ORTHOS can adapt its plasticity parameters
using evolutionary strategies to optimize performance.
"""

import numpy as np
from orthos.layers.hebbian import HebbianCore
from orthos.layers.temporal import TemporalLayer
from orthos.plasticity.controller import PlasticityController
from orthos.utils.logging import setup_logging
from orthos.utils.visualization import plot_learning_curve, plot_plasticity_parameters

def plasticity_demo():
    """Demonstration of ORTHOS's plasticity control capabilities."""
    print("🧠 ORTHOS Plasticity Demo")
    print("=" * 50)

    # Setup logging
    logger = setup_logging('gaia_plasticity', 'INFO')
    logger.info("Starting ORTHOS plasticity demo")

    # Create target modules for plasticity control
    logger.info("Creating target modules...")
    hebbian_core = HebbianCore(20, 40, plasticity_rule='hebbian')
    temporal_layer = TemporalLayer(input_size=40, hidden_size=80, time_window=5)

    print(f"HebbianCore: {hebbian_core}")
    print(f"TemporalLayer: {temporal_layer}")

    # Create plasticity controller
    logger.info("Creating PlasticityController...")
    controller = PlasticityController(
        target_modules=[hebbian_core, temporal_layer],
        adaptation_rate=0.01,
        exploration_noise=0.1
    )

    print(f"\nPlasticityController: {controller}")

    # Simulate a learning task
    def simulate_task(performance_history: list, step: int) -> float:
        """
        Simulate a learning task with performance that improves over time.

        Args:
            performance_history: History of previous performances
            step: Current step

        Returns:
            Simulated performance metric
        """
        # Base performance that improves with time
        base_performance = 0.5 + 0.01 * step

        # Add some noise
        noise = 0.05 * np.random.randn()

        # Performance depends on current plasticity parameters
        params = controller.get_current_params()
        param_quality = np.mean(params)  # Simple quality metric

        # Final performance is combination of all factors
        performance = base_performance + param_quality * 0.1 + noise

        # Ensure performance is in [0, 1] range
        return np.clip(performance, 0.0, 1.0)

    # Run adaptation loop
    performance_history = []
    num_episodes = 50

    logger.info(f"Running {num_episodes} adaptation episodes...")
    print(f"\nRunning {num_episodes} adaptation episodes...")

    for episode in range(num_episodes):
        # Simulate task performance
        performance = simulate_task(performance_history, episode)
        performance_history.append(performance)

        # Adapt plasticity parameters
        controller.adapt_plasticity(performance)

        # Log progress
        if episode % 10 == 0:
            current_params = controller.get_current_params()
            logger.info(f"Episode {episode}: Performance = {performance:.4f}, Params = {current_params}")
            print(f"Episode {episode}: Performance = {performance:.4f}")

    # Visualize results
    print("\n📊 Visualizing Results...")
    plot_learning_curve(performance_history, title="Plasticity Adaptation Learning Curve")

    # Plot parameter evolution
    param_history = controller.get_param_history()
    if param_history:
        # Convert parameter history to dict format for visualization
        param_names = ['param_' + str(i) for i in range(len(param_history[0]))]
        param_dict_history = [{param_names[j]: param_history[i][j] for j in range(len(param_history[i]))}
                             for i in range(len(param_history))]

        plot_plasticity_parameters(param_dict_history, title="Plasticity Parameter Evolution")

    # Display final results
    final_params = controller.get_current_params()
    print(f"\n📋 Final Results:")
    print(f"Final Performance: {performance_history[-1]:.4f}")
    print(f"Initial Performance: {performance_history[0]:.4f}")
    print(f"Improvement: {performance_history[-1] - performance_history[0]:.4f}")
    print(f"Final Parameters: {final_params}")

    # Evaluate plasticity controller configuration
    config = controller.get_config()
    print(f"\n📝 Controller Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\n✅ Plasticity demo completed successfully!")
    logger.info("ORTHOS plasticity demo completed successfully")

def plasticity_rule_comparison():
    """Compare different plasticity rules."""
    print("\n🔬 Plasticity Rule Comparison")
    print("=" * 50)

    # Test different plasticity rules
    rules = ['hebbian', 'oja', 'bcm']
    results = {}

    for rule in rules:
        print(f"\nTesting {rule.upper()} rule...")

        # Create HebbianCore with specific rule
        core = HebbianCore(20, 40, plasticity_rule=rule)

        # Simple test: measure weight changes over time
        weight_history = []
        input_data = np.random.randn(100, 20)  # 100 time steps

        for t in range(100):
            output = core.forward(input_data[t])
            core.update()
            weight_history.append(core.get_weights().copy())

        # Calculate weight change statistics
        weight_changes = np.array([np.linalg.norm(weight_history[i+1] - weight_history[i])
                                 for i in range(len(weight_history)-1)])

        results[rule] = {
            'mean_change': np.mean(weight_changes),
            'std_change': np.std(weight_changes),
            'max_change': np.max(weight_changes),
            'final_norm': np.linalg.norm(weight_history[-1])
        }

        print(f"{rule.upper()} results: {results[rule]}")

    # Display comparison
    print("\n📊 Plasticity Rule Comparison Results:")
    for rule, metrics in results.items():
        print(f"{rule.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    plasticity_demo()
    plasticity_rule_comparison()