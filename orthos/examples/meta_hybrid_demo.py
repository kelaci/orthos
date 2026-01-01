import numpy as np
import matplotlib.pyplot as plt
from orthos.hierarchy.filtered_level import FilteredHierarchicalLevel
from orthos.layers.hebbian import HebbianCore

def run_meta_simulation():
    print("🚀 Starting ORTHOS Hybrid Meta-Learning Simulation...")
    
    # 1. Setup Environment
    n_steps = 200
    input_size = 10
    output_size = 5
    
    # Generate a simple signal with a "Stationary" period and a "Storm" (noisy/chaotic) period
    def get_signal(t):
        # Base signal: Sine wave
        sig = np.sin(0.1 * t) * np.ones(input_size)
        # Noise
        if 50 < t < 150: # The Storm
            noise_level = 0.5
        else:
            noise_level = 0.05
        return sig + np.random.randn(input_size) * noise_level

    # 2. Setup ORTHOS Level
    level = FilteredHierarchicalLevel(
        level_id=1,
        input_size=input_size,
        output_size=output_size,
        filter_type='kalman',
        process_noise=0.01,
        obs_noise=0.1,
        use_meta_bandit=True
    )
    
    # Add a Hebbian layer for feature learning
    level.add_layer(HebbianCore(input_size, output_size, plasticity_rule='oja'))
    
    # 3. Simulation Loop
    history = {
        'error': [],
        'uncertainty': [],
        'adaptation_rate': [],
        'obs_noise_scale': []
    }
    
    print("📈 Running simulation steps...")
    for t in range(n_steps):
        data = get_signal(t)
        
        # Forward pass (this triggers meta-adaptation internally)
        pred, unc = level.forward_filtered(data)
        
        # Record metrics
        meta_params = level.meta_manager.active_params
        history['error'].append(float(np.linalg.norm(data[:output_size] - pred)))
        history['uncertainty'].append(unc)
        history['adaptation_rate'].append(meta_params.get('adaptation_rate', 0))
        history['obs_noise_scale'].append(meta_params.get('obs_noise_scale', 0))
        
        if (t + 1) % 50 == 0:
            print(f"   Step {t+1}/{n_steps} | Err: {history['error'][-1]:.3f} | Unc: {unc:.3f}")

    # 4. Results
    print("\n✅ Simulation complete.")
    
    # Plotting
    try:
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        
        axes[0].plot(history['error'], color='#e74c3c', label='Prediction Error')
        axes[0].axvspan(50, 150, alpha=0.2, color='gray', label='The Storm')
        axes[0].set_ylabel('Error')
        axes[0].legend()
        
        axes[1].plot(history['uncertainty'], color='#3498db', label='KF Uncertainty')
        axes[1].set_ylabel('Uncertainty')
        axes[1].legend()
        
        axes[2].plot(history['adaptation_rate'], color='#2ecc71', label='Meta Adaptation Rate (η)')
        axes[2].set_ylabel('η')
        axes[2].legend()
        
        axes[3].plot(history['obs_noise_scale'], color='#f1c40f', label='Meta Obs Noise Scale (R)')
        axes[3].set_ylabel('R scale')
        axes[3].set_xlabel('Time Step')
        axes[3].legend()
        
        plt.tight_layout()
        plt.savefig('meta_hybrid_verification.png')
        print("📊 Plot saved to meta_hybrid_verification.png")
    except Exception as e:
        print(f"⚠️ Could not generate plot: {e}")
        
    # Validation check
    # We expect the Bandit to have modulated parameters during the run
    rates = np.array(history['adaptation_rate'])
    noise_scales = np.array(history['obs_noise_scale'])
    
    rate_variance = np.var(rates)
    noise_variance = np.var(noise_scales)
    
    print(f"\nFinal Adaptation Rate: {rates[-1]:.6f}")
    print(f"Adaptation Rate Variance: {rate_variance:.8f}")
    print(f"Noise Scale Variance: {noise_variance:.8f}")
    
    if rate_variance > 0 or noise_variance > 0:
        print("🌟 Success: Meta-learning is actively modulating parameters over time!")
    else:
        print("❌ Warning: Meta-parameters remained static. Check if learning rate or exploration is too low.")

if __name__ == "__main__":
    run_meta_simulation()
