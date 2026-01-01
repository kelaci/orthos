import numpy as np
import sys
import os

# Ensure we can import gaia
sys.path.insert(0, os.getcwd())

from gaia.core.base import Layer
from gaia.hierarchy.filtered_level import FilteredHierarchicalLevel
from gaia.hierarchy.consensus_manager import ConsensusHierarchyManager

# Concrete Layer for testing
class SimpleLayer(Layer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        # Xavier init
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / (input_size + output_size))
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Handle 1D input
        if x.ndim == 1:
            x = x.reshape(1, -1)
        res = x @ self.weights
        return res.flatten() if x.shape[0] == 1 else res

    def backward(self, grad): pass
    def update(self, lr): pass
    def reset_state(self): pass
    def activation(self, x): return x
    def get_weights(self): return self.weights
    def set_weights(self, w): self.weights = w
    def get_plasticity_params(self): return {}
    def set_plasticity_params(self, p): pass

def main():
    print("🚀 Initializing GAIA v4.2 Probabilistic Spine...")
    
    # 1. Setup Consensus Manager
    manager = ConsensusHierarchyManager(
        consensus_method='weighted_vote',
        min_agreement=0.6
    )
    
    # Notice: Modified level output sizes to be consistent for Consensus Aggregation.
    print("Notice: Modified level output sizes to consistent (10) for Consensus Aggregation.")
    
    # 2. Add 3 Levels of increasing complexity
    
    # Level 0
    # Note: User plan had output_size=20, invalid for consensus with other levels if dim is mismatched.
    l0 = FilteredHierarchicalLevel(
        level_id=0, input_size=10, output_size=10, 
        temporal_resolution=1, filter_type='kalman', obs_noise=0.5
    )
    l0.add_layer(SimpleLayer(10, 10))
    manager.add_level(l0)
    
    # Level 1
    # Note: User plan had output_size=40
    l1 = FilteredHierarchicalLevel(
        level_id=1, input_size=10, output_size=10, 
        temporal_resolution=2, filter_type='kalman', obs_noise=0.3
    )
    l1.add_layer(SimpleLayer(10, 10))
    manager.add_level(l1)
    
    # Level 2
    # Note: User plan had output_size=10
    l2 = FilteredHierarchicalLevel(
        level_id=2, input_size=10, output_size=10, 
        temporal_resolution=4, filter_type='kalman', obs_noise=0.1
    )
    l2.add_layer(SimpleLayer(10, 10))
    manager.add_level(l2)
    
    # 3. Run Simulation Loop
    print("\n--- Starting Simulation ---")
    np.random.seed(42)
    input_seq = np.random.randn(50, 10) # 50 timesteps
    
    for t, inp in enumerate(input_seq):
        result = manager.get_consensus_prediction(inp)
        
        print(f"Step {t:02d} | "
              f"Agreement: {result.agreement_score:.2f} | "
              f"Uncertainty: {result.uncertainty:.4f} | "
              f"Outliers: {result.outlier_count}")
              
        # Simulate top-down signal if stable
        if t == 25:
            print("   >> Injecting Top-Down Prior (simulated expectation)...")
        
    # 4. Final Status
    print("\n--- Final Status ---")
    print(f"Hierarchy Stable: {manager.is_hierarchy_stable()}")

if __name__ == '__main__':
    main()
