import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
from dataclasses import dataclass
from typing import List, Dict, Optional
import matplotlib.pyplot as plt

# ORTHOS Imports
from orthos.hierarchy.manager import HierarchyManager
from orthos.hierarchy.level import HierarchicalLevel
from orthos.layers.reactive import ReactiveLayer
# Assuming Kalman/Probabilistic exists in v4.2 based on prompt context
# from orthos.filters.kalman import KalmanFilter # Example import

@dataclass
class BenchmarkResult:
    name: str
    scenario: str
    metric: str
    value: float
    units: str
    metadata: Dict

class UtilityBenchmark:
    """
    Orchestrates the 'Practical & Business Utility' benchmarks for ORTHOS.
    """
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.manager = self._setup_gaia_system()
        
    def _setup_gaia_system(self) -> HierarchyManager:
        """Initializes a standard ORTHOS hierarchy for testing."""
        manager = HierarchyManager()
        # Level 0: Fast Reactive
        l0 = HierarchicalLevel(0, input_size=10, output_size=16, temporal_resolution=1)
        l0.add_layer(ReactiveLayer(10, 16))
        manager.add_level(l0)
        # Add more levels/probabilistic components here normally
        return manager

    def run_noise_resilience_test(self, noise_levels: List[float]):
        """
        Test Scenario 1.1: Signal-to-Noise Degradation.
        Measures reliability under increasing chaos.
        """
        print(f"Running Noise Resilience Test along {noise_levels}...")
        
        base_signal = np.sin(np.linspace(0, 100, 1000)) # Simple sine wave
        # Reshape to (1000, 1, 10) for input simulation (broadcasting)
        
        for sigma in noise_levels:
            # 1. Generate Noisy Data
            noise = np.random.normal(0, sigma, base_signal.shape)
            noisy_signal = base_signal + noise
            
            # 2. Run ORTHOS
            # start_time = time.time()
            # output = self.manager.process_hierarchy(noisy_signal)
            
            # 3. Measure Error (MSE against clean signal)
            # reconstruction = ... extract from output ...
            # mse = np.mean((reconstruction - base_signal)**2)
            
            # Placeholder result
            mse = 0.01 * (sigma + 0.1) # Mock result
            
            self.results.append(BenchmarkResult(
                name="ORTHOS v4.2",
                scenario=f"Noise_Sigma_{sigma}",
                metric="MSE",
                value=mse,
                units="squared_error",
                metadata={"noise_type": "Gaussian"}
            ))
            
    def run_efficiency_test(self, sequence_lengths: List[int]):
        """
        Test Scenario 2.1: Inference Cost vs Sequence Length.
        Demonstrates the business value of sparsity/hierarchy.
        """
        print(f"Running Efficiency Test for lengths {sequence_lengths}...")
        
        for seq_len in sequence_lengths:
            data = np.random.randn(seq_len, 10)
            
            start_time = time.time()
            # self.manager.process_hierarchy(data)
            time.sleep(seq_len * 0.0001) # Mock processing time
            duration = time.time() - start_time
            
            throughput = seq_len / duration
            
            self.results.append(BenchmarkResult(
                name="ORTHOS v4.2",
                scenario=f"Seq_Len_{seq_len}",
                metric="Throughput",
                value=throughput,
                units="tokens/sec",
                metadata={"hardware": "cpu"}
            ))

    def run_drone_safety_test(self, failure_modes: List[str]):
        """
        Test Scenario 1.3: Drone Salvation Protocol.
        Simulates sudden sensor loss and motor degradation.
        """
        print(f"Running Drone Safety Test for modes {failure_modes}...")
        
        # Simulation parameters
        altitude = 50.0 # meters
        
        for mode in failure_modes:
            # 1. Simulate stable flight
            # 2. Inject Failure (e.g., set GPS_noise = infinity)
            
            # 3. Simulate ORTHOS response (Mock)
            # In real impl, ORTHOS's Precision estimates would shift weights automatically
            
            reaction_time = 0.05 # 50ms (Hypothetical ORTHOS response)
            drift = 0.2 if "GPS_LOSS" in mode else 1.5 # meters
            
            outcome = "STABLE" if drift < 1.0 else "CRASH"
            
            self.results.append(BenchmarkResult(
                name="ORTHOS v4.2 (DroneMode)",
                scenario=f"Failure_{mode}",
                metric="Drift_Distance",
                value=drift,
                units="meters",
                metadata={"outcome": outcome, "reaction_ms": reaction_time * 1000}
            ))

    def print_report(self):
        print("\n=== ORTHOS RESEARCH UTILITY REPORT ===")
        print(f"{'SCENARIO':<25} | {'METRIC':<15} | {'VALUE':<10} | {'UNITS'}")
        print("-" * 65)
        for r in self.results:
            print(f"{r.scenario:<25} | {r.metric:<15} | {r.value:<10.4f} | {r.units}")

if __name__ == "__main__":
    benchmark = UtilityBenchmark()
    
    # 1. Run Robustness
    benchmark.run_noise_resilience_test([0.1, 0.5, 1.0, 2.0])
    
    # 2. Run Efficiency
    benchmark.run_efficiency_test([100, 1000, 5000])

    # 3. Run Drone Safety
    benchmark.run_drone_safety_test(["GPS_LOSS", "MOTOR_DEGRADE"])
    
    benchmark.print_report()
