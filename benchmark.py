import time
import numpy as np
import matplotlib.pyplot as plt
from orthos.hierarchy.manager import HierarchyManager
from orthos.hierarchy.level import HierarchicalLevel
from orthos.layers.reactive import ReactiveLayer
from orthos.layers.hebbian import HebbianCore

def benchmark_hierarchy(depths=[1, 3, 5, 10], steps=1000):
    print(f"🚀 Running Scale Benchmark (Steps={steps})")
    print("-" * 50)
    print(f"{'Depth':<10} | {'Time (s)':<15} | {'Steps/sec':<15}")
    print("-" * 50)
    
    results = []
    
    for depth in depths:
        manager = HierarchyManager()
        input_size = 10
        
        # Build hierarchy
        for i in range(depth):
            width = 10
            level = HierarchicalLevel(i, input_size=width, output_size=width, temporal_resolution=1)
            if i == 0:
                level.add_layer(ReactiveLayer(width, width))
            else:
                level.add_layer(HebbianCore(width, width))
            manager.add_level(level)
            
        # Generate data
        data = np.random.randn(steps, 10)
        
        # Run
        start = time.time()
        manager.process_hierarchy(data, steps)
        elapsed = time.time() - start
        
        sps = steps / elapsed
        results.append(sps)
        
        print(f"{depth:<10} | {elapsed:<15.4f} | {sps:<15.2f}")
        
    print("-" * 50)
    return depths, results

if __name__ == "__main__":
    depths, sps = benchmark_hierarchy()
    
    # Simple plot
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(depths, sps, 'o-')
        plt.xlabel('Hierarchy Depth')
        plt.ylabel('Steps per Second')
        plt.title('ORTHOS Performance Benchmark')
        plt.grid(True)
        plt.savefig('benchmark_results.png')
        print("📊 Benchmark plot saved to benchmark_results.png")
    except Exception as e:
        print(f"Could not save plot: {e}")
