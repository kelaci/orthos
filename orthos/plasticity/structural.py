"""
Structural plasticity and sparsity management for ORTHOS.

This module provides the core engines for managing sparse connectivity,
synaptic turnover (pruning/regrowth), and structural guardrails.
"""

from typing import Dict, Optional, Tuple, Any, List, Union
import numpy as np
from orthos.core.gpu_utils import get_array_module, to_cpu, to_gpu, zeros, mean, abs, sum

class SynapticConsolidation:
    """
    Protects important synapses from pruning.
    
    Synapses with high weight magnitude, high age, or high importance
    (proxy for Fisher Information) are protected.
    """
    def __init__(self, importance_decay: float = 0.99,
                 protection_threshold: float = 0.5):
        self.importance_decay = importance_decay
        self.protection_threshold = protection_threshold
        self.importance_scores = None
        
    def update_importance(self, layer_weights: Any, weight_grads: Optional[Any] = None) -> None:
        """Update importance scores based on gradient magnitude."""
        xp = get_array_module()
        
        if self.importance_scores is None:
            self.importance_scores = xp.zeros_like(layer_weights)
        
        # Decay old importance
        self.importance_scores *= self.importance_decay
        
        # Add new importance from gradients (Fisher Information proxy)
        if weight_grads is not None:
            self.importance_scores += xp.square(weight_grads)
            
    def get_protection_mask(self, mask: Any, synaptic_age: Any) -> Any:
        """Return mask of synapses that should NOT be pruned."""
        xp = get_array_module()
        
        if self.importance_scores is None:
            return xp.zeros_like(mask, dtype=bool)
        
        # Normalize importance
        max_imp = xp.max(self.importance_scores)
        norm_importance = self.importance_scores / (max_imp + 1e-8)
        
        # Combine with age-based protection
        max_age = xp.max(synaptic_age)
        age_factor = synaptic_age / (max_age + 1)
        
        # Heuristic combination
        protection_score = 0.5 * norm_importance + 0.5 * age_factor
        
        return protection_score > self.protection_threshold

class SparsityScheduler:
    """
    Controls the target density over training progress.
    """
    def __init__(self, initial_density: float = 1.0,
                 target_density: float = 0.1,
                 warmup_steps: int = 1000,
                 total_steps: int = 10000,
                 schedule: str = 'cosine'):
        self.initial_density = initial_density
        self.target_density = target_density
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.schedule = schedule
        self.current_step = 0
        
    def get_density(self) -> float:
        """Get current target density based on schedule."""
        if self.current_step < self.warmup_steps:
            return self.initial_density
        
        progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        
        if self.schedule == 'linear':
            density = self.initial_density - progress * (self.initial_density - self.target_density)
        elif self.schedule == 'cosine':
            density = self.target_density + 0.5 * (self.initial_density - self.target_density) * (1 + np.cos(np.pi * progress))
        elif self.schedule == 'step':
            if progress < 0.33:
                density = self.initial_density
            elif progress < 0.66:
                density = (self.initial_density + self.target_density) / 2
            else:
                density = self.target_density
        else:
            density = self.target_density
            
        return float(density)
    
    def step(self) -> None:
        self.current_step += 1

class StructuralGuardrails:
    """
    Prevents catastrophic topology changes and ensures minimum connectivity.
    """
    def __init__(self, min_density: float = 0.05,
                 max_turnover: float = 0.1,
                 warmup_steps: int = 1000):
        self.min_density = min_density
        self.max_turnover = max_turnover
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
    def should_rewire(self) -> bool:
        """Check if rewiring is allowed."""
        return self.current_step >= self.warmup_steps
    
    def validate_rewire(self, mask: Any, proposed_prunes: Any) -> Any:
        """Validate and potentially limit pruning to maintain minimum density."""
        xp = get_array_module()
        
        current_density = float(xp.mean(mask))
        prune_fraction = float(xp.mean(proposed_prunes))
        post_prune_density = current_density - prune_fraction
        
        if post_prune_density < self.min_density:
            # We need to prune fewer synapses
            allowed_prune_fraction = current_density - self.min_density
            if allowed_prune_fraction <= 0:
                return xp.zeros_like(proposed_prunes, dtype=bool)
            
            # Probability of keeping a proposed prune
            keep_prob = allowed_prune_fraction / (prune_fraction + 1e-8)
            random_gate = xp.random.random(proposed_prunes.shape) < keep_prob
            return proposed_prunes & random_gate
                
        return proposed_prunes
    
    def step(self) -> None:
        self.current_step += 1

class SparsityMonitor:
    """
    Tracks and logs sparsity-related metrics for observability.
    """
    def __init__(self):
        self.history = {
            'density': [],
            'turnover': [],
            'entropy': [],
            'pruned': [],
            'grown': []
        }
        self._last_mask = None
        
    def log(self, mask: Any, rewire_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Log metrics for a layer mask."""
        xp = get_array_module()
        metrics = {}
        
        # Connectivity Density
        density = float(xp.mean(mask))
        metrics['density'] = density
        self.history['density'].append(density)
        
        # Synaptic Turnover
        if self._last_mask is not None:
            diff = mask != self._last_mask
            turnover = float(xp.mean(diff))
            metrics['turnover'] = turnover
            self.history['turnover'].append(turnover)
        else:
            metrics['turnover'] = 0.0
            
        self._last_mask = mask.copy()
        
        # Mask Entropy
        p = max(1e-8, min(1.0 - 1e-8, density))
        entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)
        metrics['entropy'] = entropy
        self.history['entropy'].append(entropy)
        
        if rewire_metrics:
            self.history['pruned'].append(rewire_metrics.get('pruned', 0))
            self.history['grown'].append(rewire_metrics.get('grown', 0))
            metrics.update(rewire_metrics)
            
        return metrics

class StructuralOptimizer:
    """
    Manages the synaptic pruning and regrowth cycle.
    """
    def __init__(self, 
                 prune_threshold: float = 0.01,
                 regrow_fraction: float = 0.05,
                 consolidation_strength: float = 0.5,
                 maturity_age: int = 50):
        self.prune_threshold = prune_threshold
        self.regrow_fraction = regrow_fraction
        self.consolidation_strength = consolidation_strength
        self.maturity_age = maturity_age
        
        self.consolidation = SynapticConsolidation(protection_threshold=consolidation_strength)
        self.guardrails = StructuralGuardrails()
        self.monitor = SparsityMonitor()
        
    def rewire(self, layer: Any) -> Dict[str, Any]:
        """
        Execute one pruning/regrowth cycle on a MaskedLinear layer.
        """
        xp = get_array_module()
        metrics = {}
        
        if not self.guardrails.should_rewire():
            self.guardrails.step()
            return {'status': 'warmup'}

        # 1. Update importance for consolidation
        self.consolidation.update_importance(layer.weights, getattr(layer, 'weight_grads', None))
        
        # 2. PRUNING
        # Candidates: weak magnitude AND mature AND currently active
        prune_candidates = (
            (xp.abs(layer.weights) < self.prune_threshold) &
            (layer.synaptic_age > self.maturity_age) &
            (layer.mask == 1)
        )
        
        # Protect important synapses
        protection_mask = self.consolidation.get_protection_mask(layer.mask, layer.synaptic_age)
        proposed_prunes = prune_candidates & (~protection_mask)
        
        # Validate with guardrails
        actual_prunes = self.guardrails.validate_rewire(layer.mask, proposed_prunes)
        
        # Apply pruning
        layer.mask[actual_prunes] = 0
        layer.weights[actual_prunes] = 0
        layer.synaptic_age[actual_prunes] = 0
        metrics['pruned'] = int(xp.sum(actual_prunes))
        
        # 3. REGROWTH
        current_density = float(xp.mean(layer.mask))
        if current_density < layer.density:
            target_regrow_count = int(self.regrow_fraction * layer.mask.size)
            
            # Find inactive positions
            inactive_mask = (layer.mask == 0)
            n_inactive = int(xp.sum(inactive_mask))
            
            if n_inactive > 0:
                n_grow = min(target_regrow_count, n_inactive)
                
                # Selection logic (simplified for both CPU/GPU)
                # On GPU, we might use a random mask over the inactive mask
                prob_grow = n_grow / n_inactive
                grow_mask = (xp.random.random(layer.mask.shape) < prob_grow) & inactive_mask
                
                # Apply regrowth
                layer.mask[grow_mask] = 1
                # Initialize with small random weights
                layer.weights[grow_mask] = xp.random.randn(*layer.weights.shape)[grow_mask] * 0.01
                layer.synaptic_age[grow_mask] = 0
                metrics['grown'] = int(xp.sum(grow_mask))
        else:
            metrics['grown'] = 0
            
        # 4. Aging
        layer.synaptic_age += layer.mask
        
        # 5. Logging
        log_metrics = self.monitor.log(layer.mask, metrics)
        self.guardrails.step()
        
        return log_metrics
