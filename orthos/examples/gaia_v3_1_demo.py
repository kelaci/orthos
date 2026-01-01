# ============================================================================
# 🔬 ORTHOS PROTOCOL v3.1 — ANALYSIS & ENHANCEMENT SUGGESTIONS
# ============================================================================

"""
ARCHITECTURAL ANALYSIS:

This is a sophisticated hybrid implementation combining:
1. Hebbian plasticity (fast/slow trace dynamics)
2. Active Inference (Free Energy Principle)
3. BitNet-style quantization
4. Ensemble uncertainty estimation

KEY INNOVATIONS:
- Dual-timescale plasticity mimics biological memory consolidation
- Quantized backbone with plastic modulation (hybrid digital/analog)
- Epistemic exploration through uncertainty quantification
- Homeostatic normalization prevents runaway dynamics

STABILITY PROPERTIES:
✅ Trace decay prevents unbounded growth
✅ Homeostatic clipping at norm=5.0
✅ Sigmoid-weighted policy updates
✅ Separate optimizers for WM and policy

POTENTIAL ENHANCEMENTS BELOW:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")

# ============================================================================
# ENHANCED CONFIGURATION WITH DIAGNOSTIC FEATURES
# ============================================================================

@dataclass
class GaiaConfigEnhanced:
    # Core dimensions
    state_dim: int = 1
    action_dim: int = 1
    hidden_dim: int = 64
    n_ensemble: int = 5
    
    # Hebbian plasticity
    fast_trace_decay: float = 0.95
    fast_trace_lr: float = 0.05
    slow_trace_decay: float = 0.99
    slow_trace_lr: float = 0.01
    homeostatic_target: float = 5.0
    
    # Active Inference
    planning_samples: int = 30
    exploration_weight: float = 0.5
    temperature: float = 0.1
    
    # Learning
    weight_scale: float = 5.0
    wm_lr: float = 1e-3
    policy_lr: float = 1e-3
    
    # Diagnostics
    track_metrics: bool = True
    trace_history_len: int = 1000

cfg = GaiaConfigEnhanced()

# ============================================================================
# ENHANCED PLASTIC LINEAR WITH DIAGNOSTICS
# ============================================================================

class DiagnosticPlasticLinear(nn.Module):
    """Enhanced version with detailed tracking for research analysis"""
    
    def __init__(self, in_features: int, out_features: int, cfg: GaiaConfigEnhanced):
        super().__init__()
        self.cfg = cfg
        self.in_features = in_features
        self.out_features = out_features
        
        # Statikus súlyok
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        
        # Plasztikus nyomok
        self.register_buffer("fast_trace", torch.zeros(out_features, in_features))
        self.register_buffer("slow_trace", torch.zeros(out_features, in_features))
        
        # Diagnostics
        self.plasticity_enabled = True
        if cfg.track_metrics:
            self.register_buffer("trace_norm_history", 
                               torch.zeros(cfg.trace_history_len))
            self.register_buffer("update_magnitude_history", 
                               torch.zeros(cfg.trace_history_len))
            self.step_counter = 0
    
    def bitnet_quantize(self, w: torch.Tensor) -> torch.Tensor:
        """1.58-bit kvantálás: {-1, 0, 1} with per-row scaling"""
        scale = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
        w_normalized = w / scale
        w_quantized = w_normalized.round().clamp(-1, 1)
        return w_quantized * scale
    
    def update_traces(self, x: torch.Tensor, y: torch.Tensor):
        """Enhanced trace update with diagnostics"""
        if not self.plasticity_enabled:
            return
        
        with torch.no_grad():
            # Hebbian update: csak aktív neuronok
            y_active = F.relu(y)
            delta = torch.matmul(y_active.t(), x) / x.shape[0]
            
            # Track update magnitude
            if self.cfg.track_metrics:
                update_mag = delta.norm().item()
                # Use scalar indexing for tensor writing if needed, but here simple assignment is fine
                # However, direct assignment to tensor buffer element works in torch
                idx = self.step_counter % self.cfg.trace_history_len
                self.update_magnitude_history[idx] = update_mag
            
            # Gyors nyom frissítése
            self.fast_trace.mul_(self.cfg.fast_trace_decay)
            self.fast_trace.add_(delta, alpha=self.cfg.fast_trace_lr)
            
            # Homeostatic normalization
            fast_norm = self.fast_trace.norm()
            if fast_norm > self.cfg.homeostatic_target:
                self.fast_trace.mul_(self.cfg.homeostatic_target / (fast_norm + 1e-6))
            
            # Lassú nyom (konszolidáció)
            self.slow_trace.mul_(self.cfg.slow_trace_decay)
            self.slow_trace.add_(self.fast_trace, alpha=self.cfg.slow_trace_lr)
            
            # Track trace norm
            if self.cfg.track_metrics:
                self.trace_norm_history[idx] = fast_norm.item()
                self.step_counter += 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Kvantált váz
        w_static = self.bitnet_quantize(self.weight)
        
        # Plasztikus moduláció
        w_plastic = 0.1 * self.fast_trace + 0.05 * self.slow_trace
        
        # Effektív súly
        w_effective = w_static + w_plastic
        y = F.linear(x, w_effective)
        
        # Nyomok frissítése
        self.update_traces(x, y)
        
        return y
    
    def get_diagnostics(self) -> Dict[str, float]:
        """Return current diagnostic metrics"""
        if not self.cfg.track_metrics:
            return {}
        
        valid_steps = min(self.step_counter, self.cfg.trace_history_len)
        if valid_steps == 0:
            return {}
        
        return {
            "fast_trace_norm": self.fast_trace.norm().item(),
            "slow_trace_norm": self.slow_trace.norm().item(),
            "mean_trace_norm": self.trace_norm_history[:valid_steps].mean().item(),
            "mean_update_mag": self.update_magnitude_history[:valid_steps].mean().item(),
            "weight_static_norm": self.weight.norm().item(),
        }

# ============================================================================
# ENHANCED DEEP PLASTIC MEMBER
# ============================================================================

class EnhancedDeepPlasticMember(nn.Module):
    def __init__(self, cfg: GaiaConfigEnhanced):
        super().__init__()
        inp = cfg.state_dim + cfg.action_dim
        h = cfg.hidden_dim
        
        self.l1 = DiagnosticPlasticLinear(inp, h, cfg)
        self.l2 = DiagnosticPlasticLinear(h, h, cfg)
        self.l3 = nn.Linear(h, cfg.state_dim)
        
        # Layer normalization for stability
        self.ln1 = nn.LayerNorm(h)
        self.ln2 = nn.LayerNorm(h)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        h = self.ln1(F.relu(self.l1(x)))
        h = self.ln2(F.relu(self.l2(h)))
        return self.l3(h)
    
    def get_all_diagnostics(self) -> Dict[str, float]:
        """Aggregate diagnostics from all plastic layers"""
        diag = {}
        for name, layer in [("l1", self.l1), ("l2", self.l2)]:
            layer_diag = layer.get_diagnostics()
            diag.update({f"{name}_{k}": v for k, v in layer_diag.items()})
        return diag

class EnhancedEnsembleWorldModel(nn.Module):
    def __init__(self, cfg: GaiaConfigEnhanced):
        super().__init__()
        self.models = nn.ModuleList(
            [EnhancedDeepPlasticMember(cfg) for _ in range(cfg.n_ensemble)]
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        preds = torch.stack([m(state, action) for m in self.models])
        return preds.mean(dim=0), preds.std(dim=0)
    
    def get_ensemble_diagnostics(self) -> Dict[str, float]:
        """Get diagnostics from first ensemble member as representative"""
        return self.models[0].get_all_diagnostics()

# ============================================================================
# STOCHASTIC ACTION MODEL (unchanged, already well-designed)
# ============================================================================

class StochasticActionModel(nn.Module):
    def __init__(self, cfg: GaiaConfigEnhanced):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, cfg.action_dim * 2)
        )
    
    def forward(self, state: torch.Tensor):
        params = self.net(state)
        mean, log_std = params.chunk(2, dim=-1)
        return mean, log_std.exp().clamp(0.01, 1.0)
    
    def log_prob(self, state: torch.Tensor, action: torch.Tensor):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).sum(dim=-1)

# ============================================================================
# ENHANCED ORTHOS AGENT WITH COMPREHENSIVE DIAGNOSTICS
# ============================================================================

class GaiaAgentEnhanced:
    def __init__(self, cfg: GaiaConfigEnhanced):
        self.cfg = cfg
        self.wm = EnhancedEnsembleWorldModel(cfg).to(device)
        self.am = StochasticActionModel(cfg).to(device)
        
        self.wm_opt = optim.AdamW(self.wm.parameters(), lr=cfg.wm_lr)
        self.act_opt = optim.AdamW(self.am.parameters(), lr=cfg.policy_lr)
        
        self.preferred_state = torch.zeros(1, cfg.state_dim).to(device)
        
        # Metrics tracking
        self.metrics = {
            "wm_loss": [],
            "epistemic_uncertainty": [],
            "policy_improvement": [],
            "trace_norms": [],
        }
    
    def approximate_efe(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Expected Free Energy: pragmatic + epistemic value"""
        pragmatic = F.mse_loss(mean, self.preferred_state, reduction="none").mean(-1)
        epistemic = std.mean(-1)
        return pragmatic - self.cfg.exploration_weight * epistemic
    
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Active Inference action selection"""
        with torch.no_grad():
            s_rep = state.repeat(self.cfg.planning_samples, 1)
            mean, std = self.am(s_rep)
            actions = torch.tanh(mean + std * torch.randn_like(mean))
            
            mean_next, std_next = self.wm(s_rep, actions)
            efe = self.approximate_efe(mean_next, std_next)
            
            probs = F.softmax(-efe / self.cfg.temperature, dim=0)
            idx = torch.multinomial(probs, 1).item()
            
            return actions[idx].unsqueeze(0)
    
    def learn(self, state, action, next_state):
        """Learning step with comprehensive metrics"""
        # World model update
        preds = torch.stack([m(state, action) for m in self.wm.models])
        target = next_state.unsqueeze(0).expand_as(preds)
        wm_loss = F.mse_loss(preds, target)
        
        self.wm_opt.zero_grad()
        wm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)  # Gradient clipping
        self.wm_opt.step()
        
        # Policy update with stability
        with torch.no_grad():
            before = F.mse_loss(state, self.preferred_state)
            after = F.mse_loss(next_state, self.preferred_state)
            improvement = (before - after).item()
        
        if improvement > 0:
            logp = self.am.log_prob(state, action)
            weight = torch.sigmoid(torch.tensor(improvement * self.cfg.weight_scale))
            loss = -(logp * weight)
            
            self.act_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.am.parameters(), 1.0)
            self.act_opt.step()
        
        # Track metrics
        self.metrics["wm_loss"].append(wm_loss.item())
        self.metrics["epistemic_uncertainty"].append(preds.std().item())
        self.metrics["policy_improvement"].append(improvement)
        
        # Get plasticity diagnostics
        diag = self.wm.get_ensemble_diagnostics()
        if "l1_fast_trace_norm" in diag:
            self.metrics["trace_norms"].append(diag["l1_fast_trace_norm"])
        
        return wm_loss.item(), preds.std().item()
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics"""
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[f"{key}_mean"] = np.mean(values[-100:])  # Last 100 steps
                summary[f"{key}_std"] = np.std(values[-100:])
        return summary

# ============================================================================
# COMPREHENSIVE VALIDATION SUITE
# ============================================================================

def run_comprehensive_validation():
    print("\n" + "="*80)
    print("🔬 ORTHOS PROTOCOL v3.1 — COMPREHENSIVE VALIDATION")
    print("="*80 + "\n")
    
    agent = GaiaAgentEnhanced(cfg)
    state = torch.tensor([[0.0]], device=device)
    
    print("📊 Running 200-step simulation with diagnostics...\n")
    
    for step in range(200):
        action = agent.select_action(state)
        next_state = state * 0.9 + action + 0.01 * torch.randn_like(state)
        wm_loss, uncertainty = agent.learn(state, action, next_state)
        state = next_state
        
        if (step + 1) % 50 == 0:
            summary = agent.get_summary()
            print(f"Step {step+1}:")
            print(f"  WM Loss: {summary.get('wm_loss_mean', 0):.4f} ± {summary.get('wm_loss_std', 0):.4f}")
            print(f"  Epistemic: {summary.get('epistemic_uncertainty_mean', 0):.4f}")
            print(f"  Trace Norm: {summary.get('trace_norms_mean', 0):.4f}")
            print(f"  State: {state.item():.4f}\n")
    
    # Final diagnostics
    print("="*80)
    print("✅ FINAL DIAGNOSTICS")
    print("="*80)
    
    diag = agent.wm.get_ensemble_diagnostics()
    for key, value in diag.items():
        print(f"  {key}: {value:.4f}")
    
    # Stability check
    final_trace_norm = diag.get("l1_fast_trace_norm", 0)
    stable = 0.0 < final_trace_norm < cfg.homeostatic_target * 1.1
    
    print(f"\n{'✅ STABLE' if stable else '❌ UNSTABLE'}")
    print(f"Final trace norm: {final_trace_norm:.4f} (target < {cfg.homeostatic_target})")
    
    return agent

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    agent = run_comprehensive_validation()
    
    print("\n" + "="*80)
    print("🎯 ASSESSMENT")
    print("="*80)
    print("""
    STRENGTHS:
    ✅ Biologically-inspired dual-timescale plasticity
    ✅ Quantization-ready architecture (deployment efficiency)
    ✅ Uncertainty-driven exploration (Active Inference)
    ✅ Homeostatic stability mechanisms
    ✅ Clean separation of fast/slow memory systems
    
    RESEARCH EXTENSIONS TO CONSIDER:
    🔬 Meta-plasticity: learning-to-learn the trace decay rates
    🔬 Attention mechanisms over trace history
    🔬 Multi-scale temporal hierarchies (add ultra-slow traces)
    🔬 Synaptic tagging: selective consolidation
    🔬 Neuromodulation: context-dependent plasticity rates
    
    """)
