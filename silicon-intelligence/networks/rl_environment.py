
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import os
import tempfile

class DesignOptimizationEnv:
    """
    A custom reinforcement learning environment for RTL optimization.
    Treats the optimization process as a Markov Decision Process (MDP).
    """
    
    def __init__(self, initial_rtl: str, optimizer: Any, design_name: str = "sandbox"):
        self.initial_rtl = initial_rtl
        self.current_rtl = initial_rtl
        self.optimizer = optimizer
        self.design_name = design_name
        self.step_count = 0
        self.max_steps = 10
        self.history = []
        
        # Action space: Indices corresponding to strategies
        self.strategies = [
            "pipeline_critical_paths",
            "cluster_congested_areas",
            "reduce_register_count",
            "optimize_fanout",
            "balance_area_power",
            "clock_gating"
        ]
        
        # Initial analysis to establish baseline
        self.baseline_analysis = self.optimizer.design_intelligence.analyze_design(self.initial_rtl, f"{design_name}_baseline")
        self.last_analysis = self.baseline_analysis

    def reset(self) -> np.ndarray:
        """Reset the environment to the initial design state"""
        self.current_rtl = self.initial_rtl
        self.step_count = 0
        self.last_analysis = self.baseline_analysis
        self.history = [self.baseline_analysis]
        return self._get_observation(self.baseline_analysis)

    def _get_observation(self, analysis: Dict[str, Any]) -> np.ndarray:
        """Extract a feature vector from design analysis"""
        stats = analysis.get('physical_ir_stats', {})
        ppa = analysis.get('openroad_results', {}).get('overall_ppa', {})
        
        # Normalize and vectorize features
        obs = [
            stats.get('num_nodes', 0) / 1000.0,
            stats.get('num_edges', 0) / 2000.0,
            stats.get('max_fanout', 0) / 100.0,
            ppa.get('area_um2', 0) / 50000.0,
            ppa.get('power_mw', 0) / 10.0,
            ppa.get('timing_ns', 0) / 5.0,
            self.step_count / float(self.max_steps)
        ]
        return np.array(obs, dtype=np.float32)

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Apply an optimization strategy and return results"""
        if action_idx < 0 or action_idx >= len(self.strategies):
            raise ValueError(f"Invalid action index: {action_idx}")
            
        strategy_name = self.strategies[action_idx]
        self.step_count += 1
        
        # We need to map generic names to the optimizer's strategy enum if it exists
        # In our case, we'll just call the optimizer's apply_strategy
        from autonomous_optimizer import OptimizationStrategy
        strategy_enum = OptimizationStrategy(strategy_name)
        
        # For simplicity in RL, we provide default params for strategies
        params = self._get_default_params(strategy_enum)
        
        # Apply transformation
        try:
            new_rtl = self.optimizer._apply_strategy(self.current_rtl, strategy_enum, params)
            
            # Analyze new state
            new_analysis = self.optimizer.design_intelligence.analyze_design(new_rtl, f"{self.design_name}_step_{self.step_count}")
            
            # Calculate reward
            reward = self._calculate_reward(self.last_analysis, new_analysis)
            
            # Update state
            self.current_rtl = new_rtl
            self.last_analysis = new_analysis
            self.history.append(new_analysis)
            
            obs = self._get_observation(new_analysis)
            done = self.step_count >= self.max_steps
            
            info = {
                'strategy': strategy_name,
                'reward_components': self._get_reward_components(self.last_analysis, new_analysis)
            }
            
            return obs, reward, done, info
            
        except Exception as e:
            # Illegal move or transformation failure
            print(f"RL Step Failed: {e}")
            return self._get_observation(self.last_analysis), -5.0, False, {'error': str(e)}

    def _get_default_params(self, strategy: Any) -> Dict:
        """Heuristic for default params during exploration"""
        # In a more advanced version, the agent could select these params too
        return {
            'module_name': 'top_module', # This should be parsed in reality
            'target_signal': 'clk',
            'targets': ['data_in'],
            'enable_signal': 'en'
        }

    def _calculate_reward(self, old: Dict, new: Dict) -> float:
        """Multi-objective reward for PPA improvement"""
        old_ppa = old['openroad_results']['overall_ppa']
        new_ppa = new['openroad_results']['overall_ppa']
        
        area_imp = (old_ppa['area_um2'] - new_ppa['area_um2']) / max(old_ppa['area_um2'], 1.0)
        power_imp = (old_ppa['power_mw'] - new_ppa['power_mw']) / max(old_ppa['power_mw'], 0.1)
        timing_imp = (old_ppa['timing_ns'] - new_ppa['timing_ns']) / max(old_ppa['timing_ns'], 0.1)
        
        # Reward = Weighted sum of improvements
        # Note: We penalize area/power/timing degradation (negative improvement)
        reward = (area_imp * 10.0) + (power_imp * 10.0) + (timing_imp * 5.0)
        
        # Small penalty for taking a step to encourage efficiency
        reward -= 0.1
        
        return reward

    def _get_reward_components(self, old: Dict, new: Dict) -> Dict:
        old_ppa = old['openroad_results']['overall_ppa']
        new_ppa = new['openroad_results']['overall_ppa']
        return {
            'area_delta': old_ppa['area_um2'] - new_ppa['area_um2'],
            'power_delta': old_ppa['power_mw'] - new_ppa['power_mw'],
            'timing_delta': old_ppa['timing_ns'] - new_ppa['timing_ns']
        }
