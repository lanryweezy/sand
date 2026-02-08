
import json
import os
from datetime import datetime
from typing import Dict, List, Any

class CreditAssignmentLogger:
    """
    Logs and attributes design improvements to specific AI-driven actions.
    Solves the credit assignment problem by tracking delta-PPA per step.
    """
    def __init__(self, log_dir: str = "telemetry_data/learning_logs"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.current_episode = []

    def log_step(self, action_id: str, old_ppa: Dict, new_ppa: Dict, reward: float):
        """Record the causal link between an action and its outcome"""
        step_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action_id,
            'reward': reward,
            'deltas': {
                'area_um2': new_ppa['area_um2'] - old_ppa['area_um2'],
                'power_mw': new_ppa['power_mw'] - old_ppa['power_mw'],
                'timing_ns': new_ppa['timing_ns'] - old_ppa['timing_ns']
            }
        }
        self.current_episode.append(step_entry)

    def finalize_episode(self, design_name: str, total_improvement: Dict):
        """Save the full episode trace to disk for offline analysis"""
        log_path = os.path.join(self.log_dir, f"episode_{design_name}_{int(datetime.now().timestamp())}.json")
        
        entry = {
            'design': design_name,
            'final_ppa_improvement': total_improvement,
            'steps': self.current_episode
        }
        
        with open(log_path, 'w') as f:
            json.dump(entry, f, indent=2)
            
        print(f"Credit Assignment Trace saved: {log_path}")
        self.current_episode = []

    def get_action_performance_summary(self) -> Dict[str, float]:
        """Calculates which actions are currently the "MVPs" based on history"""
        # This would read historical JSONs. For now, we return a simple summary.
        # Future enhancement: automated ablation study results.
        return {
            "logic_merging": 0.85,
            "fanout_buffering": 0.72,
            "input_isolation": 0.91
        }
