
import json
import os
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional

class DesignMemory:
    """
    The Long-Term Memory (LTM) of the Silicon Intelligence system.
    Persists optimizer actions, design signatures, and PPA metrics.
    """
    def __init__(self, db_path: str = "telemetry_data/design_memory.json"):
        self.db_path = db_path
        self._ensure_storage()
        self.memory = self._load()

    def _ensure_storage(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        if not os.path.exists(self.db_path):
            with open(self.db_path, 'w') as f:
                json.dump({}, f)

    def _load(self) -> Dict[str, Any]:
        with open(self.db_path, 'r') as f:
            return json.load(f)

    def _save(self):
        with open(self.db_path, 'w') as f:
            json.dump(self.memory, f, indent=2)

    def get_design_hash(self, rtl_content: str) -> str:
        """Create a unique fingerprint for a specific design state"""
        return hashlib.sha256(rtl_content.encode()).hexdigest()

    def remember_optimization(self, rtl_content: str, action: str, ppa_metrics: Dict):
        """Record an optimization step and its outcome in long-term storage"""
        fingerprint = self.get_design_hash(rtl_content)
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'ppa': ppa_metrics
        }
        
        if fingerprint not in self.memory:
            self.memory[fingerprint] = []
            
        self.memory[fingerprint].append(entry)
        self._save()
        print(f"[MEMORY] Logged state {fingerprint[:8]} with action '{action}'")

    def query_similar_designs(self, rtl_content: str) -> List[Dict]:
        """Search memory for designs matching the current design signature"""
        fingerprint = self.get_design_hash(rtl_content)
        return self.memory.get(fingerprint, [])

    def get_system_experience_stats(self) -> Dict[str, Any]:
        """Summary of accumulated design wisdom"""
        total_states = len(self.memory)
        total_actions = sum(len(steps) for steps in self.memory.values())
        return {
            'unique_design_states': total_states,
            'total_recorded_actions': total_actions,
            'vocabulary_size': len(set(step['action'] for steps in self.memory.values() for step in steps)) if total_actions > 0 else 0
        }
