"""
Atomic Node Physics Engine - Sub-5nm GAAFET (Gate-All-Around) Modeling.
Specialized for 3nm, 2nm and beyond.
Street Heart Technologies Proprietary.
"""

from typing import Dict, Any

class GAAFETPhysicsModel:
    """
    Models the topological and electrical characteristics of nanosheet GAAFETs.
    Essential for sub-5nm PPA accuracy where FinFET physics fail.
    """
    
    def __init__(self, node_nm: int = 3):
        self.node_nm = node_nm
        # Physical parameters for nanosheet stacks
        self.params = self._get_node_params(node_nm)

    def _get_node_params(self, node: int) -> Dict[str, float]:
        if node == 3:
            return {
                'sheet_count': 3.0,
                'gate_length': 12.0, # nm
                'oxide_thickness': 1.1, # nm
                'quantum_leakage_coeff': 1.45e-3,
                'nanosheet_width': 30.0,
                'metal_pitch': 24.0
            }
        elif node == 2:
            return {
                'sheet_count': 4.0,
                'gate_length': 10.0,
                'oxide_thickness': 0.9,
                'quantum_leakage_coeff': 2.8e-3, # Tunneling increases significantly
                'nanosheet_width': 25.0,
                'metal_pitch': 21.0
            }
        return {}

    def calculate_parasitic_leakage(self, gate_count: int, activity_factor: float) -> float:
        """
        Calculates quantum tunneling leakage for GAAFET nanosheet stacks.
        """
        base_leakage = self.params['quantum_leakage_coeff'] * self.params['sheet_count']
        # Quantum tunneling scales exponentially as oxide thickness drops
        thickness_penalty = 1.0 / self.params['oxide_thickness']
        
        total_leakage = gate_count * base_leakage * thickness_penalty * (1.0 - activity_factor)
        return total_leakage

    def predict_delay_at_atomic_scale(self, wire_length_um: float, stack_count: int) -> float:
        """
        Predicts logic delay considering EUV lithography variations.
        """
        # Resistance scales drastically at 2nm
        resistance_coeff = 2.5 if self.node_nm == 2 else 1.8
        base_delay = wire_length_um * resistance_coeff * (1.0 / stack_count)
        
        # EUV variation factor
        euv_margin = 1.05 if self.node_nm == 3 else 1.12
        return base_delay * euv_margin

if __name__ == "__main__":
    model_3nm = GAAFETPhysicsModel(3)
    model_2nm = GAAFETPhysicsModel(2)
    
    print(f"3nm Leakage (1k gates): {model_3nm.calculate_parasitic_leakage(1000, 0.1):.6f} mW")
    print(f"2nm Leakage (1k gates): {model_2nm.calculate_parasitic_leakage(1000, 0.1):.6f} mW")
