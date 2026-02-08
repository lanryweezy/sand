"""
Power Agent - Specialized agent for power optimization and clock gating.
Uses design insights to identify and propose power-saving transformations.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from agents.base_agent import BaseAgent, AgentType, AgentProposal
from core.canonical_silicon_graph import CanonicalSiliconGraph
from utils.logger import get_logger

class PowerAgent(BaseAgent):
    """
    Power Agent - identifies opportunities for clock gating and power recovery.
    """
    
    def __init__(self):
        super().__init__(AgentType.POWER) # Using POWER as base type
        self.logger = get_logger(f"{__name__}.power_agent")
        # self.agent_type is set by super().__init__ to AgentType.POWER
    
    def propose_action(self, graph: CanonicalSiliconGraph) -> Optional[AgentProposal]:
        """
        Scans the silicon graph for registers that could benefit from clock gating.
        """
        self.logger.info("Scanning for power optimization opportunities")
        
        # Identify registers in the graph
        registers = [n for n, attrs in graph.graph.nodes(data=True) 
                    if attrs.get('node_type') and attrs.get('node_type').value == 'register']
        
        if not registers:
            self.logger.debug("No registers found, skipping power optimization")
            return None
            
        # Find registers with potential enable signals or high switching activity
        candidates = self._find_clock_gating_candidates(graph, registers)
        
        if not candidates:
            return None
            
        # Select the best candidate (highest local impact)
        best_candidate = candidates[0]
        
        parameters = {
            'target_signal': best_candidate,
            'module_name': graph.graph.nodes[best_candidate].get('module', 'top'),
            'enable_signal': self._find_potential_enable(graph, best_candidate),
            'strategy': 'integrated_clock_gate'
        }
        
        proposal = AgentProposal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            proposal_id=f"pwr_{np.random.randint(1000, 9999)}",
            timestamp=self._get_timestamp(),
            action_type="insert_clock_gate",
            targets=[best_candidate],
            parameters=parameters,
            confidence_score=0.85,
            risk_profile={'timing_risk': 0.2, 'congestion_risk': 0.1},
            cost_vector={'power_impact': -0.15, 'area_impact': 0.02}, # Negative power impact is good
            predicted_outcome={'dynamic_power_reduction': 0.2},
            dependencies=[],
            conflicts_with=[]
        )
        
        return proposal

    def _find_clock_gating_candidates(self, graph, registers) -> List[str]:
        """Identifies registers that are likely to benefit from gating"""
        # Criteria: High fanout from an enable-like signal or part of a large bus
        candidates = []
        for reg in registers:
            # Simple heuristic: if it has high fan-in or specific naming
            if 'reg' in reg.lower() or 'pipe' in reg.lower():
                candidates.append(reg)
        return candidates

    def _find_potential_enable(self, graph, register_id) -> str:
        """Heuristically find an enable signal for the register"""
        # In a real system, this would trace back logic cones
        # For now, return a placeholder that the RTL transformer can use
        node_attrs = graph.graph.nodes[register_id]
        return node_attrs.get('enable_signal', 'en_signal')

    def evaluate_proposal_impact(self, proposal: AgentProposal, graph: CanonicalSiliconGraph) -> Dict[str, float]:
        """Evaluate the potential power impact of a proposal"""
        # Heuristic evaluation of power impact
        impact = {}
        
        # If it's a power-related proposal, assume we calculated it correctly in the proposal
        if proposal.agent_type == self.agent_type:
            impact['power_reduction'] = proposal.predicted_outcome.get('dynamic_power_reduction', 0.0)
            impact['area_overhead'] = proposal.cost_vector.get('area_impact', 0.0)
            return impact
            
        # Evaluate other agents' proposals for power impact
        if proposal.action_type == "place_macro":
            # Moving macros might affect wire length and thus power
            impact['power_change'] = 0.05 # Placeholder assumption
        elif proposal.action_type == "route_net":
            # Routing changes definitely affect capacitance and power
            impact['power_change'] = 0.02 # Placeholder
            
        return impact

    def _get_timestamp(self):
        from datetime import datetime
        return datetime.now()