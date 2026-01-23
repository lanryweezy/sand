"""
Agent Negotiation Protocol - Framework for agent communication and coordination

This module implements the negotiation protocol that allows specialist agents
to communicate, propose solutions, and coordinate their activities through
the canonical silicon graph.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid
from datetime import datetime
import numpy as np
from core.canonical_silicon_graph import CanonicalSiliconGraph
from utils.logger import get_logger


class ProposalStatus(Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PARTIAL_ACCEPTANCE = "partial_acceptance"


class AgentType(Enum):
    FLOORPLAN = "floorplan"
    PLACEMENT = "placement"
    CLOCK = "clock"
    ROUTING = "routing"
    POWER = "power"
    YIELD = "yield"


@dataclass
class AgentProposal:
    """Represents a proposal from an agent"""
    agent_id: str
    agent_type: AgentType
    proposal_id: str
    timestamp: datetime
    action_type: str  # e.g., "place_macro", "route_net", "modify_constraint"
    targets: List[str]  # List of node IDs affected
    parameters: Dict[str, Any]  # Specific parameters for the action
    confidence_score: float  # 0.0 to 1.0
    risk_profile: Dict[str, float]  # Risk assessment (timing, power, area, etc.)
    cost_vector: Dict[str, float]  # PPA impact (Power, Performance, Area, Yield, Schedule)
    predicted_outcome: Dict[str, float]  # Expected results
    dependencies: List[str]  # Other proposals this depends on
    conflicts_with: List[str]  # Proposals this conflicts with


@dataclass
class NegotiationResult:
    """Result of a negotiation round"""
    accepted_proposals: List[AgentProposal]
    rejected_proposals: List[AgentProposal]
    partially_accepted_proposals: List[AgentProposal]
    updated_graph: CanonicalSiliconGraph
    global_metrics: Dict[str, float]  # Overall PPA metrics
    conflict_resolution_log: List[Dict[str, Any]]


class BaseAgent(ABC):
    """Base class for all specialist agents"""
    
    def __init__(self, agent_type: AgentType, agent_id: str = None):
        self.agent_type = agent_type
        self.agent_id = agent_id or str(uuid.uuid4())
        self.logger = get_logger(f"{__name__}.{self.agent_type.value}_agent")
        self.authority_level = 1.0  # Starts at full authority
        self.performance_history = []  # Track success/failure rate
    
    @abstractmethod
    def propose_action(self, graph: CanonicalSiliconGraph) -> Optional[AgentProposal]:
        """Generate a proposal based on the current graph state"""
        pass
    
    @abstractmethod
    def evaluate_proposal_impact(self, proposal: AgentProposal, graph: CanonicalSiliconGraph) -> Dict[str, float]:
        """Evaluate the potential impact of a proposal"""
        pass
    
    def update_authority(self, success: bool, penalty_factor: float = 0.1):
        """Update agent's authority based on success/failure"""
        if success:
            self.authority_level = min(self.authority_level + 0.05, 1.0)
        else:
            self.authority_level = max(self.authority_level - penalty_factor, 0.1)
        
        self.performance_history.append({
            'timestamp': datetime.now(),
            'success': success,
            'authority_after': self.authority_level
        })
    
    def get_recent_performance(self, window: int = 10) -> float:
        """Get recent performance ratio"""
        recent = self.performance_history[-window:]
        if not recent:
            return 1.0
        successes = sum(1 for record in recent if record['success'])
        return successes / len(recent)


class AgentNegotiator:
    """
    Coordinator that manages agent negotiations and resolves conflicts
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.agents: List[BaseAgent] = []
        self.proposal_history: List[AgentProposal] = []
        self.negotiation_rounds = 0
        
    def register_agent(self, agent: BaseAgent):
        """Register a new agent with the negotiator"""
        self.agents.append(agent)
        self.logger.info(f"Registered agent {agent.agent_id} of type {agent.agent_type.value}")
    
    def run_negotiation_round(self, graph: CanonicalSiliconGraph) -> NegotiationResult:
        """Run one round of agent negotiations"""
        self.negotiation_rounds += 1
        self.logger.info(f"Starting negotiation round {self.negotiation_rounds}")
        
        # Collect proposals from all agents
        all_proposals = []
        for agent in self.agents:
            try:
                proposal = agent.propose_action(graph)
                if proposal:
                    # Adjust proposal confidence based on agent authority
                    proposal.confidence_score *= agent.authority_level
                    all_proposals.append(proposal)
            except Exception as e:
                self.logger.error(f"Agent {agent.agent_id} failed to generate proposal: {str(e)}")
        
        if not all_proposals:
            self.logger.warning("No proposals generated in this round")
            return NegotiationResult([], [], [], graph, {}, [])
        
        # Evaluate all proposals
        for proposal in all_proposals:
            self.proposal_history.append(proposal)
        
        # Perform conflict resolution and acceptance
        result = self._resolve_proposals(all_proposals, graph)
        
        self.logger.info(f"Round {self.negotiation_rounds}: "
                        f"Accepted {len(result.accepted_proposals)}, "
                        f"Rejected {len(result.rejected_proposals)}, "
                        f"Partially accepted {len(result.partially_accepted_proposals)}")
        
        return result
    
    def _resolve_proposals(self, proposals: List[AgentProposal], graph: CanonicalSiliconGraph) -> NegotiationResult:
        """Resolve conflicts between proposals and determine acceptance"""
        accepted = []
        rejected = []
        partially_accepted = []
        conflict_log = []
        
        # Sort proposals by weighted priority (confidence * authority)
        sorted_proposals = sorted(proposals, 
                                 key=lambda p: p.confidence_score * self._get_agent_authority(p.agent_id),
                                 reverse=True)
        
        # Keep track of applied changes to the graph
        working_graph = graph.copy() if hasattr(graph, 'copy') else self._deep_copy_graph(graph)
        
        for proposal in sorted_proposals:
            # Check for conflicts with already accepted proposals
            conflicts = self._check_conflicts(proposal, accepted, working_graph)
            
            if conflicts:
                # Log the conflict
                conflict_log.append({
                    'proposal_id': proposal.proposal_id,
                    'conflicts_with': [p.proposal_id for p in conflicts],
                    'conflict_details': [str(c) for c in conflicts]
                })
                
                # Try partial acceptance or reject
                partial_result = self._attempt_partial_acceptance(proposal, conflicts, working_graph)
                
                if partial_result:
                    partially_accepted.append(partial_result)
                    self._apply_proposal(partial_result, working_graph)
                else:
                    rejected.append(proposal)
                    # Update agent authority negatively
                    self._update_agent_authority(proposal.agent_id, success=False)
            else:
                # Check if proposal improves overall metrics
                if self._proposal_improves_metrics(proposal, working_graph):
                    accepted.append(proposal)
                    self._apply_proposal(proposal, working_graph)
                    # Update agent authority positively
                    self._update_agent_authority(proposal.agent_id, success=True)
                else:
                    rejected.append(proposal)
                    # Update agent authority negatively
                    self._update_agent_authority(proposal.agent_id, success=False)
        
        # Calculate global metrics after applying accepted proposals
        global_metrics = self._calculate_global_metrics(working_graph)
        
        return NegotiationResult(
            accepted_proposals=accepted,
            rejected_proposals=rejected,
            partially_accepted_proposals=partially_accepted,
            updated_graph=working_graph,
            global_metrics=global_metrics,
            conflict_resolution_log=conflict_log
        )
    
    def _check_conflicts(self, proposal: AgentProposal, accepted_proposals: List[AgentProposal], 
                         graph: CanonicalSiliconGraph) -> List[AgentProposal]:
        """Check if a proposal conflicts with already accepted proposals"""
        conflicts = []
        
        for accepted_prop in accepted_proposals:
            # Check for target overlap
            if set(proposal.targets) & set(accepted_prop.targets):
                conflicts.append(accepted_prop)
            
            # Check for resource conflicts (e.g., same routing resources)
            if self._check_resource_conflict(proposal, accepted_prop, graph):
                conflicts.append(accepted_prop)
        
        return conflicts
    
    def _check_resource_conflict(self, prop1: AgentProposal, prop2: AgentProposal, 
                                 graph: CanonicalSiliconGraph) -> bool:
        """Check for resource conflicts between proposals"""
        # This is a simplified check - in reality would check for physical space,
        # routing resources, power delivery, etc.
        
        # For routing proposals, check if they use overlapping resources
        if prop1.action_type == "route_net" and prop2.action_type == "route_net":
            # Check if they're routing nets that are physically close and might interfere
            shared_resources = set(prop1.parameters.get('layers', [])) & set(prop2.parameters.get('layers', []))
            if shared_resources:
                # Check if they're routing in the same area
                # This is a simplified check
                return True
        
        return False
    
    def _attempt_partial_acceptance(self, proposal: AgentProposal, conflicts: List[AgentProposal], 
                                    graph: CanonicalSiliconGraph) -> Optional[AgentProposal]:
        """Attempt to partially accept a proposal by modifying it to avoid conflicts"""
        self.logger.info(f"Attempting partial acceptance for proposal {proposal.proposal_id}")

        # Identify all targets that are in conflict
        conflicting_targets = set()
        for conflict_prop in conflicts:
            conflicting_targets.update(conflict_prop.targets)
        
        # Create a new list of targets for the partial proposal, excluding conflicting ones
        salvageable_targets = [target for target in proposal.targets if target not in conflicting_targets]
        
        if not salvageable_targets:
            self.logger.debug(f"No salvageable targets for proposal {proposal.proposal_id}, returning None.")
            return None # No targets can be salvaged, effectively a full rejection
        
        # Create a new proposal with reduced scope and confidence
        partial_proposal = AgentProposal(
            agent_id=proposal.agent_id,
            agent_type=proposal.agent_type,
            proposal_id=f"{proposal.proposal_id}_partial",
            timestamp=proposal.timestamp,
            action_type=proposal.action_type,
            targets=salvageable_targets, # Only non-conflicting targets
            parameters=proposal.parameters, # Parameters remain the same for now, might need refinement
            confidence_score=proposal.confidence_score * 0.7, # Reduce confidence due to partial acceptance
            risk_profile=proposal.risk_profile, # Re-evaluate risk for partial proposal in a real system
            cost_vector=proposal.cost_vector, # Re-evaluate cost for partial proposal in a real system
            predicted_outcome=proposal.predicted_outcome, # Re-evaluate outcome for partial proposal
            dependencies=proposal.dependencies,
            conflicts_with=[] # Conflicts are now resolved for this partial proposal
        )
        
        self.logger.info(f"Generated partial proposal {partial_proposal.proposal_id} with {len(salvageable_targets)} targets.")
        return partial_proposal

    
    def _proposal_improves_metrics(self, proposal: AgentProposal, graph: CanonicalSiliconGraph) -> bool:
        """Check if a proposal improves overall design metrics considering trade-offs"""
        current_metrics = self._calculate_global_metrics(graph) # Use global metrics for current state
        predicted_outcome = proposal.predicted_outcome
        
        # Define primary and secondary metrics for improvement
        # Higher values for performance/yield/schedule_reduction are good. Lower for power/area/congestion.
        primary_metrics_good_high = ['expected_performance_improvement', 'expected_yield_improvement']
        primary_metrics_good_low = ['expected_power_reduction', 'expected_area_reduction', 'expected_congestion_reduction', 'expected_timing_improvement', 'expected_skew_reduction', 'expected_ir_drop_reduction', 'expected_em_improvement', 'expected_hotspot_reduction']
        
        # Define tolerance for degradation in other metrics
        degradation_tolerance_percentage = 0.05 # 5% degradation tolerance for non-primary metrics

        # Assume improvement initially
        net_improvement = False
        significant_degradation = False

        # Evaluate if any primary metric is improved
        for metric_key in primary_metrics_good_high:
            if metric_key in predicted_outcome and predicted_outcome[metric_key] > 0: # Check for positive improvement
                net_improvement = True
                break
        
        if not net_improvement: # If no 'good_high' improvement, check 'good_low'
            for metric_key in primary_metrics_good_low:
                if metric_key in predicted_outcome and predicted_outcome[metric_key] > 0: # Check for positive reduction
                    net_improvement = True
                    break

        if not net_improvement:
            return False # No improvement in any primary metric

        # Now check if any other metric significantly degrades
        # This requires mapping predicted outcomes to global metrics
        
        # Simplified mapping for demonstration (this would be more complex in a real system)
        outcome_to_metric_map_good_high = {
            'expected_performance_stability': 'avg_timing_criticality', # Lower criticality is better
            'expected_yield_improvement': 'yield',
        }
        outcome_to_metric_map_good_low = {
            'expected_power_increase': 'total_power',
            'expected_area_increase': 'total_area',
            'expected_congestion_reduction': 'avg_congestion',
            'expected_ir_drop_reduction': 'power_integrity_risk',
            'expected_hotspot_reduction': 'hotspot_risk'
            # Add more mappings as needed
        }

        for outcome_key, global_metric_key in outcome_to_metric_map_good_high.items():
            if outcome_key in predicted_outcome and global_metric_key in current_metrics:
                # If outcome is negative (degradation) and it degrades beyond tolerance
                # For "good_high" metrics, degradation means predicted_outcome < 0
                if predicted_outcome[outcome_key] < 0 and \
                   abs(predicted_outcome[outcome_key]) > current_metrics[global_metric_key] * degradation_tolerance_percentage:
                    significant_degradation = True
                    break

        if not significant_degradation:
            for outcome_key, global_metric_key in outcome_to_metric_map_good_low.items():
                if outcome_key in predicted_outcome and global_metric_key in current_metrics:
                    # For "good_low" metrics, degradation means predicted_outcome > 0
                    if predicted_outcome[outcome_key] > 0 and \
                       predicted_outcome[outcome_key] > current_metrics[global_metric_key] * degradation_tolerance_percentage:
                        significant_degradation = True
                        break

        return net_improvement and not significant_degradation

    
    def _apply_proposal(self, proposal: AgentProposal, graph: CanonicalSiliconGraph):
        """Apply a proposal to the graph"""
        if proposal.action_type == "place_macro":
            # Apply placement changes
            for target in proposal.targets:
                if target in graph.graph.nodes:
                    for param, value in proposal.parameters.items():
                        graph.graph.nodes[target][param] = value
        
        elif proposal.action_type == "route_net":
            # Apply routing changes
            for target in proposal.targets:
                if target in graph.graph.nodes:
                    for param, value in proposal.parameters.items():
                        graph.graph.nodes[target][param] = value
        
        elif proposal.action_type == "modify_constraint":
            # Apply constraint modifications
            for target in proposal.targets:
                if target in graph.graph.nodes:
                    for param, value in proposal.parameters.items():
                        graph.graph.nodes[target][param] = value
    
    def _calculate_global_metrics(self, graph: CanonicalSiliconGraph) -> Dict[str, float]:
        """Calculate global design metrics"""
        metrics = {}
        
        # Calculate area utilization
        total_area = sum(attrs.get('area', 0) for _, attrs in graph.graph.nodes(data=True))
        metrics['total_area'] = total_area
        
        # Calculate power consumption
        total_power = sum(attrs.get('power', 0) for _, attrs in graph.graph.nodes(data=True))
        metrics['total_power'] = total_power
        
        # Calculate timing criticality
        avg_criticality = np.mean([attrs.get('timing_criticality', 0) 
                                  for _, attrs in graph.graph.nodes(data=True)])
        metrics['avg_timing_criticality'] = avg_criticality
        
        # Calculate congestion estimate
        avg_congestion = np.mean([attrs.get('estimated_congestion', 0) 
                                 for _, attrs in graph.graph.nodes(data=True)])
        metrics['avg_congestion'] = avg_congestion
        
        return metrics
    
    def _calculate_current_metrics(self, graph: CanonicalSiliconGraph) -> Dict[str, float]:
        """Calculate current metrics for comparison"""
        return self._calculate_global_metrics(graph)
    
    def _get_agent_authority(self, agent_id: str) -> float:
        """Get authority level for a specific agent"""
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent.authority_level
        return 1.0  # Default authority
    
    def _update_agent_authority(self, agent_id: str, success: bool):
        """Update authority for a specific agent"""
        for agent in self.agents:
            if agent.agent_id == agent_id:
                agent.update_authority(success)
                break
    
    def _deep_copy_graph(self, graph: CanonicalSiliconGraph) -> CanonicalSiliconGraph:
        """Create a deep copy of the graph"""
        import copy
        new_graph = CanonicalSiliconGraph()
        new_graph.graph = copy.deepcopy(graph.graph)
        new_graph.metadata = copy.deepcopy(graph.metadata)
        return new_graph


class ProposalEvaluator:
    """
    Evaluates proposals using Pareto analysis and risk assessment
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def evaluate_proposals_pareto(self, proposals: List[AgentProposal]) -> List[AgentProposal]:
        """Perform Pareto analysis to identify non-dominated proposals"""
        # A proposal A dominates proposal B if A is better in at least one objective
        # and not worse in any other objective
        
        non_dominated = []
        
        for proposal in proposals:
            is_dominated = False
            for other in proposals:
                if self._dominates(other, proposal):
                    is_dominated = True
                    break
            
            if not is_dominated:
                non_dominated.append(proposal)
        
        return non_dominated
    
    def _dominates(self, proposal_a: AgentProposal, proposal_b: AgentProposal) -> bool:
        """Check if proposal_a dominates proposal_b"""
        # Define objectives (lower is better for all)
        objectives = ['power', 'area', 'timing_slack', 'congestion']
        
        a_better_in_any = False
        
        for obj in objectives:
            a_val = proposal_a.cost_vector.get(obj, 0)
            b_val = proposal_b.cost_vector.get(obj, 0)
            
            if a_val < b_val:  # A is better
                a_better_in_any = True
            elif a_val > b_val:  # A is worse
                return False  # Cannot dominate if worse in any objective
        
        return a_better_in_any
    
    def penalize_late_stage_risk(self, proposals: List[AgentProposal], current_stage: str) -> List[AgentProposal]:
        """Apply penalties to proposals that introduce late-stage risks"""
        stage_order = ['floorplan', 'placement', 'cts', 'route', 'signoff']
        current_idx = stage_order.index(current_stage) if current_stage in stage_order else 0
        
        for proposal in proposals:
            # Check if proposal introduces risks that are difficult to fix later
            if current_idx > 0:  # Past floorplan stage
                # Penalize proposals that affect already-stabilized aspects
                if proposal.action_type in ['move_macro', 'change_floorplan']:
                    proposal.confidence_score *= 0.5  # Significant penalty
        
        return proposals