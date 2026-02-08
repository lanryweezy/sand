"""
Advanced Conflict Resolution and Partial Acceptance for Silicon Intelligence System

This module implements sophisticated algorithms for conflict detection, resolution,
and partial acceptance of agent proposals to enhance system flexibility and
optimization capabilities.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass
from enum import Enum
import networkx as nx
from agents.base_agent import AgentNegotiator, AgentProposal, NegotiationResult
from core.canonical_silicon_graph import CanonicalSiliconGraph
from utils.logger import get_logger


class ConflictType(Enum):
    """Types of conflicts between proposals"""
    RESOURCE_CONFLICT = "resource_conflict"
    SPATIAL_CONFLICT = "spatial_conflict"
    TIMING_CONFLICT = "timing_conflict"
    POWER_CONFLICT = "power_conflict"
    DRC_CONFLICT = "drc_conflict"
    LOGICAL_CONFLICT = "logical_conflict"


class ResolutionStrategy(Enum):
    """Strategies for resolving conflicts"""
    PARTIAL_ACCEPTANCE = "partial_acceptance"
    MODIFIED_ACCEPTANCE = "modified_acceptance"
    PRIORITY_BASED = "priority_based"
    COOPERATIVE_ADJUSTMENT = "cooperative_adjustment"
    COMPROMISE_BIDDING = "compromise_bidding"


@dataclass
class Conflict:
    """Represents a conflict between proposals"""
    proposal_a: AgentProposal
    proposal_b: AgentProposal
    conflict_type: ConflictType
    severity: float  # 0.0 to 1.0, higher is more severe
    affected_resources: List[str]
    resolution_strategy: Optional[ResolutionStrategy] = None


@dataclass
class ModifiedProposal:
    """Represents a modified version of an original proposal"""
    original_proposal: AgentProposal
    modified_parameters: Dict[str, Any]
    modification_reason: str
    confidence_after_modification: float
    expected_impact: Dict[str, float]


class AdvancedConflictResolver:
    """
    Advanced conflict resolution system with multiple resolution strategies
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.conflict_detection_rules = self._initialize_conflict_rules()
        self.resolution_strategies = self._initialize_resolution_strategies()
    
    def _initialize_conflict_rules(self) -> Dict[str, Any]:
        """Initialize conflict detection rules"""
        return {
            'resource_conflict': {
                'overlap_threshold': 0.1,
                'resource_types': ['routing_tracks', 'cell_sites', 'power_rails', 'clock_resources']
            },
            'spatial_conflict': {
                'distance_threshold': 5.0,  # microns
                'region_overlap': 0.2
            },
            'timing_conflict': {
                'slack_reduction_threshold': 0.1,
                'critical_path_intersection': 0.3
            },
            'power_conflict': {
                'current_density_threshold': 0.8,
                'voltage_drop_threshold': 0.05
            },
            'drc_conflict': {
                'rule_violation_probability': 0.7
            },
            'logical_conflict': {
                'dependency_cycle': True,
                'mutually_exclusive': True
            }
        }
    
    def _initialize_resolution_strategies(self) -> Dict[ResolutionStrategy, callable]:
        """Initialize resolution strategy functions"""
        return {
            ResolutionStrategy.PARTIAL_ACCEPTANCE: self._resolve_partial_acceptance,
            ResolutionStrategy.MODIFIED_ACCEPTANCE: self._resolve_modified_acceptance,
            ResolutionStrategy.PRIORITY_BASED: self._resolve_priority_based,
            ResolutionStrategy.COOPERATIVE_ADJUSTMENT: self._resolve_cooperative_adjustment,
            ResolutionStrategy.COMPROMISE_BIDDING: self._resolve_compromise_bidding
        }
    
    def detect_conflicts(self, proposals: List[AgentProposal], 
                       graph: CanonicalSiliconGraph) -> List[Conflict]:
        """
        Detect conflicts between proposals using sophisticated analysis
        """
        conflicts = []
        
        for i, prop_a in enumerate(proposals):
            for j, prop_b in enumerate(proposals[i+1:], i+1):
                conflict = self._analyze_conflict(prop_a, prop_b, graph)
                if conflict:
                    conflicts.append(conflict)
        
        return conflicts
    
    def _analyze_conflict(self, prop_a: AgentProposal, prop_b: AgentProposal, 
                         graph: CanonicalSiliconGraph) -> Optional[Conflict]:
        """Analyze potential conflict between two proposals"""
        # Check for target overlap
        target_overlap = set(prop_a.targets) & set(prop_b.targets)
        
        if target_overlap:
            # Analyze the type and severity of conflict
            conflict_type, severity = self._classify_conflict(prop_a, prop_b, target_overlap, graph)
            
            if severity > 0.1:  # Only consider significant conflicts
                return Conflict(
                    proposal_a=prop_a,
                    proposal_b=prop_b,
                    conflict_type=conflict_type,
                    severity=severity,
                    affected_resources=list(target_overlap)
                )
        
        # Check for resource conflicts (even without target overlap)
        resource_conflict = self._check_resource_conflict(prop_a, prop_b, graph)
        if resource_conflict:
            return resource_conflict
        
        # Check for logical conflicts
        logical_conflict = self._check_logical_conflict(prop_a, prop_b)
        if logical_conflict:
            return logical_conflict
        
        return None
    
    def _classify_conflict(self, prop_a: AgentProposal, prop_b: AgentProposal,
                          overlap_targets: Set[str], graph: CanonicalSiliconGraph) -> Tuple[ConflictType, float]:
        """Classify the type and severity of conflict"""
        # Analyze based on action types
        action_pair = (prop_a.action_type, prop_b.action_type)
        
        if action_pair in [('place_cells', 'place_cells'), ('route_nets', 'route_nets')]:
            # Spatial/resource conflicts
            severity = self._calculate_spatial_conflict_severity(prop_a, prop_b, overlap_targets, graph)
            return ConflictType.SPATIAL_CONFLICT, severity
        
        elif action_pair in [('synthesize_clock_tree', 'place_cells'), ('place_cells', 'synthesize_clock_tree')]:
            # Timing conflicts
            severity = self._calculate_timing_conflict_severity(prop_a, prop_b, overlap_targets, graph)
            return ConflictType.TIMING_CONFLICT, severity
        
        elif action_pair in [('optimize_power', 'place_cells'), ('place_cells', 'optimize_power')]:
            # Power conflicts
            severity = self._calculate_power_conflict_severity(prop_a, prop_b, overlap_targets, graph)
            return ConflictType.POWER_CONFLICT, severity
        
        elif action_pair in [('route_nets', 'optimize_power'), ('optimize_power', 'route_nets')]:
            # DRC conflicts
            severity = self._calculate_drc_conflict_severity(prop_a, prop_b, overlap_targets, graph)
            return ConflictType.DRC_CONFLICT, severity
        
        else:
            # General resource conflict
            severity = self._calculate_general_conflict_severity(prop_a, prop_b, overlap_targets, graph)
            return ConflictType.RESOURCE_CONFLICT, severity
    
    def _calculate_spatial_conflict_severity(self, prop_a: AgentProposal, prop_b: AgentProposal,
                                           overlap_targets: Set[str], graph: CanonicalSiliconGraph) -> float:
        """Calculate severity of spatial conflicts"""
        # For placement conflicts, severity depends on criticality and overlap
        max_criticality = 0.0
        for target in overlap_targets:
            if target in graph.graph.nodes():
                attrs = graph.graph.nodes[target]
                criticality = attrs.get('timing_criticality', 0.0)
                max_criticality = max(max_criticality, criticality)
        
        # Consider the confidence scores of both proposals
        avg_confidence = (prop_a.confidence_score + prop_b.confidence_score) / 2.0
        
        # Severity is higher for critical cells with high-confidence conflicting proposals
        severity = max_criticality * avg_confidence * len(overlap_targets) / 10.0
        return min(severity, 1.0)
    
    def _calculate_timing_conflict_severity(self, prop_a: AgentProposal, prop_b: AgentProposal,
                                         overlap_targets: Set[str], graph: CanonicalSiliconGraph) -> float:
        """Calculate severity of timing conflicts"""
        # Analyze timing impact
        timing_risk_a = prop_a.risk_profile.get('timing_risk', 0.0)
        timing_risk_b = prop_b.risk_profile.get('timing_risk', 0.0)
        
        # Consider the cost vectors for timing
        perf_cost_a = prop_a.cost_vector.get('performance', 0.5)
        perf_cost_b = prop_b.cost_vector.get('performance', 0.5)
        
        # Severity is higher when both proposals have high timing risk
        severity = (timing_risk_a + timing_risk_b) * (perf_cost_a + perf_cost_b) / 2.0
        return min(severity, 1.0)
    
    def _calculate_power_conflict_severity(self, prop_a: AgentProposal, prop_b: AgentProposal,
                                        overlap_targets: Set[str], graph: CanonicalSiliconGraph) -> float:
        """Calculate severity of power conflicts"""
        # Analyze power impact
        power_risk_a = prop_a.risk_profile.get('power_risk', 0.0)
        power_risk_b = prop_b.risk_profile.get('power_risk', 0.0)
        
        # Consider the cost vectors for power
        power_cost_a = prop_a.cost_vector.get('power', 0.5)
        power_cost_b = prop_b.cost_vector.get('power', 0.5)
        
        # Severity is higher when both proposals have high power risk
        severity = (power_risk_a + power_risk_b) * (power_cost_a + power_cost_b) / 2.0
        return min(severity, 1.0)
    
    def _calculate_drc_conflict_severity(self, prop_a: AgentProposal, prop_b: AgentProposal,
                                      overlap_targets: Set[str], graph: CanonicalSiliconGraph) -> float:
        """Calculate severity of DRC conflicts"""
        # Analyze DRC risk
        drc_risk_a = prop_a.risk_profile.get('drc_risk', 0.0)
        drc_risk_b = prop_b.risk_profile.get('drc_risk', 0.0)
        
        # Consider the cost vectors for yield (related to DRC)
        yield_cost_a = prop_a.cost_vector.get('yield', 0.5)
        yield_cost_b = prop_b.cost_vector.get('yield', 0.5)
        
        # Severity is higher when both proposals have high DRC risk
        severity = (drc_risk_a + drc_risk_b) * (yield_cost_a + yield_cost_b) / 2.0
        return min(severity, 1.0)
    
    def _calculate_general_conflict_severity(self, prop_a: AgentProposal, prop_b: AgentProposal,
                                          overlap_targets: Set[str], graph: CanonicalSiliconGraph) -> float:
        """Calculate general conflict severity"""
        # Use a combination of risk profiles and cost vectors
        risks_a = [prop_a.risk_profile.get(k, 0.0) for k in ['timing_risk', 'congestion_risk', 'power_risk', 'drc_risk']]
        risks_b = [prop_b.risk_profile.get(k, 0.0) for k in ['timing_risk', 'congestion_risk', 'power_risk', 'drc_risk']]
        
        costs_a = [prop_a.cost_vector.get(k, 0.5) for k in ['power', 'performance', 'area', 'yield']]
        costs_b = [prop_b.cost_vector.get(k, 0.5) for k in ['power', 'performance', 'area', 'yield']]
        
        avg_risk_a = np.mean(risks_a)
        avg_risk_b = np.mean(risks_b)
        avg_cost_a = np.mean(costs_a)
        avg_cost_b = np.mean(costs_b)
        
        severity = (avg_risk_a + avg_risk_b) * (avg_cost_a + avg_cost_b) / 2.0
        return min(severity, 1.0)
    
    def _check_resource_conflict(self, prop_a: AgentProposal, prop_b: AgentProposal, 
                               graph: CanonicalSiliconGraph) -> Optional[Conflict]:
        """Check for resource conflicts even without target overlap"""
        # This would check for conflicts in shared resources like routing tracks, power grids, etc.
        # For now, return None - would be implemented with detailed resource modeling
        return None
    
    def _check_logical_conflict(self, prop_a: AgentProposal, prop_b: AgentProposal) -> Optional[Conflict]:
        """Check for logical conflicts between proposals"""
        # Check if proposals are mutually exclusive
        if (prop_a.action_type == 'place_macro' and prop_b.action_type == 'place_macro' and
            prop_a.parameters.get('region') == prop_b.parameters.get('region')):
            # Two macros being placed in the same region
            return Conflict(
                proposal_a=prop_a,
                proposal_b=prop_b,
                conflict_type=ConflictType.LOGICAL_CONFLICT,
                severity=0.9,
                affected_resources=[prop_a.parameters.get('region', 'unknown')],
                resolution_strategy=ResolutionStrategy.PRIORITY_BASED
            )
        
        return None
    
    def resolve_conflicts(self, conflicts: List[Conflict], 
                        proposals: List[AgentProposal]) -> List[ModifiedProposal]:
        """
        Resolve conflicts using appropriate strategies
        """
        resolved_proposals = []
        
        for conflict in conflicts:
            # Determine the best resolution strategy
            strategy = self._select_resolution_strategy(conflict)
            conflict.resolution_strategy = strategy
            
            # Apply the resolution strategy
            resolved = self.resolution_strategies[strategy](conflict, proposals)
            resolved_proposals.extend(resolved)
        
        return resolved_proposals
    
    def _select_resolution_strategy(self, conflict: Conflict) -> ResolutionStrategy:
        """Select the most appropriate resolution strategy for a conflict"""
        # Select strategy based on conflict type and severity
        if conflict.severity > 0.8:
            # High severity conflicts need priority-based resolution
            return ResolutionStrategy.PRIORITY_BASED
        elif conflict.conflict_type in [ConflictType.SPATIAL_CONFLICT, ConflictType.RESOURCE_CONFLICT]:
            # Spatial/resource conflicts can often be resolved with partial acceptance
            return ResolutionStrategy.PARTIAL_ACCEPTANCE
        elif conflict.conflict_type == ConflictType.TIMING_CONFLICT:
            # Timing conflicts may need cooperative adjustment
            return ResolutionStrategy.COOPERATIVE_ADJUSTMENT
        elif conflict.conflict_type == ConflictType.POWER_CONFLICT:
            # Power conflicts may need compromise bidding
            return ResolutionStrategy.COMPROMISE_BIDDING
        else:
            # Default to modified acceptance
            return ResolutionStrategy.MODIFIED_ACCEPTANCE
    
    def _resolve_partial_acceptance(self, conflict: Conflict, 
                                  proposals: List[AgentProposal]) -> List[ModifiedProposal]:
        """Resolve conflict through partial acceptance of proposals"""
        # Calculate how much of each proposal can be accepted
        prop_a = conflict.proposal_a
        prop_b = conflict.proposal_b
        
        # Determine overlap percentage
        overlap_targets = set(prop_a.targets) & set(prop_b.targets)
        overlap_percentage = len(overlap_targets) / max(len(prop_a.targets), len(prop_b.targets), 1)
        
        # Adjust confidence based on conflict severity
        adjusted_conf_a = prop_a.confidence_score * (1 - conflict.severity * 0.5)
        adjusted_conf_b = prop_b.confidence_score * (1 - conflict.severity * 0.5)
        
        # Create modified proposals with reduced scope
        modified_proposals = []
        
        if adjusted_conf_a > 0.3:  # Still worth accepting partially
            partial_targets_a = [t for t in prop_a.targets if t not in overlap_targets]
            partial_targets_a.extend(
                np.random.choice(list(overlap_targets), 
                               size=int(len(overlap_targets) * (1 - conflict.severity)), 
                               replace=False)
            ) if overlap_targets else []
            
            modified_proposals.append(ModifiedProposal(
                original_proposal=prop_a,
                modified_parameters={**prop_a.parameters, 'targets': partial_targets_a},
                modification_reason=f"Partial acceptance due to conflict with {prop_b.agent_id}",
                confidence_after_modification=adjusted_conf_a,
                expected_impact=self._calculate_modified_impact(prop_a, len(partial_targets_a) / len(prop_a.targets))
            ))
        
        if adjusted_conf_b > 0.3:  # Still worth accepting partially
            partial_targets_b = [t for t in prop_b.targets if t not in overlap_targets]
            partial_targets_b.extend(
                np.random.choice(list(overlap_targets), 
                               size=int(len(overlap_targets) * (1 - conflict.severity)), 
                               replace=False)
            ) if overlap_targets else []
            
            modified_proposals.append(ModifiedProposal(
                original_proposal=prop_b,
                modified_parameters={**prop_b.parameters, 'targets': partial_targets_b},
                modification_reason=f"Partial acceptance due to conflict with {prop_a.agent_id}",
                confidence_after_modification=adjusted_conf_b,
                expected_impact=self._calculate_modified_impact(prop_b, len(partial_targets_b) / len(prop_b.targets))
            ))
        
        return modified_proposals
    
    def _resolve_modified_acceptance(self, conflict: Conflict, 
                                   proposals: List[AgentProposal]) -> List[ModifiedProposal]:
        """Resolve conflict by modifying proposal parameters"""
        prop_a = conflict.proposal_a
        prop_b = conflict.proposal_b
        
        modified_proposals = []
        
        # Modify parameters to reduce conflict
        modified_params_a = self._modify_parameters_for_conflict(prop_a, conflict)
        modified_params_b = self._modify_parameters_for_conflict(prop_b, conflict)
        
        modified_proposals.append(ModifiedProposal(
            original_proposal=prop_a,
            modified_parameters=modified_params_a,
            modification_reason=f"Parameters modified to resolve conflict with {prop_b.agent_id}",
            confidence_after_modification=prop_a.confidence_score * (1 - conflict.severity * 0.3),
            expected_impact=self._calculate_modified_impact(prop_a, 0.8)  # 80% effectiveness
        ))
        
        modified_proposals.append(ModifiedProposal(
            original_proposal=prop_b,
            modified_parameters=modified_params_b,
            modification_reason=f"Parameters modified to resolve conflict with {prop_a.agent_id}",
            confidence_after_modification=prop_b.confidence_score * (1 - conflict.severity * 0.3),
            expected_impact=self._calculate_modified_impact(prop_b, 0.8)  # 80% effectiveness
        ))
        
        return modified_proposals
    
    def _resolve_priority_based(self, conflict: Conflict, 
                               proposals: List[AgentProposal]) -> List[ModifiedProposal]:
        """Resolve conflict based on agent priorities and authority"""
        prop_a = conflict.proposal_a
        prop_b = conflict.proposal_b
        
        # Determine which proposal has higher priority
        # Priority is based on confidence, authority, and criticality
        priority_a = prop_a.confidence_score * self._get_agent_authority(prop_a.agent_id)
        priority_b = prop_b.confidence_score * self._get_agent_authority(prop_b.agent_id)
        
        if priority_a > priority_b:
            # Accept prop_a, reject/modify prop_b
            return [
                ModifiedProposal(
                    original_proposal=prop_a,
                    modified_parameters=prop_a.parameters,
                    modification_reason="Higher priority proposal",
                    confidence_after_modification=prop_a.confidence_score,
                    expected_impact=prop_a.predicted_outcome
                )
            ]
        else:
            # Accept prop_b, reject/modify prop_a
            return [
                ModifiedProposal(
                    original_proposal=prop_b,
                    modified_parameters=prop_b.parameters,
                    modification_reason="Higher priority proposal",
                    confidence_after_modification=prop_b.confidence_score,
                    expected_impact=prop_b.predicted_outcome
                )
            ]
    
    def _resolve_cooperative_adjustment(self, conflict: Conflict, 
                                      proposals: List[AgentProposal]) -> List[ModifiedProposal]:
        """Resolve conflict through cooperative adjustment of both proposals"""
        prop_a = conflict.proposal_a
        prop_b = conflict.proposal_b
        
        # Both proposals are adjusted cooperatively
        adjusted_params_a = self._cooperative_adjustment(prop_a, prop_b, conflict)
        adjusted_params_b = self._cooperative_adjustment(prop_b, prop_a, conflict)
        
        return [
            ModifiedProposal(
                original_proposal=prop_a,
                modified_parameters=adjusted_params_a,
                modification_reason="Cooperative adjustment with competing proposal",
                confidence_after_modification=prop_a.confidence_score * (1 - conflict.severity * 0.2),
                expected_impact=self._calculate_modified_impact(prop_a, 0.9)  # 90% effectiveness
            ),
            ModifiedProposal(
                original_proposal=prop_b,
                modified_parameters=adjusted_params_b,
                modification_reason="Cooperative adjustment with competing proposal",
                confidence_after_modification=prop_b.confidence_score * (1 - conflict.severity * 0.2),
                expected_impact=self._calculate_modified_impact(prop_b, 0.9)  # 90% effectiveness
            )
        ]
    
    def _resolve_compromise_bidding(self, conflict: Conflict, 
                                  proposals: List[AgentProposal]) -> List[ModifiedProposal]:
        """Resolve conflict through compromise bidding mechanism"""
        prop_a = conflict.proposal_a
        prop_b = conflict.proposal_b
        
        # Calculate "bids" for resources based on importance
        bid_a = self._calculate_resource_bid(prop_a, conflict.affected_resources)
        bid_b = self._calculate_resource_bid(prop_b, conflict.affected_resources)
        
        # Allocate resources based on bids
        total_bid = bid_a + bid_b
        if total_bid > 0:
            allocation_a = bid_a / total_bid
            allocation_b = bid_b / total_bid
        else:
            allocation_a = allocation_b = 0.5  # Equal split
        
        # Adjust proposals based on allocation
        modified_params_a = self._adjust_proposal_by_allocation(prop_a, allocation_a)
        modified_params_b = self._adjust_proposal_by_allocation(prop_b, allocation_b)
        
        return [
            ModifiedProposal(
                original_proposal=prop_a,
                modified_parameters=modified_params_a,
                modification_reason=f"Compromise allocation: {allocation_a:.2f}",
                confidence_after_modification=prop_a.confidence_score * allocation_a,
                expected_impact=self._calculate_modified_impact(prop_a, allocation_a)
            ),
            ModifiedProposal(
                original_proposal=prop_b,
                modified_parameters=modified_params_b,
                modification_reason=f"Compromise allocation: {allocation_b:.2f}",
                confidence_after_modification=prop_b.confidence_score * allocation_b,
                expected_impact=self._calculate_modified_impact(prop_b, allocation_b)
            )
        ]
    
    def _modify_parameters_for_conflict(self, proposal: AgentProposal, 
                                      conflict: Conflict) -> Dict[str, Any]:
        """Modify proposal parameters to reduce conflict"""
        modified_params = proposal.parameters.copy()
        
        # Adjust based on conflict type
        if conflict.conflict_type == ConflictType.SPATIAL_CONFLICT:
            # Add spacing or adjust placement
            if 'spacing_margin' in modified_params:
                modified_params['spacing_margin'] *= (1 + conflict.severity * 0.2)
            else:
                modified_params['spacing_margin'] = 0.1 * (1 + conflict.severity * 0.2)
        
        elif conflict.conflict_type == ConflictType.TIMING_CONFLICT:
            # Adjust timing constraints
            if 'timing_constraints' in modified_params:
                constraints = modified_params['timing_constraints'].copy()
                for key, value in constraints.items():
                    if isinstance(value, (int, float)):
                        constraints[key] = value * (1 + conflict.severity * 0.1)
                modified_params['timing_constraints'] = constraints
        
        elif conflict.conflict_type == ConflictType.POWER_CONFLICT:
            # Adjust power optimization parameters
            if 'power_grid_config' in modified_params:
                config = modified_params['power_grid_config'].copy()
                if 'mesh_width' in config:
                    config['mesh_width'] *= (1 - conflict.severity * 0.1)  # Tighter mesh
                modified_params['power_grid_config'] = config
        
        return modified_params
    
    def _cooperative_adjustment(self, primary_prop: AgentProposal, 
                               secondary_prop: AgentProposal, 
                               conflict: Conflict) -> Dict[str, Any]:
        """Adjust proposal cooperatively with another proposal"""
        modified_params = primary_prop.parameters.copy()
        
        # Make adjustments that consider the other proposal
        if conflict.conflict_type == ConflictType.SPATIAL_CONFLICT:
            # Adjust placement to accommodate both
            if 'region' in modified_params:
                base_region = modified_params['region']
                modified_params['region'] = f"{base_region}_adjusted_for_{secondary_prop.agent_id[:8]}"
        
        elif conflict.conflict_type == ConflictType.TIMING_CONFLICT:
            # Coordinate timing optimization
            if 'optimization_goals' in modified_params:
                goals = modified_params['optimization_goals'].copy()
                # Reduce aggressiveness to allow cooperation
                for goal, value in goals.items():
                    if isinstance(value, bool) and value:
                        goals[goal] = conflict.severity < 0.7  # Be less aggressive if high conflict
                modified_params['optimization_goals'] = goals
        
        return modified_params
    
    def _calculate_resource_bid(self, proposal: AgentProposal, resources: List[str]) -> float:
        """Calculate the bid value for resources"""
        # Bid based on criticality, confidence, and resource importance
        base_bid = proposal.confidence_score * proposal.risk_profile.get('timing_criticality', 0.5)
        
        # Adjust based on cost vector
        cost_factors = [proposal.cost_vector.get(k, 0.5) for k in ['power', 'performance', 'area', 'yield']]
        avg_cost_factor = np.mean(cost_factors)
        
        return base_bid * avg_cost_factor * len(resources)
    
    def _adjust_proposal_by_allocation(self, proposal: AgentProposal, 
                                     allocation: float) -> Dict[str, Any]:
        """Adjust proposal based on resource allocation"""
        modified_params = proposal.parameters.copy()
        
        # Scale parameters based on allocation
        if 'optimization_goals' in modified_params:
            goals = modified_params['optimization_goals'].copy()
            # Scale goal intensity based on allocation
            for goal, value in goals.items():
                if isinstance(value, bool) and value:
                    # Convert to intensity if possible
                    if allocation < 0.5:
                        goals[goal] = False  # Skip this goal if low allocation
            modified_params['optimization_goals'] = goals
        
        return modified_params
    
    def _get_agent_authority(self, agent_id: str) -> float:
        """Get agent authority (would normally come from agent system)"""
        # For now, return a default value
        return 1.0
    
    def _calculate_modified_impact(self, original_proposal: AgentProposal, 
                                 effectiveness: float) -> Dict[str, float]:
        """Calculate the impact of a modified proposal"""
        # Scale the original predicted outcome by effectiveness
        modified_impact = {}
        for key, value in original_proposal.predicted_outcome.items():
            if isinstance(value, (int, float)):
                modified_impact[key] = value * effectiveness
            else:
                modified_impact[key] = value
        
        return modified_impact


class EnhancedAgentNegotiator(AgentNegotiator):
    """
    Enhanced negotiator with advanced conflict resolution
    """
    
    def __init__(self):
        super().__init__()
        self.conflict_resolver = AdvancedConflictResolver()
    
    def _resolve_proposals(self, proposals: List[AgentProposal], 
                          graph: CanonicalSiliconGraph) -> NegotiationResult:
        """Enhanced proposal resolution with advanced conflict handling"""
        accepted = []
        rejected = []
        partially_accepted = []
        conflict_log = []
        
        # Sort proposals by weighted priority (confidence * authority)
        sorted_proposals = sorted(proposals, 
                                 key=lambda p: p.confidence_score * self._get_agent_authority(p.agent_id),
                                 reverse=True)
        
        # Detect conflicts
        conflicts = self.conflict_resolver.detect_conflicts(sorted_proposals, graph)
        
        if conflicts:
            # Resolve conflicts
            resolved_proposals = self.conflict_resolver.resolve_conflicts(conflicts, sorted_proposals)
            
            # Apply resolved proposals
            for mod_prop in resolved_proposals:
                if mod_prop.confidence_after_modification > 0.3:  # Threshold for acceptance
                    # Create a new proposal with modified parameters
                    new_proposal = AgentProposal(
                        agent_id=mod_prop.original_proposal.agent_id,
                        agent_type=mod_prop.original_proposal.agent_type,
                        proposal_id=f"mod_{mod_prop.original_proposal.proposal_id}",
                        timestamp=mod_prop.original_proposal.timestamp,
                        action_type=mod_prop.original_proposal.action_type,
                        targets=mod_prop.original_proposal.targets,  # May need to update targets too
                        parameters=mod_prop.modified_parameters,
                        confidence_score=mod_prop.confidence_after_modification,
                        risk_profile=mod_prop.original_proposal.risk_profile,
                        cost_vector=mod_prop.original_proposal.cost_vector,
                        predicted_outcome=mod_prop.expected_impact,
                        dependencies=mod_prop.original_proposal.dependencies,
                        conflicts_with=mod_prop.original_proposal.conflicts_with
                    )
                    partially_accepted.append(new_proposal)
                else:
                    rejected.append(mod_prop.original_proposal)
            
            # Log conflicts
            for conflict in conflicts:
                conflict_log.append({
                    'proposal_a': conflict.proposal_a.proposal_id,
                    'proposal_b': conflict.proposal_b.proposal_id,
                    'conflict_type': conflict.conflict_type.value,
                    'severity': conflict.severity,
                    'resolution_strategy': conflict.resolution_strategy.value if conflict.resolution_strategy else 'none'
                })
        else:
            # No conflicts, proceed with original logic
            for proposal in sorted_proposals:
                # Check if proposal improves overall metrics
                if self._proposal_improves_metrics(proposal, graph):
                    accepted.append(proposal)
                    self._apply_proposal(proposal, graph)
                    # Update agent authority positively
                    self._update_agent_authority(proposal.agent_id, success=True)
                else:
                    rejected.append(proposal)
                    # Update agent authority negatively
                    self._update_agent_authority(proposal.agent_id, success=False)
        
        # Calculate global metrics after applying accepted proposals
        global_metrics = self._calculate_global_metrics(graph)
        
        return NegotiationResult(
            accepted_proposals=accepted,
            rejected_proposals=rejected,
            partially_accepted_proposals=partially_accepted,
            updated_graph=graph,
            global_metrics=global_metrics,
            conflict_resolution_log=conflict_log
        )


# Example usage
def example_conflict_resolution():
    """Example of using the advanced conflict resolution"""
    logger = get_logger(__name__)
    
    # Initialize the enhanced negotiator
    negotiator = EnhancedAgentNegotiator()
    logger.info("Enhanced negotiator initialized with advanced conflict resolution")
    
    # The negotiator can now handle conflicts more intelligently
    logger.info("Advanced conflict resolution system ready for use")
    logger.info("The system can now resolve conflicts through partial acceptance, cooperative adjustment, and other strategies")


if __name__ == "__main__":
    example_conflict_resolution()