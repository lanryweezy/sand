"""
Routing Agent - Specialized agent for routing optimization

This agent focuses on intelligent routing decisions considering congestion, DRC,
timing, and manufacturing constraints at advanced process nodes.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from agents.base_agent import BaseAgent, AgentType, AgentProposal
from core.canonical_silicon_graph import CanonicalSiliconGraph
from utils.logger import get_logger


class RoutingAgent(BaseAgent):
    """
    Routing Agent - intelligent routing optimization considering multiple constraints
    """
    
    def __init__(self):
        super().__init__(AgentType.ROUTING)
        self.logger = get_logger(f"{__name__}.routing_agent")
        self.routing_strategies = [
            'maze_expansion', 'a_star', 'channel_based', 'global_then_detail',
            'ant_colony_optimization', 'machine_learning_guided'
        ]
    
    def propose_action(self, graph: CanonicalSiliconGraph) -> Optional['AgentProposal']:
        """
        Generate a routing proposal based on the current graph state
        """
        self.logger.info("Generating routing optimization proposal")
        
        # Identify unrouted nets and congestion areas
        unrouted_nets = self._identify_unrouted_nets(graph)
        congestion_areas = self._identify_congestion_areas(graph)
        timing_critical_nets = self._identify_timing_critical_nets(graph)
        drc_sensitive_areas = self._identify_drc_sensitive_areas(graph)
        
        if not unrouted_nets:
            self.logger.debug("No unrouted nets found, skipping routing proposal")
            return None
        
        # Select optimal routing strategy based on design characteristics
        strategy = self._select_routing_strategy(
            len(unrouted_nets), len(congestion_areas), len(timing_critical_nets)
        )
        
        # Generate routing parameters
        parameters = self._generate_routing_parameters(
            strategy, unrouted_nets, congestion_areas, timing_critical_nets, drc_sensitive_areas
        )
        
        # Calculate risk profile
        risk_profile = self._assess_routing_risk(
            parameters, congestion_areas, timing_critical_nets, drc_sensitive_areas
        )
        
        # Calculate cost vector (PPA impact)
        cost_vector = self._calculate_routing_cost_vector(parameters)
        
        # Estimate outcome
        predicted_outcome = self._predict_routing_outcome(parameters)
        
        proposal = AgentProposal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            proposal_id=f"rt_{np.random.randint(1000, 9999)}",
            timestamp=self._get_timestamp(),
            action_type="route_nets",
            targets=unrouted_nets[:20],  # Limit targets for proposal size
            parameters=parameters,
            confidence_score=self._calculate_confidence(graph),
            risk_profile=risk_profile,
            cost_vector=cost_vector,
            predicted_outcome=predicted_outcome,
            dependencies=[],
            conflicts_with=[]
        )
        
        self.logger.info(f"Generated routing proposal for {len(unrouted_nets)} nets")
        return proposal
    
    def _identify_unrouted_nets(self, graph: CanonicalSiliconGraph) -> List[str]:
        """Identify nets that need routing"""
        self.logger.debug("Identifying unrouted nets.")
        unrouted = []
        
        for node, attrs in graph.graph.nodes(data=True):
            if attrs.get('node_type') == 'signal':
                # In a real implementation, this would check routing status
                # For now, we'll assume a hypothetical 'is_routed' attribute
                if not attrs.get('is_routed', False):
                    unrouted.append(node)
        
        self.logger.debug(f"Identified {len(unrouted)} unrouted nets.")
        return unrouted[:100]  # Limit for performance
    
    def _identify_congestion_areas(self, graph: CanonicalSiliconGraph) -> List[Dict[str, Any]]:
        """Identify areas of the design with high routing congestion"""
        self.logger.debug("Identifying congestion areas.")
        congestion_areas = []
        
        # Analyze the graph for high-connectivity regions and high estimated_congestion
        for node, attrs in graph.graph.nodes(data=True):
            # Calculate connectivity density around this node
            neighbors = list(graph.graph.neighbors(node))
            node_estimated_congestion = attrs.get('estimated_congestion', 0.0)

            # A node is congestion-prone if it has high connectivity OR high estimated_congestion
            if len(neighbors) > 15 or node_estimated_congestion > 0.7:  # Adjusted thresholds
                congestion_areas.append({
                    'center_node': node,
                    'connectivity': len(neighbors),
                    'region': attrs.get('region', 'unknown'),
                    'estimated_congestion': node_estimated_congestion # Use the graph's estimate
                })
                self.logger.debug(f"Node '{node}' identified as congestion-prone (connectivity: {len(neighbors)}, estimated_congestion: {node_estimated_congestion:.2f}).")
        
        # Further refine by grouping into larger congested areas if needed (e.g., using clustering)
        # For simplicity, returning the list of congestion-prone nodes/regions
        return congestion_areas
    
    def _identify_timing_critical_nets(self, graph: CanonicalSiliconGraph) -> List[str]:
        """Identify nets that are on timing-critical paths"""
        self.logger.debug("Identifying timing critical nets.")
        critical_nets = []
        
        # Look for nets connecting high-timing-critical cells
        for node, attrs in graph.graph.nodes(data=True):
            if attrs.get('timing_criticality', 0) > 0.7: # High timing criticality threshold
                # This node is timing critical, so connecting nets are too
                for successor in graph.graph.successors(node):
                    if successor in graph.graph.nodes():
                        succ_attrs = graph.graph.nodes[successor]
                        # Ensure it's a signal node
                        if succ_attrs.get('node_type') == 'signal':
                            # Check the edge type to ensure it's a real connection
                            edge_data = graph.graph.get_edge_data(node, successor)
                            # Iterate through parallel edges if present (MultiDiGraph)
                            for key in edge_data:
                                if edge_data[key].get('edge_type') == EdgeType.CONNECTION.value:
                                    critical_nets.append(successor)
                                    break # Only need one connection to count it
        
        critical_nets = list(set(critical_nets)) # Remove duplicates
        self.logger.debug(f"Identified {len(critical_nets)} timing critical nets.")
        return critical_nets    
    def _identify_drc_sensitive_areas(self, graph: CanonicalSiliconGraph) -> List[Dict[str, Any]]:
        """Identify areas sensitive to DRC violations"""
        self.logger.debug("Identifying DRC sensitive areas.")
        drc_sensitive = []
        
        # Look for areas with tight spacing requirements or high density
        for node, attrs in graph.graph.nodes(data=True):
            region = attrs.get('region', 'default')
            power = attrs.get('power', 0.0)
            timing_criticality = attrs.get('timing_criticality', 0.0)
            estimated_congestion = attrs.get('estimated_congestion', 0.0)
            cell_type = attrs.get('cell_type', '').lower()
            
            # Calculate DRC sensitivity score
            drc_sensitivity = (power * 0.2 + timing_criticality * 0.3 + estimated_congestion * 0.3)
            
            # Boost sensitivity for known problematic cell types or contexts
            if 'memory' in cell_type or 'sram' in cell_type:
                drc_sensitivity += 0.2
            if 'pll' in cell_type or 'dll' in cell_type:
                drc_sensitivity += 0.15
            if region == 'io_pad': # IO pads can have specific DRC challenges
                drc_sensitivity += 0.1
            
            # Apply a clamping to ensure score is within reasonable bounds, e.g., 0 to 1.0
            drc_sensitivity = min(max(drc_sensitivity, 0.0), 1.0)
            
            if drc_sensitivity > 0.6: # Adjusted threshold for sensitivity
                drc_sensitive.append({
                    'node': node,
                    'region': region,
                    'sensitivity_score': drc_sensitivity,
                    'factors': {
                        'power_factor': power,
                        'timing_factor': timing_criticality,
                        'congestion_factor': estimated_congestion,
                        'cell_type': cell_type
                    }
                })
                self.logger.debug(f"Node '{node}' identified as DRC sensitive (score: {drc_sensitivity:.2f}).")
        
        return drc_sensitive
    
    def _select_routing_strategy(self, num_nets: int, num_congested: int, 
                               num_timing_critical: int) -> str:
        """Select the optimal routing strategy based on design characteristics"""
        self.logger.debug(f"Selecting routing strategy based on {num_nets} nets, {num_congested} congested areas, {num_timing_critical} critical nets.")

        # Prioritize timing-critical scenarios
        if num_timing_critical > num_nets * 0.25 and num_timing_critical > 50:
            self.logger.debug("Selecting 'a_star' due to many timing critical nets.")
            return 'a_star'  # Better for timing optimization, explores shortest paths
        
        # Prioritize congestion scenarios
        elif num_congested > 100 or (num_congested > 5 and num_nets > 5000): # High absolute congestion or moderate in large designs
            self.logger.debug("Selecting 'maze_expansion' due to high congestion.")
            return 'maze_expansion'  # Good for detailed congestion avoidance, but can be slow
        
        # For very large designs, hierarchical approach is generally efficient
        elif num_nets > 50000: # Very large designs
            self.logger.debug("Selecting 'global_then_detail' for very large designs.")
            return 'global_then_detail' # Breaks problem into global and detailed stages
        
        # For designs with moderate complexity and emphasis on routability, channel-based can be effective
        elif num_nets > 10000:
            self.logger.debug("Selecting 'channel_based' for moderate complexity and routability.")
            return 'channel_based' # Good for regular structures, efficient
        
        # If there's a need for advanced exploration or a more global optimal solution
        elif num_nets > 1000:
            self.logger.debug("Selecting 'ant_colony_optimization' for advanced exploration.")
            return 'ant_colony_optimization' # Metaheuristic, can find good global solutions
        
        # For scenarios where previous routing attempts failed or require learning from past data
        elif num_nets > 500:
            self.logger.debug("Selecting 'machine_learning_guided' for designs benefiting from historical data.")
            return 'machine_learning_guided' # Leverages learned patterns for faster convergence

        else:
            self.logger.debug("Selecting 'channel_based' as a default for smaller or less complex designs.")
            return 'channel_based' # Fallback for general routability    
    def _generate_routing_parameters(self, strategy: str, unrouted_nets: List[str],
                                   congestion_areas: List[Dict], 
                                   timing_critical_nets: List[str],
                                   drc_sensitive_areas: List[Dict]) -> Dict[str, Any]:
        """Generate routing parameters based on strategy and conditions"""
        self.logger.debug(f"Generating routing parameters for strategy: {strategy}")
        parameters = {
            'strategy': strategy,
            'routing_layers': ['metal1', 'metal2', 'metal3', 'metal4', 'metal5', 'metal6', 'metal7'], # Expanded layers
            'congestion_aware': len(congestion_areas) > 0,
            'timing_driven': len(timing_critical_nets) > 0,
            'drc_aware': len(drc_sensitive_areas) > 0,
            'net_priorities': {},
            'layer_assignment_policy': 'balanced',
            'spacing_rules': 'default',
            'via_minimization': True,
            'redundant_via_insertion': False # Added for manufacturability
        }
        
        # Adjust parameters based on strategy
        if strategy == 'a_star':
            parameters['timing_driven'] = True
            parameters['search_heuristic'] = 'manhattan_with_timing'
            parameters['timing_weight'] = 0.8 # Higher weight for timing
            parameters['global_routing_grid_size'] = 10 # Finer grid for precision
        elif strategy == 'maze_expansion':
            parameters['congestion_aware'] = True
            parameters['search_heuristic'] = 'congestion_based'
            parameters['congestion_weight'] = 0.9 # Higher weight for congestion
            parameters['routing_iterations'] = 5 # More iterations for convergence
            parameters['path_cost_model'] = 'area_and_congestion'
        elif strategy == 'global_then_detail':
            parameters['hierarchical'] = True
            parameters['global_routing_passes'] = 3
            parameters['detail_refinement_passes'] = 2
            parameters['global_router_type'] = 'fast_gr'
            parameters['detail_router_type'] = 'track_based_dr'
        elif strategy == 'channel_based':
            parameters['channel_width_multiplier'] = 1.2 # Wider channels for easier routing
            parameters['channel_assignment_algorithm'] = 'greedy_with_lookahead'
            parameters['use_power_rails_as_channels'] = True
        elif strategy == 'ant_colony_optimization':
            parameters['pheromone_decay_rate'] = 0.1
            parameters['pheromone_deposit_amount'] = 1.0
            parameters['num_ants'] = 50
            parameters['num_iterations'] = 100
            parameters['cost_function_weights'] = {'length': 0.4, 'vias': 0.3, 'congestion': 0.3}
        elif strategy == 'machine_learning_guided':
            parameters['ml_model_path'] = '/models/routing_ml_model.pkl'
            parameters['ml_features_to_use'] = ['congestion', 'timing_criticality', 'drc_hotspot']
            parameters['learning_rate_adjustment'] = True
        
        # Set priorities for critical nets
        for net in timing_critical_nets:
            parameters['net_priorities'][net] = 'high'
        
        # Prioritize nets in congested areas
        for area in congestion_areas:
            if area['estimated_congestion'] > 0.8:
                # Find nets within this area and give them higher priority
                # This is a simplified example
                for net_in_area in unrouted_nets: # Heuristic
                    if net_in_area not in parameters['net_priorities']:
                        parameters['net_priorities'][net_in_area] = 'medium'
        
        for net in unrouted_nets:
            if net not in parameters['net_priorities']:
                parameters['net_priorities'][net] = 'normal'
        
        # Adjust spacing rules for DRC sensitive areas
        if len(drc_sensitive_areas) > 0:
            parameters['spacing_rules'] = 'enhanced'
            parameters['drc_safe_routing'] = True
            parameters['redundant_via_insertion'] = True # Add redundant vias in sensitive areas
        
        return parameters
    
    def _assess_routing_risk(self, parameters: Dict[str, Any],
                           congestion_areas: List[Dict],
                           timing_critical_nets: List[str],
                           drc_sensitive_areas: List[Dict]) -> Dict[str, float]:
        """Assess the risk profile of the proposed routing"""
        self.logger.debug(f"Assessing routing risk for strategy: {parameters['strategy']}")
        risk_profile = {
            'congestion_risk': 0.0,
            'timing_risk': 0.0,
            'drc_risk': 0.0,
            'manufacturability_risk': 0.0,
            'thermal_impact_risk': 0.0 # New: Thermal risk for routing choices
        }
        
        strategy = parameters['strategy']
        num_congested_areas = len(congestion_areas)
        num_timing_critical = len(timing_critical_nets)
        num_drc_sensitive = len(drc_sensitive_areas)

        # Congestion risk based on congestion areas and strategy
        if strategy in ['maze_expansion', 'global_then_detail']: # These aim to reduce congestion
            congestion_risk = min(num_congested_areas / 50.0, 0.4)
        else:
            congestion_risk = min(num_congested_areas / 20.0, 0.7) # Others might be higher risk
        risk_profile['congestion_risk'] = congestion_risk
        
        # Timing risk based on critical nets and strategy
        if strategy == 'a_star' or parameters.get('timing_driven', False): # Timing-driven strategy
            timing_risk = min(num_timing_critical / 100.0, 0.3)
        else:
            timing_risk = min(num_timing_critical / 50.0, 0.6)
        risk_profile['timing_risk'] = timing_risk
        
        # DRC risk based on sensitive areas and strategy
        if parameters.get('drc_safe_routing', False): # Strategy explicitly handles DRC
            drc_risk = min(num_drc_sensitive / 50.0, 0.2)
        else:
            drc_risk = min(num_drc_sensitive / 20.0, 0.5)
        risk_profile['drc_risk'] = drc_risk
        
        # Manufacturability risk (related to complex routing, redundant vias etc.)
        if strategy in ['ant_colony_optimization', 'machine_learning_guided']:
            manufacturability_risk = 0.5 # Can be complex to ensure manufacturability
        elif parameters.get('redundant_via_insertion', False):
            manufacturability_risk = 0.3 # Improves manufacturability but adds complexity
        else:
            manufacturability_risk = 0.2
        risk_profile['manufacturability_risk'] = manufacturability_risk

        # Thermal impact risk (routing can create heat concentration, or spread it)
        if num_congested_areas > 0 and parameters.get('power_mesh_aware', False): # If power mesh aware, better thermal
            thermal_risk = 0.2
        else:
            thermal_risk = 0.4
        risk_profile['thermal_impact_risk'] = thermal_risk
        
        return risk_profile
    
    def _calculate_routing_cost_vector(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Calculate the cost vector (PPA impact) of the routing proposal"""
        self.logger.debug(f"Calculating cost vector for strategy: {parameters['strategy']}")
        strategy = parameters['strategy']
        
        if strategy == 'a_star':
            return {
                'power_impact': 0.05,      # Moderate power increase due to potential detours
                'performance_impact': -0.15, # Excellent timing performance improvement
                'area_impact': 0.08,       # Moderate area impact due to longer routes
                'yield_impact': -0.05,     # Good yield with timing awareness
                'schedule_impact': 0.1,    # More computationally intensive
                'thermal_impact': 0.05     # Slight increase due to potential wire density
            }
        elif strategy == 'maze_expansion':
            return {
                'power_impact': 0.03,      # Good for congestion, can result in lower power
                'performance_impact': -0.08, # Good performance
                'area_impact': 0.05,       # Good area utilization
                'yield_impact': -0.06,     # Good yield with congestion awareness
                'schedule_impact': 0.08,   # Moderate computational cost
                'thermal_impact': 0.03     # Slight increase due to routing density
            }
        elif strategy == 'global_then_detail':
            return {
                'power_impact': 0.05,      # Balanced approach
                'performance_impact': -0.1, # Good performance
                'area_impact': 0.05,       # Good area utilization
                'yield_impact': -0.04,     # Good yield
                'schedule_impact': 0.15,   # Most computationally intensive
                'thermal_impact': 0.05     # Balanced thermal impact
            }
        elif strategy == 'channel_based':
            return {
                'power_impact': 0.07,      # Decent power efficiency
                'performance_impact': -0.05, # Moderate performance
                'area_impact': 0.06,       # Good area efficiency
                'yield_impact': -0.03,     # Good yield
                'schedule_impact': -0.08,  # Fast execution
                'thermal_impact': 0.04     # Moderate thermal impact
            }
        elif strategy == 'ant_colony_optimization':
            return {
                'power_impact': 0.08,      # Can be higher due to potentially complex routes
                'performance_impact': -0.12, # Good for exploring performance optima
                'area_impact': 0.1,        # Can be higher
                'yield_impact': -0.07,     # Good for exploring manufacturability optima
                'schedule_impact': 0.15,   # Highly computationally intensive
                'thermal_impact': 0.08     # Higher thermal impact due to varied routes
            }
        else:  # machine_learning_guided
            return {
                'power_impact': -0.05,     # Can learn to optimize power
                'performance_impact': -0.15, # Can learn to optimize performance
                'area_impact': -0.05,      # Can learn to optimize area
                'yield_impact': -0.08,     # Can learn to optimize yield
                'schedule_impact': 0.05,   # Can be faster if model is efficient
                'thermal_impact': -0.03    # Can learn to optimize thermal
            }
    
    def _predict_routing_outcome(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Predict the outcome of applying this routing"""
        self.logger.debug(f"Predicting outcome for strategy: {parameters['strategy']}")
        strategy = parameters['strategy']
        
        if strategy == 'a_star':
            return {
                'expected_congestion_reduction': 0.15, # Moderate reduction
                'expected_timing_improvement': 0.35, # Significant timing improvement
                'expected_drc_violation_reduction': 0.1, # Minor DRC reduction
                'routability_score': 0.9,
                'expected_thermal_impact_reduction': 0.05, # Slight thermal improvement
                'expected_manufacturability_improvement': 0.1 # Slight manufacturability improvement
            }
        elif strategy == 'maze_expansion':
            return {
                'expected_congestion_reduction': 0.4, # Significant congestion reduction
                'expected_timing_improvement': 0.1, # Moderate timing improvement
                'expected_drc_violation_reduction': 0.15, # Moderate DRC reduction
                'routability_score': 0.88,
                'expected_thermal_impact_reduction': 0.08,
                'expected_manufacturability_improvement': 0.12
            }
        elif strategy == 'global_then_detail':
            return {
                'expected_congestion_reduction': 0.35,
                'expected_timing_improvement': 0.25,
                'expected_drc_violation_reduction': 0.2,
                'routability_score': 0.95, # Excellent routability
                'expected_thermal_impact_reduction': 0.1,
                'expected_manufacturability_improvement': 0.15
            }
        elif strategy == 'channel_based':
            return {
                'expected_congestion_reduction': 0.2,
                'expected_timing_improvement': 0.08,
                'expected_drc_violation_reduction': 0.08,
                'routability_score': 0.85,
                'expected_thermal_impact_reduction': 0.03,
                'expected_manufacturability_improvement': 0.08
            }
        elif strategy == 'ant_colony_optimization':
            return {
                'expected_congestion_reduction': 0.25,
                'expected_timing_improvement': 0.2,
                'expected_drc_violation_reduction': 0.18,
                'routability_score': 0.92,
                'expected_thermal_impact_reduction': 0.07,
                'expected_manufacturability_improvement': 0.1
            }
        else:  # machine_learning_guided
            return {
                'expected_congestion_reduction': 0.3, # Can be very good
                'expected_timing_improvement': 0.3, # Can be very good
                'expected_drc_violation_reduction': 0.25, # Can be very good
                'routability_score': 0.98, # Potentially excellent
                'expected_thermal_impact_reduction': 0.12,
                'expected_manufacturability_improvement': 0.18
            }
    
    def _calculate_confidence(self, graph: CanonicalSiliconGraph) -> float:
        """Calculate confidence in the routing proposal"""
        base_confidence = 0.82
        
        # Adjust based on recent performance
        recent_perf = self.get_recent_performance()
        adjusted_confidence = base_confidence * recent_perf * self.authority_level
        
        return max(min(adjusted_confidence, 1.0), 0.1)  # Clamp to [0.1, 1.0]
    
    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now()
    
    def evaluate_proposal_impact(self, proposal: 'AgentProposal', 
                               graph: CanonicalSiliconGraph) -> Dict[str, float]:
        """Evaluate the potential impact of a routing proposal"""
        impact = {
            'congestion_improvement': 0.0,
            'timing_improvement': 0.0,
            'routability': 0.0,
            'drc_compliance': 0.0
        }
        
        # Based on the strategy, estimate improvements
        strategy = proposal.parameters.get('strategy', 'channel_based')
        
        if strategy == 'a_star':
            impact['timing_improvement'] = 0.35
            impact['routability'] = 0.9
            impact['drc_compliance'] = 0.8
        elif strategy == 'maze_expansion':
            impact['congestion_improvement'] = 0.4
            impact['routability'] = 0.85
            impact['drc_compliance'] = 0.85
        elif strategy == 'global_then_detail':
            impact['congestion_improvement'] = 0.35
            impact['timing_improvement'] = 0.25
            impact['routability'] = 0.95
        else:  # channel_based
            impact['congestion_improvement'] = 0.25
            impact['routability'] = 0.8
            impact['drc_compliance'] = 0.75
        
        return impact