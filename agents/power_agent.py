"""
Power Agent - Specialized agent for power optimization

This agent focuses on IR drop, EM (electromigration), and dynamic power optimization
to ensure power integrity across the design.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from agents.base_agent import BaseAgent, AgentType, AgentProposal
from core.canonical_silicon_graph import CanonicalSiliconGraph
from utils.logger import get_logger


class PowerAgent(BaseAgent):
    """
    Power Agent - optimizes for IR drop, EM, and dynamic power
    """
    
    def __init__(self):
        super().__init__(AgentType.POWER)
        self.logger = get_logger(f"{__name__}.power_agent")
        self.power_strategies = [
            'uniform_grid', 'adaptive_grid', 'hierarchical', 'ring_spine', 'hybrid'
        ]
    
    def propose_action(self, graph: CanonicalSiliconGraph) -> Optional['AgentProposal']:
        """
        Generate a power optimization proposal based on the current graph state
        """
        self.logger.info("Generating power optimization proposal")
        
        # Identify high-power nodes and critical power paths
        high_power_nodes = graph.get_high_power_nodes(threshold=0.05)
        if not high_power_nodes:
            # Find nodes with highest power consumption
            all_nodes = [(n, graph.graph.nodes[n].get('power', 0.0)) 
                        for n in graph.graph.nodes() 
                        if 'power' in graph.graph.nodes[n]]
            all_nodes.sort(key=lambda x: x[1], reverse=True)
            high_power_nodes = [n for n, p in all_nodes[:10]]  # Top 10 power consumers
        
        if not high_power_nodes:
            self.logger.debug("No high-power nodes found, skipping power proposal")
            return None
        
        # Analyze power distribution and identify hotspots
        power_hotspots = self._identify_power_hotspots(graph, high_power_nodes)
        ir_drop_risk_nodes = self._identify_ir_drop_risk_nodes(graph, high_power_nodes)
        em_risk_nodes = self._identify_em_risk_nodes(graph, high_power_nodes)
        
        # Select optimal power strategy
        strategy = self._select_power_strategy(power_hotspots, ir_drop_risk_nodes, em_risk_nodes)
        
        # Generate power optimization parameters
        parameters = self._generate_power_parameters(
            strategy, power_hotspots, ir_drop_risk_nodes, em_risk_nodes
        )
        
        # Calculate risk profile
        risk_profile = self._assess_power_risk(
            parameters, power_hotspots, ir_drop_risk_nodes, em_risk_nodes
        )
        
        # Calculate cost vector (PPA impact)
        cost_vector = self._calculate_power_cost_vector(parameters)
        
        # Estimate outcome
        predicted_outcome = self._predict_power_outcome(parameters)
        
        proposal = AgentProposal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            proposal_id=f"pw_{np.random.randint(1000, 9999)}",
            timestamp=self._get_timestamp(),
            action_type="optimize_power",
            targets=high_power_nodes,
            parameters=parameters,
            confidence_score=self._calculate_confidence(graph),
            risk_profile=risk_profile,
            cost_vector=cost_vector,
            predicted_outcome=predicted_outcome,
            dependencies=[],
            conflicts_with=[]
        )
        
        self.logger.info(f"Generated power optimization proposal for {len(high_power_nodes)} nodes")
        return proposal
    
    def _identify_power_hotspots(self, graph: CanonicalSiliconGraph, high_power_nodes: List[str]) -> List[Dict[str, Any]]:
        """Identify power hotspots in the design"""
        self.logger.debug("Identifying power hotspots.")
        hotspots = []
        
        # Group high-power nodes by region
        region_power = {}
        total_design_power = sum(attrs.get('power', 0.0) for _, attrs in graph.graph.nodes(data=True) if 'power' in attrs)
        average_node_power = total_design_power / max(1, len(graph.graph.nodes()))

        for node in high_power_nodes:
            if node in graph.graph.nodes():
                attrs = graph.graph.nodes[node]
                region = attrs.get('region', 'default')
                power = attrs.get('power', 0.0)
                
                if region not in region_power:
                    region_power[region] = {'nodes': [], 'total_power': 0.0, 'area': 0.0} # Added area
                region_power[region]['nodes'].append(node)
                region_power[region]['total_power'] += power
                region_power[region]['area'] += attrs.get('area', 0.0) # Accumulate area
        
        # Identify regions with high power density
        # Threshold for hotspot: 2x average node power density or region total power > 1.0 (arbitrary)
        for region, data in region_power.items():
            # Estimate area of the region (simple sum of node areas for now)
            region_area = data['area']
            if region_area == 0: region_area = 0.001 # Avoid division by zero
            
            region_power_density = data['total_power'] / region_area

            # A simple heuristic for a hotspot: if region power density is significantly higher than average
            # Or if total power in a region is simply high.
            # Here, using a fixed threshold for simplicity, but more dynamic in a real system.
            if region_power_density > (average_node_power * 10) or data['total_power'] > 1.0: # Example thresholds
                hotspot_info = {
                    'region': region,
                    'nodes': data['nodes'],
                    'total_power': data['total_power'],
                    'node_count': len(data['nodes']),
                    'power_density': region_power_density
                }
                hotspots.append(hotspot_info)
                self.logger.debug(f"Identified hotspot in region '{region}' with power density {region_power_density:.2f}")
        
        return hotspots
    
    def _identify_ir_drop_risk_nodes(self, graph: CanonicalSiliconGraph, high_power_nodes: List[str]) -> List[str]:
        """Identify nodes at risk for IR drop issues"""
        self.logger.debug("Identifying IR drop risk nodes.")
        ir_drop_risk = []
        
        # Calculate max possible IR risk score for normalization
        max_ir_risk_score = 0.0
        all_ir_risk_scores = []

        for node in high_power_nodes:
            if node in graph.graph.nodes():
                attrs = graph.graph.nodes[node]
                
                power = attrs.get('power', 0.0)
                switching_activity = attrs.get('switching_activity', 0.5)
                fanout = len(list(graph.graph.successors(node)))
                
                # Calculate IR drop risk score
                ir_risk_score = power * switching_activity * (1 + np.log(fanout + 1))
                all_ir_risk_scores.append(ir_risk_score)
                max_ir_risk_score = max(max_ir_risk_score, ir_risk_score)
        
        # Normalize and identify risky nodes
        if max_ir_risk_score == 0: # Avoid division by zero
            return []

        for node in high_power_nodes:
            if node in graph.graph.nodes():
                attrs = graph.graph.nodes[node]
                power = attrs.get('power', 0.0)
                switching_activity = attrs.get('switching_activity', 0.5)
                fanout = len(list(graph.graph.successors(node)))
                ir_risk_score = power * switching_activity * (1 + np.log(fanout + 1))
                
                normalized_ir_risk = ir_risk_score / max_ir_risk_score
                
                # If risk score exceeds threshold, add to risky nodes
                if normalized_ir_risk > 0.7:  # Normalized threshold
                    ir_drop_risk.append(node)
                    self.logger.debug(f"Node '{node}' identified as IR drop risk (score: {normalized_ir_risk:.2f}).")
        
        return ir_drop_risk
    
    def _identify_em_risk_nodes(self, graph: CanonicalSiliconGraph, high_power_nodes: List[str]) -> List[str]:
        """Identify nodes at risk for electromigration issues"""
        self.logger.debug("Identifying EM risk nodes.")
        em_risk = []
        
        # Calculate average current estimate for normalization (simplified)
        all_current_estimates = []
        for node in high_power_nodes:
            if node in graph.graph.nodes():
                attrs = graph.graph.nodes[node]
                power = attrs.get('power', 0.0)
                switching_activity = attrs.get('switching_activity', 0.5)
                all_current_estimates.append(power * switching_activity)
        
        avg_current_estimate = np.mean(all_current_estimates) if all_current_estimates else 0.0
        em_risk_threshold = avg_current_estimate * 1.5 # Example: 1.5x average is risky

        for node in high_power_nodes:
            if node in graph.graph.nodes():
                attrs = graph.graph.nodes[node]
                
                # EM risk factors: high current density paths
                power = attrs.get('power', 0.0)
                switching_activity = attrs.get('switching_activity', 0.5)
                
                # Look at connecting edges for current density
                for successor in graph.graph.successors(node):
                    # In a more detailed graph, edge_attrs might contain width/length
                    edge_attrs = graph.graph.get_edge_data(node, successor)
                    if edge_attrs:
                        # Simplified current estimate
                        current_estimate = power * switching_activity
                        
                        # If current estimate exceeds threshold, add to risky nodes
                        if current_estimate > em_risk_threshold and em_risk_threshold > 0:
                            em_risk.append(node)
                            self.logger.debug(f"Node '{node}' identified as EM risk (current: {current_estimate:.2f}).")
                            break  # Found one risky connection, no need to check other successors
        
        return list(set(em_risk))  # Remove duplicates
    
    def _select_power_strategy(self, power_hotspots: List[Dict], 
                             ir_drop_risk_nodes: List[str], 
                             em_risk_nodes: List[str]) -> str:
        """Select the optimal power strategy based on risk profile"""
        self.logger.debug("Selecting optimal power strategy.")
        num_hotspots = len(power_hotspots)
        num_ir_risk = len(ir_drop_risk_nodes)
        num_em_risk = len(em_risk_nodes)

        # Prioritize strategies based on the most severe or prevalent risks
        if num_hotspots > 5 or num_ir_risk > 10:
            # If there are many hotspots or significant IR drop issues
            self.logger.debug(f"Selecting 'adaptive_grid' due to {num_hotspots} hotspots and {num_ir_risk} IR drop risks.")
            return 'adaptive_grid' # Best for fine-grained power distribution and IR drop mitigation

        elif num_em_risk > 5:
            # If EM risks are a primary concern
            self.logger.debug(f"Selecting 'hierarchical' due to {num_em_risk} EM risks.")
            return 'hierarchical' # Good for current density management and EM reliability

        elif num_hotspots > 1 and num_ir_risk > 3:
            # Moderate number of hotspots and IR risks, combined
            self.logger.debug(f"Selecting 'ring_spine' due to {num_hotspots} hotspots and {num_ir_risk} IR drop risks.")
            return 'ring_spine' # Offers good power distribution and can mitigate both

        elif num_hotspots == 0 and num_ir_risk == 0 and num_em_risk == 0:
            # If no significant power issues, opt for a simple and efficient approach
            self.logger.debug("Selecting 'uniform_grid' as no major power issues detected.")
            return 'uniform_grid' # Simpler, less area overhead

        else:
            # Default to a balanced hybrid approach for moderate issues
            self.logger.debug("Selecting 'hybrid' as a balanced approach for moderate power issues.")
            return 'hybrid'
    
    def _generate_power_parameters(self, strategy: str, power_hotspots: List[Dict], 
                                 ir_drop_risk_nodes: List[str], 
                                 em_risk_nodes: List[str]) -> Dict[str, Any]:
        """Generate power optimization parameters based on strategy"""
        self.logger.debug(f"Generating power parameters for strategy: {strategy}")
        parameters = {
            'strategy': strategy,
            'power_grid_config': {},
            'decap_placement': [],
            'power_island_config': {},
            'dynamic_power_management': {}
        }
        
        num_hotspots = len(power_hotspots)
        num_ir_risk = len(ir_drop_risk_nodes)
        num_em_risk = len(em_risk_nodes)

        if strategy == 'uniform_grid':
            parameters['power_grid_config'] = {
                'mesh_width_um': 10.0, # Larger width for less critical designs
                'via_stacks': 1,       # Simpler via structures
                'metal_layers': ['metal1', 'metal2', 'metal3', 'metal4'],
                'supply_rails': True,
                'power_ring_around_periphery': False
            }
        elif strategy == 'adaptive_grid':
            parameters['power_grid_config'] = {
                'mesh_width_um': 2.0,  # Finer mesh, especially in hotspot areas
                'via_stacks': max(2, num_hotspots // 2), # Scale via stacks with hotspots
                'metal_layers': ['metal1', 'metal2', 'metal3', 'metal4', 'metal5', 'metal6'],
                'adaptive_density': True,
                'hotspot_reinforcement_regions': [h['region'] for h in power_hotspots],
                'pg_strap_width_multiplier': 1.5 # Thicker straps in critical areas
            }
        elif strategy == 'hierarchical':
            parameters['power_grid_config'] = {
                'primary_rings_width_um': 20.0,
                'secondary_mesh_width_um': 5.0,
                'hierarchical_vdd_rails': True,
                'hierarchical_vss_rails': True,
                'power_domains_isolated': True
            }
        elif strategy == 'ring_spine':
            parameters['power_grid_config'] = {
                'perimeter_rings_count': 2,
                'internal_spines_count': max(2, num_ir_risk // 5), # Scale with IR drop
                'cross_connections_density': 0.7, # Moderate density
                'power_rings_width_um': 15.0
            }
        elif strategy == 'hybrid':
            parameters['power_grid_config'] = {
                'adaptive_regions': [h['region'] for h in power_hotspots[:2]],
                'standard_periphery': True,
                'reinforced_hotspots': [h['region'] for h in power_hotspots],
                'flexible_layer_assignment': True
            }
        
        # Add decoupling capacitor placement recommendations
        for hotspot in power_hotspots:
            if hotspot['total_power'] > 0.8:  # Very high power hotspots
                parameters['decap_placement'].extend([{'node': n, 'type': 'high_cap'} for n in hotspot['nodes'][:5]]) # Place high-cap decaps
            elif hotspot['total_power'] > 0.5:
                parameters['decap_placement'].extend([{'node': n, 'type': 'std_cap'} for n in hotspot['nodes'][:3]]) # Place std-cap decaps
        
        # Configure power islands if needed
        if num_hotspots > 1 or num_ir_risk > 5:
            parameters['power_island_config'] = {
                'enable_islands': True,
                'island_boundaries_regions': [h['region'] for h in power_hotspots],
                'isolation_cells': True,
                'supply_switching_frequency_ghz': 0.5
            }
        
        # Dynamic power management
        if num_ir_risk > 0 or num_em_risk > 0:
            parameters['dynamic_power_management'] = {
                'clock_gating_recommendations': ir_drop_risk_nodes[:min(5, num_ir_risk)],
                'power_switching_zones': [h['region'] for h in power_hotspots if h['power_density'] > 0.8],
                'dvfs_zones': [h['region'] for h in power_hotspots if h['node_count'] > 10],
                'adaptive_body_biasing_targets': em_risk_nodes[:min(5, num_em_risk)]
            }
        
        return parameters
    
    def _assess_power_risk(self, parameters: Dict[str, Any], 
                          power_hotspots: List[Dict], 
                          ir_drop_risk_nodes: List[str], 
                          em_risk_nodes: List[str]) -> Dict[str, float]:
        """Assess the risk profile of the proposed power optimization"""
        self.logger.debug(f"Assessing power risk for strategy: {parameters['strategy']}")
        risk_profile = {
            'ir_drop_risk': 0.0,
            'em_risk': 0.0,
            'power_integrity_risk': 0.0,
            'area_overhead_risk': 0.0,
            'thermal_impact_risk': 0.0 # New: Reflects potential thermal impact
        }
        
        strategy = parameters['strategy']
        num_ir_risk = len(ir_drop_risk_nodes)
        num_em_risk = len(em_risk_nodes)
        num_hotspots = len(power_hotspots)

        # IR drop risk assessment
        if strategy == 'adaptive_grid':
            ir_risk = min(num_ir_risk / 20.0, 0.4) # Adaptive grid is good at mitigating IR drop
        elif strategy == 'ring_spine':
            ir_risk = min(num_ir_risk / 15.0, 0.6) # Ring-spine is also effective
        else:
            ir_risk = min(num_ir_risk / 10.0, 0.8) # Other strategies might have higher base risk
        risk_profile['ir_drop_risk'] = ir_risk
        
        # EM risk assessment
        if strategy == 'hierarchical':
            em_risk = min(num_em_risk / 10.0, 0.4) # Hierarchical power distribution helps EM
        elif strategy == 'adaptive_grid':
            em_risk = min(num_em_risk / 8.0, 0.5) # Adaptive also helps
        else:
            em_risk = min(num_em_risk / 5.0, 0.7) # Others might be higher risk
        risk_profile['em_risk'] = em_risk
        
        # Power integrity risk (combination of IR and EM)
        risk_profile['power_integrity_risk'] = (ir_risk * 0.6 + em_risk * 0.4) # Weighted average
        
        # Area overhead risk from power grid
        if strategy in ['adaptive_grid', 'hierarchical']:
            area_risk = 0.7 + (num_hotspots * 0.05) # Higher area overhead, scales with hotspots
        elif strategy == 'uniform_grid':
            area_risk = 0.5 - (num_hotspots * 0.02) # Lower area overhead if few hotspots
        else:
            area_risk = 0.4 + (num_hotspots * 0.03) # Moderate area overhead
        risk_profile['area_overhead_risk'] = min(area_risk, 1.0)
        
        # Thermal impact risk (dense power grids can trap heat, or spread it)
        if strategy == 'adaptive_grid' and num_hotspots > 0:
            thermal_risk = 0.2 # Can help spread heat
        elif strategy == 'uniform_grid' and num_hotspots > 0:
            thermal_risk = 0.6 # May worsen hotspots if not adaptive
        else:
            thermal_risk = 0.4 # Moderate impact
        risk_profile['thermal_impact_risk'] = thermal_risk
        
        return risk_profile
    
    def _calculate_power_cost_vector(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Calculate the cost vector (PPA impact) of the power proposal"""
        self.logger.debug(f"Calculating cost vector for strategy: {parameters['strategy']}")
        strategy = parameters['strategy']
        
        if strategy == 'adaptive_grid':
            return {
                'power_impact': -0.15,     # Excellent power optimization
                'performance_impact': -0.05, # Good performance (reduced IR drop)
                'area_impact': 0.15,       # High area cost for adaptive grid
                'yield_impact': -0.05,     # Good yield (better power integrity)
                'schedule_impact': 0.08,   # Moderate schedule impact
                'thermal_impact': -0.08    # Good thermal impact (spreading heat)
            }
        elif strategy == 'hierarchical':
            return {
                'power_impact': -0.1,      # Good power optimization
                'performance_impact': -0.03, # Good performance
                'area_impact': 0.1,        # Medium area cost
                'yield_impact': -0.04,     # Good yield
                'schedule_impact': 0.05,   # Moderate schedule impact
                'thermal_impact': -0.05    # Moderate thermal impact
            }
        elif strategy == 'ring_spine':
            return {
                'power_impact': -0.08,     # Good power optimization
                'performance_impact': -0.02, # Good performance
                'area_impact': 0.08,       # Medium area cost
                'yield_impact': -0.03,     # Good yield
                'schedule_impact': 0.03,   # Lower schedule impact
                'thermal_impact': -0.04    # Moderate thermal impact
            }
        elif strategy == 'hybrid':
            return {
                'power_impact': -0.07,     # Good power optimization
                'performance_impact': -0.02, # Good performance
                'area_impact': 0.05,       # Lower area cost
                'yield_impact': -0.02,     # Good yield
                'schedule_impact': 0.02,   # Lower schedule impact
                'thermal_impact': -0.03    # Moderate thermal impact
            }
        else:  # uniform_grid
            return {
                'power_impact': 0.05,      # Moderate power optimization (less effective)
                'performance_impact': 0.03, # Moderate performance impact (may worsen IR)
                'area_impact': -0.02,      # Lower area cost (efficient)
                'yield_impact': 0.02,      # Moderate yield (less robust)
                'schedule_impact': -0.05,  # Fast implementation
                'thermal_impact': 0.05     # May worsen thermal hotspots
            }
    
    def _predict_power_outcome(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Predict the outcome of applying this power optimization"""
        self.logger.debug(f"Predicting outcome for strategy: {parameters['strategy']}")
        strategy = parameters['strategy']
        
        if strategy == 'adaptive_grid':
            return {
                'expected_ir_drop_reduction': 0.7, # 70% reduction
                'expected_em_improvement': 0.6,    # 60% improvement
                'expected_dynamic_power_reduction': 0.15, # 15% reduction
                'expected_area_increase': 0.25, # 25% area increase (trade-off)
                'expected_power_integrity_improvement': 0.8, # Overall strong improvement
                'expected_thermal_impact_reduction': 0.15 # Helps spread heat, reduces hotspots
            }
        elif strategy == 'hierarchical':
            return {
                'expected_ir_drop_reduction': 0.6,
                'expected_em_improvement': 0.5,
                'expected_dynamic_power_reduction': 0.12,
                'expected_area_increase': 0.15,
                'expected_power_integrity_improvement': 0.7,
                'expected_thermal_impact_reduction': 0.1
            }
        elif strategy == 'ring_spine':
            return {
                'expected_ir_drop_reduction': 0.5,
                'expected_em_improvement': 0.4,
                'expected_dynamic_power_reduction': 0.1,
                'expected_area_increase': 0.1,
                'expected_power_integrity_improvement': 0.6,
                'expected_thermal_impact_reduction': 0.08
            }
        elif strategy == 'hybrid':
            return {
                'expected_ir_drop_reduction': 0.55,
                'expected_em_improvement': 0.45,
                'expected_dynamic_power_reduction': 0.08,
                'expected_area_increase': 0.08,
                'expected_power_integrity_improvement': 0.65,
                'expected_thermal_impact_reduction': 0.1
            }
        else:  # uniform_grid
            return {
                'expected_ir_drop_reduction': 0.4,
                'expected_em_improvement': 0.3,
                'expected_dynamic_power_reduction': 0.05,
                'expected_area_increase': 0.05,
                'expected_power_integrity_improvement': 0.5,
                'expected_thermal_impact_reduction': -0.05 # May worsen thermal impact slightly
            }
    
    def _calculate_confidence(self, graph: CanonicalSiliconGraph) -> float:
        """Calculate confidence in the power proposal"""
        # Confidence based on design characteristics and agent performance
        base_confidence = 0.85
        
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
        """Evaluate the potential impact of a power optimization proposal"""
        impact = {
            'ir_drop_improvement': 0.0,
            'em_improvement': 0.0,
            'power_efficiency': 0.0,
            'area_overhead': 0.0
        }
        
        # Based on the strategy, estimate improvements
        strategy = proposal.parameters.get('strategy', 'uniform_grid')
        
        if strategy == 'adaptive_grid':
            impact['ir_drop_improvement'] = 0.7
            impact['em_improvement'] = 0.6
            impact['power_efficiency'] = 0.8
            impact['area_overhead'] = 0.25
        elif strategy == 'hierarchical':
            impact['ir_drop_improvement'] = 0.6
            impact['em_improvement'] = 0.5
            impact['power_efficiency'] = 0.7
            impact['area_overhead'] = 0.15
        elif strategy == 'ring_spine':
            impact['ir_drop_improvement'] = 0.5
            impact['em_improvement'] = 0.4
            impact['power_efficiency'] = 0.6
            impact['area_overhead'] = 0.1
        elif strategy == 'hybrid':
            impact['ir_drop_improvement'] = 0.55
            impact['em_improvement'] = 0.45
            impact['power_efficiency'] = 0.75
            impact['area_overhead'] = 0.08
        else:  # uniform_grid
            impact['ir_drop_improvement'] = 0.4
            impact['em_improvement'] = 0.3
            impact['power_efficiency'] = 0.5
            impact['area_overhead'] = 0.05
        
        return impact