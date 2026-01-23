"""
Clock Agent - Specialized agent for clock tree synthesis

This agent designs clock trees with focus on skew, latency, and variation
at advanced process nodes.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from agents.base_agent import BaseAgent, AgentType, AgentProposal
from core.canonical_silicon_graph import CanonicalSiliconGraph
from utils.logger import get_logger


class ClockAgent(BaseAgent):
    """
    Clock Agent - designs clock trees with focus on skew, latency, and variation
    """
    
    def __init__(self):
        super().__init__(AgentType.CLOCK)
        self.logger = get_logger(f"{__name__}.clock_agent")
        self.clock_strategies = [
            'balanced_tree', 'fishbone', 'h_tree', 'spine', 'custom_topology'
        ]
    
    def propose_action(self, graph: CanonicalSiliconGraph) -> Optional['AgentProposal']:
        """
        Generate a clock tree synthesis proposal based on the current graph state
        """
        self.logger.info("Generating clock tree synthesis proposal")
        
        # Identify clock sources and sinks
        clock_sources = graph.get_clock_roots()
        if not clock_sources:
            # If no explicit clock roots, find potential clock sources
            clock_sources = self._identify_potential_clock_sources(graph)
        
        # Identify clock sink nodes (sequential elements)
        clock_sinks = self._identify_clock_sinks(graph)
        
        if not clock_sources or not clock_sinks:
            self.logger.debug("No clock sources or sinks found, skipping clock proposal")
            return None
        
        # Analyze variation sensitivity
        variation_sensitive_nodes = self._identify_variation_sensitive_nodes(graph, clock_sinks)
        
        # Select optimal clock strategy
        strategy = self._select_clock_strategy(
            clock_sources, clock_sinks, variation_sensitive_nodes
        )
        
        # Generate clock tree parameters
        parameters = self._generate_clock_parameters(
            strategy, clock_sources, clock_sinks, variation_sensitive_nodes
        )
        
        # Calculate risk profile
        risk_profile = self._assess_clock_risk(
            parameters, clock_sinks, variation_sensitive_nodes
        )
        
        # Calculate cost vector (PPA impact)
        cost_vector = self._calculate_clock_cost_vector(parameters)
        
        # Estimate outcome
        predicted_outcome = self._predict_clock_outcome(parameters)
        
        proposal = AgentProposal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            proposal_id=f"clk_{np.random.randint(1000, 9999)}",
            timestamp=self._get_timestamp(),
            action_type="synthesize_clock_tree",
            targets=list(set(clock_sources + clock_sinks)),
            parameters=parameters,
            confidence_score=self._calculate_confidence(graph),
            risk_profile=risk_profile,
            cost_vector=cost_vector,
            predicted_outcome=predicted_outcome,
            dependencies=[],
            conflicts_with=[]
        )
        
        self.logger.info(f"Generated clock tree proposal with {len(clock_sources)} sources and {len(clock_sinks)} sinks")
        return proposal
    
    def _identify_potential_clock_sources(self, graph: CanonicalSiliconGraph) -> List[str]:
        """Identify potential clock sources in the design"""
        potential_sources = []
        
        # First, check for nodes explicitly marked as clock roots
        for node, attrs in graph.graph.nodes(data=True):
            if attrs.get('is_clock_root', False):
                potential_sources.append(node)
        
        if potential_sources:
            self.logger.debug(f"Identified {len(potential_sources)} explicit clock roots.")
            return potential_sources
        
        # If no explicit clock roots, use heuristic based on cell type and fanout
        self.logger.debug("No explicit clock roots found, falling back to heuristic identification.")
        for node, attrs in graph.graph.nodes(data=True):
            cell_type = attrs.get('cell_type', '').lower()
            
            # Look for typical clock source cells
            if any(keyword in cell_type for keyword in ['clk', 'clock', 'osc', 'buf']):
                # Check if it has high fanout (indicating it might be a clock source)
                fanout = len(list(graph.graph.successors(node)))
                if fanout > 5:  # Threshold for potential clock source
                    potential_sources.append(node)
        
        return potential_sources
    
    def _identify_clock_sinks(self, graph: CanonicalSiliconGraph) -> List[str]:
        """Identify nodes that are clock sinks (sequential elements)"""
        clock_sinks = []
        
        for node, attrs in graph.graph.nodes(data=True):
            cell_type = attrs.get('cell_type', '').lower()
            node_type = attrs.get('node_type')
            
            # Look for sequential elements that need clock and are standard cells
            if node_type == NodeType.CELL.value and any(keyword in cell_type for keyword in ['dff', 'ff', 'latch', 'reg', 'flipflop']):
                clock_sinks.append(node)
        
        return clock_sinks
    
    def _identify_variation_sensitive_nodes(self, graph: CanonicalSiliconGraph, 
                                         clock_sinks: List[str]) -> List[str]:
        """Identify nodes sensitive to process, voltage, temperature variations"""
        variation_sensitive = []
        
        for sink in clock_sinks:
            attrs = graph.graph.nodes[sink]
            # If it's in a high-performance region or has high timing criticality
            # Also consider nodes with high power consumption as they are sensitive to thermal variations
            if (attrs.get('timing_criticality', 0) > 0.7 or 
                attrs.get('region', '') in ['high_performance', 'timing_critical', 'power_hotspot'] or
                attrs.get('power', 0) > 0.05): # Assuming power > 0.05 is high power
                variation_sensitive.append(sink)
        
        return variation_sensitive
    
    def _select_clock_strategy(self, clock_sources: List[str], clock_sinks: List[str], 
                             variation_sensitive_nodes: List[str]) -> str:
        """Select the optimal clock strategy based on design characteristics"""
        num_clock_sources = len(clock_sources)
        num_clock_sinks = len(clock_sinks)
        num_variation_sensitive_nodes = len(variation_sensitive_nodes)

        # Prioritize custom topology if multiple clock sources are present
        if num_clock_sources > 1:
            self.logger.debug(f"Selecting 'custom_topology' due to multiple clock sources ({num_clock_sources}).")
            return 'custom_topology'

        # Prioritize H-tree for high variation sensitivity, as it offers better symmetry
        if num_variation_sensitive_nodes > num_clock_sinks * 0.25: # Tuned threshold
            self.logger.debug(f"Selecting 'h_tree' due to high variation sensitivity ({num_variation_sensitive_nodes}/{num_clock_sinks} sinks sensitive).")
            return 'h_tree'
        
        # For large designs, spine structure can be efficient for distribution
        if num_clock_sinks > 5000: # Tuned threshold for large designs
            self.logger.debug(f"Selecting 'spine' for large designs ({num_clock_sinks} sinks).")
            return 'spine'
        
        # For smaller designs, a simple balanced tree is often sufficient
        if num_clock_sinks < 500: # Tuned threshold for small designs
            self.logger.debug(f"Selecting 'balanced_tree' for smaller designs ({num_clock_sinks} sinks).")
            return 'balanced_tree'
        
        # Default for medium-sized designs or moderate sensitivity
        self.logger.debug(f"Selecting 'fishbone' as a balanced option ({num_clock_sinks} sinks).")
        return 'fishbone'
    
    def _generate_clock_parameters(self, strategy: str, clock_sources: List[str], 
                                 clock_sinks: List[str], 
                                 variation_sensitive_nodes: List[str]) -> Dict[str, Any]:
        """Generate clock tree synthesis parameters based on strategy"""
        self.logger.debug(f"Generating clock parameters for strategy: {strategy}")
        parameters = {
            'strategy': strategy,
            'clock_topology': {},
            'buffer_insertion_points': [],
            'skew_requirements': {},
            'latency_targets': {},
            'variation_mitigation': {}
        }
        
        # Determine base skew target based on number of sensitive nodes
        base_target_ppm = 80 # Default
        if variation_sensitive_nodes:
            # If many sensitive nodes, aim for tighter skew
            base_target_ppm = max(20, base_target_ppm - (len(variation_sensitive_nodes) // 100))

        # Set strategy-specific parameters
        if strategy == 'balanced_tree':
            parameters['clock_topology'] = {
                'type': 'balanced_binary',
                'max_fanout_per_level': 8,
                'buffer_types': ['CLKBUF_X1', 'BUF_X2']
            }
            parameters['skew_requirements'] = {'target_ppm': max(30, base_target_ppm * 0.75), 'max_local_skew_ps': 10}
            
        elif strategy == 'fishbone':
            parameters['clock_topology'] = {
                'type': 'fishbone_spine',
                'spine_count': max(2, len(clock_sinks) // 500),  # Scale with design size
                'buffer_types': ['CLKBUF_X4', 'CLKBUF_X8', 'BUF_X4']
            }
            parameters['skew_requirements'] = {'target_ppm': max(50, base_target_ppm * 0.9), 'max_local_skew_ps': 15}
            
        elif strategy == 'h_tree':
            parameters['clock_topology'] = {
                'type': 'symmetric_h_tree',
                'levels': int(np.ceil(np.log2(max(1, len(clock_sinks)/4)))),
                'buffer_types': ['CLKBUF_X1', 'CLKBUF_X2', 'CLKBUF_X4']
            }
            parameters['skew_requirements'] = {'target_ppm': max(15, base_target_ppm * 0.5), 'max_local_skew_ps': 5}
            parameters['variation_mitigation']['symmetry_priority'] = True
            
        elif strategy == 'spine':
            parameters['clock_topology'] = {
                'type': 'central_spine',
                'spine_buffer_count': max(4, len(clock_sinks) // 200),
                'buffer_types': ['SPINE_BUF_X12', 'CLOCK_BUF_X8']
            }
            parameters['skew_requirements'] = {'target_ppm': max(60, base_target_ppm * 0.95), 'max_local_skew_ps': 20}
            
        elif strategy == 'custom_topology':
            parameters['clock_topology'] = {
                'type': 'multi_source_custom',
                'sources': clock_sources,
                'buffer_types': ['CLKBUF_X1', 'CLKBUF_X2', 'CLKBUF_X4', 'CUST_CLKBUF_X1', 'CUST_CLKBUF_X2'] # More specific
            }
            parameters['skew_requirements'] = {'target_ppm': max(40, base_target_ppm * 0.8), 'max_local_skew_ps': 12}
        
        # Set latency targets based on design speed requirements
        # This would normally come from timing constraints
        parameters['latency_targets'] = {
            'max_latency_ps': 1200,  # 1.2ns for high-performance designs, adjustable
            'typical_latency_ps': 600
        }
        
        # Add variation mitigation for sensitive nodes
        if variation_sensitive_nodes:
            parameters['variation_mitigation']['enable'] = True
            parameters['variation_mitigation']['sensitive_nodes'] = variation_sensitive_nodes
            parameters['variation_mitigation']['extra_margin_ps'] = 8 # Increased margin for sensitive nodes
            parameters['variation_mitigation']['redundant_paths'] = True
            parameters['variation_mitigation']['on_chip_variation_aware'] = True # Explicitly state
        
        return parameters
    
    def _assess_clock_risk(self, parameters: Dict[str, Any], clock_sinks: List[str], 
                          variation_sensitive_nodes: List[str]) -> Dict[str, float]:
        """Assess the risk profile of the proposed clock tree"""
        self.logger.debug(f"Assessing clock risk for strategy: {parameters['strategy']}")
        risk_profile = {
            'skew_risk': 0.0,
            'latency_risk': 0.0,
            'power_risk': 0.0,
            'yield_risk': 0.0,
            'thermal_risk': 0.0 # New: Thermal risk for clock network
        }
        
        strategy = parameters['strategy']
        num_clock_sinks = len(clock_sinks)
        num_variation_sensitive = len(variation_sensitive_nodes)

        # Skew risk assessment
        if strategy == 'h_tree':
            skew_risk = 0.1 + (num_variation_sensitive / num_clock_sinks * 0.1 if num_clock_sinks > 0 else 0) # Slightly increases with sensitive nodes
        elif strategy == 'balanced_tree':
            skew_risk = 0.2 + (num_variation_sensitive / num_clock_sinks * 0.15 if num_clock_sinks > 0 else 0)
        elif strategy == 'fishbone':
            skew_risk = 0.3 + (num_variation_sensitive / num_clock_sinks * 0.2 if num_clock_sinks > 0 else 0)
        elif strategy == 'spine':
            skew_risk = 0.4 + (num_variation_sensitive / num_clock_sinks * 0.25 if num_clock_sinks > 0 else 0)
        else:  # custom_topology
            skew_risk = 0.35 + (num_variation_sensitive / num_clock_sinks * 0.3 if num_clock_sinks > 0 else 0) # Potentially higher if not well controlled
        risk_profile['skew_risk'] = min(skew_risk, 1.0) # Cap at 1.0
        
        # Latency risk assessment
        if num_clock_sinks > 20000:
            latency_risk = 0.7  # High risk for very large designs
        elif num_clock_sinks > 5000:
            latency_risk = 0.5  # Medium risk
        else:
            latency_risk = 0.3  # Low risk
        
        # H-tree and spine might have slightly higher base latency
        if strategy in ['h_tree', 'spine']:
            latency_risk += 0.05
        
        risk_profile['latency_risk'] = min(latency_risk, 1.0)
        
        # Power risk (clock networks typically consume significant power)
        # More complex trees (H-tree, Custom) generally consume more power
        base_power_risk = 0.6
        if strategy in ['h_tree', 'custom_topology']:
            power_risk = base_power_risk + 0.15
        elif strategy in ['balanced_tree', 'fishbone']:
            power_risk = base_power_risk - 0.1
        else:
            power_risk = base_power_risk
        risk_profile['power_risk'] = min(power_risk, 1.0)
        
        # Yield risk (complex clock trees can have manufacturing issues)
        # Custom topology might be higher risk, balanced tree lower
        base_yield_risk = 0.4
        if strategy == 'custom_topology':
            yield_risk = base_yield_risk + 0.15
        elif strategy == 'balanced_tree':
            yield_risk = base_yield_risk - 0.1
        else:
            yield_risk = base_yield_risk
        risk_profile['yield_risk'] = min(yield_risk, 1.0)

        # Thermal risk (clock buffers can be hotspots)
        base_thermal_risk = 0.3
        if strategy in ['h_tree', 'spine', 'custom_topology']: # More buffers/complexity
            thermal_risk = base_thermal_risk + 0.15
        else:
            thermal_risk = base_thermal_risk
        risk_profile['thermal_risk'] = min(thermal_risk, 1.0)
        
        return risk_profile
    
    def _calculate_clock_cost_vector(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Calculate the cost vector (PPA impact) of the clock proposal"""
        self.logger.debug(f"Calculating cost vector for strategy: {parameters['strategy']}")
        strategy = parameters['strategy']
        
        if strategy == 'h_tree':
            return {
                'power_impact': 0.15,      # High power due to symmetric buffers
                'performance_impact': -0.2, # Excellent performance due to low skew
                'area_impact': 0.1,        # High area due to symmetric structure
                'yield_impact': -0.05,     # Good yield due to regular structure
                'schedule_impact': 0.05,   # Moderate runtime impact
                'thermal_impact': 0.1      # Higher thermal impact due to buffer density
            }
        elif strategy == 'balanced_tree':
            return {
                'power_impact': 0.08,      # Moderate power impact
                'performance_impact': -0.15, # Good performance
                'area_impact': 0.05,       # Moderate area impact
                'yield_impact': -0.03,     # Good yield
                'schedule_impact': -0.05,  # Faster runtime
                'thermal_impact': 0.05     # Moderate thermal impact
            }
        elif strategy == 'fishbone':
            return {
                'power_impact': 0.06,      # Lower power than H-tree
                'performance_impact': -0.12, # Good performance
                'area_impact': 0.04,       # Lower area than H-tree
                'yield_impact': -0.04,     # Good yield
                'schedule_impact': -0.03,  # Fast runtime
                'thermal_impact': 0.04     # Moderate thermal impact
            }
        elif strategy == 'spine':
            return {
                'power_impact': 0.1,       # Higher power due to spine structure
                'performance_impact': -0.08, # Moderate performance
                'area_impact': 0.03,       # Efficient area usage
                'yield_impact': -0.03,     # Good yield
                'schedule_impact': 0.08,   # Moderate runtime
                'thermal_impact': 0.08     # Higher thermal impact
            }
        else:  # custom_topology
            return {
                'power_impact': 0.2,       # Highest power due to complexity
                'performance_impact': -0.18, # Potentially excellent performance
                'area_impact': 0.15,       # Highest area due to complexity
                'yield_impact': 0.05,      # Variable yield, potentially lower
                'schedule_impact': 0.1,    # Longest runtime
                'thermal_impact': 0.12     # High thermal impact
            }
    
    def _predict_clock_outcome(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Predict the outcome of applying this clock tree synthesis"""
        self.logger.debug(f"Predicting outcome for strategy: {parameters['strategy']}")
        strategy = parameters['strategy']
        
        if strategy == 'h_tree':
            return {
                'expected_skew_reduction': 0.8,  # 80% improvement
                'expected_latency_improvement': 0.6,  # 60% improvement
                'expected_power_increase': 0.15,  # 15% increase (trade-off)
                'expected_yield_improvement': 0.2,  # 20% improvement
                'expected_thermal_impact_reduction': 0.05 # Small thermal reduction due to regularity
            }
        elif strategy == 'balanced_tree':
            return {
                'expected_skew_reduction': 0.6,
                'expected_latency_improvement': 0.5,
                'expected_power_increase': 0.08,
                'expected_yield_improvement': 0.15,
                'expected_thermal_impact_reduction': 0.03
            }
        elif strategy == 'fishbone':
            return {
                'expected_skew_reduction': 0.5,
                'expected_latency_improvement': 0.55,
                'expected_power_increase': 0.06,
                'expected_yield_improvement': 0.18,
                'expected_thermal_impact_reduction': 0.04
            }
        elif strategy == 'spine':
            return {
                'expected_skew_reduction': 0.4,
                'expected_latency_improvement': 0.45,
                'expected_power_increase': 0.12,
                'expected_yield_improvement': 0.12,
                'expected_thermal_impact_reduction': 0.06
            }
        else:  # custom_topology
            return {
                'expected_skew_reduction': 0.7,
                'expected_latency_improvement': 0.7,
                'expected_power_increase': 0.2,
                'expected_yield_improvement': 0.1,
                'expected_thermal_impact_reduction': 0.07 # Can be good if designed for it
            }
    
    def _calculate_confidence(self, graph: CanonicalSiliconGraph) -> float:
        """Calculate confidence in the clock proposal"""
        # Confidence based on design characteristics and agent performance
        base_confidence = 0.9  # Clock agent typically has high confidence
        
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
        """Evaluate the potential impact of a clock tree proposal"""
        impact = {
            'skew_reduction': 0.0,
            'latency_improvement': 0.0,
            'power_efficiency': 0.0,
            'yield_improvement': 0.0
        }
        
        # Based on the strategy, estimate improvements
        strategy = proposal.parameters.get('strategy', 'balanced_tree')
        
        if strategy == 'h_tree':
            impact['skew_reduction'] = 0.8
            impact['latency_improvement'] = 0.6
            impact['yield_improvement'] = 0.2
        elif strategy == 'balanced_tree':
            impact['skew_reduction'] = 0.6
            impact['latency_improvement'] = 0.5
            impact['yield_improvement'] = 0.15
        elif strategy == 'fishbone':
            impact['skew_reduction'] = 0.5
            impact['latency_improvement'] = 0.55
            impact['yield_improvement'] = 0.18
        elif strategy == 'spine':
            impact['skew_reduction'] = 0.4
            impact['latency_improvement'] = 0.45
            impact['yield_improvement'] = 0.12
        else:  # custom_topology
            impact['skew_reduction'] = 0.7
            impact['latency_improvement'] = 0.7
            impact['yield_improvement'] = 0.1
        
        return impact