"""
Yield Agent - Specialized agent for yield and manufacturability optimization

This agent focuses on reducing random defect sensitivity and improving
manufacturability at advanced process nodes.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from agents.base_agent import BaseAgent, AgentType, AgentProposal
from core.canonical_silicon_graph import CanonicalSiliconGraph
from utils.logger import get_logger


class YieldAgent(BaseAgent):
    """
    Yield Agent - optimizes for yield and manufacturability
    """

    def __init__(self):
        super().__init__(AgentType.YIELD)
        self.logger = get_logger(f"{__name__}.yield_agent")
        self.yield_strategies = [
            'defect_aware_placement', 'guard_ring_protection', 'spacing_enhancement',
            'redundancy_insertion', 'statistical_buffering'
        ]

    def propose_action(self, graph: CanonicalSiliconGraph) -> Optional['AgentProposal']:
        """
        Generate a yield optimization proposal based on the current graph state
        """
        self.logger.info("Generating yield optimization proposal")

        # Identify yield-critical nodes and structures
        yield_critical_nodes = self._identify_yield_critical_nodes(graph)
        if not yield_critical_nodes:
            # Find nodes with high timing criticality or in sensitive regions
            all_nodes = [(n, graph.graph.nodes[n].get('timing_criticality', 0.0))
                        for n in graph.graph.nodes()
                        if graph.graph.nodes[n].get('timing_criticality', 0.0) > 0.5]
            all_nodes.sort(key=lambda x: x[1], reverse=True)
            yield_critical_nodes = [n for n, crit in all_nodes[:15]]  # Top 15 critical nodes

        if not yield_critical_nodes:
            self.logger.debug("No yield-critical nodes found, skipping yield proposal")
            return None

        # Analyze manufacturing sensitivity
        defect_sensitivity_zones = self._analyze_defect_sensitivity(graph, yield_critical_nodes)
        process_variation_zones = self._analyze_process_variation_sensitivity(graph, yield_critical_nodes)
        thermal_sensitivity_zones = self._analyze_thermal_sensitivity(graph, yield_critical_nodes)

        # Select optimal yield strategy
        strategy = self._select_yield_strategy(
            defect_sensitivity_zones, process_variation_zones, thermal_sensitivity_zones
        )

        # Generate yield optimization parameters
        parameters = self._generate_yield_parameters(
            strategy, defect_sensitivity_zones, process_variation_zones, thermal_sensitivity_zones
        )

        # Calculate risk profile
        risk_profile = self._assess_yield_risk(
            parameters, defect_sensitivity_zones, process_variation_zones, thermal_sensitivity_zones
        )

        # Calculate cost vector (PPA impact)
        cost_vector = self._calculate_yield_cost_vector(parameters)

        # Estimate outcome
        predicted_outcome = self._predict_yield_outcome(parameters)

        proposal = AgentProposal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            proposal_id=f"yd_{np.random.randint(1000, 9999)}",
            timestamp=self._get_timestamp(),
            action_type="optimize_yield",
            targets=yield_critical_nodes,
            parameters=parameters,
            confidence_score=self._calculate_confidence(graph),
            risk_profile=risk_profile,
            cost_vector=cost_vector,
            predicted_outcome=predicted_outcome,
            dependencies=[],
            conflicts_with=[]
        )

        self.logger.info(f"Generated yield optimization proposal for {len(yield_critical_nodes)} nodes")
        return proposal

    def _identify_yield_critical_nodes(self, graph: CanonicalSiliconGraph) -> List[str]:
        """Identify nodes that are critical for yield"""
        self.logger.debug("Identifying yield critical nodes.")
        yield_critical_nodes = []

        all_timing_criticalities = [attrs.get('timing_criticality', 0.0) for _, attrs in graph.graph.nodes(data=True)]
        avg_timing_criticality = np.mean(all_timing_criticalities) if all_timing_criticalities else 0.0

        # Dynamic threshold for timing criticality: e.g., 1.2x average
        timing_criticality_threshold = max(0.6, avg_timing_criticality * 1.2)

        for node, attrs in graph.graph.nodes(data=True):
            # Factors that make nodes yield-critical
            timing_criticality = attrs.get('timing_criticality', 0.0)
            region = attrs.get('region', 'default')
            cell_type = attrs.get('cell_type', '').lower()

            # High timing criticality makes nodes yield-sensitive
            if timing_criticality > timing_criticality_threshold:
                yield_critical_nodes.append(node)
                self.logger.debug(f"Node '{node}' is yield-critical due to high timing criticality ({timing_criticality:.2f}).")
            # Memory or analog cells are often yield-sensitive
            elif any(keyword in cell_type for keyword in ['ram', 'dac', 'adc', 'pll', 'dll', 'serdes']):
                yield_critical_nodes.append(node)
                self.logger.debug(f"Node '{node}' is yield-critical due to cell type '{cell_type}'.")
            # Cells in sensitive regions
            elif region in ['io', 'pll_zone', 'analog']:
                yield_critical_nodes.append(node)
                self.logger.debug(f"Node '{node}' is yield-critical due to region '{region}'.")

        return list(set(yield_critical_nodes)) # Remove duplicates

    def _analyze_defect_sensitivity(self, graph: CanonicalSiliconGraph,
                                  yield_critical_nodes: List[str]) -> List[Dict[str, Any]]:
        """Analyze sensitivity to random defects"""
        self.logger.debug("Analyzing defect sensitivity.")
        defect_sensitivity_zones = []

        all_defect_sensitivities = []
        # First pass to calculate raw sensitivities for normalization
        for node in yield_critical_nodes:
            if node in graph.graph.nodes():
                attrs = graph.graph.nodes[node]
                timing_criticality = attrs.get('timing_criticality', 0.0)
                region = attrs.get('region', 'default')
                cell_type = attrs.get('cell_type', '').lower()
                fanout = len(list(graph.graph.successors(node)))

                defect_sensitivity = timing_criticality
                if any(keyword in cell_type for keyword in ['ram', 'dac', 'adc', 'pll', 'dll']):
                    defect_sensitivity *= 1.5
                if region == 'io':
                    defect_sensitivity *= 1.3
                if fanout > 10:
                    defect_sensitivity *= 1.2
                all_defect_sensitivities.append(defect_sensitivity)

        max_defect_sensitivity = max(all_defect_sensitivities) if all_defect_sensitivities else 0.001 # Avoid div by zero

        defect_sensitivity_zones = []
        for node in yield_critical_nodes:
            if node in graph.graph.nodes():
                attrs = graph.graph.nodes[node]
                timing_criticality = attrs.get('timing_criticality', 0.0)
                region = attrs.get('region', 'default')
                cell_type = attrs.get('cell_type', '').lower()
                fanout = len(list(graph.graph.successors(node)))

                defect_sensitivity = timing_criticality
                if any(keyword in cell_type for keyword in ['ram', 'dac', 'adc', 'pll', 'dll']):
                    defect_sensitivity *= 1.5
                if region == 'io':
                    defect_sensitivity *= 1.3
                if fanout > 10:
                    defect_sensitivity *= 1.2

                normalized_sensitivity = defect_sensitivity / max_defect_sensitivity

                if normalized_sensitivity > 0.6:  # Normalized threshold
                    defect_sensitivity_zones.append({
                        'node': node,
                        'sensitivity_score': normalized_sensitivity,
                        'region': region,
                        'cell_type': cell_type,
                        'timing_criticality': timing_criticality
                    })
                    self.logger.debug(f"Node '{node}' identified as defect sensitive (normalized score: {normalized_sensitivity:.2f}).")

        # Sort by sensitivity score
        defect_sensitivity_zones.sort(key=lambda x: x['sensitivity_score'], reverse=True)
        return defect_sensitivity_zones
    
    def _analyze_process_variation_sensitivity(self, graph: CanonicalSiliconGraph,
                                             yield_critical_nodes: List[str]) -> List[Dict[str, Any]]:
        """Analyze sensitivity to process variations"""
        self.logger.debug("Analyzing process variation sensitivity.")
        variation_sensitivity_zones = []

        all_variation_sensitivities = []
        # First pass to calculate raw sensitivities for normalization
        for node in yield_critical_nodes:
            if node in graph.graph.nodes():
                attrs = graph.graph.nodes[node]
                timing_criticality = attrs.get('timing_criticality', 0.0)
                region = attrs.get('region', 'default')
                cell_type = attrs.get('cell_type', '').lower()

                variation_sensitivity = timing_criticality
                if any(keyword in cell_type for keyword in ['dac', 'adc', 'pll', 'dll', 'comp', 'opamp']):
                    variation_sensitivity *= 1.8
                elif 'dff' in cell_type or 'ff' in cell_type:
                    variation_sensitivity *= 1.4
                if region in ['io', 'periphery']:
                    variation_sensitivity *= 1.2
                all_variation_sensitivities.append(variation_sensitivity)

        max_variation_sensitivity = max(all_variation_sensitivities) if all_variation_sensitivities else 0.001 # Avoid div by zero

        variation_sensitivity_zones = []
        for node in yield_critical_nodes:
            if node in graph.graph.nodes():
                attrs = graph.graph.nodes[node]
                timing_criticality = attrs.get('timing_criticality', 0.0)
                region = attrs.get('region', 'default')
                cell_type = attrs.get('cell_type', '').lower()

                variation_sensitivity = timing_criticality
                if any(keyword in cell_type for keyword in ['dac', 'adc', 'pll', 'dll', 'comp', 'opamp']):
                    variation_sensitivity *= 1.8
                elif 'dff' in cell_type or 'ff' in cell_type:
                    variation_sensitivity *= 1.4
                if region in ['io', 'periphery']:
                    variation_sensitivity *= 1.2

                normalized_sensitivity = variation_sensitivity / max_variation_sensitivity

                if normalized_sensitivity > 0.5:  # Normalized threshold
                    variation_sensitivity_zones.append({
                        'node': node,
                        'sensitivity_score': normalized_sensitivity,
                        'region': region,
                        'cell_type': cell_type,
                        'timing_criticality': timing_criticality
                    })
                    self.logger.debug(f"Node '{node}' identified as variation sensitive (normalized score: {normalized_sensitivity:.2f}).")

        # Sort by sensitivity score
        variation_sensitivity_zones.sort(key=lambda x: x['sensitivity_score'], reverse=True)
        return variation_sensitivity_zones

    def _analyze_thermal_sensitivity(self, graph: CanonicalSiliconGraph,
                                   yield_critical_nodes: List[str]) -> List[Dict[str, Any]]:
        """Analyze sensitivity to thermal effects"""
        self.logger.debug("Analyzing thermal sensitivity.")
        thermal_sensitivity_zones = []

        all_thermal_sensitivities = []
        # First pass to calculate raw sensitivities for normalization
        for node in yield_critical_nodes:
            if node in graph.graph.nodes():
                attrs = graph.graph.nodes[node]
                power = attrs.get('power', 0.0)
                timing_criticality = attrs.get('timing_criticality', 0.0)
                region = attrs.get('region', 'default')
                cell_type = attrs.get('cell_type', '').lower()

                thermal_sensitivity = power * (1 + timing_criticality)
                if any(keyword in cell_type for keyword in ['dac', 'adc', 'pll', 'dll', 'refgen']):
                    thermal_sensitivity *= 1.6
                if region in ['high_power', 'compute']:
                    thermal_sensitivity *= 1.3
                all_thermal_sensitivities.append(thermal_sensitivity)

        max_thermal_sensitivity = max(all_thermal_sensitivities) if all_thermal_sensitivities else 0.001 # Avoid div by zero

        thermal_sensitivity_zones = []
        for node in yield_critical_nodes:
            if node in graph.graph.nodes():
                attrs = graph.graph.nodes[node]
                power = attrs.get('power', 0.0)
                timing_criticality = attrs.get('timing_criticality', 0.0)
                region = attrs.get('region', 'default')
                cell_type = attrs.get('cell_type', '').lower()

                thermal_sensitivity = power * (1 + timing_criticality)
                if any(keyword in cell_type for keyword in ['dac', 'adc', 'pll', 'dll', 'refgen']):
                    thermal_sensitivity *= 1.6
                if region in ['high_power', 'compute']:
                    thermal_sensitivity *= 1.3

                normalized_sensitivity = thermal_sensitivity / max_thermal_sensitivity

                if normalized_sensitivity > 0.5:  # Normalized threshold
                    thermal_sensitivity_zones.append({
                        'node': node,
                        'sensitivity_score': normalized_sensitivity,
                        'region': region,
                        'cell_type': cell_type,
                        'power': power,
                        'timing_criticality': timing_criticality
                    })
                    self.logger.debug(f"Node '{node}' identified as thermal sensitive (normalized score: {normalized_sensitivity:.2f}).")

        # Sort by sensitivity score
        thermal_sensitivity_zones.sort(key=lambda x: x['sensitivity_score'], reverse=True)
        return thermal_sensitivity_zones
    
    def _select_yield_strategy(self, defect_zones: List[Dict],
                             variation_zones: List[Dict],
                             thermal_zones: List[Dict]) -> str:
        """Select the optimal yield strategy based on sensitivity analysis"""
        self.logger.debug("Selecting optimal yield strategy.")
        num_defect_sensitive = len(defect_zones)
        num_variation_sensitive = len(variation_zones)
        num_thermal_sensitive = len(thermal_zones)

        # Prioritize based on the most prevalent or severe yield issues
        if num_defect_sensitive > 10 or (num_defect_sensitive > 5 and num_variation_sensitive > 5):
            # High number of nodes sensitive to random defects, or combination with PV
            self.logger.debug(f"Selecting 'defect_aware_placement' due to high defect sensitivity ({num_defect_sensitive}).")
            return 'defect_aware_placement' # Focus on layout to reduce defect impact

        elif num_variation_sensitive > 15:
            # Many nodes sensitive to process variations
            self.logger.debug(f"Selecting 'statistical_buffering' due to many variation sensitive nodes ({num_variation_sensitive}).")
            return 'statistical_buffering' # Addresses PV directly through robust buffering

        elif num_thermal_sensitive > 8:
            # Significant thermal sensitivity, often requires physical isolation
            self.logger.debug(f"Selecting 'guard_ring_protection' due to high thermal sensitivity ({num_thermal_sensitive}).")
            return 'guard_ring_protection' # Isolates critical areas from thermal stress

        elif num_defect_sensitive > 3 or num_variation_sensitive > 5:
            # Moderate defect or variation sensitivity, spacing can help
            self.logger.debug(f"Selecting 'spacing_enhancement' for moderate defect/variation sensitivity.")
            return 'spacing_enhancement' # Simple layout rule changes to improve yield

        else:
            # Default to redundancy for general yield improvement or if specific issues are less severe
            self.logger.debug(f"Selecting 'redundancy_insertion' as a general yield improvement or default.")
            return 'redundancy_insertion' # Adds robustness through redundant elements
    
    def _generate_yield_parameters(self, strategy: str, defect_zones: List[Dict],
                                 variation_zones: List[Dict],
                                 thermal_zones: List[Dict]) -> Dict[str, Any]:
        """Generate yield optimization parameters based on strategy"""
        self.logger.debug(f"Generating yield parameters for strategy: {strategy}")
        parameters = {
            'strategy': strategy,
            'defect_mitigation': {},
            'variation_mitigation': {},
            'thermal_mitigation': {},
            'layout_hardening': {},
            'redundancy_config': {},
            'common_techniques': {} # Added here from previous location
        }

        num_defect_sensitive = len(defect_zones)
        num_variation_sensitive = len(variation_zones)
        num_thermal_sensitive = len(thermal_zones)

        if strategy == 'defect_aware_placement':
            parameters['defect_mitigation'] = {
                'avoid_edge_placement': True,
                'minimum_spacing_rules': 'enhanced',
                'sensitive_cell_shielding': [dz['node'] for dz in defect_zones[:min(num_defect_sensitive, 5)]],  # Top N sensitive
                'defect_immune_placement': True,
                'max_routing_layers_for_sensitive_nets': 3 # Use fewer layers
            }
        elif strategy == 'guard_ring_protection':
            parameters['thermal_mitigation'] = { # thermal_mitigation moved from main parameters
                'guard_ring_insertion': [tz['node'] for tz in thermal_zones[:min(num_thermal_sensitive, 3)]],  # Top N thermal
                'isolation_wells': True,
                'thermal_via_ladders': True,
                'heat_spreading_structures': True,
                'guard_ring_spacing_um': 0.5 # Example spacing
            }
        elif strategy == 'spacing_enhancement':
            parameters['defect_mitigation'] = { # defect_mitigation moved from main parameters
                'enhanced_spacing_rules': True,
                'minimum_line_end_spacing_um': 0.18,  # Increased for yield
                'minimum_poly_spacing_um': 0.15,
                'minimum_active_spacing_um': 0.12,
                'critical_net_spacing_multiplier': 1.5 # Extra spacing for critical nets
            }
        elif strategy == 'redundancy_insertion':
            parameters['redundancy_config'] = { # redundancy_config moved from main parameters
                'dual_sequential_cells': [vz['node'] for vz in variation_zones[:min(num_variation_sensitive, 5)]
                                        if 'dff' in vz.get('cell_type', '')],  # Only FFs, top N sensitive
                'redundant_routing': True,
                'backup_power_rails': True,
                'error_correction_codes': {'enabled': True, 'level': 'medium'}
            }
        elif strategy == 'statistical_buffering':
            parameters['variation_mitigation'] = { # variation_mitigation moved from main parameters
                'statistical_timing_buffers': [vz['node'] for vz in variation_zones[:min(num_variation_sensitive, 5)]],
                'process_variation_aware_sizing': True,
                'robust_circuit_techniques': True,
                'adaptive_biasing': True,
                'buffer_strength_variation_range': 0.1 # Example range
            }

        # Layout hardening measures (apply regardless of strategy)
        parameters['layout_hardening'] = {
            'drc_enhancement': True,
            'lvs_verification_enhancement': True,
            'stress_migration_awareness': True,
            'antenna_rule_enhancement': True,
            'fat_metal_for_power_nets': True
        }

        # Common yield enhancement techniques
        parameters['common_techniques'] = {
            'optical_proximity_correction_enabled': True,
            'resolution_enhancement_techniques_enabled': True,
            'design_for_testability_level': 'high',
            'built_in_self_test_enabled': True,
            'critical_area_analysis_enabled': True
        }

        return parameters

    def _assess_yield_risk(self, parameters: Dict[str, Any],
                          defect_zones: List[Dict],
                          variation_zones: List[Dict],
                          thermal_zones: List[Dict]) -> Dict[str, float]:
        """Assess the risk profile of the proposed yield optimization"""
        self.logger.debug(f"Assessing yield risk for strategy: {parameters['strategy']}")
        risk_profile = {
            'defect_risk': 0.0,
            'variation_risk': 0.0,
            'thermal_risk': 0.0,
            'area_overhead_risk': 0.0,
            'performance_degradation_risk': 0.0,
            'schedule_impact_risk': 0.0 # New: Schedule impact risk
        }

        strategy = parameters['strategy']
        num_defect_sensitive = len(defect_zones)
        num_variation_sensitive = len(variation_zones)
        num_thermal_sensitive = len(thermal_zones)

        # Defect risk assessment
        if strategy == 'defect_aware_placement':
            defect_risk = min(num_defect_sensitive / 20.0, 0.3) # Good at mitigating
        elif strategy == 'spacing_enhancement':
            defect_risk = min(num_defect_sensitive / 15.0, 0.4) # Also good
        else:
            defect_risk = min(num_defect_sensitive / 10.0, 0.6) # Higher base risk
        risk_profile['defect_risk'] = defect_risk

        # Variation risk assessment
        if strategy == 'statistical_buffering':
            variation_risk = min(num_variation_sensitive / 25.0, 0.3) # Good at mitigating
        elif strategy == 'redundancy_insertion':
            variation_risk = min(num_variation_sensitive / 20.0, 0.4) # Helps with hard failures from variation
        else:
            variation_risk = min(num_variation_sensitive / 15.0, 0.6) # Higher base risk
        risk_profile['variation_risk'] = variation_risk

        # Thermal risk assessment (indirectly, via yield)
        if strategy == 'guard_ring_protection':
            thermal_risk = min(num_thermal_sensitive / 15.0, 0.3) # Good for thermal isolation
        else:
            thermal_risk = min(num_thermal_sensitive / 10.0, 0.5) # Higher base risk
        risk_profile['thermal_risk'] = thermal_risk

        # Area overhead risk from yield enhancements
        if strategy == 'redundancy_insertion':
            area_risk = 0.7 # Highest area overhead
        elif strategy == 'guard_ring_protection':
            area_risk = 0.6
        elif strategy == 'spacing_enhancement':
            area_risk = 0.4
        else:
            area_risk = 0.2 # Lower area overhead
        risk_profile['area_overhead_risk'] = area_risk

        # Performance degradation risk from yield enhancements
        if strategy == 'redundancy_insertion':
            perf_risk = 0.5 # Redundancy can impact performance
        elif strategy == 'statistical_buffering':
            perf_risk = 0.3 # Buffering can have some impact
        else:
            perf_risk = 0.1 # Minimal impact
        risk_profile['performance_degradation_risk'] = perf_risk

        # Schedule impact risk
        if strategy in ['redundancy_insertion', 'statistical_buffering']:
            schedule_risk = 0.7 # Complex implementation
        elif strategy == 'spacing_enhancement':
            schedule_risk = 0.2 # Easier to implement
        else:
            schedule_risk = 0.4 # Moderate
        risk_profile['schedule_impact_risk'] = schedule_risk

        return risk_profile

    def _calculate_yield_cost_vector(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Calculate the cost vector (PPA impact) of the yield proposal"""
        self.logger.debug(f"Calculating cost vector for strategy: {parameters['strategy']}")
        strategy = parameters['strategy']

        if strategy == 'redundancy_insertion':
            return {
                'power_impact': 0.1,       # High power due to redundancy
                'performance_impact': 0.05, # Moderate performance impact (delay due to redundant logic)
                'area_impact': 0.15,       # High area cost
                'yield_impact': -0.15,     # Excellent yield improvement
                'schedule_impact': 0.1     # Higher schedule impact (complexity)
            }
        elif strategy == 'guard_ring_protection':
            return {
                'power_impact': 0.05,      # Moderate power impact
                'performance_impact': 0.02, # Low performance impact
                'area_impact': 0.12,       # High area cost for guard rings
                'yield_impact': -0.12,     # Good yield improvement
                'schedule_impact': 0.08    # Moderate schedule impact
            }
        elif strategy == 'spacing_enhancement':
            return {
                'power_impact': 0.02,      # Low power impact
                'performance_impact': 0.01, # Very low performance impact
                'area_impact': 0.08,       # Moderate area cost
                'yield_impact': -0.08,     # Good yield improvement
                'schedule_impact': 0.03    # Low schedule impact
            }
        elif strategy == 'defect_aware_placement':
            return {
                'power_impact': 0.03,      # Low power impact
                'performance_impact': 0.02, # Low performance impact
                'area_impact': 0.05,       # Low area cost
                'yield_impact': -0.1,      # Good yield improvement
                'schedule_impact': 0.06    # Moderate schedule impact
            }
        else:  # statistical_buffering
            return {
                'power_impact': 0.08,      # Moderate power impact
                'performance_impact': 0.03, # Low performance impact
                'area_impact': 0.06,       # Low area cost
                'yield_impact': -0.13,     # Excellent yield improvement
                'schedule_impact': 0.12    # Higher schedule impact for complex buffering
            }

    def _predict_yield_outcome(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Predict the outcome of applying this yield optimization"""
        strategy = parameters['strategy']

        if strategy == 'redundancy_insertion':
            return {
                'expected_yield_improvement': 0.15,  # 15% yield improvement
                'expected_defect_reduction': 0.12,  # 12% defect reduction
                'expected_variation_tolerance': 0.18,  # 18% improvement in variation tolerance
                'expected_thermal_stability': 0.08,  # 8% improvement in thermal stability
                'expected_area_increase': 0.12  # 12% area increase
            }
        elif strategy == 'guard_ring_protection':
            return {
                'expected_yield_improvement': 0.12,  # 12% yield improvement
                'expected_defect_reduction': 0.08,  # 8% defect reduction
                'expected_variation_tolerance': 0.10,  # 10% improvement in variation tolerance
                'expected_thermal_stability': 0.20,  # 20% improvement in thermal stability
                'expected_area_increase': 0.10  # 10% area increase
            }
        elif strategy == 'spacing_enhancement':
            return {
                'expected_yield_improvement': 0.08,  # 8% yield improvement
                'expected_defect_reduction': 0.15,  # 15% defect reduction
                'expected_variation_tolerance': 0.05,  # 5% improvement in variation tolerance
                'expected_thermal_stability': 0.05,  # 5% improvement in thermal stability
                'expected_area_increase': 0.08  # 8% area increase
            }
        elif strategy == 'defect_aware_placement':
            return {
                'expected_yield_improvement': 0.10,  # 10% yield improvement
                'expected_defect_reduction': 0.20,  # 20% defect reduction
                'expected_variation_tolerance': 0.08,  # 8% improvement in variation tolerance
                'expected_thermal_stability': 0.03,  # 3% improvement in thermal stability
                'expected_area_increase': 0.05  # 5% area increase
            }
        else:  # statistical_buffering
            return {
                'expected_yield_improvement': 0.18,  # 18% yield improvement
                'expected_defect_reduction': 0.10,  # 10% defect reduction
                'expected_variation_tolerance': 0.25,  # 25% improvement in variation tolerance
                'expected_thermal_stability': 0.12,  # 12% improvement in thermal stability
                'expected_area_increase': 0.06  # 6% area increase
            }

    def _calculate_confidence(self, graph: CanonicalSiliconGraph) -> float:
        """Calculate confidence in the yield proposal"""
        # Confidence based on design characteristics and agent performance
        base_confidence = 0.75  # Yield optimization typically has moderate confidence

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
        """Evaluate the potential impact of a yield optimization proposal"""
        impact = {
            'yield_improvement': 0.0,
            'defect_reduction': 0.0,
            'variation_tolerance': 0.0,
            'thermal_stability': 0.0,
            'area_overhead': 0.0
        }

        # Based on the strategy, estimate improvements
        strategy = proposal.parameters.get('strategy', 'defect_aware_placement')

        if strategy == 'redundancy_insertion':
            impact['yield_improvement'] = 0.15
            impact['defect_reduction'] = 0.12
            impact['variation_tolerance'] = 0.18
            impact['thermal_stability'] = 0.08
            impact['area_overhead'] = 0.12
        elif strategy == 'guard_ring_protection':
            impact['yield_improvement'] = 0.12
            impact['defect_reduction'] = 0.08
            impact['variation_tolerance'] = 0.10
            impact['thermal_stability'] = 0.20
            impact['area_overhead'] = 0.10
        elif strategy == 'spacing_enhancement':
            impact['yield_improvement'] = 0.08
            impact['defect_reduction'] = 0.15
            impact['variation_tolerance'] = 0.05
            impact['thermal_stability'] = 0.05
            impact['area_overhead'] = 0.08
        elif strategy == 'defect_aware_placement':
            impact['yield_improvement'] = 0.10
            impact['defect_reduction'] = 0.20
            impact['variation_tolerance'] = 0.08
            impact['thermal_stability'] = 0.03
            impact['area_overhead'] = 0.05
        else:  # statistical_buffering
            impact['yield_improvement'] = 0.18
            impact['defect_reduction'] = 0.10
            impact['variation_tolerance'] = 0.25
            impact['thermal_stability'] = 0.12
            impact['area_overhead'] = 0.06

        return impact