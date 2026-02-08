"""
Thermal Agent - Specialized agent for thermal optimization

This agent focuses on thermal management and heat dissipation optimization
to ensure reliable operation and prevent thermal issues at advanced nodes.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from agents.base_agent import BaseAgent, AgentType, AgentProposal
from core.canonical_silicon_graph import CanonicalSiliconGraph
from utils.logger import get_logger


class ThermalAgent(BaseAgent):
    """
    Thermal Agent - thermal management and heat dissipation optimization
    """
    
    def __init__(self):
        super().__init__(AgentType.POWER)  # Using POWER type since thermal relates to power
        self.logger = get_logger(f"{__name__}.thermal_agent")
        self.thermal_strategies = [
            'hotspot_aware_placement', 'thermal_via_insertion', 'power_binning',
            'thermal_guard_ring', 'adaptive_body_biasing', 'clock_frequency_scaling'
        ]
    
    def propose_action(self, graph: CanonicalSiliconGraph) -> Optional['AgentProposal']:
        """
        Generate a thermal optimization proposal based on the current graph state
        """
        self.logger.info("Generating thermal optimization proposal")
        
        # Identify thermal hotspots and critical thermal paths
        thermal_hotspots = self._identify_thermal_hotspots(graph)
        thermal_gradient_zones = self._identify_thermal_gradients(graph)
        thermal_sensitive_nodes = self._identify_thermal_sensitive_nodes(graph)
        
        if not thermal_hotspots and not thermal_sensitive_nodes:
            self.logger.debug("No thermal issues detected, skipping thermal proposal")
            return None
        
        # Select optimal thermal strategy
        strategy = self._select_thermal_strategy(
            thermal_hotspots, thermal_gradient_zones, thermal_sensitive_nodes
        )
        
        # Generate thermal optimization parameters
        parameters = self._generate_thermal_parameters(
            strategy, thermal_hotspots, thermal_gradient_zones, thermal_sensitive_nodes
        )
        
        # Calculate risk profile
        risk_profile = self._assess_thermal_risk(
            parameters, thermal_hotspots, thermal_gradient_zones, thermal_sensitive_nodes
        )
        
        # Calculate cost vector (PPA impact)
        cost_vector = self._calculate_thermal_cost_vector(parameters)
        
        # Estimate outcome
        predicted_outcome = self._predict_thermal_outcome(parameters)
        
        # Determine targets for the proposal
        targets = []
        for hotspot in thermal_hotspots[:5]:  # Limit to top 5 hotspots
            targets.extend(hotspot.get('nodes', []))
        for node in thermal_sensitive_nodes[:10]:  # Limit to top 10 sensitive nodes
            targets.append(node)
        
        proposal = AgentProposal(
            agent_id=self.agent_id,
            agent_type=AgentType.POWER,  # Using POWER type
            proposal_id=f"th_{np.random.randint(1000, 9999)}",
            timestamp=self._get_timestamp(),
            action_type="optimize_thermal",
            targets=list(set(targets)),  # Remove duplicates
            parameters=parameters,
            confidence_score=self._calculate_confidence(graph),
            risk_profile=risk_profile,
            cost_vector=cost_vector,
            predicted_outcome=predicted_outcome,
            dependencies=[],
            conflicts_with=[]
        )
        
        self.logger.info(f"Generated thermal optimization proposal for {len(set(targets))} nodes")
        return proposal
    
    def _identify_thermal_hotspots(self, graph: CanonicalSiliconGraph) -> List[Dict[str, Any]]:
        """Identify thermal hotspots in the design"""
        self.logger.debug("Identifying thermal hotspots.")
        hotspots = []
        
        # Group nodes by region and calculate thermal density
        region_power_and_area = {}
        total_design_power = 0.0
        total_design_area = 0.0

        for node, attrs in graph.graph.nodes(data=True):
            region = attrs.get('region', 'default')
            power = attrs.get('power', 0.0)
            area = attrs.get('area', 0.0)
            thermal_sensitivity = attrs.get('thermal_sensitivity', 0.0) # Assume 0.0 if not explicitly set

            total_design_power += power
            total_design_area += area
            
            if region not in region_power_and_area:
                region_power_and_area[region] = {'nodes': [], 'total_power': 0.0, 'total_area': 0.0, 'max_thermal_sensitivity': 0.0}
            
            region_power_and_area[region]['nodes'].append(node)
            region_power_and_area[region]['total_power'] += power
            region_power_and_area[region]['total_area'] += area
            region_power_and_area[region]['max_thermal_sensitivity'] = max(
                region_power_and_area[region]['max_thermal_sensitivity'], thermal_sensitivity
            )
        
        # Calculate average thermal density for the entire design
        overall_thermal_density = total_design_power / max(1.0, total_design_area) # Avoid division by zero
        
        # Identify regions with high thermal density
        for region, data in region_power_and_area.items():
            region_area = max(1.0, data['total_area']) # Avoid division by zero
            thermal_density = data['total_power'] / region_area

            # A region is a hotspot if its thermal density is significantly above average
            # or if it contains highly thermal-sensitive nodes.
            hotspot_threshold = overall_thermal_density * 1.5 # 1.5x average density
            sensitive_node_threshold = 0.6 # Max sensitivity in region above this is risky

            if thermal_density > hotspot_threshold or data['max_thermal_sensitivity'] > sensitive_node_threshold:
                hotspot_info = {
                    'region': region,
                    'nodes': data['nodes'],
                    'total_power': data['total_power'],
                    'thermal_density': thermal_density,
                    'thermal_sensitivity': data['max_thermal_sensitivity'],
                    'node_count': len(data['nodes'])
                }
                hotspots.append(hotspot_info)
                self.logger.debug(f"Identified hotspot in region '{region}' (density: {thermal_density:.2f}, max_sensitivity: {data['max_thermal_sensitivity']:.2f}).")
        
        # Sort by thermal impact (density * sensitivity)
        hotspots.sort(key=lambda x: x['thermal_density'] * x['thermal_sensitivity'], reverse=True)
        return hotspots
    
    def _identify_thermal_gradients(self, graph: CanonicalSiliconGraph) -> List[Dict[str, Any]]:
        """Identify areas with high thermal gradients"""
        self.logger.debug("Identifying thermal gradients.")
        gradients = []
        
        region_power = {}
        for node, attrs in graph.graph.nodes(data=True):
            region = attrs.get('region', 'default')
            power = attrs.get('power', 0.0)
            
            if region not in region_power:
                region_power[region] = {'total_power': 0.0, 'node_count': 0}
            region_power[region]['total_power'] += power
            region_power[region]['node_count'] += 1
        
        # Calculate average power per region
        region_avg_power = {r: data['total_power']/max(data['node_count'], 1) 
                           for r, data in region_power.items()}
        
        # Identify adjacent regions with high power difference (gradient)
        regions = list(region_avg_power.keys())
        all_power_diffs = []

        # Assuming a simplified adjacency model (all pairs for now)
        # In a real system, this would involve spatial adjacency checks
        for i in range(len(regions)):
            for j in range(i+1, len(regions)):
                power_diff = abs(region_avg_power[regions[i]] - region_avg_power[regions[j]])
                all_power_diffs.append(power_diff)
        
        avg_power_diff = np.mean(all_power_diffs) if all_power_diffs else 0.0
        gradient_threshold = avg_power_diff * 1.5 # Example: 1.5x average power difference is significant

        for i in range(len(regions)):
            for j in range(i+1, len(regions)):
                power_diff = abs(region_avg_power[regions[i]] - region_avg_power[regions[j]])
                
                if power_diff > gradient_threshold and gradient_threshold > 0:  # Dynamic threshold
                    gradient_severity = min(power_diff / (gradient_threshold * 2), 1.0)  # Normalize
                    gradients.append({
                        'region1': regions[i],
                        'region2': regions[j],
                        'power_difference': power_diff,
                        'gradient_severity': gradient_severity
                    })
                    self.logger.debug(f"Identified thermal gradient between '{regions[i]}' and '{regions[j]}' (severity: {gradient_severity:.2f}).")
        
        return gradients
    
    def _identify_thermal_sensitive_nodes(self, graph: CanonicalSiliconGraph) -> List[str]:
        """Identify nodes that are sensitive to thermal effects"""
        self.logger.debug("Identifying thermal sensitive nodes.")
        sensitive_nodes = []
        
        all_thermal_sensitivities = []
        # First pass to gather all sensitivity scores for normalization
        for node, attrs in graph.graph.nodes(data=True):
            power = attrs.get('power', 0.0)
            timing_criticality = attrs.get('timing_criticality', 0.0)
            cell_type = attrs.get('cell_type', '').lower()
            
            thermal_sensitivity = power * (1 + timing_criticality)
            if any(keyword in cell_type for keyword in ['dac', 'adc', 'pll', 'dll', 'comp', 'opamp', 'ref']):
                thermal_sensitivity *= 2.0
            elif 'dff' in cell_type or 'ff' in cell_type:
                thermal_sensitivity *= 1.5
            
            all_thermal_sensitivities.append(thermal_sensitivity)
        
        max_thermal_sensitivity = max(all_thermal_sensitivities) if all_thermal_sensitivities else 0.001 # Avoid div by zero

        for node, attrs in graph.graph.nodes(data=True):
            power = attrs.get('power', 0.0)
            timing_criticality = attrs.get('timing_criticality', 0.0)
            cell_type = attrs.get('cell_type', '').lower()
            
            thermal_sensitivity = power * (1 + timing_criticality)
            if any(keyword in cell_type for keyword in ['dac', 'adc', 'pll', 'dll', 'comp', 'opamp', 'ref']):
                thermal_sensitivity *= 2.0
            elif 'dff' in cell_type or 'ff' in cell_type:
                thermal_sensitivity *= 1.5
            
            normalized_thermal_sensitivity = thermal_sensitivity / max_thermal_sensitivity
            
            if normalized_thermal_sensitivity > 0.6 or attrs.get('thermal_sensitivity', 0.0) > 0.5: # Hybrid condition
                sensitive_nodes.append(node)
                self.logger.debug(f"Node '{node}' identified as thermal sensitive (normalized score: {normalized_thermal_sensitivity:.2f}).")
        
        return sensitive_nodes
    
    def _select_thermal_strategy(self, hotspots: List[Dict], gradients: List[Dict], 
                               sensitive_nodes: List[str]) -> str:
        """Select the optimal thermal strategy based on conditions"""
        self.logger.debug("Selecting optimal thermal strategy.")
        num_hotspots = len(hotspots)
        num_gradients = len(gradients)
        num_sensitive_nodes = len(sensitive_nodes)

        # Prioritize strategies based on the most severe or prevalent thermal issues
        if num_hotspots > 5 and num_gradients > 3:
            # Many hotspots and significant gradients suggest comprehensive solution
            self.logger.debug("Selecting 'hotspot_aware_placement' due to many hotspots and gradients.")
            return 'hotspot_aware_placement' # Spreading heat sources is often key

        elif num_gradients > 5 or (num_gradients > 2 and num_hotspots > 2):
            # High thermal gradients, often due to localized heat concentration
            self.logger.debug("Selecting 'thermal_via_insertion' due to significant thermal gradients.")
            return 'thermal_via_insertion' # Direct heat extraction through vias

        elif num_sensitive_nodes > 20 and num_hotspots > 0:
            # Many sensitive nodes in proximity to hotspots
            self.logger.debug("Selecting 'thermal_guard_ring' for protection of sensitive nodes near hotspots.")
            return 'thermal_guard_ring' # Isolates sensitive areas

        elif num_hotspots > 0 and (num_sensitive_nodes > 10 or num_gradients > 0):
            # General hotspot management with some sensitive nodes
            self.logger.debug("Selecting 'power_binning' for general hotspot management.")
            return 'power_binning' # Redistributes power/thermal load

        elif num_hotspots == 0 and num_gradients == 0 and num_sensitive_nodes > 0:
            # No major hotspots, but sensitive nodes could benefit from fine-grained control
            self.logger.debug("Selecting 'adaptive_body_biasing' for fine-grained thermal control of sensitive nodes.")
            return 'adaptive_body_biasing' # Proactive thermal management

        else:
            # Default or if issues are less severe, consider frequency scaling as a last resort performance trade-off
            self.logger.debug("Selecting 'clock_frequency_scaling' as a fallback or for less severe cases.")
            return 'clock_frequency_scaling'    
    def _generate_thermal_parameters(self, strategy: str, hotspots: List[Dict], 
                                   gradients: List[Dict], sensitive_nodes: List[str]) -> Dict[str, Any]:
        """Generate thermal optimization parameters based on strategy"""
        self.logger.debug(f"Generating thermal parameters for strategy: {strategy}")
        parameters = {
            'strategy': strategy,
            'hotspot_mitigation': {},
            'thermal_management': {},
            'power_optimization': {},
            'layout_modifications': {},
            'common_features': {}
        }
        
        num_hotspots = len(hotspots)
        num_gradients = len(gradients)
        num_sensitive_nodes = len(sensitive_nodes)

        if strategy == 'hotspot_aware_placement':
            parameters['hotspot_mitigation'] = {
                'spread_hot_blocks': True,
                'thermal_aware_placement': True,
                'hotspot_regions': [hs['region'] for hs in hotspots[:min(num_hotspots, 5)]],  # Top N hotspots
                'cooling_requirements_watt_per_sqmm': [hs['thermal_density'] * 1.5 for hs in hotspots[:min(num_hotspots, 5)]], # Scale with density
                'min_spacing_multiplier': 1.2 # Increase spacing for hot blocks
            }
        elif strategy == 'thermal_via_insertion':
            parameters['thermal_management'] = {
                'thermal_via_insertion': True,
                'via_placement_regions': [(g['region1'], g['region2']) for g in gradients[:min(num_gradients, 5)]], # Top N gradients
                'thermal_conductivity_enhancement': True,
                'via_density_multiplier': 1.5 + (num_gradients * 0.1) # Scale density with gradients
            }
        elif strategy == 'power_binning':
            parameters['power_optimization'] = {
                'power_binning_enabled': True,
                'sensitive_node_grouping': sensitive_nodes[:min(num_sensitive_nodes, 10)],  # Top N sensitive
                'power_smoothing_required': True,
                'binning_threshold_power_mw': 5.0 # Example threshold
            }
        elif strategy == 'thermal_guard_ring':
            parameters['layout_modifications'] = {
                'guard_ring_insertion': [hs['region'] for hs in hotspots if hs['thermal_sensitivity'] > 0.7], # For highly sensitive hotspots
                'isolation_wells': True,
                'heat_spreading_structures': True,
                'guard_ring_width_um': 5.0 # Example width
            }
        elif strategy == 'adaptive_body_biasing':
            parameters['power_optimization'] = {
                'adaptive_biasing_enabled': True,
                'biasing_targets': sensitive_nodes[:min(num_sensitive_nodes, 15)],  # Top N sensitive
                'temperature_monitoring_frequency_hz': 100,
                'biasing_voltage_range_mv': 100 # Example range
            }
        elif strategy == 'clock_frequency_scaling':
            parameters['thermal_management'] = {
                'dynamic_frequency_scaling': True,
                'thermal_throttling_zones': [hs['region'] for hs in hotspots if hs['thermal_density'] > 0.8], # Only for very hot regions
                'performance_scaling_factors': [0.75 + (0.05 * (1 - hs['thermal_density'])) for hs in hotspots if hs['thermal_density'] > 0.8], # Dynamic scaling factor
                'monitoring_interval_ms': 50 # How often to check
            }
        
        # Common thermal management features
        parameters['common_features'] = {
            'thermal_monitoring_sensors': max(2, num_hotspots + num_sensitive_nodes // 5), # Scale sensors
            'thermal_analysis_required': True,
            'cooling_solution_consideration': num_hotspots > 0 or num_gradients > 0
        }
        
        return parameters
        
        return parameters
    
    def _assess_thermal_risk(self, parameters: Dict[str, Any], hotspots: List[Dict], 
                           gradients: List[Dict], sensitive_nodes: List[str]) -> Dict[str, float]:
        """Assess the thermal risk profile of the proposed optimization"""
        self.logger.debug(f"Assessing thermal risk for strategy: {parameters['strategy']}")
        risk_profile = {
            'hotspot_risk': 0.0,
            'thermal_gradient_risk': 0.0,
            'device_lifetime_risk': 0.0,
            'performance_degradation_risk': 0.0,
            'area_overhead_risk': 0.0,      # New: Area overhead risk
            'schedule_impact_risk': 0.0     # New: Schedule impact risk
        }
        
        strategy = parameters['strategy']
        num_hotspots = len(hotspots)
        num_gradients = len(gradients)
        num_sensitive_nodes = len(sensitive_nodes)

        # Hotspot risk
        if strategy == 'hotspot_aware_placement':
            hotspot_risk = min(num_hotspots / 10.0, 0.4) # Good at mitigating hotspots
        elif strategy == 'clock_frequency_scaling':
            hotspot_risk = 0.1 # Very effective
        else:
            hotspot_risk = min(num_hotspots / 5.0, 0.7) # Higher base risk
        risk_profile['hotspot_risk'] = hotspot_risk
        
        # Gradient risk
        if strategy == 'thermal_via_insertion':
            gradient_risk = min(num_gradients / 10.0, 0.3) # Good at mitigating gradients
        else:
            gradient_risk = min(num_gradients / 5.0, 0.6) # Higher base risk
        risk_profile['thermal_gradient_risk'] = gradient_risk
        
        # Device lifetime risk (related to thermal stress)
        # Strategies that reduce temperature effectively will improve lifetime
        if strategy in ['clock_frequency_scaling', 'adaptive_body_biasing']:
            device_lifetime_risk = 0.1
        elif strategy == 'thermal_guard_ring':
            device_lifetime_risk = 0.2
        else:
            device_lifetime_risk = 0.4
        risk_profile['device_lifetime_risk'] = device_lifetime_risk
        
        # Performance degradation risk from thermal effects
        # Clock frequency scaling has highest degradation
        if strategy == 'clock_frequency_scaling':
            performance_degradation_risk = 0.8
        elif strategy == 'thermal_via_insertion':
            performance_degradation_risk = 0.1 # Minimal
        else:
            performance_degradation_risk = 0.3 # Moderate
        risk_profile['performance_degradation_risk'] = performance_degradation_risk

        # Area overhead risk
        if strategy in ['thermal_guard_ring', 'hotspot_aware_placement']:
            area_overhead_risk = 0.6 # High area overhead
        elif strategy == 'thermal_via_insertion':
            area_overhead_risk = 0.4 # Moderate
        else:
            area_overhead_risk = 0.1 # Low
        risk_profile['area_overhead_risk'] = area_overhead_risk

        # Schedule impact risk
        if strategy in ['adaptive_body_biasing', 'thermal_guard_ring']:
            schedule_impact_risk = 0.7 # Complex implementation
        elif strategy == 'clock_frequency_scaling':
            schedule_impact_risk = 0.2 # Easier to implement
        else:
            schedule_impact_risk = 0.4 # Moderate
        risk_profile['schedule_impact_risk'] = schedule_impact_risk
        
        return risk_profile
    
    def _calculate_thermal_cost_vector(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Calculate the cost vector (PPA impact) of the thermal proposal"""
        self.logger.debug(f"Calculating cost vector for strategy: {parameters['strategy']}")
        strategy = parameters['strategy']
        
        if strategy == 'hotspot_aware_placement':
            return {
                'power_impact': 0.05,      # Slight power increase due to wirelength spreading
                'performance_impact': 0.0, # Neutral performance impact
                'area_impact': 0.1,        # High area increase for spreading
                'yield_impact': -0.05,     # Good yield improvement due to reduced thermal stress
                'schedule_impact': 0.08,   # Medium schedule impact (placement adjustments)
                'thermal_impact': -0.15    # Excellent thermal reduction
            }
        elif strategy == 'thermal_via_insertion':
            return {
                'power_impact': 0.02,      # Slight power increase for vias
                'performance_impact': -0.02, # Slight performance gain due to stable temperature
                'area_impact': 0.08,       # Moderate area increase for vias
                'yield_impact': -0.08,     # Good yield improvement
                'schedule_impact': 0.05,   # Moderate schedule impact
                'thermal_impact': -0.12    # Excellent thermal reduction
            }
        elif strategy == 'power_binning':
            return {
                'power_impact': -0.08,     # Good power optimization (reduces peak power)
                'performance_impact': 0.0, # Neutral performance impact
                'area_impact': 0.01,       # Low area impact
                'yield_impact': -0.07,     # Good yield improvement
                'schedule_impact': 0.03,   # Low schedule impact
                'thermal_impact': -0.1     # Good thermal reduction
            }
        elif strategy == 'thermal_guard_ring':
            return {
                'power_impact': 0.03,      # Slight power increase for guard rings
                'performance_impact': 0.0, # Neutral performance impact
                'area_impact': 0.12,       # High area increase for guard rings
                'yield_impact': -0.1,      # Excellent yield improvement (isolation)
                'schedule_impact': 0.07,   # Moderate schedule impact
                'thermal_impact': -0.08    # Good thermal reduction
            }
        elif strategy == 'adaptive_body_biasing':
            return {
                'power_impact': -0.1,      # Excellent power optimization
                'performance_impact': 0.0, # Neutral performance impact
                'area_impact': 0.02,       # Low area impact (control circuitry)
                'yield_impact': -0.06,     # Good yield improvement
                'schedule_impact': 0.1,    # High schedule impact (complex control)
                'thermal_impact': -0.12    # Excellent thermal reduction
            }
        else:  # clock_frequency_scaling
            return {
                'power_impact': -0.15,     # Excellent power reduction
                'performance_impact': 0.15, # Significant performance degradation
                'area_impact': 0.01,       # Low area impact
                'yield_impact': -0.08,     # Good yield improvement
                'schedule_impact': 0.02,   # Low schedule impact
                'thermal_impact': -0.18    # Excellent thermal reduction
            }

    def _predict_thermal_outcome(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Predict the outcome of applying this thermal optimization"""
        self.logger.debug(f"Predicting outcome for strategy: {parameters['strategy']}")
        strategy = parameters['strategy']
        
        if strategy == 'hotspot_aware_placement':
            return {
                'expected_hotspot_reduction': 0.6,
                'expected_thermal_gradient_improvement': 0.4,
                'expected_device_lifetime_extension': 0.3,
                'expected_performance_stability': 0.5,
                'expected_area_change': 0.1, # 10% area increase
                'expected_schedule_impact': 0.05 # 5% schedule increase
            }
        elif strategy == 'thermal_via_insertion':
            return {
                'expected_hotspot_reduction': 0.4,
                'expected_thermal_gradient_improvement': 0.7,
                'expected_device_lifetime_extension': 0.4,
                'expected_performance_stability': 0.6,
                'expected_area_change': 0.08,
                'expected_schedule_impact': 0.03
            }
        elif strategy == 'power_binning':
            return {
                'expected_hotspot_reduction': 0.5,
                'expected_thermal_gradient_improvement': 0.5,
                'expected_device_lifetime_extension': 0.5,
                'expected_performance_stability': 0.4,
                'expected_area_change': 0.02,
                'expected_schedule_impact': 0.02
            }
        elif strategy == 'thermal_guard_ring':
            return {
                'expected_hotspot_reduction': 0.3,
                'expected_thermal_gradient_improvement': 0.6,
                'expected_device_lifetime_extension': 0.7,
                'expected_performance_stability': 0.7,
                'expected_area_change': 0.15,
                'expected_schedule_impact': 0.07
            }
        elif strategy == 'adaptive_body_biasing':
            return {
                'expected_hotspot_reduction': 0.7,
                'expected_thermal_gradient_improvement': 0.5,
                'expected_device_lifetime_extension': 0.8,
                'expected_performance_stability': 0.8,
                'expected_area_change': 0.03,
                'expected_schedule_impact': 0.1
            }
        else:  # clock_frequency_scaling
            return {
                'expected_hotspot_reduction': 0.8,
                'expected_thermal_gradient_improvement': 0.8,
                'expected_device_lifetime_extension': 0.9,
                'expected_performance_stability': 0.3,  # Lower due to frequency scaling
                'expected_area_change': 0.01,
                'expected_schedule_impact': 0.02
            }
    
    def _calculate_confidence(self, graph: CanonicalSiliconGraph) -> float:
        """Calculate confidence in the thermal proposal"""
        base_confidence = 0.78  # Thermal analysis can be complex
        
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
        """Evaluate the potential impact of a thermal optimization proposal"""
        impact = {
            'hotspot_reduction': 0.0,
            'thermal_stability': 0.0,
            'device_lifetime': 0.0,
            'performance_stability': 0.0
        }
        
        # Based on the strategy, estimate improvements
        strategy = proposal.parameters.get('strategy', 'adaptive_body_biasing')
        
        if strategy == 'hotspot_aware_placement':
            impact['hotspot_reduction'] = 0.6
            impact['thermal_stability'] = 0.5
            impact['device_lifetime'] = 0.3
            impact['performance_stability'] = 0.5
        elif strategy == 'thermal_via_insertion':
            impact['hotspot_reduction'] = 0.4
            impact['thermal_stability'] = 0.7
            impact['device_lifetime'] = 0.4
            impact['performance_stability'] = 0.6
        elif strategy == 'power_binning':
            impact['hotspot_reduction'] = 0.5
            impact['thermal_stability'] = 0.5
            impact['device_lifetime'] = 0.5
            impact['performance_stability'] = 0.4
        elif strategy == 'thermal_guard_ring':
            impact['hotspot_reduction'] = 0.3
            impact['thermal_stability'] = 0.6
            impact['device_lifetime'] = 0.7
            impact['performance_stability'] = 0.7
        elif strategy == 'adaptive_body_biasing':
            impact['hotspot_reduction'] = 0.7
            impact['thermal_stability'] = 0.5
            impact['device_lifetime'] = 0.8
            impact['performance_stability'] = 0.8
        else:  # clock_frequency_scaling
            impact['hotspot_reduction'] = 0.8
            impact['thermal_stability'] = 0.8
            impact['device_lifetime'] = 0.9
            impact['performance_stability'] = 0.3
        
        return impact