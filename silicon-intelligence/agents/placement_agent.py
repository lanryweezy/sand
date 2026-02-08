"""
Placement Agent - Specialized agent for cell placement optimization

This agent focuses on congestion-aware and clock-aware placement
to optimize for wirelength, timing, and routing resources.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from agents.base_agent import BaseAgent, AgentType, AgentProposal
from core.canonical_silicon_graph import CanonicalSiliconGraph
from utils.logger import get_logger


class PlacementAgent(BaseAgent):
    """
    Placement Agent - congestion-aware and clock-aware placement optimization
    """
    
    def __init__(self):
        super().__init__(AgentType.PLACEMENT)
        self.logger = get_logger(f"{__name__}.placement_agent")
        self.placement_strategies = [
            'analytical', 'partitioning', 'force_directed', 'simulated_annealing'
        ]
    
    def propose_action(self, graph: CanonicalSiliconGraph) -> Optional['AgentProposal']:
        """
        Generate a placement proposal based on the current graph state
        """
        self.logger.info("Generating placement proposal")
        
        # Identify cells that need placement optimization
        cells = [n for n, attrs in graph.graph.nodes(data=True) 
                if attrs.get('node_type').value == 'cell']
        
        if not cells:
            self.logger.debug("No cells found, skipping placement proposal")
            return None
        
        # Analyze current congestion and timing criticality
        congestion_nodes = self._identify_congestion_prone_areas(graph)
        timing_critical_cells = [c for c in cells 
                                if graph.graph.nodes[c].get('timing_criticality', 0) > 0.5]
        clock_related_cells = self._identify_clock_related_cells(graph, cells)
        
        # Generate a placement strategy
        strategy = self._select_placement_strategy(
            congestion_nodes, timing_critical_cells, clock_related_cells
        )
        
        # Create placement parameters
        parameters = self._generate_placement_parameters(
            strategy, cells, congestion_nodes, timing_critical_cells, clock_related_cells
        )
        
        # Calculate risk profile
        risk_profile = self._assess_placement_risk(
            parameters, congestion_nodes, timing_critical_cells
        )
        
        # Calculate cost vector (PPA impact)
        cost_vector = self._calculate_placement_cost_vector(parameters)
        
        # Estimate outcome
        predicted_outcome = self._predict_placement_outcome(parameters)
        
        proposal = AgentProposal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            proposal_id=f"pl_{np.random.randint(1000, 9999)}",
            timestamp=self._get_timestamp(),
            action_type="place_cells",
            targets=cells,
            parameters=parameters,
            confidence_score=self._calculate_confidence(graph),
            risk_profile=risk_profile,
            cost_vector=cost_vector,
            predicted_outcome=predicted_outcome,
            dependencies=[],
            conflicts_with=[]
        )
        
        self.logger.info(f"Generated placement proposal for {len(cells)} cells")
        return proposal
    
    def _identify_congestion_prone_areas(self, graph: CanonicalSiliconGraph) -> List[str]:
        """Identify nodes in areas prone to congestion"""
        # For now, identify nodes with high connectivity (fanout) or in high-density regions
        congestion_prone = []
        
        for node, attrs in graph.graph.nodes(data=True):
            # Check connectivity (fanout)
            fanout = len(list(graph.graph.successors(node)))
            if fanout > 10:  # Threshold for high fanout
                congestion_prone.append(node)
            
            # Check if in a region with high node density
            estimated_congestion = attrs.get('estimated_congestion', 0.0)
            if estimated_congestion > 0.6:
                congestion_prone.append(node)
        
        return list(set(congestion_prone))  # Remove duplicates
    
    def _identify_clock_related_cells(self, graph: CanonicalSiliconGraph, cells: List[str]) -> List[str]:
        """Identify cells that are related to clock distribution"""
        clock_cells = []
        
        for cell in cells:
            attrs = graph.graph.nodes[cell]
            cell_type = attrs.get('cell_type', '')
            
            # Clock buffer cells
            if 'buf' in cell_type.lower() or 'clk' in cell_type.lower():
                clock_cells.append(cell)
            # Sequential cells that need clock
            elif any(t in cell_type.lower() for t in ['dff', 'ff', 'latch', 'reg']):
                clock_cells.append(cell)
        
        return clock_cells
    
    def _select_placement_strategy(self, congestion_nodes: List[str], 
                                 timing_critical_cells: List[str], 
                                 clock_cells: List[str]) -> str:
        """Select the optimal placement strategy based on design characteristics"""
        num_congestion_nodes = len(congestion_nodes)
        num_timing_critical_cells = len(timing_critical_cells)
        num_clock_cells = len(clock_cells)
        
        # Prioritize based on severity of design challenges
        if num_congestion_nodes > (num_timing_critical_cells * 1.5) and num_congestion_nodes > 100:
            # If congestion is significantly high and a major issue
            self.logger.debug(f"Selecting 'partitioning' due to high congestion ({num_congestion_nodes} nodes).")
            return 'partitioning'  # Good for managing congestion, aims to balance density

        elif num_timing_critical_cells > 50 and num_timing_critical_cells > (num_congestion_nodes * 1.5):
            # If timing is significantly critical for many cells
            self.logger.debug(f"Selecting 'analytical' due to many timing critical paths ({num_timing_critical_cells} cells).")
            return 'analytical'  # Good for timing optimization, focuses on wirelength

        elif num_clock_cells > (num_timing_critical_cells * 1.2) and num_clock_cells > 30:
            # If clock-aware placement is more important due to many clock-related cells
            self.logger.debug(f"Selecting 'force_directed' due to many clock-related cells ({num_clock_cells} cells).")
            return 'force_directed'  # Good for clock tree considerations, balances forces

        else:
            # Default to a balanced, thorough approach if no single concern dominates
            self.logger.debug("Selecting 'simulated_annealing' as a balanced approach.")
            return 'simulated_annealing'  # Good overall optimizer, explores solution space broadly
    
    def _generate_placement_parameters(self, strategy: str, cells: List[str], 
                                     congestion_nodes: List[str], 
                                     timing_critical_cells: List[str], 
                                     clock_cells: List[str]) -> Dict[str, Any]:
        """Generate placement parameters based on strategy"""
        parameters = {
            'strategy': strategy,
            'cell_placements': {},
            'constraints': {},
            'optimization_goals': {}
        }
        
        # Set optimization goals based on strategy
        if strategy == 'analytical':
            parameters['optimization_goals'] = {
                'minimize_wirelength': True,
                'balance_congestion': False, # Analytical can sometimes worsen congestion
                'preserve_timing': True,
                'high_density_regions': False
            }
        elif strategy == 'partitioning':
            parameters['optimization_goals'] = {
                'minimize_cut_size': True, # Key for partitioning
                'balance_partition_sizes': True,
                'reduce_congestion': True,
                'preserve_timing': False # Not primary goal
            }
        elif strategy == 'force_directed':
            parameters['optimization_goals'] = {
                'minimize_repulsion': True, # Spreading out components
                'optimize_clock_distribution': True, # Good for clock placement
                'balance_attraction': True, # Keep connected components close
                'reduce_overlap': True
            }
        elif strategy == 'simulated_annealing':
            parameters['optimization_goals'] = {
                'global_optimization': True, # Explores entire solution space
                'escape_local_minima': True,
                'balanced_tradeoffs': True, # Tries to balance all PPA
                'robustness_to_noise': True
            }
        
        # Assign special treatment to critical cells
        for cell in cells:
            cell_params = {
                'priority': 1.0, # 1.0 = Normal priority
                'movability': True,
                'region_constraint': None,
                'orientation_preference': 'N'
            }
            
            if cell in timing_critical_cells:
                cell_params['priority'] = 2.0  # 2.0 = Very High priority for timing
                cell_params['movability'] = True  # Must be movable for timing optimization
            elif cell in congestion_nodes:
                cell_params['priority'] = 1.5  # 1.5 = Medium-High priority for congestion relief
                cell_params['movability'] = True
                cell_params['spread_factor'] = 0.8 # Suggest spreading
            elif cell in clock_cells:
                cell_params['priority'] = 1.8  # 1.8 = High priority for clock network integrity
                cell_params['region_constraint'] = 'clock_zone'  # Prefer specific regions for clock cells
                cell_params['orientation_preference'] = 'N_S' # Allow N or S for better clock routing
            
            parameters['cell_placements'][cell] = cell_params
        
        # Add global constraints
        parameters['constraints'] = {
            'aspect_ratio': 1.0, # Maintain square chip aspect ratio
            'boundary_constraints': True, # Keep cells within boundaries
            'keepout_zones': [], # Define areas to avoid for placement
            'alignment_grids': True, # Snap to placement grids
            'power_mesh_aware': True # Consider power mesh during placement
        }
        
        return parameters
    
    def _assess_placement_risk(self, parameters: Dict[str, Any], 
                             congestion_nodes: List[str], 
                             timing_critical_cells: List[str]) -> Dict[str, float]:
        """Assess the risk profile of the proposed placement"""
        risk_profile = {
            'timing_risk': 0.0,
            'congestion_risk': 0.0,
            'power_risk': 0.0,
            'yield_risk': 0.0
        }
        
        strategy = parameters['strategy']
        num_timing_critical = len(timing_critical_cells)
        num_congestion_prone = len(congestion_nodes)

        # Timing risk based on number of critical cells and strategy
        if strategy == 'analytical':
            timing_risk = min(num_timing_critical / 150.0, 0.6) # Analytical aims to improve timing
        elif strategy == 'simulated_annealing':
            timing_risk = min(num_timing_critical / 100.0, 0.7) # SA is good overall
        else:
            timing_risk = min(num_timing_critical / 75.0, 0.8) # Other strategies might have higher base timing risk
        risk_profile['timing_risk'] = timing_risk
        
        # Congestion risk based on congestion-prone nodes and strategy
        if strategy == 'partitioning':
            congestion_risk = min(num_congestion_prone / 300.0, 0.5) # Partitioning aims to reduce congestion
        elif strategy == 'simulated_annealing':
            congestion_risk = min(num_congestion_prone / 200.0, 0.6) # SA is good overall
        else:
            congestion_risk = min(num_congestion_prone / 150.0, 0.7) # Other strategies might have higher base congestion risk
        risk_profile['congestion_risk'] = congestion_risk
        
        # Power risk (related to placement density and switching activity)
        # Force-directed can sometimes spread cells out, reducing hot spots but increasing wirelength power
        if strategy == 'force_directed':
            power_risk = 0.35 # Slightly lower due to spreading
        else:
            power_risk = 0.45  # Moderate default risk, might increase with dense placement
        risk_profile['power_risk'] = power_risk
        
        # Yield risk (related to manufacturing complexity from placement)
        # Analytical and SA tend to produce more regular patterns, good for yield
        if strategy in ['analytical', 'simulated_annealing']:
            yield_risk = 0.2  # Lower risk
        else:
            yield_risk = 0.35  # Moderate risk
        risk_profile['yield_risk'] = yield_risk
        
        return risk_profile
    
    def _calculate_placement_cost_vector(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Calculate the cost vector (PPA impact) of the placement proposal"""
        strategy = parameters['strategy']
        
        if strategy == 'analytical':
            return {
                'power_impact': 0.05,      # Slight increase due to wirelength focus
                'performance_impact': -0.1, # Good timing improvement
                'area_impact': 0.03,       # Slight area increase
                'yield_impact': -0.02,      # Slight yield improvement
                'schedule_impact': -0.05    # Faster convergence
            }
        elif strategy == 'partitioning':
            return {
                'power_impact': 0.03,      # Slight power increase
                'performance_impact': -0.05, # Moderate timing improvement
                'area_impact': -0.05,       # Good area efficiency
                'yield_impact': -0.03,      # Good yield improvement
                'schedule_impact': -0.03    # Moderate runtime
            }
        elif strategy == 'force_directed':
            return {
                'power_impact': 0.0,       # Neutral power impact (spreading vs wirelength)
                'performance_impact': -0.08, # Good timing due to proximity
                'area_impact': 0.08,       # May use more area
                'yield_impact': -0.02,      # Good yield
                'schedule_impact': 0.05     # Slower convergence
            }
        else:  # simulated annealing
            return {
                'power_impact': -0.08,     # Excellent power optimization
                'performance_impact': -0.12, # Excellent timing optimization
                'area_impact': -0.08,       # Good area utilization
                'yield_impact': -0.05,      # Excellent yield
                'schedule_impact': 0.1      # Slowest but most thorough
            }
    
    def _predict_placement_outcome(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Predict the outcome of applying this placement"""
        strategy = parameters['strategy']
        
        # Outcomes are typically expressed as percentage changes or absolute improvements
        if strategy == 'analytical':
            return {
                'expected_wirelength_reduction': 0.25, # 25% reduction
                'expected_timing_improvement': 0.15, # 15% improvement
                'expected_congestion_reduction': 0.1, # 10% reduction
                'expected_power_reduction': 0.05, # 5% reduction
                'expected_area_change': 0.02 # 2% increase (trade-off)
            }
        elif strategy == 'partitioning':
            return {
                'expected_wirelength_reduction': 0.18,
                'expected_timing_improvement': 0.1,
                'expected_congestion_reduction': 0.25, # Strong congestion reduction
                'expected_power_reduction': 0.03,
                'expected_area_change': -0.05 # 5% area reduction
            }
        elif strategy == 'force_directed':
            return {
                'expected_wirelength_reduction': 0.2,
                'expected_timing_improvement': 0.12,
                'expected_congestion_reduction': 0.08,
                'expected_power_reduction': 0.0, # Neutral power change
                'expected_area_change': 0.05 # 5% area increase (spreading)
            }
        else:  # simulated annealing
            return {
                'expected_wirelength_reduction': 0.35,
                'expected_timing_improvement': 0.25,
                'expected_congestion_reduction': 0.20,
                'expected_power_reduction': 0.12,
                'expected_area_change': 0.01 # 1% increase
            }
    
    def _calculate_confidence(self, graph: CanonicalSiliconGraph) -> float:
        """Calculate confidence in the placement proposal"""
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
        """Evaluate the potential impact of a placement proposal"""
        impact = {
            'wirelength_improvement': 0.0,
            'timing_improvement': 0.0,
            'congestion_reduction': 0.0,
            'power_efficiency': 0.0
        }
        
        # Based on the strategy, estimate improvements
        strategy = proposal.parameters.get('strategy', 'analytical')
        
        if strategy == 'analytical':
            impact['wirelength_improvement'] = 0.25
            impact['timing_improvement'] = 0.15
            impact['congestion_reduction'] = 0.1
        elif strategy == 'partitioning':
            impact['wirelength_improvement'] = 0.2
            impact['timing_improvement'] = 0.1
            impact['congestion_reduction'] = 0.25
        elif strategy == 'force_directed':
            impact['wirelength_improvement'] = 0.3
            impact['timing_improvement'] = 0.2
            impact['congestion_reduction'] = 0.15
        else:  # simulated annealing
            impact['wirelength_improvement'] = 0.35
            impact['timing_improvement'] = 0.25
            impact['congestion_reduction'] = 0.2
        
        return impact