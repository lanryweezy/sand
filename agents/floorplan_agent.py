"""
Floorplan Agent - Macro placement and floorplan optimization

This agent specializes in:
- Macro placement and floorplan generation
- Hierarchical design organization
- Thermal-aware placement
- Power-aware floorplanning
"""

import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from agents.base_agent import BaseAgent, AgentType, AgentProposal
from core.canonical_silicon_graph import CanonicalSiliconGraph
from utils.logger import get_logger


class FloorplanAgent(BaseAgent):
    """
    Floorplan Agent - Optimizes macro placement and floorplan
    
    Strategies:
    - Hierarchical clustering: Group related macros
    - Linear arrangement: Minimize wirelength
    - Grid-based: Regular grid placement
    - Thermal-aware: Minimize hotspots
    - Power-aware: Optimize power distribution
    """
    
    def __init__(self, agent_id: str = None):
        super().__init__(AgentType.FLOORPLAN, agent_id)
        self.strategies = [
            'hierarchical_clustering',
            'linear_arrangement',
            'grid_based',
            'thermal_aware',
            'power_aware'
        ]
        self.strategy_index = 0
    
    def propose_action(self, graph: CanonicalSiliconGraph) -> Optional[AgentProposal]:
        """Generate a floorplan proposal"""
        
        # Get macros from graph
        macros = graph.get_macros()
        if not macros:
            self.logger.debug("No macros found in graph")
            return None
        
        # Select strategy (round-robin through strategies)
        strategy = self.strategies[self.strategy_index % len(self.strategies)]
        self.strategy_index += 1
        
        # Generate proposal based on strategy
        if strategy == 'hierarchical_clustering':
            return self._propose_hierarchical_clustering(graph, macros)
        elif strategy == 'linear_arrangement':
            return self._propose_linear_arrangement(graph, macros)
        elif strategy == 'grid_based':
            return self._propose_grid_based(graph, macros)
        elif strategy == 'thermal_aware':
            return self._propose_thermal_aware(graph, macros)
        elif strategy == 'power_aware':
            return self._propose_power_aware(graph, macros)
        
        return None
    
    def _propose_hierarchical_clustering(self, graph: CanonicalSiliconGraph, 
                                        macros: List[str]) -> AgentProposal:
        """Propose hierarchical clustering floorplan"""
        
        # Group macros by connectivity
        clusters = self._cluster_macros_by_connectivity(graph, macros)
        
        # Generate positions for clusters
        positions = self._generate_cluster_positions(clusters)
        
        # Calculate metrics for predicted outcome
        total_area = sum(graph.graph.nodes[m].get('area', 0) for m in macros)
        total_power = sum(graph.graph.nodes[m].get('power', 0) for m in macros)
        
        # Estimate wirelength based on positions
        estimated_wirelength = self._estimate_wirelength(positions)

        proposal = AgentProposal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            proposal_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            action_type='place_macros',
            targets=macros,
            parameters={
                'positions': positions,
                'strategy': 'hierarchical_clustering',
                'clusters': clusters
            },
            confidence_score=self._calculate_confidence(graph),
            risk_profile={
                'timing_risk': 0.15,  # Hierarchical clustering often improves timing
                'power_risk': 0.1,    # Can also improve power delivery
                'area_risk': 0.2,     # Might have slight area overhead due to separation
                'congestion_risk': 0.1, # Reduces congestion by grouping
                'thermal_risk': 0.15 # Can help manage thermal hotspots through grouping
            },
            cost_vector={
                'power_impact': -0.05, # Negative means improvement (reduction)
                'performance_impact': 0.1, # Positive means improvement (faster)
                'area_impact': 0.05,  # Positive means increase
                'yield_impact': -0.02, # Negative means improvement
                'schedule_impact': 0.05 # Increase in schedule
            },
            predicted_outcome={
                'total_area_change': 0.02 * total_area, # Slight increase
                'total_power_change': -0.05 * total_power, # Small decrease
                'estimated_congestion_reduction': 0.2, # Significant reduction
                'timing_slack_improvement': 0.15, # Good improvement
                'wirelength_reduction': 0.18 * estimated_wirelength, # Significant wirelength reduction
            },
            dependencies=[],
            conflicts_with=[]
        )
        
        self.logger.info(f"Generated hierarchical clustering proposal with {len(clusters)} clusters")
        return proposal
    
    def _propose_linear_arrangement(self, graph: CanonicalSiliconGraph, 
                                   macros: List[str]) -> AgentProposal:
        """Propose linear arrangement floorplan"""
        
        # Arrange macros in a line to minimize wirelength for specific scenarios
        positions = {}
        for i, macro in enumerate(macros):
            # Simple linear arrangement on x-axis
            positions[macro] = (i * 100.0, 0.0)  
        
        # Calculate metrics for predicted outcome
        total_area = sum(graph.graph.nodes[m].get('area', 0) for m in macros)
        total_power = sum(graph.graph.nodes[m].get('power', 0) for m in macros)
        estimated_wirelength = self._estimate_wirelength(positions)
        
        proposal = AgentProposal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            proposal_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            action_type='place_macros',
            targets=macros,
            parameters={
                'positions': positions,
                'strategy': 'linear_arrangement'
            },
            confidence_score=self._calculate_confidence(graph) * 0.9, # Slightly lower confidence than hierarchical
            risk_profile={
                'timing_risk': 0.4, # Can be higher due to long interconnections
                'power_risk': 0.2,
                'area_risk': 0.3, # Can be less efficient in area
                'congestion_risk': 0.5, # Can be higher if not carefully planned
                'thermal_risk': 0.2 # Moderate thermal risk
            },
            cost_vector={
                'power_impact': 0.05,
                'performance_impact': -0.05, # Can have negative impact
                'area_impact': 0.1,
                'yield_impact': 0.05,
                'schedule_impact': -0.05 # Faster to generate
            },
            predicted_outcome={
                'total_area_change': 0.05 * total_area, # Moderate increase
                'total_power_change': 0.03 * total_power, # Slight increase
                'estimated_congestion_reduction': -0.1, # Might increase congestion
                'timing_slack_improvement': -0.05, # Might degrade timing
                'wirelength_reduction': 0.1 * estimated_wirelength, # Some wirelength reduction in one dimension
            },
            dependencies=[],
            conflicts_with=[]
        )
        
        self.logger.info("Generated linear arrangement proposal")
        return proposal
    
    def _propose_grid_based(self, graph: CanonicalSiliconGraph, 
                           macros: List[str]) -> AgentProposal:
        """Propose grid-based floorplan"""
        
        # Arrange macros in a grid
        grid_size = int(np.ceil(np.sqrt(len(macros))))
        positions = {}
        
        # Simple grid-based placement
        for i, macro in enumerate(macros):
            row = i // grid_size
            col = i % grid_size
            positions[macro] = (col * 100.0, row * 100.0)
        
        # Calculate metrics for predicted outcome
        total_area = sum(graph.graph.nodes[m].get('area', 0) for m in macros)
        total_power = sum(graph.graph.nodes[m].get('power', 0) for m in macros)
        estimated_wirelength = self._estimate_wirelength(positions)
        
        proposal = AgentProposal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            proposal_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            action_type='place_macros',
            targets=macros,
            parameters={
                'positions': positions,
                'strategy': 'grid_based',
                'grid_size': grid_size
            },
            confidence_score=self._calculate_confidence(graph) * 0.95, # Good confidence
            risk_profile={
                'timing_risk': 0.25, # Moderate timing risk
                'power_risk': 0.15,
                'area_risk': 0.20, # Balanced area utilization
                'congestion_risk': 0.30, # Can have hotspots if not optimized
                'thermal_risk': 0.18
            },
            cost_vector={
                'power_impact': 0.03,
                'performance_impact': 0.08,
                'area_impact': 0.05,
                'yield_impact': -0.01,
                'schedule_impact': -0.03 # Relatively fast
            },
            predicted_outcome={
                'total_area_change': 0.01 * total_area, # Slight area change
                'total_power_change': 0.01 * total_power, # Slight power change
                'estimated_congestion_reduction': 0.1, # Moderate congestion reduction
                'timing_slack_improvement': 0.1, # Moderate timing improvement
                'wirelength_reduction': 0.15 * estimated_wirelength, # Moderate wirelength reduction
            },
            dependencies=[],
            conflicts_with=[]
        )
        
        self.logger.info(f"Generated grid-based proposal ({grid_size}x{grid_size})")
        return proposal
    
    def _propose_thermal_aware(self, graph: CanonicalSiliconGraph, 
                              macros: List[str]) -> AgentProposal:
        """Propose thermal-aware floorplan"""
        
        # Sort macros by power consumption (high power first)
        high_power_macros = sorted(macros, 
                                  key=lambda m: graph.graph.nodes[m].get('power', 0),
                                  reverse=True)
        
        positions = {}
        # Simple strategy: try to spread out high-power macros
        # This is a heuristic; real thermal-aware placement is complex
        chip_width = 10000.0
        chip_height = 10000.0
        
        num_macros = len(high_power_macros)
        grid_dim = int(np.ceil(np.sqrt(num_macros)))
        
        for i, macro in enumerate(high_power_macros):
            row = i // grid_dim
            col = i % grid_dim
            
            # Simple spread-out logic: place high-power macros further apart
            # For demonstration, distribute them in a grid
            x = (col / grid_dim) * chip_width + (chip_width / (2 * grid_dim))
            y = (row / grid_dim) * chip_height + (chip_height / (2 * grid_dim))
            positions[macro] = (x, y)
        
        # Calculate metrics for predicted outcome
        total_area = sum(graph.graph.nodes[m].get('area', 0) for m in macros)
        total_power = sum(graph.graph.nodes[m].get('power', 0) for m in macros)
        estimated_wirelength = self._estimate_wirelength(positions)
        
        proposal = AgentProposal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            proposal_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            action_type='place_macros',
            targets=macros,
            parameters={
                'positions': positions,
                'strategy': 'thermal_aware'
            },
            confidence_score=self._calculate_confidence(graph) * 0.9, # Good confidence for thermal benefits
            risk_profile={
                'timing_risk': 0.3, # May increase timing risk due to spreading
                'power_risk': 0.05, # Low power risk
                'area_risk': 0.25, # May consume more area due to spreading
                'congestion_risk': 0.2, # Good for congestion if spread
                'thermal_risk': 0.05 # Low thermal risk
            },
            cost_vector={
                'power_impact': -0.05, # Reduces peak temperature, thus improves power efficiency
                'performance_impact': -0.05, # Could slightly degrade performance due to longer wires
                'area_impact': 0.1, # Likely increases area
                'yield_impact': -0.05, # Improves yield by reducing thermal stress
                'schedule_impact': 0.05 # Moderate schedule impact
            },
            predicted_outcome={
                'total_area_change': 0.08 * total_area, # Significant area increase
                'total_power_change': -0.03 * total_power, # Small power decrease due to lower temperature
                'estimated_congestion_reduction': 0.15, # Moderate congestion reduction
                'timing_slack_improvement': -0.08, # Moderate timing degradation
                'wirelength_reduction': 0.05 * estimated_wirelength, # Some wirelength reduction due to spreading
                'max_temperature_reduction': 0.15 # Significant max temperature reduction
            },
            dependencies=[],
            conflicts_with=[]
        )
        
        self.logger.info("Generated thermal-aware proposal")
        return proposal
    
    def _propose_power_aware(self, graph: CanonicalSiliconGraph, 
                            macros: List[str]) -> AgentProposal:
        """Propose power-aware floorplan"""
        
        # Sort macros by power consumption (high power first)
        high_power_macros = sorted(macros, 
                                  key=lambda m: graph.graph.nodes[m].get('power', 0),
                                  reverse=True)
        
        positions = {}
        # Simple strategy: try to align high-power macros near power rails (simplified)
        # Or distribute them to balance power density across regions.
        chip_width = 10000.0
        chip_height = 10000.0
        
        num_macros = len(high_power_macros)
        
        # Distribute macros in a more power-aware way, e.g., balancing power density
        # For demonstration, a simple alternating pattern
        for i, macro in enumerate(high_power_macros):
            row = i // 2 # Two rows for example
            col = i % 2
            
            x_offset = col * (chip_width / 2.0)
            y_offset = row * (chip_height / (num_macros / 2.0)) if (num_macros / 2.0) > 0 else 0
            
            positions[macro] = (x_offset + 500.0, y_offset + 500.0) # Add some padding
        
        # Calculate metrics for predicted outcome
        total_area = sum(graph.graph.nodes[m].get('area', 0) for m in macros)
        total_power = sum(graph.graph.nodes[m].get('power', 0) for m in macros)
        estimated_wirelength = self._estimate_wirelength(positions)
        
        proposal = AgentProposal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            proposal_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            action_type='place_macros',
            targets=macros,
            parameters={
                'positions': positions,
                'strategy': 'power_aware'
            },
            confidence_score=self._calculate_confidence(graph) * 0.92, # Good confidence for power benefits
            risk_profile={
                'timing_risk': 0.25, # Moderate timing risk
                'power_risk': 0.08, # Low power risk (IR drop, EM)
                'area_risk': 0.20, # Moderate area impact
                'congestion_risk': 0.22, # Moderate congestion risk
                'thermal_risk': 0.15 # Can help with thermal too
            },
            cost_vector={
                'power_impact': -0.08, # Significant power improvement
                'performance_impact': -0.05, # Slight performance degradation due to spreading
                'area_impact': 0.07, # Slight area increase
                'yield_impact': -0.03, # Small yield improvement
                'schedule_impact': 0.03 # Slight schedule impact
            },
            predicted_outcome={
                'total_area_change': 0.05 * total_area, # Slight area increase
                'total_power_change': -0.07 * total_power, # Significant power decrease
                'estimated_congestion_reduction': 0.1, # Moderate congestion reduction
                'timing_slack_improvement': -0.05, # Slight timing degradation
                'wirelength_reduction': 0.1 * estimated_wirelength, # Some wirelength reduction due to spreading
                'ir_drop_reduction': 0.20, # Significant IR drop reduction
                'em_reduction': 0.15 # Moderate EM reduction
            },
            dependencies=[],
            conflicts_with=[]
        )
        
        self.logger.info("Generated power-aware proposal")
        return proposal
    
    def _cluster_macros_by_connectivity(self, graph: CanonicalSiliconGraph, 
                                       macros: List[str]) -> Dict[str, List[str]]:
        """Cluster macros by connectivity"""
        clusters = {}
        
        # Simple clustering: group by connectivity
        for i, macro in enumerate(macros):
            cluster_id = f"cluster_{i // 3}"  # 3 macros per cluster
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(macro)
        
        return clusters
    
    def _generate_cluster_positions(self, clusters: Dict[str, List[str]]) -> Dict[str, Tuple[float, float]]:
        """Generate positions for clusters"""
        positions = {}
        
        for i, (cluster_id, macros) in enumerate(clusters.items()):
            # Simple grid layout for clusters
            x = (i % 4) * 1000
            y = (i // 4) * 1000
            
            # Position macros within cluster
            for j, macro in enumerate(macros):
                macro_x = x + (j % 2) * 300
                macro_y = y + (j // 2) * 300
                positions[macro] = (macro_x, macro_y)
        
        return positions
    
    def _estimate_wirelength(self, positions: Dict[str, Tuple[float, float]]) -> float:
        """Estimate total wirelength"""
        if len(positions) < 2:
            return 0.0
        
        # Simple estimate: sum of distances between consecutive positions
        pos_list = list(positions.values())
        total_length = 0.0
        
        for i in range(len(pos_list) - 1):
            x1, y1 = pos_list[i]
            x2, y2 = pos_list[i + 1]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            total_length += distance
        
        return total_length
    
    def evaluate_proposal_impact(self, proposal: 'AgentProposal', 
                               graph: CanonicalSiliconGraph) -> Dict[str, float]:
        """Evaluate the potential impact of a floorplan proposal"""
        impact = {
            'congestion_improvement': 0.0,
            'timing_improvement': 0.0,
            'area_efficiency': 0.0,
            'power_efficiency': 0.0
        }
        
        # Based on the strategy, estimate improvements
        strategy = proposal.parameters.get('strategy', 'compact')
        
        if strategy == 'hierarchical':
            impact['congestion_improvement'] = 0.25
            impact['timing_improvement'] = 0.2
        elif strategy == 'ring':
            impact['congestion_improvement'] = 0.3
            impact['timing_improvement'] = 0.1
        else:  # compact
            impact['area_efficiency'] = 0.2
            impact['congestion_improvement'] = 0.1
        
        return impact

    def _cluster_macros_by_connectivity(self, graph: CanonicalSiliconGraph, macros: List[str]) -> List[List[str]]:
        """
        Groups macros by their connectivity using a simplified clustering approach.
        For demonstration, use connected components in a subgraph of macros.
        """
        if not macros:
            return []
        
        # Create a subgraph containing only macros and their direct connections
        macro_subgraph = graph.graph.subgraph(macros).copy()
        
        # Add edges between macros if they share a common net (indirect connection)
        for i, m1 in enumerate(macros):
            for m2 in macros[i+1:]:
                # Check if m1 and m2 share common neighbors (nets)
                common_neighbors = set(graph.graph.neighbors(m1)) & set(graph.graph.neighbors(m2))
                # Only consider net-type neighbors
                common_nets = [n for n in common_neighbors if graph.graph.nodes[n].get('node_type') == 'signal']
                if common_nets:
                    # Add a proxy edge in the macro subgraph to denote strong connectivity
                    macro_subgraph.add_edge(m1, m2)
        
        # Find connected components in this macro subgraph
        clusters = list(nx.connected_components(macro_subgraph))
        return [list(cluster) for cluster in clusters]
    
    def _generate_cluster_positions(self, clusters: List[List[str]]) -> Dict[str, Tuple[float, float]]:
        """Assigns approximate (x, y) positions to the clusters of macros."""
        positions = {}
        base_x, base_y = 0.0, 0.0
        spacing = 100.0 # Arbitrary spacing
        
        for i, cluster in enumerate(clusters):
            cluster_center_x = base_x + (i * spacing)
            cluster_center_y = base_y
            
            # Distribute macros within the cluster around this center
            for j, macro in enumerate(cluster):
                # Simple linear distribution for macros within a cluster
                positions[macro] = (cluster_center_x + (j * (spacing / len(cluster))), cluster_center_y)
                
        return positions
    
    def _estimate_wirelength(self, positions: Dict[str, Tuple[float, float]]) -> float:
        """
        Estimates the total wirelength based on macro positions.
        This is a simplified heuristic.
        """
        if not positions:
            return 0.0
        
        # For simplicity, assume a fixed "connection" between all placed macros
        # In a real system, this would iterate through actual nets and connections
        # in the graph to calculate wirelength more accurately.
        total_wirelength = 0.0
        macros = list(positions.keys())
        
        # Calculate sum of distances from a central point (for a very rough estimate)
        if macros:
            avg_x = np.mean([positions[m][0] for m in macros])
            avg_y = np.mean([positions[m][1] for m in macros])
            
            for macro_pos in positions.values():
                total_wirelength += abs(macro_pos[0] - avg_x) + abs(macro_pos[1] - avg_y)
                
        return total_wirelength * 0.001 # Scale down for more reasonable numbers

