"""
Timing Analyzer - Analyzes timing risks in the design

This module implements a model to analyze timing paths and predict timing risk zones
in the physical implementation.
"""

from typing import Dict, List, Any
import numpy as np
from core.canonical_silicon_graph import CanonicalSiliconGraph, NodeType
from utils.logger import get_logger


class TimingAnalyzer:
    """
    Timing Analyzer - analyzes timing paths and predicts timing risk zones
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.timing_weights = self._initialize_weights()
        self.model_weights = 1.0 # General weight for prediction confidence
    
    def _initialize_weights(self) -> Dict[str, float]:
        """Initialize weights for timing analysis factors"""
        return {
            'path_length': 0.3,
            'logic_depth': 0.25,
            'fanout_load': 0.2,
            'criticality': 0.15,
            'variation_sensitivity': 0.1
        }
    
    def analyze(self, silicon_graph: CanonicalSiliconGraph, constraints: Dict) -> Dict[str, Any]:
        """
        Analyze timing paths and identify risk zones
        
        Args:
            silicon_graph: The canonical silicon graph to analyze
            constraints: Design constraints (clocks, timing paths, etc.)
            
        Returns:
            Dictionary with list of timing risk zones and overall prediction confidence
        """
        self.logger.info("Analyzing timing paths for risk zones")
        
        timing_risks = []
        
        # Extract clock information from constraints
        clocks = constraints.get('clocks', [])
        clock_periods = {clk.get('name'): clk.get('period', 10.0) for clk in clocks}
        
        # Find critical paths in the graph
        critical_paths = self._find_critical_paths(silicon_graph, clock_periods)
        
        # Analyze each critical path
        for path_info in critical_paths:
            path_nodes = path_info['nodes']
            path_delay = path_info['delay']
            clock_name = path_info['clock']
            
            # Calculate timing slack
            period = clock_periods.get(clock_name, 10.0)
            slack = period - path_delay
            
            # Determine risk level
            risk_level = self._determine_timing_risk_level(slack, path_delay, period)
            
            # Calculate risk factors
            risk_factors = self._calculate_risk_factors(silicon_graph, path_nodes)
            
            timing_risk = {
                'path': path_nodes,
                'delay': path_delay,
                'clock': clock_name,
                'period': period,
                'slack': slack,
                'risk_level': risk_level,
                'criticality_score': 1.0 - (slack / period) if period > 0 else 0.5,
                'risk_factors': risk_factors,
                'recommended_fix': self._suggest_timing_fix(path_nodes, slack)
            }
            
            timing_risks.append(timing_risk)
        
        # Also analyze individual nodes for timing sensitivity
        node_timing_risks = self._analyze_node_timing_sensitivity(silicon_graph, constraints)
        timing_risks.extend(node_timing_risks)
        
        confidence = self._calculate_prediction_confidence()
        
        self.logger.info(f"Identified {len(timing_risks)} timing risk zones with confidence: {confidence:.2f}")
        return {
            'timing_risks': timing_risks,
            'confidence': confidence
        }
    
    def _find_critical_paths(self, graph: CanonicalSiliconGraph, clock_periods: Dict[str, float]) -> List[Dict]:
        """Find potentially critical timing paths in the graph"""
        self.logger.debug("Finding critical timing paths.")
        critical_paths = []
        
        # For simplicity, we'll identify paths based on:
        # 1. Paths between sequential elements (FFs, latches)
        # 2. Long paths with many logic gates
        # 3. High-fanout paths
        
        # Find all sequential elements (potential path endpoints)
        seq_elements = []
        for node, attrs in graph.graph.nodes(data=True):
            cell_type = attrs.get('cell_type', '').lower()
            if any(keyword in cell_type for keyword in ['dff', 'ff', 'latch', 'reg']):
                seq_elements.append(node)
        
        self.logger.debug(f"Identified {len(seq_elements)} sequential elements.")

        # For each sequential element, find paths to other sequential elements
        for start_seq in seq_elements:  # Removed hardcoded limit
            for end_seq in seq_elements:
                if start_seq != end_seq:
                    try:
                        # Find a path between sequential elements
                        path = self._find_path_with_timing_info(graph, start_seq, end_seq)
                        if path and len(path) > 2:  # At least start -> logic -> end
                            estimated_delay = self._estimate_path_delay(graph, path)
                            clock_name = self._find_associated_clock(graph, start_seq, end_seq)
                            
                            critical_paths.append({
                                'nodes': path,
                                'delay': estimated_delay,
                                'clock': clock_name,
                                'start': start_seq,
                                'end': end_seq
                            })
                            self.logger.debug(f"Found path from {start_seq} to {end_seq} with estimated delay {estimated_delay:.2f}.")
                    except nx.NetworkXNoPath:
                        self.logger.debug(f"No path found between {start_seq} and {end_seq}.")
                        continue
                    except Exception as e:
                        self.logger.warning(f"Error finding path between {start_seq} and {end_seq}: {e}")
                        continue
        
        # Sort by estimated delay (most critical first)
        critical_paths.sort(key=lambda x: x['delay'], reverse=True)
        
        # Return all found critical paths (or a reasonable top N, e.g., top 100 for practical reasons)
        return critical_paths # Removed hardcoded limit, can add a practical limit later if needed    
    def _find_path_with_timing_info(self, graph: CanonicalSiliconGraph, start: str, end: str):
        """Find a path between two nodes with timing considerations"""
        try:
            # Use NetworkX to find a path
            import networkx as nx
            path = nx.shortest_path(graph.graph, start, end)
            return path
        except nx.NetworkXNoPath:
            return None
    
    def _estimate_path_delay(self, graph: CanonicalSiliconGraph, path: List[str]) -> float:
        """Estimate the delay of a path based on node characteristics and interconnects"""
        self.logger.debug(f"Estimating delay for path: {path}")
        total_node_delay = 0.0
        total_interconnect_delay = 0.0
        
        for i, node in enumerate(path):
            attrs = graph.graph.nodes[node]
            # Add delay based on node type and characteristics
            node_delay = attrs.get('delay', 0.1)  # Default delay for node
            
            # Add extra delay for high-fanout nodes (logic cells)
            if attrs.get('node_type') == NodeType.CELL.value:
                fanout = len(list(graph.graph.successors(node)))
                if fanout > 5:
                    node_delay *= (1 + np.log10(fanout) * 0.1)
            
            total_node_delay += node_delay
            
            # Add interconnect delay from this node to the next in the path
            if i < len(path) - 1:
                next_node = path[i+1]
                edge_data = graph.graph.get_edge_data(node, next_node)
                
                # Assume a single edge for simplicity, or iterate if MultiDiGraph
                if edge_data:
                    # In a MultiDiGraph, edge_data is a dict of {key: attributes}
                    # We need to find the relevant edge (e.g., of type CONNECTION)
                    for key in edge_data:
                        edge_attrs = edge_data[key]
                        if edge_attrs.get('edge_type') == EdgeType.CONNECTION.value:
                            interconnect_delay = edge_attrs.get('delay', 0.05) # Default interconnect delay
                            interconnect_length = edge_attrs.get('length', 0.0) # Assume length might be available
                            
                            # Factor in length if available, a simple RC model could be used
                            total_interconnect_delay += interconnect_delay + (interconnect_length * 0.001) # Simple length factor
                            break # Found the connection edge
        
        total_delay = total_node_delay + total_interconnect_delay
        self.logger.debug(f"Estimated total delay for path: {total_delay:.2f} ps (nodes: {total_node_delay:.2f}, interconnect: {total_interconnect_delay:.2f})")
        
        return total_delay    
    def _find_associated_clock(self, graph: CanonicalSiliconGraph, start_node: str, end_node: str) -> str:
        """Find the clock associated with a path by tracing back to a clock source"""
        self.logger.debug(f"Finding associated clock for path from {start_node} to {end_node}.")
        
        # Helper to trace back from a node to find a clock source
        def _trace_for_clock_source(node: str, visited: set) -> Optional[str]:
            if node in visited:
                return None
            visited.add(node)

            attrs = graph.graph.nodes.get(node)
            if attrs and attrs.get('node_type') == NodeType.CLOCK.value:
                return node # Found a clock source
            
            # Recursively check predecessors
            for predecessor in graph.graph.predecessors(node):
                clock_source = _trace_for_clock_source(predecessor, visited)
                if clock_source:
                    return clock_source
            return None

        # Try to find clock source from end_node first (closer to clock capture)
        end_clock_source = _trace_for_clock_source(end_node, set())
        if end_clock_source:
            self.logger.debug(f"Found clock source '{end_clock_source}' for end node '{end_node}'.")
            return end_clock_source

        # If not found from end_node, try from start_node
        start_clock_source = _trace_for_clock_source(start_node, set())
        if start_clock_source:
            self.logger.debug(f"Found clock source '{start_clock_source}' for start node '{start_node}'.")
            return start_clock_source

        # Fallback to existing heuristic if no explicit clock source found
        start_attrs = graph.graph.nodes.get(start_node, {})
        end_attrs = graph.graph.nodes.get(end_node, {})
        
        start_clk_domain = start_attrs.get('clock_domain', 'default')
        end_clk_domain = end_attrs.get('clock_domain', 'default')
        
        final_clock = start_clk_domain if start_clk_domain != 'default' else end_clk_domain
        self.logger.debug(f"Using heuristic fallback clock domain: '{final_clock}'.")
        return final_clock
    
    def _determine_timing_risk_level(self, slack: float, delay: float, period: float) -> str:
        """Determine the timing risk level based on slack and other factors"""
        if slack < 0:
            return 'critical'  # Already violating
        elif slack < period * 0.1:  # Less than 10% of period
            return 'high'
        elif slack < period * 0.2:  # Less than 20% of period
            return 'medium'
        else:
            return 'low'
    
    def _calculate_risk_factors(self, graph: CanonicalSiliconGraph, path_nodes: List[str]) -> Dict[str, float]:
        """Calculate various risk factors for a timing path"""
        risk_factors = {}
        
        # Path length factor
        path_length = len(path_nodes)
        risk_factors['path_length_factor'] = min(path_length / 20.0, 1.0)  # Normalize
        
        # Logic depth (count of combinational logic)
        logic_depth = 0
        for node in path_nodes:
            attrs = graph.graph.nodes[node]
            cell_type = attrs.get('cell_type', '').lower()
            # Count combinational logic (not FFs or buffers)
            if not any(keyword in cell_type for keyword in ['dff', 'ff', 'latch', 'reg', 'buf', 'clk']):
                logic_depth += 1
        
        risk_factors['logic_depth_factor'] = min(logic_depth / 15.0, 1.0)
        
        # Fanout load along the path
        avg_fanout = np.mean([len(list(graph.graph.successors(node))) for node in path_nodes 
                             if node in graph.graph.nodes()])
        risk_factors['fanout_load_factor'] = min(avg_fanout / 10.0, 1.0)
        
        # Variation sensitivity (based on node types and regions)
        var_sens_count = 0
        for node in path_nodes:
            attrs = graph.graph.nodes[node]
            if attrs.get('timing_criticality', 0) > 0.7:
                var_sens_count += 1
        
        risk_factors['variation_sensitivity_factor'] = min(var_sens_count / len(path_nodes), 1.0)
        
        return risk_factors
    
    def _suggest_timing_fix(self, path_nodes: List[str], slack: float) -> str:
        """Suggest a timing fix based on the path and slack"""
        if slack < 0:
            return "Insert buffers to break long nets" if len(path_nodes) > 5 else "Upsize critical cells"
        elif slack < 0.5:
            return "Consider cell upsizing for critical path"
        elif slack < 1.0:
            return "Review path for potential optimizations"
        else:
            return "Path timing looks acceptable"
    
    def _analyze_node_timing_sensitivity(self, graph: CanonicalSiliconGraph, 
                                       constraints: Dict) -> List[Dict[str, Any]]:
        """Analyze individual nodes for timing sensitivity"""
        node_risks = []
        
        # Identify nodes with high timing criticality
        for node, attrs in graph.graph.nodes(data=True):
            criticality = attrs.get('timing_criticality', 0.0)
            
            if criticality > 0.5:  # High criticality threshold
                node_risk = {
                    'path': [node],  # Single node path
                    'delay': attrs.get('delay', 0.1),
                    'clock': attrs.get('clock_domain', 'default'),
                    'period': 10.0,  # Default period
                    'slack': -1.0,  # Placeholder
                    'risk_level': 'medium' if criticality < 0.7 else 'high',
                    'criticality_score': criticality,
                    'risk_factors': {
                        'path_length_factor': 0.0,
                        'logic_depth_factor': 0.0,
                        'fanout_load_factor': min(len(list(graph.graph.successors(node))) / 10.0, 1.0),
                        'variation_sensitivity_factor': criticality
                    },
                    'recommended_fix': f"Review cell type: {attrs.get('cell_type', 'unknown')} for timing optimization"
                }
                
                node_risks.append(node_risk)
        
        return node_risks

    def train(self, training_data: List[Dict]):
        """
        Train the timing analysis model based on historical feedback.
        
        Args:
            training_data: List of dictionaries containing features and actual timing values.
                           Each sample should have 'features' (e.g., path_length, logic_depth)
                           and 'actual_slack'.
        """
        self.logger.info(f"Training timing analyzer with {len(training_data)} samples")

        if not training_data:
            self.logger.warning("No training data provided for timing analyzer.")
            return

        # Simplified training: adjust weights based on correlation with actual slack
        # In a real scenario, this would involve a more sophisticated ML model.
        feature_factors = ['path_length', 'logic_depth', 'fanout_load', 'criticality', 'variation_sensitivity']

        for factor_key in feature_factors:
            factor_values = []
            actual_slacks = []

            for sample in training_data:
                features = sample.get('features', {})
                actual_slack = sample.get('actual_slack') # Assuming 'actual_slack' is provided directly

                # Attempt to extract relevant feature for correlation
                if factor_key == 'path_length' and 'path_length_factor' in features:
                    factor_values.append(features['path_length_factor'])
                elif factor_key == 'logic_depth' and 'logic_depth_factor' in features:
                    factor_values.append(features['logic_depth_factor'])
                elif factor_key == 'fanout_load' and 'fanout_load_factor' in features:
                    factor_values.append(features['fanout_load_factor'])
                elif factor_key == 'criticality' and 'criticality_score' in features:
                    factor_values.append(features['criticality_score'])
                elif factor_key == 'variation_sensitivity' and 'variation_sensitivity_factor' in features:
                    factor_values.append(features['variation_sensitivity_factor'])
                else:
                    continue # Skip if feature not available in sample

                if actual_slack is not None:
                    actual_slacks.append(actual_slack)
            
            if len(factor_values) > 1 and len(actual_slacks) > 1:
                # Calculate correlation between feature and actual slack
                # We expect a negative correlation if high factor value means worse slack
                corr = np.corrcoef(factor_values, actual_slacks)[0, 1]
                
                # Adjust weight: positive correlation (high factor -> good slack) decreases weight
                # Negative correlation (high factor -> bad slack) increases weight
                # Damping factor of 0.1 for slow adjustment
                current_weight = self.timing_weights.get(factor_key, 0.0)
                new_weight = current_weight - (corr * 0.1) # Subtract correlation to increase weight for negative corr
                
                # Clamp weights to a reasonable range, e.g., 0.0 to 1.0
                self.timing_weights[factor_key] = max(0.0, min(new_weight, 1.0))
                self.logger.debug(f"Adjusted timing_weights['{factor_key}'] to {self.timing_weights[factor_key]:.3f} (correlation: {corr:.2f})")
            elif len(factor_values) > 0 and len(actual_slacks) > 0:
                self.logger.debug(f"Not enough data points to calculate correlation for '{factor_key}', skipping weight adjustment.")
            else:
                self.logger.debug(f"No valid data for '{factor_key}' in training samples.")

        self.logger.info("Timing analyzer training completed.")

    def _calculate_prediction_confidence(self) -> float:
        """Calculate overall prediction confidence based on model weights"""
        base_confidence = 0.78 # A reasonable starting point for timing analysis confidence
        adjusted_confidence = base_confidence * self.model_weights
        return max(0.1, min(adjusted_confidence, 1.0))