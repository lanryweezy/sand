"""
Congestion Predictor - Predicts routing congestion based on design characteristics

This module implements a model to predict where routing congestion will occur
in the physical implementation based on the RTL structure and connectivity.
"""

from typing import Dict, List, Any
import numpy as np
from core.canonical_silicon_graph import CanonicalSiliconGraph, NodeType
from utils.logger import get_logger


class CongestionPredictor:
    """
    Congestion Predictor - predicts routing congestion in design
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.model_trained = False
        self.congestion_weights = self._initialize_weights()
        self.model_weights = 1.0 # General weight for prediction confidence
    
    def _initialize_weights(self) -> Dict[str, float]:
        """Initialize weights for congestion prediction factors"""
        return {
            'connectivity_density': 0.3,
            'fanout': 0.25,
            'net_complexity': 0.2,
            'region_density': 0.15,
            'hierarchical_depth': 0.1
        }
    
    def predict(self, silicon_graph: CanonicalSiliconGraph) -> Dict[str, Any]:
        """
        Predict congestion levels for different regions/nodes in the graph,
        and provide global, regional, and layer-specific congestion metrics.
        
        Args:
            silicon_graph: The canonical silicon graph to analyze
            
        Returns:
            Dictionary with comprehensive congestion analysis:
            'node_congestion_map': Node/region IDs to congestion probability (0-1)
            'global_congestion': Overall congestion score for the design
            'region_congestion_map': Congestion scores per region
            'layer_congestion_map': Congestion scores per metal layer
            'hotspots': Identified congestion hotspots
            'confidence': Overall prediction confidence
        """
        self.logger.info("Predicting congestion levels")
        
        node_congestion_map = {}
        
        # Calculate congestion for each node
        for node, attrs in silicon_graph.graph.nodes(data=True):
            if attrs.get('node_type') != NodeType.PORT.value:  # Skip ports
                congestion_score = self._calculate_congestion_score(silicon_graph, node, attrs)
                node_congestion_map[node] = min(max(congestion_score, 0.0), 1.0)  # Clamp to [0, 1]
        
        # Calculate global congestion
        global_congestion = np.mean(list(node_congestion_map.values())) if node_congestion_map else 0.0
        self.logger.info(f"Global congestion: {global_congestion:.2f}")

        # Calculate regional congestion
        region_congestion_map = self._calculate_regional_congestion(silicon_graph, node_congestion_map)
        self.logger.info(f"Regional congestion for {len(region_congestion_map)} regions calculated.")

        # Calculate layer congestion
        layer_congestion_map = self._calculate_layer_congestion(silicon_graph, node_congestion_map)
        self.logger.info(f"Layer congestion for {len(layer_congestion_map)} layers calculated.")

        # Identify hotspots
        hotspots = self._identify_hotspots(node_congestion_map, region_congestion_map, layer_congestion_map)
        self.logger.info(f"Identified {len(hotspots)} congestion hotspots.")

        confidence = self._calculate_prediction_confidence()
        
        self.logger.info(f"Predicted congestion for {len(node_congestion_map)} elements with confidence: {confidence:.2f}")
        return {
            'node_congestion_map': node_congestion_map,
            'global_congestion': global_congestion,
            'region_congestion_map': region_congestion_map,
            'layer_congestion_map': layer_congestion_map,
            'hotspots': hotspots,
            'confidence': confidence
        }
        congestion_map.update(region_congestion)
        
        self.logger.info(f"Predicted congestion for {len(congestion_map)} elements")
        return congestion_map
    
    def _calculate_congestion_score(self, graph: CanonicalSiliconGraph, node: str, attrs: Dict) -> float:
        """Calculate congestion score for a specific node"""
        scores = {}
        
        # 1. Connectivity density - how connected is this node?
        connectivity = self._calculate_connectivity_score(graph, node)
        scores['connectivity'] = connectivity * self.congestion_weights['connectivity_density']
        
        # 2. Fanout - how many outputs does this node drive?
        fanout = self._calculate_fanout_score(graph, node)
        scores['fanout'] = fanout * self.congestion_weights['fanout']
        
        # 3. Net complexity - complexity of nets connected to this node
        net_complexity = self._calculate_net_complexity_score(graph, node)
        scores['net_complexity'] = net_complexity * self.congestion_weights['net_complexity']
        
        # 4. Region density - how dense is the region where this node is located?
        region_density = self._calculate_region_density_score(graph, node, attrs)
        scores['region_density'] = region_density * self.congestion_weights['region_density']
        
        # 5. Hierarchical depth - deeper hierarchy may mean more complex routing
        hier_depth = self._calculate_hier_depth_score(graph, node)
        scores['hier_depth'] = hier_depth * self.congestion_weights['hierarchical_depth']
        
        # Combine all scores
        total_score = sum(scores.values())
        
        # Normalize to 0-1 range
        normalized_score = min(total_score, 1.0)
        
        return normalized_score
    
    def _calculate_connectivity_score(self, graph: CanonicalSiliconGraph, node: str) -> float:
        """Calculate connectivity-based congestion score"""
        # Count total connections (both incoming and outgoing)
        total_degree = graph.graph.degree(node)
        
        # Normalize based on average degree in the graph
        all_degrees = [graph.graph.degree(n) for n in graph.graph.nodes()]
        avg_degree = np.mean(all_degrees) if all_degrees else 1.0
        max_degree = max(all_degrees) if all_degrees else 1.0
        
        if max_degree > 0:
            # Score based on how much more connected this node is compared to average
            connectivity_score = min(total_degree / avg_degree, max_degree / avg_degree)
            # Normalize to 0-1
            connectivity_score = min(connectivity_score / 5.0, 1.0)  # Cap at 5x average
        else:
            connectivity_score = 0.0
        
        return connectivity_score
    
    def _calculate_fanout_score(self, graph: CanonicalSiliconGraph, node: str) -> float:
        """Calculate fanout-based congestion score"""
        # Count number of successors (outputs)
        fanout = len(list(graph.graph.successors(node)))
        
        # Find max fanout in the graph to normalize
        all_fanouts = [len(list(graph.graph.successors(n))) for n in graph.graph.nodes()]
        max_fanout = max(all_fanouts) if all_fanouts else 1.0
        
        if max_fanout > 0:
            fanout_score = fanout / max_fanout
        else:
            fanout_score = 0.0
        
        return fanout_score
    
    def _calculate_net_complexity_score(self, graph: CanonicalSiliconGraph, node: str) -> float:
        """Calculate net complexity-based congestion score"""
        # Complexity based on the number of nodes connected to nets this node connects to
        complexity = 0.0
        
        # Find all nets this node connects to
        for successor in graph.graph.successors(node):
            # If successor is a net (based on naming convention or attributes)
            succ_attrs = graph.graph.nodes[successor]
            if succ_attrs.get('cell_type') == 'NET':
                # Count how many other nodes connect to this net
                net_fanout = len(list(graph.graph.successors(successor)))
                complexity += net_fanout
        
        # Find max complexity in the graph to normalize
        all_complexities = []
        for n in graph.graph.nodes():
            node_complexity = 0
            for succ in graph.graph.successors(n):
                succ_attrs = graph.graph.nodes[succ]
                if succ_attrs.get('cell_type') == 'NET':
                    net_fanout = len(list(graph.graph.successors(succ)))
                    node_complexity += net_fanout
            all_complexities.append(node_complexity)
        
        max_complexity = max(all_complexities) if all_complexities else 1.0
        
        if max_complexity > 0:
            complexity_score = min(complexity / max_complexity, 1.0)
        else:
            complexity_score = 0.0
        
        return complexity_score
    
    def _calculate_region_density_score(self, graph: CanonicalSiliconGraph, node: str, attrs: Dict) -> float:
        """Calculate region density-based congestion score"""
        region = attrs.get('region', 'default')
        
        # Count nodes in the same region
        region_nodes = [n for n, n_attrs in graph.graph.nodes(data=True) 
                       if n_attrs.get('region') == region]
        
        # Calculate density as fraction of total nodes
        total_nodes = len(graph.graph.nodes())
        region_density = len(region_nodes) / total_nodes if total_nodes > 0 else 0.0
        
        # Also consider area utilization in the region if available
        # For now, we'll just use node count as proxy
        
        return region_density
    
    def _calculate_hier_depth_score(self, graph: CanonicalSiliconGraph, node: str) -> float:
        """Calculate hierarchical depth-based congestion score"""
        # For now, use a simple proxy: nodes with many connections to other regions
        # might be at higher hierarchy levels
        
        attrs = graph.graph.nodes[node]
        current_region = attrs.get('region', 'default')
        
        # Count connections to other regions
        inter_region_connections = 0
        for neighbor in graph.graph.neighbors(node):
            neighbor_attrs = graph.graph.nodes[neighbor]
            neighbor_region = neighbor_attrs.get('region', 'default')
            if neighbor_region != current_region:
                inter_region_connections += 1
        
        # Normalize by total connections
        total_connections = graph.graph.degree(node)
        if total_connections > 0:
            hier_score = inter_region_connections / total_connections
        else:
            hier_score = 0.0
        
        return hier_score
    
    def _calculate_regional_congestion(self, graph: CanonicalSiliconGraph, 
                                     node_congestion: Dict[str, float]) -> Dict[str, float]:
        """Calculate regional congestion scores"""
        region_congestion = {}
        
        # Group node congestions by region
        region_nodes = {}
        for node, cong_score in node_congestion.items():
            if node in graph.graph.nodes():
                attrs = graph.graph.nodes[node]
                region = attrs.get('region', 'default')
                
                if region not in region_nodes:
                    region_nodes[region] = []
                region_nodes[region].append(cong_score)
        
        # Calculate average congestion per region
        for region, cong_scores in region_nodes.items():
            if cong_scores:
                region_congestion[f"region_{region}"] = np.mean(cong_scores)
        
        return region_congestion
    
    def train(self, training_data: List[Dict]):
        """
        Train the congestion prediction model
        
        Args:
            training_data: List of dictionaries containing features and actual congestion values
        """
        self.logger.info(f"Training congestion predictor with {len(training_data)} samples")
        
        # This is a simplified training approach
        # In a real implementation, this would use ML techniques
        
        if not training_data:
            self.logger.warning("No training data provided")
            return
        
        # Calculate feature correlations with congestion
        feature_names = ['connectivity', 'fanout', 'net_complexity', 'region_density', 'hierarchical_depth']
        
        for feat_name in feature_names:
            values = []
            congestion_vals = []
            
            for sample in training_data:
                # Need to map sample's 'features' to the keys used in congestion_weights
                # Assuming training_data samples now contain 'features' with raw scores
                features = sample.get('features', {})
                if feat_name in features and 'actual_congestion' in sample:
                    values.append(features[feat_name])
                    congestion_vals.append(sample['actual_congestion'])
            
            if len(values) > 1 and len(congestion_vals) > 1:
                # Calculate correlation and adjust weight
                corr = np.corrcoef(values, congestion_vals)[0, 1]
                # Update weight based on correlation (with damping factor)
                # Ensure the feature name matches the key in self.congestion_weights
                weight_key = f'{feat_name.replace("_", "")}_density' if 'density' not in feat_name else feat_name
                weight_key = weight_key.replace('path', 'connectivity') # handle path_length_factor -> connectivity_density
                
                # Dynamic adjustment of weights: positive correlation to good outcome increases weight
                if weight_key in self.congestion_weights:
                    current_weight = self.congestion_weights[weight_key]
                    # Simple heuristic: positive correlation (feature higher, congestion higher) increases weight
                    # Negative correlation (feature higher, congestion lower) decreases weight
                    new_weight = current_weight + (corr * 0.05) # Small learning rate
                    self.congestion_weights[weight_key] = max(0.01, min(new_weight, 1.0)) # Clamp to [0.01, 1.0]
                    self.logger.debug(f"Adjusted congestion_weights['{weight_key}'] to {self.congestion_weights[weight_key]:.3f} (correlation: {corr:.2f})")
                else:
                    self.logger.warning(f"Feature '{feat_name}' does not have a corresponding weight '{weight_key}'.")
            elif len(values) > 0 and len(congestion_vals) > 0:
                self.logger.debug(f"Not enough data points to calculate correlation for '{feat_name}', skipping weight adjustment.")
            else:
                self.logger.debug(f"No valid data for '{feat_name}' in training samples.")
        
        self.model_trained = True
        self.logger.info("Congestion predictor training completed")

    def _calculate_prediction_confidence(self) -> float:
        """Calculate overall prediction confidence based on model weights"""
        base_confidence = 0.75 # A reasonable starting point for congestion confidence
        adjusted_confidence = base_confidence * self.model_weights
        return max(0.1, min(adjusted_confidence, 1.0))

    def _calculate_layer_congestion(self, graph: CanonicalSiliconGraph, node_congestion_map: Dict[str, float]) -> Dict[str, float]:
        """
        Estimates congestion for each metal layer.
        This is a heuristic and would depend on actual routing information.
        """
        layer_congestion = {
            'metal1': 0.0, 'metal2': 0.0, 'metal3': 0.0, 
            'metal4': 0.0, 'metal5': 0.0, 'metal6': 0.0
        }
        layer_node_counts = {layer: 0 for layer in layer_congestion.keys()}

        # For simplicity, assume nodes in congested areas contribute to congestion on certain layers
        # In a real tool, this would be based on actual routing density per layer
        for node, congestion_score in node_congestion_map.items():
            # Heuristic: if a node has high congestion, it contributes to congestion on adjacent layers
            if congestion_score > 0.6: 
                # Assign to a primary layer, and some to adjacent layers
                primary_layer = 'metal3' # Example: assume mid-layer is most used
                if 'metal' in node and len(node) > 5 and node[5].isdigit(): # Simple heuristic for nodes with layer info
                    primary_layer = f"metal{node[5]}"
                
                if primary_layer in layer_congestion:
                    layer_congestion[primary_layer] += congestion_score
                    layer_node_counts[primary_layer] += 1
                    # Spread to adjacent layers
                    layer_num = int(primary_layer.replace('metal', ''))
                    if layer_num > 1:
                        layer_congestion[f'metal{layer_num-1}'] += congestion_score * 0.2
                    if layer_num < 6:
                        layer_congestion[f'metal{layer_num+1}'] += congestion_score * 0.2
        
        # Average the congestion per layer
        for layer in layer_congestion:
            if layer_node_counts[layer] > 0:
                layer_congestion[layer] /= layer_node_counts[layer]
            else:
                layer_congestion[layer] = 0.1 # Baseline if no nodes contributing

        return layer_congestion

    def _identify_hotspots(self, node_congestion_map: Dict[str, float], 
                           region_congestion_map: Dict[str, float], 
                           layer_congestion_map: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Identifies congestion hotspots based on node, region, and layer congestion maps.
        """
        self.logger.debug("Identifying congestion hotspots.")
        hotspots = []
        
        # Node-level hotspots
        for node_id, score in node_congestion_map.items():
            if score > 0.8: # Threshold for high congestion at node level
                hotspots.append({'type': 'node', 'id': node_id, 'score': score})
        
        # Region-level hotspots
        for region_id, score in region_congestion_map.items():
            if score > 0.7: # Threshold for high congestion at region level
                hotspots.append({'type': 'region', 'id': region_id, 'score': score})
        
        # Layer-level hotspots
        for layer_id, score in layer_congestion_map.items():
            if score > 0.6: # Threshold for high congestion at layer level
                hotspots.append({'type': 'layer', 'id': layer_id, 'score': score})
        
        # Sort hotspots by score in descending order
        hotspots.sort(key=lambda x: x['score'], reverse=True)
        
        self.logger.debug(f"Found {len(hotspots)} hotspots.")
        return hotspots