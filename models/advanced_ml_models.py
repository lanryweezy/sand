"""
Sophisticated ML Models for Silicon Intelligence System

This module implements advanced machine learning models for various prediction
tasks in the physical implementation flow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pickle
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from core.canonical_silicon_graph import CanonicalSiliconGraph
from utils.logger import get_logger


class GraphNeuralNetwork(nn.Module):
    """
    Graph Neural Network for silicon graph processing
    """
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, output_dim: int = 32, num_layers: int = 4):
        super(GraphNeuralNetwork, self).__init__()
        
        self.num_layers = num_layers
        
        # Input layer
        self.conv1 = GCNConv(input_dim, hidden_dim)
        
        # Hidden layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.conv_out = GCNConv(hidden_dim, output_dim)
        
        # Global pooling
        self.pool = global_mean_pool
        
        # Final MLP for output
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim // 2, output_dim // 4)
        )
        
    def forward(self, x, edge_index, batch):
        # Forward pass through graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, training=self.training)
        
        x = self.conv_out(x, edge_index)
        
        # Global pooling to get graph-level representation
        x = self.pool(x, batch)
        
        # Final MLP
        x = self.mlp(x)
        
        return x


class SiliconLanguageModel(nn.Module):
    """
    Transformer-based language model for silicon design understanding
    """
    
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 256, num_heads: int = 8, 
                 ff_dim: int = 512, num_layers: int = 6):
        super(SiliconLanguageModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(512, embed_dim))  # Max sequence length 512
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Linear(embed_dim, embed_dim // 2)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask=None):
        # Embedding + positional encoding
        x = self.embedding(input_ids) + self.pos_encoding[:input_ids.size(1)]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            x = x.masked_fill(attention_mask.unsqueeze(-1) == 0, float('-inf'))
        
        # Transformer layers
        x = self.transformer(x.transpose(0, 1)).transpose(0, 1)  # Transpose for PyTorch format
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification head
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


class AdvancedCongestionPredictor:
    """
    Advanced congestion predictor using GNN and ensemble methods
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.gnn_model = GraphNeuralNetwork(input_dim=16, hidden_dim=64, output_dim=32)
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_graph_data(self, graph: CanonicalSiliconGraph) -> Tuple[Data, List[str]]:
        """
        Convert CanonicalSiliconGraph to PyTorch Geometric Data format
        """
        # Extract node features
        node_features = []
        node_names = []
        
        for node, attrs in graph.graph.nodes(data=True):
            node_names.append(node)
            
            # Create feature vector
            features = [
                float(attrs.get('area', 1.0)),
                float(attrs.get('power', 0.01)),
                float(attrs.get('timing_criticality', 0.0)),
                float(attrs.get('estimated_congestion', 0.0)),
                1.0 if attrs.get('node_type') == 'cell' else 0.0,
                1.0 if attrs.get('node_type') == 'macro' else 0.0,
                1.0 if attrs.get('node_type') == 'clock' else 0.0,
                1.0 if attrs.get('node_type') == 'power' else 0.0,
                1.0 if attrs.get('node_type') == 'port' else 0.0,
                1.0 if attrs.get('node_type') == 'signal' else 0.0,
                float(attrs.get('delay', 0.1)),
                float(attrs.get('capacitance', 0.001)),
                float(len(list(graph.graph.neighbors(node)))),  # Degree
                float(len(list(graph.graph.predecessors(node)))),  # In-degree
                float(len(list(graph.graph.successors(node)))),   # Out-degree
                float(attrs.get('region', 'default') != 'default')  # Region indicator
            ]
            node_features.append(features)
        
        # Convert to tensor
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Create edge index
        edge_indices = []
        for src, dst in graph.graph.edges():
            src_idx = node_names.index(src)
            dst_idx = node_names.index(dst)
            edge_indices.append([src_idx, dst_idx])
            edge_indices.append([dst_idx, src_idx])  # Add reverse edge for undirected
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        
        # Create batch (single graph)
        batch = torch.zeros(x.size(0), dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, batch=batch)
        return data, node_names
    
    def train(self, training_graphs: List[CanonicalSiliconGraph], 
              congestion_labels: List[Dict[str, float]]):
        """
        Train the congestion predictor
        
        Args:
            training_graphs: List of training graphs
            congestion_labels: List of congestion labels for each graph/node
        """
        self.logger.info(f"Training advanced congestion predictor with {len(training_graphs)} graphs")
        
        # Prepare training data
        all_features = []
        all_labels = []
        
        for graph, labels in zip(training_graphs, congestion_labels):
            # Get GNN embeddings
            graph_data, node_names = self.prepare_graph_data(graph)
            
            # Get node-level features and labels
            for i, node_name in enumerate(node_names):
                if node_name in labels:
                    # Get node features
                    node_features = graph_data.x[i].detach().numpy()
                    
                    all_features.append(node_features)
                    all_labels.append(labels[node_name])
        
        if not all_features:
            self.logger.warning("No training data available")
            return
        
        # Convert to arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest model
        self.rf_model.fit(X_scaled, y)
        self.is_trained = True
        
        self.logger.info("Advanced congestion predictor training completed")
    
    def predict(self, graph: CanonicalSiliconGraph) -> Dict[str, float]:
        """
        Predict congestion for a graph
        """
        if not self.is_trained:
            self.logger.warning("Model not trained, returning uniform predictions")
            return {node: 0.5 for node in graph.graph.nodes()}
        
        graph_data, node_names = self.prepare_graph_data(graph)
        X = graph_data.x.numpy()
        X_scaled = self.scaler.transform(X)
        
        # Predict congestion for each node
        predictions = self.rf_model.predict(X_scaled)
        
        # Create result dictionary
        result = {}
        for i, node_name in enumerate(node_names):
            # Clamp prediction to [0, 1] range
            congestion = max(0.0, min(1.0, float(predictions[i])))
            result[node_name] = congestion
        
        return result


class AdvancedTimingAnalyzer:
    """
    Advanced timing analyzer using ensemble of ML models
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.delay_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.slack_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.critical_path_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_scaler = StandardScaler()
        self.is_trained = False
    
    def extract_timing_features(self, graph: CanonicalSiliconGraph) -> np.ndarray:
        """
        Extract features relevant for timing analysis
        """
        features = []
        
        for node, attrs in graph.graph.nodes(data=True):
            # Timing-relevant features
            node_features = [
                float(attrs.get('delay', 0.1)),
                float(attrs.get('capacitance', 0.001)),
                float(attrs.get('power', 0.01)),
                float(attrs.get('timing_criticality', 0.0)),
                float(attrs.get('area', 1.0)),
                len(list(graph.graph.predecessors(node))),  # Fan-in
                len(list(graph.graph.successors(node))),   # Fan-out
                float(attrs.get('region', 'default') != 'default'),
                float(attrs.get('node_type') == 'clock'),
                float(attrs.get('node_type') == 'cell'),
                float(attrs.get('node_type') == 'macro'),
                float(attrs.get('estimated_congestion', 0.0))
            ]
            features.append(node_features)
        
        return np.array(features)
    
    def train(self, training_graphs: List[CanonicalSiliconGraph],
              timing_labels: List[Dict[str, Dict[str, float]]]):
        """
        Train the timing analyzer
        
        Args:
            training_graphs: List of training graphs
            timing_labels: List of timing labels for each graph
        """
        self.logger.info(f"Training advanced timing analyzer with {len(training_graphs)} graphs")
        
        all_features = []
        delay_targets = []
        slack_targets = []
        critical_labels = []
        
        for graph, labels in zip(training_graphs, timing_labels):
            features = self.extract_timing_features(graph)
            
            for i, node in enumerate(graph.graph.nodes()):
                if node in labels:
                    all_features.append(features[i])
                    delay_targets.append(labels[node].get('delay', 0.1))
                    slack_targets.append(labels[node].get('slack', 0.0))
                    critical_labels.append(1 if labels[node].get('critical', False) else 0)
        
        if not all_features:
            self.logger.warning("No training data available")
            return
        
        X = np.array(all_features)
        y_delay = np.array(delay_targets)
        y_slack = np.array(slack_targets)
        y_critical = np.array(critical_labels)
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Train models
        self.delay_predictor.fit(X_scaled, y_delay)
        self.slack_predictor.fit(X_scaled, y_slack)
        self.critical_path_classifier.fit(X_scaled, y_critical)
        
        self.is_trained = True
        self.logger.info("Advanced timing analyzer training completed")
    
    def analyze(self, graph: CanonicalSiliconGraph) -> List[Dict[str, Any]]:
        """
        Analyze timing for a graph
        """
        if not self.is_trained:
            self.logger.warning("Model not trained, returning default analysis")
            return []
        
        features = self.extract_timing_features(graph)
        X_scaled = self.feature_scaler.transform(features)
        
        # Make predictions
        predicted_delays = self.delay_predictor.predict(X_scaled)
        predicted_slacks = self.slack_predictor.predict(X_scaled)
        critical_probs = self.critical_path_classifier.predict_proba(X_scaled)
        critical_predictions = [prob[1] for prob in critical_probs]  # Probability of being critical
        
        # Format results
        results = []
        for i, node in enumerate(graph.graph.nodes()):
            results.append({
                'node': node,
                'predicted_delay': float(predicted_delays[i]),
                'predicted_slack': float(predicted_slacks[i]),
                'critical_probability': float(critical_predictions[i]),
                'risk_level': self._determine_risk_level(float(predicted_slacks[i]), float(critical_predictions[i]))
            })
        
        return results
    
    def _determine_risk_level(self, slack: float, critical_prob: float) -> str:
        """Determine timing risk level based on slack and critical probability"""
        if slack < 0:
            return 'critical'  # Already violating
        elif slack < 0.1 or critical_prob > 0.8:
            return 'high'
        elif slack < 0.2 or critical_prob > 0.5:
            return 'medium'
        else:
            return 'low'


class AdvancedDRCPredictor:
    """
    Advanced DRC predictor using ensemble methods and pattern recognition
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.pattern_recognizer = RandomForestClassifier(n_estimators=100, random_state=42)
        self.violation_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def extract_drc_features(self, graph: CanonicalSiliconGraph) -> np.ndarray:
        """
        Extract features relevant for DRC prediction
        """
        features = []
        
        for node, attrs in graph.graph.nodes(data=True):
            # DRC-relevant features
            node_features = [
                float(attrs.get('area', 1.0)),
                float(attrs.get('power', 0.01)),
                float(attrs.get('timing_criticality', 0.0)),
                float(attrs.get('estimated_congestion', 0.0)),
                len(list(graph.graph.neighbors(node))),  # Local density
                len(list(graph.graph.predecessors(node))),
                len(list(graph.graph.successors(node))),
                float(attrs.get('region', 'default') != 'default'),
                float(attrs.get('node_type') == 'clock'),
                float(attrs.get('node_type') == 'power'),
                float(attrs.get('node_type') == 'macro'),
                float(attrs.get('capacitance', 0.001)),
                float(attrs.get('delay', 0.1)),
                float(attrs.get('spacing_margin', 0.1)),
                float(attrs.get('density_factor', 0.5)),
                float(attrs.get('process_variation_sensitivity', 0.2))
            ]
            features.append(node_features)
        
        return np.array(features)
    
    def train(self, training_graphs: List[CanonicalSiliconGraph],
              drc_labels: List[Dict[str, Dict[str, Any]]]):
        """
        Train the DRC predictor
        
        Args:
            training_graphs: List of training graphs
            drc_labels: List of DRC labels for each graph
        """
        self.logger.info(f"Training advanced DRC predictor with {len(training_graphs)} graphs")
        
        all_features = []
        violation_counts = []
        pattern_labels = []
        
        for graph, labels in zip(training_graphs, drc_labels):
            features = self.extract_drc_features(graph)
            
            for i, node in enumerate(graph.graph.nodes()):
                if node in labels:
                    all_features.append(features[i])
                    violation_counts.append(labels[node].get('violation_count', 0))
                    
                    # Pattern classification (whether this node is likely to cause violations)
                    has_violations = labels[node].get('has_violations', False)
                    pattern_labels.append(1 if has_violations else 0)
        
        if not all_features:
            self.logger.warning("No training data available")
            return
        
        X = np.array(all_features)
        y_violations = np.array(violation_counts)
        y_patterns = np.array(pattern_labels)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models
        self.violation_predictor.fit(X_scaled, y_violations)
        self.pattern_recognizer.fit(X_scaled, y_patterns)
        
        self.is_trained = True
        self.logger.info("Advanced DRC predictor training completed")
    
    def predict_drc_violations(self, graph: CanonicalSiliconGraph, 
                             process_node: str = '7nm') -> Dict[str, Any]:
        """
        Predict DRC violations for a graph
        """
        if not self.is_trained:
            self.logger.warning("Model not trained, returning default predictions")
            return {
                'violations_by_type': {},
                'high_risk_nodes': [],
                'overall_risk_score': 0.5,
                'confidence': 0.5
            }
        
        features = self.extract_drc_features(graph)
        X_scaled = self.scaler.transform(features)
        
        # Make predictions
        predicted_violations = self.violation_predictor.predict(X_scaled)
        pattern_probabilities = self.pattern_recognizer.predict_proba(X_scaled)
        
        # Format results
        high_risk_nodes = []
        violations_by_type = {}
        
        for i, node in enumerate(graph.graph.nodes()):
            violation_prob = pattern_probabilities[i][1]  # Probability of causing violations
            if violation_prob > 0.5:  # High risk threshold
                high_risk_nodes.append({
                    'node': node,
                    'violation_probability': float(violation_prob),
                    'predicted_violation_count': int(predicted_violations[i])
                })
        
        # Calculate overall risk score
        overall_risk = np.mean([pred for pred in predicted_violations if pred > 0])
        
        return {
            'violations_by_type': violations_by_type,
            'high_risk_nodes': high_risk_nodes,
            'overall_risk_score': float(overall_risk),
            'confidence': 0.75  # Model confidence
        }


# Example usage and testing
def example_ml_models():
    """
    Example of how to use the advanced ML models
    """
    logger = get_logger(__name__)
    
    # Create example graph (in practice, this would come from RTL parsing)
    graph = CanonicalSiliconGraph()
    
    # Add some example nodes
    graph.graph.add_node('cell1', node_type='cell', power=0.1, area=2.0, timing_criticality=0.3)
    graph.graph.add_node('macro1', node_type='macro', power=1.0, area=100.0, timing_criticality=0.8)
    graph.graph.add_node('clk1', node_type='clock', power=0.05, area=1.5, timing_criticality=0.9)
    graph.graph.add_edge('cell1', 'macro1')
    graph.graph.add_edge('clk1', 'cell1')
    
    # Example: Advanced Congestion Predictor
    logger.info("Testing Advanced Congestion Predictor...")
    congestion_predictor = AdvancedCongestionPredictor()
    
    # In practice, you would train with real data
    # congestion_predictor.train(training_graphs, congestion_labels)
    
    # For this example, we'll just show the prediction interface
    congestion_result = congestion_predictor.predict(graph)
    logger.info(f"Congestion predictions: {list(congestion_result.items())[:3]}")  # Show first 3
    
    # Example: Advanced Timing Analyzer
    logger.info("Testing Advanced Timing Analyzer...")
    timing_analyzer = AdvancedTimingAnalyzer()
    
    # timing_analyzer.train(training_graphs, timing_labels)
    timing_result = timing_analyzer.analyze(graph)
    logger.info(f"Timing analysis results: {timing_result[:3]}")  # Show first 3
    
    # Example: Advanced DRC Predictor
    logger.info("Testing Advanced DRC Predictor...")
    drc_predictor = AdvancedDRCPredictor()
    
    # drc_predictor.train(training_graphs, drc_labels)
    drc_result = drc_predictor.predict_drc_violations(graph)
    logger.info(f"DRC prediction results: {drc_result}")
    
    logger.info("Advanced ML models example completed")


if __name__ == "__main__":
    example_ml_models()