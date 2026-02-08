"""
Tests for the Predictive Models (Congestion, Timing, DRC)
"""

import unittest
import numpy as np
import os
import shutil
from unittest.mock import MagicMock, patch

from silicon_intelligence.models.congestion_predictor import CongestionPredictor
from silicon_intelligence.models.timing_analyzer import TimingAnalyzer
from silicon_intelligence.models.drc_predictor import DRCPredictor
from silicon_intelligence.core.canonical_silicon_graph import CanonicalSiliconGraph, NodeType, EdgeType
from silicon_intelligence.core.learning_loop import SiliconFeedbackProcessor, ModelUpdater, LearningLoopController
from silicon_intelligence.data.training_data_pipeline import TrainingDataPipeline


class TestPredictiveModels(unittest.TestCase):

    def setUp(self):
        # Setup a mock graph for testing
        self.graph = CanonicalSiliconGraph()
        self.graph.graph.add_node("clk_source", node_type=NodeType.CLOCK.value, is_clock_root=True)
        self.graph.graph.add_node("dff1", node_type=NodeType.CELL.value, cell_type="dff", timing_criticality=0.8, power=0.1, area=10.0, region="core")
        self.graph.graph.add_node("dff2", node_type=NodeType.CELL.value, cell_type="dff", timing_criticality=0.6, power=0.08, area=8.0, region="core")
        self.graph.graph.add_node("and2_1", node_type=NodeType.CELL.value, cell_type="and2", timing_criticality=0.3, power=0.01, area=2.0, region="core")
        self.graph.graph.add_node("net1", node_type=NodeType.SIGNAL.value, is_routed=False, estimated_congestion=0.7)
        self.graph.graph.add_node("net2", node_type=NodeType.SIGNAL.value, is_routed=False, estimated_congestion=0.4)
        
        self.graph.graph.add_edge("clk_source", "dff1", edge_type=EdgeType.CONNECTION.value, delay=0.1, length=10)
        self.graph.graph.add_edge("dff1", "net1", edge_type=EdgeType.CONNECTION.value, delay=0.05, length=5)
        self.graph.graph.add_edge("net1", "and2_1", edge_type=EdgeType.CONNECTION.value, delay=0.05, length=5)
        self.graph.graph.add_edge("and2_1", "net2", edge_type=EdgeType.CONNECTION.value, delay=0.05, length=5)
        self.graph.graph.add_edge("net2", "dff2", edge_type=EdgeType.CONNECTION.value, delay=0.05, length=5)

        self.drc_rules = {
            '7nm': {
                'minimum_spacing': {'poly': 0.070, 'metal1': 0.065},
                'density_rules': {'metal1': {'min_density': 0.2, 'max_density': 0.8}},
                'aspect_ratios': {'via_aspect': 4.0}
            }
        }

    def test_congestion_prediction(self):
        predictor = CongestionPredictor()
        # Ensure model_weights is initialized for tests
        predictor.model_weights = 1.0 
        prediction_results = predictor.predict(self.graph)
        
        self.assertIsInstance(prediction_results, dict)
        self.assertIn('node_congestion_map', prediction_results)
        self.assertIn('global_congestion', prediction_results)
        self.assertIn('hotspots', prediction_results)
        self.assertGreaterEqual(prediction_results['global_congestion'], 0.0)
        self.assertLessEqual(prediction_results['global_congestion'], 1.0)
        self.assertGreaterEqual(prediction_results['confidence'], 0.1)
        self.assertLessEqual(prediction_results['confidence'], 1.0)

        # Test training
        training_data_pipeline = TrainingDataPipeline()
        synthetic_data = training_data_pipeline.generate_synthetic_data(num_samples=5)
        processed_data = training_data_pipeline.prepare_data_for_model(synthetic_data, 'congestion_predictor')
        
        initial_weights = predictor.congestion_weights.copy()
        initial_model_weight = predictor.model_weights
        predictor.train(processed_data)
        
        self.assertNotEqual(initial_weights, predictor.congestion_weights)
        # Model weight should not decrease if trained without specific feedback on error
        self.assertGreaterEqual(predictor.model_weights, initial_model_weight)


    def test_timing_analysis(self):
        analyzer = TimingAnalyzer()
        # Ensure model_weights is initialized for tests
        analyzer.model_weights = 1.0
        constraints = {'clocks': [{'name': 'clk', 'period': 1000, 'source': 'clk_source'}]}
        analysis_results = analyzer.analyze(self.graph, constraints)
        
        self.assertIsInstance(analysis_results, dict)
        self.assertIn('timing_risks', analysis_results)
        self.assertIn('confidence', analysis_results)
        self.assertGreaterEqual(analysis_results['confidence'], 0.1)
        self.assertLessEqual(analysis_results['confidence'], 1.0)
        
        if analysis_results['timing_risks']:
            self.assertIn('delay', analysis_results['timing_risks'][0])
            self.assertIn('slack', analysis_results['timing_risks'][0])
            self.assertIn('criticality_score', analysis_results['timing_risks'][0])
            self.assertIsInstance(analysis_results['timing_risks'][0]['delay'], float)

        # Test training
        training_data_pipeline = TrainingDataPipeline()
        synthetic_data = training_data_pipeline.generate_synthetic_data(num_samples=5)
        processed_data = training_data_pipeline.prepare_data_for_model(synthetic_data, 'timing_analyzer')

        initial_weights = analyzer.timing_weights.copy()
        initial_model_weight = analyzer.model_weights
        analyzer.train(processed_data)

        self.assertNotEqual(initial_weights, analyzer.timing_weights)
        self.assertGreaterEqual(analyzer.model_weights, initial_model_weight) # Should not decrease with generic data


    def test_drc_prediction(self):
        predictor = DRCPredictor()
        # Ensure model_weights is initialized for tests
        predictor.model_weights = 1.0
        prediction_results = predictor.predict_drc_violations(self.graph, process_node='7nm')
        
        self.assertIsInstance(prediction_results, dict)
        self.assertIn('spacing_violations', prediction_results)
        self.assertIn('density_violations', prediction_results)
        self.assertIn('overall_risk_score', prediction_results)
        self.assertGreaterEqual(prediction_results['overall_risk_score'], 0.0)
        self.assertLessEqual(prediction_results['overall_risk_score'], 1.0)
        self.assertGreaterEqual(prediction_results['confidence'], 0.1)
        self.assertLessEqual(prediction_results['confidence'], 1.0)

        # Test training/feedback
        initial_model_weight = predictor.model_weights
        mock_actual_violations = [{'type': 'spacing', 'nodes': ['dff1', 'and2_1']}]
        mock_predicted_violations = {'spacing_violations': {'predicted_violations': mock_actual_violations}}
        predictor.update_model_from_feedback(mock_actual_violations, mock_predicted_violations)
        self.assertGreaterEqual(predictor.model_weights, initial_model_weight) # Should increase with perfect accuracy here

    def test_model_accuracy_extraction(self):
        feedback_processor = SiliconFeedbackProcessor()
        # Mock some bringup results that align with the standardized format
        mock_bringup_results = {
            'predicted_timing': {'wns': -0.1}, 'actual_timing': {'wns': -0.11},
            'predicted_power': {'total': 1.0}, 'actual_power': {'total': 1.05},
            'predicted_congestion': 0.7, 'actual_congestion': 0.72,
            'predicted_drc': {'total_violations': 5}, 'actual_drc': {'total_violations': 6},
            'predicted_thermal': {'max_temp_c': 80.0}, 'actual_thermal': {'max_temp_c': 81.0},
            'predicted_yield': 0.95, 'actual_yield': 0.94
        }
        metrics = feedback_processor._extract_performance_metrics(mock_bringup_results)
        
        self.assertIn('timing_accuracy', metrics)
        self.assertIn('power_accuracy', metrics)
        self.assertIn('congestion_accuracy', metrics)
        self.assertIn('drc_accuracy', metrics)
        self.assertIn('thermal_accuracy', metrics)
        self.assertIn('yield_accuracy', metrics)

        self.assertGreaterEqual(metrics['timing_accuracy'], 0.0)
        self.assertLessEqual(metrics['timing_accuracy'], 1.0)

    def test_hotspot_identification(self):
        predictor = CongestionPredictor()
        # Create a graph with clear hotspots
        graph_with_hotspot = CanonicalSiliconGraph()
        graph_with_hotspot.graph.add_node("nodeA", node_type=NodeType.CELL.value, estimated_congestion=0.9, region="hot_region")
        graph_with_hotspot.graph.add_node("nodeB", node_type=NodeType.CELL.value, estimated_congestion=0.95, region="hot_region")
        graph_with_hotspot.graph.add_node("nodeC", node_type=NodeType.CELL.value, estimated_congestion=0.2, region="cold_region")
        
        # Mock node congestion map for hotspot identification (beyond what _calculate_congestion_score provides)
        mock_node_congestion_map = {
            "nodeA": 0.9, "nodeB": 0.95, "nodeC": 0.2,
            "region_hot_region": 0.92, # Placeholder, will be computed from data
            "metal1": 0.85 # Placeholder, will be computed from data
        }
        
        # Test hotspot identification directly
        hotspots = predictor._identify_hotspots(
            {'nodeA': 0.9, 'nodeB': 0.95, 'nodeC': 0.2},
            {'region_hot_region': 0.92, 'region_cold_region': 0.2},
            {'metal1': 0.85, 'metal2': 0.3}
        )

        self.assertGreater(len(hotspots), 0)
        self.assertTrue(any(h['id'] == 'nodeA' for h in hotspots))
        self.assertTrue(any(h['id'] == 'region_hot_region' for h in hotspots))
        self.assertTrue(any(h['id'] == 'metal1' for h in hotspots))

if __name__ == '__main__':
    unittest.main()
