"""
Training Data Pipeline for Predictive Models

This module provides functionality to generate synthetic training data,
load historical data, and prepare it for model training.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import random

from silicon_intelligence.utils.logger import get_logger
from silicon_intelligence.core.canonical_silicon_graph import CanonicalSiliconGraph, NodeType


class TrainingDataPipeline:
    """
    Manages the creation and preparation of training data for predictive models.
    """
    
    def __init__(self, data_storage_path: str = "./training_data"):
        self.logger = get_logger(__name__)
        self.data_storage_path = data_storage_path
        os.makedirs(data_storage_path, exist_ok=True)

    def generate_synthetic_data(self, num_samples: int = 100, 
                                 design_complexity_range: Tuple[int, int] = (100, 1000), # Number of cells
                                 process_node_options: List[str] = None) -> List[Dict[str, Any]]:
        """
        Generates synthetic training data for various predictive models.
        """
        self.logger.info(f"Generating {num_samples} synthetic training data samples.")
        
        if process_node_options is None:
            process_node_options = ['7nm', '5nm', '3nm']

        synthetic_data = []
        for i in range(num_samples):
            design_id = f"synthetic_design_{i:04d}"
            num_cells = random.randint(*design_complexity_range)
            process_node = random.choice(process_node_options)
            
            # Simulate design characteristics
            avg_fanout = random.uniform(3, 10)
            avg_connectivity_density = random.uniform(0.1, 0.8)
            avg_net_complexity = random.uniform(0.2, 0.9)
            
            # Simulate predicted vs actual metrics
            # Timing
            predicted_wns = random.uniform(-0.5, 0.2)
            actual_wns = predicted_wns + random.uniform(-0.1, 0.1) # Simulate some error
            
            # Power
            predicted_total_power = random.uniform(0.5, 5.0) # mW
            actual_total_power = predicted_total_power * random.uniform(0.9, 1.1)
            
            # Congestion
            predicted_congestion = random.uniform(0.1, 0.9)
            actual_congestion = predicted_congestion * random.uniform(0.8, 1.2)
            
            # DRC
            predicted_drc_violations = random.randint(0, 50)
            actual_drc_violations = predicted_drc_violations + random.randint(-10, 10)
            actual_drc_violations = max(0, actual_drc_violations)
            
            # Thermal
            predicted_max_temp = random.uniform(60, 90) # Celsius
            actual_max_temp = predicted_max_temp * random.uniform(0.95, 1.05)
            
            # Yield
            predicted_yield = random.uniform(0.7, 0.99)
            actual_yield = predicted_yield * random.uniform(0.9, 1.0)
            
            sample = {
                'design_id': design_id,
                'timestamp': (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat(),
                'process_node': process_node,
                'design_features': {
                    'num_cells': num_cells,
                    'avg_fanout': avg_fanout,
                    'avg_connectivity_density': avg_connectivity_density,
                    'avg_net_complexity': avg_net_complexity
                },
                'bringup_results': { # Structured to match SiliconFeedbackProcessor's expectations
                    'predicted_timing': {'wns': predicted_wns, 'tns': predicted_wns * 3},
                    'actual_timing': {'wns': actual_wns, 'tns': actual_wns * 3},
                    'predicted_power': {'total': predicted_total_power, 'leakage': predicted_total_power * 0.1},
                    'actual_power': {'total': actual_total_power, 'leakage': actual_total_power * 0.1},
                    'predicted_congestion': predicted_congestion,
                    'actual_congestion': actual_congestion,
                    'predicted_drc': {'total_violations': predicted_drc_violations},
                    'actual_drc': {'total_violations': actual_drc_violations},
                    'predicted_thermal': {'max_temp_c': predicted_max_temp},
                    'actual_thermal': {'max_temp_c': actual_max_temp},
                    'predicted_yield': predicted_yield,
                    'actual_yield': actual_yield,
                }
            }
            synthetic_data.append(sample)
        
        self.logger.info("Synthetic data generation completed.")
        return synthetic_data

    def load_historical_data(self, file_paths: List[str] = None) -> List[Dict[str, Any]]:
        """
        Loads historical training data from specified JSON files or all files in data_storage_path.
        """
        self.logger.info("Loading historical training data.")
        all_data = []
        
        if file_paths is None:
            # Load all JSON files from the data_storage_path
            file_paths = [os.path.join(self.data_storage_path, f) 
                          for f in os.listdir(self.data_storage_path) 
                          if f.endswith('.json')]
        
        for fp in file_paths:
            try:
                with open(fp, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
                self.logger.debug(f"Loaded data from {fp}")
            except Exception as e:
                self.logger.warning(f"Could not load data from {fp}: {e}")
                
        self.logger.info(f"Loaded {len(all_data)} historical data samples.")
        return all_data

    def normalize_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalizes and prepares training data.
        For now, this is a placeholder for more advanced normalization techniques.
        """
        self.logger.info("Normalizing training data.")
        # In a real scenario, this would apply min-max scaling, z-score normalization,
        # or other data preparation steps.
        
        # For demonstration, we'll just ensure all numeric values are within a reasonable range
        # and convert lists to arrays if needed for models.
        
        normalized_data = []
        for sample in data:
            # Deep copy to avoid modifying original data
            processed_sample = json.loads(json.dumps(sample))
            
            # Example: normalize some feature values if they exist
            if 'design_features' in processed_sample:
                for k, v in processed_sample['design_features'].items():
                    if isinstance(v, (int, float)):
                        # Simple min-max scaling for demonstration (assuming ranges)
                        if k == 'num_cells':
                            processed_sample['design_features'][k] = v / 1000.0
                        elif k == 'avg_fanout':
                            processed_sample['design_features'][k] = v / 10.0
                        elif k == 'avg_connectivity_density':
                            processed_sample['design_features'][k] = v # Already normalized
                        elif k == 'avg_net_complexity':
                            processed_sample['design_features'][k] = v # Already normalized
            
            # Ensure predicted/actual values are floats
            if 'bringup_results' in processed_sample:
                for key_prefix in ['predicted_', 'actual_']:
                    for metric_type in ['timing', 'power', 'congestion', 'drc', 'thermal', 'yield']:
                        full_key = f'{key_prefix}{metric_type}'
                        if full_key in processed_sample['bringup_results']:
                            value = processed_sample['bringup_results'][full_key]
                            if isinstance(value, dict): # For timing, power, drc, thermal
                                for sub_key, sub_val in value.items():
                                    if isinstance(sub_val, (int, float)):
                                        processed_sample['bringup_results'][full_key][sub_key] = float(sub_val)
                            elif isinstance(value, (int, float)): # For congestion, yield
                                processed_sample['bringup_results'][full_key] = float(value)

            normalized_data.append(processed_sample)
            
        self.logger.info("Data normalization completed.")
        return normalized_data

    def prepare_data_for_model(self, data: List[Dict[str, Any]], 
                               model_type: str) -> List[Dict[str, Any]]:
        """
        Prepares data in a specific format required by a given model.
        """
        self.logger.info(f"Preparing data for {model_type} model.")
        prepared_data = []

        for sample in data:
            features = sample.get('design_features', {})
            bringup_results = sample.get('bringup_results', {})

            if model_type == 'congestion_predictor':
                # Example: Features for congestion are design_features
                # Target is actual_congestion
                if 'actual_congestion' in bringup_results:
                    prepared_data.append({
                        'features': {
                            'connectivity_density': features.get('avg_connectivity_density', 0.0),
                            'fanout': features.get('avg_fanout', 0.0),
                            'net_complexity': features.get('avg_net_complexity', 0.0),
                            'region_density': features.get('avg_connectivity_density', 0.0), # Re-using
                            'hierarchical_depth': 0.5 # Placeholder
                        },
                        'actual_congestion': bringup_results['actual_congestion']
                    })
            elif model_type == 'drc_predictor':
                # Example: Features are design_features
                # Target is actual_drc.total_violations
                if 'actual_drc' in bringup_results and 'total_violations' in bringup_results['actual_drc']:
                    prepared_data.append({
                        'features': features, # All design features for DRC
                        'actual_drc_violations': bringup_results['actual_drc']['total_violations']
                    })
            elif model_type == 'timing_analyzer':
                # Example: Features are related to path structure, target is actual_wns
                if 'actual_timing' in bringup_results and 'wns' in bringup_results['actual_timing']:
                    prepared_data.append({
                        'features': {
                            'path_length': 0.5, # Placeholder
                            'logic_depth': 0.5, # Placeholder
                            'fanout_load': features.get('avg_fanout', 0.0),
                            'criticality': 0.5, # Placeholder
                            'variation_sensitivity': 0.5 # Placeholder
                        },
                        'actual_slack': bringup_results['actual_timing']['wns']
                    })
            # Add other model types as needed

        self.logger.info(f"Prepared {len(prepared_data)} samples for {model_type}.")
        return prepared_data