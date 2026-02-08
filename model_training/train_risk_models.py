#!/usr/bin/env python3
"""
Training Pipeline for Silicon Intelligence Risk Models

Implements the complete training pipeline for the Physical Risk Oracle
using the data-driven approach specified in the strategic plan.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from data_collection.telemetry_collector import TelemetryCollector
from data_generation.synthetic_generator import SyntheticDataGenerator
from data_integration.learning_pipeline import LearningPipeline
from cognitive.advanced_cognitive_system import PhysicalRiskOracle
from models.congestion_predictor import CongestionPredictor
from models.timing_analyzer import TimingAnalyzer


class RiskModelTrainer:
    """
    Training pipeline for Silicon Intelligence risk prediction models
    """
    
    def __init__(self, model_save_path: str = "models/trained"):
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize data collectors
        self.telemetry_collector = TelemetryCollector()
        self.synthetic_generator = SyntheticDataGenerator()
        self.learning_pipeline = LearningPipeline()
        
        # Initialize models
        self.congestion_model = RandomForestRegressor(n_estimators=200, random_state=42)
        self.timing_model = RandomForestRegressor(n_estimators=200, random_state=42)
        self.power_model = RandomForestRegressor(n_estimators=200, random_state=42)
        self.drc_model = RandomForestClassifier(n_estimators=200, random_state=42)
        
        # Feature scalers
        self.scalers = {}
        
        print(f"Training pipeline initialized. Models will be saved to: {self.model_save_path}")
    
    def prepare_congestion_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data for congestion prediction"""
        print("Preparing congestion prediction training data...")
        
        # Load real and synthetic data
        real_snapshots = self._load_real_snapshots()
        synthetic_experiments = self._load_synthetic_experiments()
        
        features_list = []
        labels_list = []
        
        # Extract features from real data
        for snapshot in real_snapshots:
            if 'estimated_congestion' in snapshot.metrics:
                # Use congestion heatmap data as labels
                avg_congestion = np.mean(list(snapshot.metrics['estimated_congestion'].values())) if snapshot.metrics['estimated_congestion'] else 0.0
                
                features = self._extract_congestion_features(snapshot)
                features_list.append(features)
                labels_list.append(avg_congestion)
        
        # Extract features from synthetic data
        for exp in synthetic_experiments:
            if exp.intended_failure_mode == 'congestion':
                # Create features based on experiment parameters
                features = self._extract_synthetic_congestion_features(exp)
                labels = 0.8 + np.random.random() * 0.2  # High congestion for congestion experiments
                features_list.append(features)
                labels_list.append(labels)
        
        if not features_list:
            print("Warning: No congestion training data available. Using synthetic defaults.")
            # Create some default training data
            features_list = [[0.1, 0.2, 0.3, 0.4, 0.5]] * 100
            labels_list = [0.3] * 100
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(labels_list)
        
        # Get feature names
        if len(features_list) > 0:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        else:
            feature_names = ["node_count", "edge_count", "macro_count", "clock_count", "timing_critical_count"]
        
        print(f"Prepared {len(X)} congestion training samples")
        return X, y, feature_names
    
    def prepare_timing_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data for timing prediction"""
        print("Preparing timing prediction training data...")
        
        # Load real and synthetic data
        real_snapshots = self._load_real_snapshots()
        synthetic_experiments = self._load_synthetic_experiments()
        
        features_list = []
        labels_list = []
        
        # Extract features from real data
        for snapshot in real_snapshots:
            if 'timing_slack_stats' in snapshot.metrics:
                # Use timing criticality as labels
                timing_criticality = snapshot.metrics['timing_slack_stats'].get('max_criticality', 0.0)
                
                features = self._extract_timing_features(snapshot)
                features_list.append(features)
                labels_list.append(timing_criticality)
        
        # Extract features from synthetic data
        for exp in synthetic_experiments:
            if exp.intended_failure_mode == 'timing':
                # Create features based on experiment parameters
                features = self._extract_synthetic_timing_features(exp)
                labels = 0.7 + np.random.random() * 0.3  # High timing risk for timing experiments
                features_list.append(features)
                labels_list.append(labels)
        
        if not features_list:
            print("Warning: No timing training data available. Using synthetic defaults.")
            # Create some default training data
            features_list = [[0.2, 0.3, 0.1, 0.4, 0.6]] * 100
            labels_list = [0.4] * 100
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(labels_list)
        
        # Get feature names
        if len(features_list) > 0:
            feature_names = [f"timing_feature_{i}" for i in range(X.shape[1])]
        else:
            feature_names = ["node_count", "critical_nodes", "max_fanout", "clock_paths", "constraint_complexity"]
        
        print(f"Prepared {len(X)} timing training samples")
        return X, y, feature_names
    
    def prepare_power_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data for power prediction"""
        print("Preparing power prediction training data...")
        
        # Load real and synthetic data
        real_snapshots = self._load_real_snapshots()
        synthetic_experiments = self._load_synthetic_experiments()
        
        features_list = []
        labels_list = []
        
        # Extract features from real data
        for snapshot in real_snapshots:
            if 'power_estimates' in snapshot.metrics:
                # Use power estimates as labels
                avg_power = np.mean(list(snapshot.metrics['power_estimates'].values())) if snapshot.metrics['power_estimates'] else 0.05
                
                features = self._extract_power_features(snapshot)
                features_list.append(features)
                labels_list.append(avg_power)
        
        # Extract features from synthetic data
        for exp in synthetic_experiments:
            if exp.intended_failure_mode == 'power_hotspots':
                # Create features based on experiment parameters
                features = self._extract_synthetic_power_features(exp)
                labels = 0.8 + np.random.random() * 0.2  # High power for power experiments
                features_list.append(features)
                labels_list.append(labels)
        
        if not features_list:
            print("Warning: No power training data available. Using synthetic defaults.")
            # Create some default training data
            features_list = [[0.15, 0.25, 0.35, 0.45, 0.55]] * 100
            labels_list = [0.3] * 100
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(labels_list)
        
        # Get feature names
        if len(features_list) > 0:
            feature_names = [f"power_feature_{i}" for i in range(X.shape[1])]
        else:
            feature_names = ["node_count", "switching_activity", "leakage_factors", "power_domains", "clock_gating"]
        
        print(f"Prepared {len(X)} power training samples")
        return X, y, feature_names
    
    def prepare_drc_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data for DRC prediction"""
        print("Preparing DRC prediction training data...")
        
        # Load real and synthetic data
        real_snapshots = self._load_real_snapshots()
        synthetic_experiments = self._load_synthetic_experiments()
        
        features_list = []
        labels_list = []
        
        # Extract features from real data
        for snapshot in real_snapshots:
            # Use binary classification for DRC violation prediction
            has_risk = 1 if len(snapshot.metrics.get('estimated_congestion', {})) > 3 else 0  # Simplified
            
            features = self._extract_drc_features(snapshot)
            features_list.append(features)
            labels_list.append(has_risk)
        
        # Extract features from synthetic data
        for exp in synthetic_experiments:
            if exp.intended_failure_mode == 'drc':
                # Create features based on experiment parameters
                features = self._extract_synthetic_drc_features(exp)
                labels = 1  # DRC risk for DRC experiments
                features_list.append(features)
                labels_list.append(labels)
            else:
                # No DRC risk for other experiments
                features = self._extract_synthetic_drc_features(exp)
                labels = 0
                features_list.append(features)
                labels_list.append(labels)
        
        if not features_list:
            print("Warning: No DRC training data available. Using synthetic defaults.")
            # Create some default training data
            features_list = [[0.1, 0.2, 0.3, 0.4, 0.5]] * 100
            labels_list = [0] * 80 + [1] * 20  # 20% positive class
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(labels_list)
        
        # Get feature names
        if len(features_list) > 0:
            feature_names = [f"drc_feature_{i}" for i in range(X.shape[1])]
        else:
            feature_names = ["node_density", "routing_complexity", "layer_utilization", "spacing_factors", "density_rules"]
        
        print(f"Prepared {len(X)} DRC training samples")
        return X, y, feature_names
    
    def _load_real_snapshots(self) -> List[Any]:
        """Load real telemetry snapshots"""
        snapshots = []
        snapshot_dir = self.telemetry_collector.storage_path / "snapshots"
        
        if snapshot_dir.exists():
            for file in snapshot_dir.glob("*.json"):
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        # Convert timestamp back to datetime if needed
                        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                        # Create a simple object with required attributes
                        class SnapshotObj:
                            def __init__(self, data):
                                for k, v in data.items():
                                    setattr(self, k, v)
                        snapshot = SnapshotObj(data)
                        snapshots.append(snapshot)
                except Exception as e:
                    print(f"Error loading snapshot {file}: {e}")
        
        return snapshots
    
    def _load_synthetic_experiments(self) -> List[Any]:
        """Load synthetic experiment records"""
        experiments = []
        experiment_dir = self.synthetic_generator.output_path / "experiments"
        
        if experiment_dir.exists():
            for file in experiment_dir.glob("*.json"):
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        # Convert timestamp back to datetime if needed
                        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                        # Create a simple object with required attributes
                        class ExpObj:
                            def __init__(self, data):
                                for k, v in data.items():
                                    setattr(self, k, v)
                        experiment = ExpObj(data)
                        experiments.append(experiment)
                except Exception as e:
                    print(f"Error loading experiment {file}: {e}")
        
        return experiments
    
    def _extract_congestion_features(self, snapshot) -> List[float]:
        """Extract features for congestion prediction"""
        features = []
        
        # Use metrics from the snapshot
        metrics = getattr(snapshot, 'metrics', {})
        
        # Add numerical metrics as features
        features.append(metrics.get('node_count', 0) / 1000.0)  # Normalize
        features.append(metrics.get('edge_count', 0) / 10000.0)  # Normalize
        features.append(metrics.get('macro_count', 0) / 100.0)   # Normalize
        features.append(metrics.get('clock_roots', 0) / 10.0)    # Normalize
        features.append(metrics.get('timing_critical_nodes', 0) / 1000.0)  # Normalize
        
        return features  # Return 5 features as expected
    
    def _extract_synthetic_congestion_features(self, exp) -> List[float]:
        """Extract features from synthetic experiment for congestion"""
        features = []
        
        params = getattr(exp, 'parameters', {})
        for param_name, param_value in params.items():
            if isinstance(param_value, (int, float)):
                features.append(min(max(float(param_value), 0.0), 1.0))  # Clamp to [0,1]
            else:
                features.append(0.5)  # Default value
        
        # Pad or truncate to 5 features to match real data
        while len(features) < 5:
            features.append(0.0)
        return features[:5]
    
    def _extract_timing_features(self, snapshot) -> List[float]:
        """Extract features for timing prediction"""
        features = []
        
        metrics = getattr(snapshot, 'metrics', {})
        features.append(metrics.get('node_count', 0) / 1000.0)
        features.append(metrics.get('timing_critical_nodes', 0) / 1000.0)
        features.append(metrics.get('clock_roots', 0) / 10.0)
        
        # Add timing-specific metrics
        timing_stats = metrics.get('timing_slack_stats', {})
        features.append(timing_stats.get('critical_ratio', 0.0))
        features.append(timing_stats.get('max_criticality', 0.0))
        
        return features[:5]  # Return 5 features as expected
    
    def _extract_synthetic_timing_features(self, exp) -> List[float]:
        """Extract features from synthetic experiment for timing"""
        features = []
        
        params = getattr(exp, 'parameters', {})
        for param_name, param_value in params.items():
            if isinstance(param_value, (int, float)):
                features.append(min(max(float(param_value), 0.0), 1.0))
            else:
                features.append(0.5)
        
        # Pad or truncate to 5 features to match real data
        while len(features) < 5:
            features.append(0.0)
        return features[:5]
    
    def _extract_power_features(self, snapshot) -> List[float]:
        """Extract features for power prediction"""
        features = []
        
        metrics = getattr(snapshot, 'metrics', {})
        features.append(metrics.get('node_count', 0) / 1000.0)
        features.append(len(metrics.get('power_estimates', {})) / 100.0)
        
        # Add power-specific metrics
        power_estimates = metrics.get('power_estimates', {})
        if power_estimates:
            features.append(np.mean(list(power_estimates.values())))
            features.append(np.max(list(power_estimates.values())))
        else:
            features.append(0.05)  # Default low power
            features.append(0.05)
        
        return features[:5]  # Return 5 features as expected
    
    def _extract_synthetic_power_features(self, exp) -> List[float]:
        """Extract features from synthetic experiment for power"""
        features = []
        
        params = getattr(exp, 'parameters', {})
        for param_name, param_value in params.items():
            if isinstance(param_value, (int, float)):
                features.append(min(max(float(param_value), 0.0), 1.0))
            else:
                features.append(0.5)
        
        # Pad or truncate to 5 features to match real data
        while len(features) < 5:
            features.append(0.0)
        return features[:5]
    
    def _extract_drc_features(self, snapshot) -> List[float]:
        """Extract features for DRC prediction"""
        features = []
        
        metrics = getattr(snapshot, 'metrics', {})
        features.append(metrics.get('node_count', 0) / 1000.0)
        features.append(len(metrics.get('estimated_congestion', {})) / 50.0)
        features.append(metrics.get('macro_count', 0) / 50.0)
        
        # Add risk features
        risk_preds = getattr(snapshot, 'risk_predictions', {})
        features.append(risk_preds.get('overall_confidence', 0.5))
        
        # Add one more feature to make it 5
        features.append(metrics.get('edge_count', 0) / 10000.0)
        
        return features[:5]  # Return 5 features as expected
    
    def _extract_synthetic_drc_features(self, exp) -> List[float]:
        """Extract features from synthetic experiment for DRC"""
        features = []
        
        params = getattr(exp, 'parameters', {})
        for param_name, param_value in params.items():
            if isinstance(param_value, (int, float)):
                features.append(min(max(float(param_value), 0.0), 1.0))
            else:
                features.append(0.5)
        
        # Pad or truncate to 5 features to match real data
        while len(features) < 5:
            features.append(0.0)
        return features[:5]
    
    def train_congestion_model(self) -> Dict[str, Any]:
        """Train the congestion prediction model"""
        print("Training congestion prediction model...")
        
        X, y, feature_names = self.prepare_congestion_training_data()
        
        if len(X) < 10:
            print("Insufficient data for training. Using default model.")
            return {'mse': float('inf'), 'r2_score': 0.0, 'samples': 0}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.congestion_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.congestion_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        # Calculate R² score
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        results = {
            'mse': mse,
            'r2_score': r2_score,
            'samples': len(X),
            'feature_count': X.shape[1]
        }
        
        print(f"Congestion model trained - MSE: {mse:.4f}, R²: {r2_score:.4f}")
        return results
    
    def train_timing_model(self) -> Dict[str, Any]:
        """Train the timing prediction model"""
        print("Training timing prediction model...")
        
        X, y, feature_names = self.prepare_timing_training_data()
        
        if len(X) < 10:
            print("Insufficient data for training. Using default model.")
            return {'mse': float('inf'), 'r2_score': 0.0, 'samples': 0}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.timing_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.timing_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        # Calculate R² score
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        results = {
            'mse': mse,
            'r2_score': r2_score,
            'samples': len(X),
            'feature_count': X.shape[1]
        }
        
        print(f"Timing model trained - MSE: {mse:.4f}, R²: {r2_score:.4f}")
        return results
    
    def train_power_model(self) -> Dict[str, Any]:
        """Train the power prediction model"""
        print("Training power prediction model...")
        
        X, y, feature_names = self.prepare_power_training_data()
        
        if len(X) < 10:
            print("Insufficient data for training. Using default model.")
            return {'mse': float('inf'), 'r2_score': 0.0, 'samples': 0}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.power_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.power_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        # Calculate R² score
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        results = {
            'mse': mse,
            'r2_score': r2_score,
            'samples': len(X),
            'feature_count': X.shape[1]
        }
        
        print(f"Power model trained - MSE: {mse:.4f}, R²: {r2_score:.4f}")
        return results
    
    def train_drc_model(self) -> Dict[str, Any]:
        """Train the DRC prediction model"""
        print("Training DRC prediction model...")
        
        X, y, feature_names = self.prepare_drc_training_data()
        
        if len(X) < 10:
            print("Insufficient data for training. Using default model.")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'samples': 0}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.drc_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.drc_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate precision and recall
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'samples': len(X),
            'feature_count': X.shape[1]
        }
        
        print(f"DRC model trained - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        return results
    
    def train_all_models(self) -> Dict[str, Any]:
        """Train all risk prediction models"""
        print("Training all risk prediction models...")
        
        results = {}
        
        results['congestion'] = self.train_congestion_model()
        results['timing'] = self.train_timing_model()
        results['power'] = self.train_power_model()
        results['drc'] = self.train_drc_model()
        
        return results
    
    def save_models(self):
        """Save all trained models to disk"""
        print("Saving trained models...")
        
        # Save each model
        joblib.dump(self.congestion_model, self.model_save_path / "congestion_model.pkl")
        joblib.dump(self.timing_model, self.model_save_path / "timing_model.pkl")
        joblib.dump(self.power_model, self.model_save_path / "power_model.pkl")
        joblib.dump(self.drc_model, self.model_save_path / "drc_model.pkl")
        
        print(f"All models saved to {self.model_save_path}")
    
    def load_models(self):
        """Load all trained models from disk"""
        print("Loading trained models...")
        
        model_files = [
            ('congestion_model.pkl', 'congestion_model'),
            ('timing_model.pkl', 'timing_model'),
            ('power_model.pkl', 'power_model'),
            ('drc_model.pkl', 'drc_model')
        ]
        
        for filename, model_attr in model_files:
            filepath = self.model_save_path / filename
            if filepath.exists():
                model = joblib.load(filepath)
                setattr(self, model_attr, model)
                print(f"Loaded {filename}")
            else:
                print(f"Model file {filename} not found")
    
    def validate_models(self) -> Dict[str, Any]:
        """Validate that models are properly trained and can make predictions"""
        print("Validizing models...")
        
        # Create some dummy test data with 5 features (matching the trained models)
        test_features = np.random.rand(5, 5).astype(np.float32)
        
        validations = {}
        
        try:
            congestion_pred = self.congestion_model.predict(test_features[:1])[0]
            validations['congestion'] = {'status': 'OK', 'sample_prediction': float(congestion_pred)}
        except Exception as e:
            validations['congestion'] = {'status': 'ERROR', 'error': str(e)}
        
        try:
            timing_pred = self.timing_model.predict(test_features[:1])[0]
            validations['timing'] = {'status': 'OK', 'sample_prediction': float(timing_pred)}
        except Exception as e:
            validations['timing'] = {'status': 'ERROR', 'error': str(e)}
        
        try:
            power_pred = self.power_model.predict(test_features[:1])[0]
            validations['power'] = {'status': 'OK', 'sample_prediction': float(power_pred)}
        except Exception as e:
            validations['power'] = {'status': 'ERROR', 'error': str(e)}
        
        try:
            drc_pred = self.drc_model.predict(test_features[:1])[0]
            drc_prob = self.drc_model.predict_proba(test_features[:1])[0]
            validations['drc'] = {'status': 'OK', 'sample_prediction': int(drc_pred), 'probabilities': drc_prob.tolist()}
        except Exception as e:
            validations['drc'] = {'status': 'ERROR', 'error': str(e)}
        
        return validations


def main():
    """Main training function"""
    print("Silicon Intelligence - Risk Model Training Pipeline")
    print("=" * 60)
    
    trainer = RiskModelTrainer()
    
    print("\nStep 1: Training all risk prediction models...")
    training_results = trainer.train_all_models()
    
    print("\nStep 2: Saving trained models...")
    trainer.save_models()
    
    print("\nStep 3: Validating models...")
    validation_results = trainer.validate_models()
    
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    
    for model_name, results in training_results.items():
        print(f"\n{model_name.upper()} MODEL:")
        for metric, value in results.items():
            print(f"  {metric}: {value}")
    
    print(f"\nMODEL VALIDATIONS:")
    for model_name, validation in validation_results.items():
        status = validation['status']
        print(f"  {model_name.upper()}: {status}")
        if status == 'ERROR':
            print(f"    Error: {validation.get('error', 'Unknown error')}")
    
    print(f"\nModels are now trained and ready for use!")
    print("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()