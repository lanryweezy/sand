#!/usr/bin/env python3
"""
Learning Pipeline for Silicon Intelligence System

Integrates real telemetry data with synthetic data to create a comprehensive
learning system that focuses on differences rather than absolutes.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from data_collection.telemetry_collector import TelemetryCollector, TelemetrySnapshot
from data_generation.synthetic_generator import SyntheticDataGenerator, SyntheticExperiment
from cognitive.advanced_cognitive_system import PhysicalRiskOracle
from core.canonical_silicon_graph import CanonicalSiliconGraph


@dataclass
class LearningSample:
    """A learning sample combining features and labels for ML training"""
    features: Dict[str, Any]
    label: str  # 'will_fail_congestion', 'will_fail_timing', 'will_pass', etc.
    design_signature: str
    source_type: str  # 'real', 'synthetic'
    context: Dict[str, Any]  # Additional context for the sample


@dataclass
class DifferenceSample:
    """A sample that focuses on differences between states rather than absolute values"""
    delta_features: Dict[str, Any]
    label: str  # 'improvement', 'deterioration', 'no_change'
    design_signature: str
    stage_transition: str  # 'initial->floorplan', 'floorplan->placement', etc.
    risk_change: float  # Change in risk prediction


class LearningPipeline:
    """
    Main learning pipeline that integrates real and synthetic data
    focusing on learning differences rather than absolutes
    """
    
    def __init__(self, model_storage_path: str = "models/trained"):
        self.model_storage_path = Path(model_storage_path)
        self.model_storage_path.mkdir(parents=True, exist_ok=True)
        
        self.telemetry_collector = TelemetryCollector()
        self.synthetic_generator = SyntheticDataGenerator()
        
        # ML models
        self.risk_prediction_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.difference_learning_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Training data
        self.absolute_samples: List[LearningSample] = []
        self.difference_samples: List[DifferenceSample] = []
        
        # Feature engineering
        self.feature_columns = []
        self.difference_columns = []
    
    def integrate_real_and_synthetic_data(self) -> Dict[str, Any]:
        """Integrate real telemetry data with synthetic experiments"""
        print("Integrating real and synthetic data...")
        
        # Load real telemetry data
        real_snapshots = self._load_real_snapshots()
        real_failures = self._load_real_failures()
        
        # Load synthetic experiments
        synthetic_experiments = self._load_synthetic_experiments()
        
        # Create learning samples from real data
        real_samples = self._create_samples_from_real_data(real_snapshots, real_failures)
        
        # Create learning samples from synthetic data
        synthetic_samples = self._create_samples_from_synthetic_data(synthetic_experiments)
        
        # Combine all samples
        self.absolute_samples.extend(real_samples)
        self.absolute_samples.extend(synthetic_samples)
        
        # Create difference samples
        difference_samples = self._create_difference_samples(real_snapshots)
        self.difference_samples.extend(difference_samples)
        
        print(f"Integrated {len(self.absolute_samples)} absolute samples")
        print(f"Created {len(self.difference_samples)} difference samples")
        
        return {
            'real_samples': len(real_samples),
            'synthetic_samples': len(synthetic_samples),
            'difference_samples': len(difference_samples),
            'total_absolute_samples': len(self.absolute_samples)
        }
    
    def _load_real_snapshots(self) -> List[TelemetrySnapshot]:
        """Load real telemetry snapshots"""
        snapshots = []
        snapshot_dir = self.telemetry_collector.storage_path / "snapshots"
        
        if snapshot_dir.exists():
            for file in snapshot_dir.glob("*.json"):
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        # Convert timestamp back to datetime
                        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                        snapshot = TelemetrySnapshot(**data)
                        snapshots.append(snapshot)
                except Exception as e:
                    print(f"Error loading snapshot {file}: {e}")
        
        return snapshots
    
    def _load_real_failures(self) -> List[Dict[str, Any]]:
        """Load real failure records"""
        failures = []
        failure_dir = self.telemetry_collector.storage_path / "failures"
        
        if failure_dir.exists():
            for file in failure_dir.glob("*.json"):
                try:
                    with open(file, 'r') as f:
                        failure_data = json.load(f)
                        failures.append(failure_data)
                except Exception as e:
                    print(f"Error loading failure {file}: {e}")
        
        return failures
    
    def _load_synthetic_experiments(self) -> List[SyntheticExperiment]:
        """Load synthetic experiment records"""
        experiments = []
        experiment_dir = self.synthetic_generator.output_path / "experiments"
        
        if experiment_dir.exists():
            for file in experiment_dir.glob("*.json"):
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        # Convert timestamp back to datetime
                        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                        experiment = SyntheticExperiment(**data)
                        experiments.append(experiment)
                except Exception as e:
                    print(f"Error loading experiment {file}: {e}")
        
        return experiments
    
    def _create_samples_from_real_data(self, snapshots: List[TelemetrySnapshot], 
                                     failures: List[Dict[str, Any]]) -> List[LearningSample]:
        """Create learning samples from real telemetry data"""
        samples = []
        
        # Map failures to design signatures for labeling
        failure_map = {}
        for failure in failures:
            # Simplified signature matching
            design_key = failure['design_name'][:8]
            if design_key not in failure_map:
                failure_map[design_key] = []
            failure_map[design_key].append(failure)
        
        for snapshot in snapshots:
            # Determine label based on whether this design had failures later
            design_key = snapshot.design_signature[:8]
            has_future_failure = design_key in failure_map
            
            if has_future_failure:
                # Label based on the type of failure that occurred
                failure_types = [f['failure_type'] for f in failure_map[design_key]]
                # Use the most critical failure type
                if 'timing' in failure_types:
                    label = 'will_fail_timing'
                elif 'congestion' in failure_types:
                    label = 'will_fail_congestion'
                elif 'drc' in failure_types:
                    label = 'will_fail_drc'
                else:
                    label = 'will_fail_other'
            else:
                label = 'will_pass'
            
            sample = LearningSample(
                features=self._extract_features_from_snapshot(snapshot),
                label=label,
                design_signature=snapshot.design_signature,
                source_type='real',
                context={
                    'stage': snapshot.stage,
                    'iteration': snapshot.iteration,
                    'timestamp': snapshot.timestamp.isoformat()
                }
            )
            samples.append(sample)
        
        return samples
    
    def _create_samples_from_synthetic_data(self, experiments: List[SyntheticExperiment]) -> List[LearningSample]:
        """Create learning samples from synthetic experiments"""
        samples = []
        
        for experiment in experiments:
            # For synthetic data, we know the intended failure mode
            label = f"will_fail_{experiment.intended_failure_mode.lower()}"
            
            # Create a representative feature vector based on the parameters
            features = self._extract_features_from_experiment(experiment)
            
            sample = LearningSample(
                features=features,
                label=label,
                design_signature=experiment.experiment_id,
                source_type='synthetic',
                context={
                    'experiment_id': experiment.experiment_id,
                    'manipulation_type': experiment.manipulation_type,
                    'intended_failure_mode': experiment.intended_failure_mode,
                    'timestamp': experiment.timestamp.isoformat()
                }
            )
            samples.append(sample)
        
        return samples
    
    def _extract_features_from_snapshot(self, snapshot: TelemetrySnapshot) -> Dict[str, Any]:
        """Extract features from a telemetry snapshot"""
        features = {}
        
        # Extract metrics
        metrics = snapshot.metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                features[f'metric_{key}'] = value
            elif isinstance(value, dict):
                # For dict values, extract summary statistics
                if value:  # Only if not empty
                    features[f'metric_{key}_count'] = len(value)
                    if all(isinstance(v, (int, float)) for v in value.values()):
                        features[f'metric_{key}_mean'] = np.mean(list(value.values()))
                        features[f'metric_{key}_std'] = np.std(list(value.values()))
                        features[f'metric_{key}_max'] = max(value.values())
        
        # Extract risk prediction features
        risk_preds = snapshot.risk_predictions
        for key, value in risk_preds.items():
            if isinstance(value, (int, float)):
                features[f'risk_{key}'] = value
            elif isinstance(value, dict):
                if value:
                    features[f'risk_{key}_count'] = len(value)
                    if all(isinstance(v, (int, float)) for v in value.values()):
                        features[f'risk_{key}_mean'] = np.mean(list(value.values()))
        
        # Extract decision features
        decisions = snapshot.decisions_made
        features['decisions_count'] = len(decisions)
        
        # Extract delta features
        deltas = snapshot.delta_from_prev
        for key, value in deltas.items():
            if isinstance(value, (int, float)):
                features[f'delta_{key}'] = value
        
        return features
    
    def _extract_features_from_experiment(self, experiment: SyntheticExperiment) -> Dict[str, Any]:
        """Extract features from a synthetic experiment"""
        features = {}
        
        # Use experiment parameters as features
        for param_name, param_value in experiment.parameters.items():
            if isinstance(param_value, (int, float)):
                features[f'param_{param_name}'] = param_value
            elif isinstance(param_value, bool):
                features[f'param_{param_name}'] = 1.0 if param_value else 0.0
            else:
                # Convert other types to float if possible
                try:
                    features[f'param_{param_name}'] = float(param_value)
                except (ValueError, TypeError):
                    features[f'param_{param_name}_str_hash'] = hash(str(param_value)) % 10000 / 10000.0
        
        return features
    
    def _create_difference_samples(self, snapshots: List[TelemetrySnapshot]) -> List[DifferenceSample]:
        """Create samples that focus on differences between consecutive states"""
        difference_samples = []
        
        # Group snapshots by design signature
        snapshot_groups = {}
        for snapshot in snapshots:
            sig = snapshot.design_signature
            if sig not in snapshot_groups:
                snapshot_groups[sig] = []
            snapshot_groups[sig].append(snapshot)
        
        # For each design, create difference samples between consecutive stages
        for design_sig, design_snapshots in snapshot_groups.items():
            # Sort by timestamp
            sorted_snapshots = sorted(design_snapshots, key=lambda x: x.timestamp)
            
            for i in range(1, len(sorted_snapshots)):
                prev_snap = sorted_snapshots[i-1]
                curr_snap = sorted_snapshots[i]
                
                # Calculate delta features
                delta_features = self._calculate_delta_features(prev_snap, curr_snap)
                
                # Determine label based on risk change
                prev_risk = prev_snap.risk_predictions.get('overall_confidence', 0.5)
                curr_risk = curr_snap.risk_predictions.get('overall_confidence', 0.5)
                
                risk_change = curr_risk - prev_risk
                
                if risk_change > 0.1:
                    label = 'deterioration'
                elif risk_change < -0.1:
                    label = 'improvement'
                else:
                    label = 'no_change'
                
                stage_transition = f"{prev_snap.stage}->{curr_snap.stage}"
                
                diff_sample = DifferenceSample(
                    delta_features=delta_features,
                    label=label,
                    design_signature=design_sig,
                    stage_transition=stage_transition,
                    risk_change=risk_change
                )
                
                difference_samples.append(diff_sample)
        
        return difference_samples
    
    def _calculate_delta_features(self, prev_snap: TelemetrySnapshot, curr_snap: TelemetrySnapshot) -> Dict[str, Any]:
        """Calculate features representing the change between two snapshots"""
        delta_features = {}
        
        # Calculate deltas for metrics
        for key in set(prev_snap.metrics.keys()) | set(curr_snap.metrics.keys()):
            prev_val = prev_snap.metrics.get(key, 0)
            curr_val = curr_snap.metrics.get(key, 0)
            
            if isinstance(prev_val, (int, float)) and isinstance(curr_val, (int, float)):
                delta_features[f'delta_metric_{key}'] = curr_val - prev_val
            elif isinstance(prev_val, dict) and isinstance(curr_val, dict):
                # For dict metrics, calculate change in summary stats
                prev_vals = list(prev_val.values()) if prev_val else [0]
                curr_vals = list(curr_val.values()) if curr_val else [0]
                
                delta_features[f'delta_metric_{key}_mean'] = np.mean(curr_vals) - np.mean(prev_vals)
                delta_features[f'delta_metric_{key}_max'] = max(curr_vals) - max(prev_vals) if prev_vals and curr_vals else 0
        
        # Calculate deltas for risk predictions
        for key in set(prev_snap.risk_predictions.keys()) | set(curr_snap.risk_predictions.keys()):
            prev_val = prev_snap.risk_predictions.get(key, 0)
            curr_val = curr_snap.risk_predictions.get(key, 0)
            
            if isinstance(prev_val, (int, float)) and isinstance(curr_val, (int, float)):
                delta_features[f'delta_risk_{key}'] = curr_val - prev_val
        
        # Calculate deltas for decisions count
        delta_features['delta_decisions_count'] = len(curr_snap.decisions_made) - len(prev_snap.decisions_made)
        
        return delta_features
    
    def train_models(self) -> Dict[str, Any]:
        """Train the ML models on the collected data"""
        print("Training ML models...")
        
        results = {}
        
        # Train absolute prediction model if we have enough data
        if len(self.absolute_samples) >= 10:  # Minimum for training
            abs_results = self._train_absolute_model()
            results['absolute_model'] = abs_results
        else:
            print("Not enough data for absolute model training")
        
        # Train difference learning model if we have enough data
        if len(self.difference_samples) >= 10:  # Minimum for training
            diff_results = self._train_difference_model()
            results['difference_model'] = diff_results
        else:
            print("Not enough data for difference model training")
        
        return results
    
    def _train_absolute_model(self) -> Dict[str, Any]:
        """Train the absolute risk prediction model"""
        print(f"Training absolute model on {len(self.absolute_samples)} samples")
        
        # Prepare features and labels
        X_raw = [sample.features for sample in self.absolute_samples]
        y_raw = [sample.label for sample in self.absolute_samples]
        
        # Convert to numeric features
        X_numeric, self.feature_columns = self._convert_features_to_numeric(X_raw)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_numeric, y_raw, test_size=0.2, random_state=42)
        
        # Train model
        self.risk_prediction_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.risk_prediction_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        results = {
            'accuracy': accuracy,
            'sample_count': len(self.absolute_samples),
            'feature_count': X_numeric.shape[1],
            'classes': list(set(y_raw))
        }
        
        print(f"Absolute model accuracy: {accuracy:.3f}")
        return results
    
    def _train_difference_model(self) -> Dict[str, Any]:
        """Train the difference learning model"""
        print(f"Training difference model on {len(self.difference_samples)} samples")
        
        # Prepare features and labels
        X_raw = [sample.delta_features for sample in self.difference_samples]
        y_raw = [sample.label for sample in self.difference_samples]
        
        # Convert to numeric features
        X_numeric, self.difference_columns = self._convert_features_to_numeric(X_raw)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_numeric, y_raw, test_size=0.2, random_state=42)
        
        # Train model
        self.difference_learning_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.difference_learning_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        results = {
            'accuracy': accuracy,
            'sample_count': len(self.difference_samples),
            'feature_count': X_numeric.shape[1],
            'classes': list(set(y_raw))
        }
        
        print(f"Difference model accuracy: {accuracy:.3f}")
        return results
    
    def _convert_features_to_numeric(self, features_list: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
        """Convert list of feature dicts to numeric array"""
        # Get all unique feature names
        all_feature_names = set()
        for features in features_list:
            all_feature_names.update(features.keys())
        
        feature_names = sorted(list(all_feature_names))
        
        # Convert to matrix
        X = np.zeros((len(features_list), len(feature_names)))
        for i, features in enumerate(features_list):
            for j, feat_name in enumerate(feature_names):
                X[i, j] = features.get(feat_name, 0.0)
        
        return X, feature_names
    
    def save_models(self):
        """Save trained models to disk"""
        # Save absolute model
        abs_model_path = self.model_storage_path / "absolute_risk_model.pkl"
        with open(abs_model_path, 'wb') as f:
            pickle.dump({
                'model': self.risk_prediction_model,
                'feature_columns': self.feature_columns
            }, f)
        
        # Save difference model
        diff_model_path = self.model_storage_path / "difference_learning_model.pkl"
        with open(diff_model_path, 'wb') as f:
            pickle.dump({
                'model': self.difference_learning_model,
                'feature_columns': self.difference_columns
            }, f)
        
        print(f"Models saved to {self.model_storage_path}")
    
    def load_models(self):
        """Load trained models from disk"""
        # Load absolute model
        abs_model_path = self.model_storage_path / "absolute_risk_model.pkl"
        if abs_model_path.exists():
            with open(abs_model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.risk_prediction_model = model_data['model']
                self.feature_columns = model_data['feature_columns']
        
        # Load difference model
        diff_model_path = self.model_storage_path / "difference_learning_model.pkl"
        if diff_model_path.exists():
            with open(diff_model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.difference_learning_model = model_data['model']
                self.difference_columns = model_data['feature_columns']
        
        print(f"Models loaded from {self.model_storage_path}")
    
    def predict_risk_changes(self, prev_snapshot: TelemetrySnapshot, curr_snapshot: TelemetrySnapshot) -> Dict[str, Any]:
        """Predict whether the design state has improved, deteriorated, or stayed the same"""
        if not hasattr(self, 'difference_learning_model'):
            return {'prediction': 'unknown', 'confidence': 0.0, 'message': 'Model not trained'}
        
        # Calculate delta features
        delta_features = self._calculate_delta_features(prev_snapshot, curr_snapshot)
        
        # Convert to numeric array
        X = np.zeros((1, len(self.difference_columns)))
        for i, feat_name in enumerate(self.difference_columns):
            X[0, i] = delta_features.get(feat_name, 0.0)
        
        # Predict
        prediction = self.difference_learning_model.predict(X)[0]
        
        # Get prediction probabilities for confidence
        proba = self.difference_learning_model.predict_proba(X)[0]
        confidence = max(proba)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {cls: proba[i] for i, cls in enumerate(self.difference_learning_model.classes_)}
        }
    
    def run_complete_pipeline(self, base_rtl_path: str = None) -> Dict[str, Any]:
        """Run the complete learning pipeline"""
        print("Running complete learning pipeline...")
        
        results = {}
        
        # If we have a base RTL path, generate synthetic data
        if base_rtl_path and Path(base_rtl_path).exists():
            print(f"Generating synthetic experiments from: {base_rtl_path}")
            synthetic_experiments = self.synthetic_generator.run_all_synthetic_experiments(base_rtl_path)
            results['synthetic_experiments'] = len(synthetic_experiments)
        
        # Integrate all data
        integration_results = self.integrate_real_and_synthetic_data()
        results['integration'] = integration_results
        
        # Train models
        training_results = self.train_models()
        results['training'] = training_results
        
        # Save models
        self.save_models()
        
        print("Learning pipeline completed successfully!")
        return results


def main():
    """Example usage of the learning pipeline"""
    print("Silicon Intelligence - Learning Pipeline")
    print("=" * 60)
    
    pipeline = LearningPipeline()
    
    print("Learning pipeline initialized.")
    print("This system learns differences rather than absolutes.")
    print("It integrates real telemetry with synthetic failures.")
    print("One intelligently generated failure is worth ten clean designs.")
    
    print(f"\nStorage path: {pipeline.model_storage_path}")
    print("Ready to run complete pipeline with real and synthetic data")


if __name__ == "__main__":
    main()