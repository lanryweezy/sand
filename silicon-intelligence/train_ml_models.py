#!/usr/bin/env python3
"""
ML Model Training for Silicon Intelligence System
Trains prediction models using the prepared training dataset
"""

import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(__file__))

class DesignPPAPredictor:
    """
    ML Model for predicting PPA (Power, Performance, Area) metrics from design features
    """
    
    def __init__(self):
        self.area_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.power_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.timing_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.drc_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        self.scaler = StandardScaler()
        self.feature_columns = [
            'num_nodes', 'num_edges', 'total_area', 'total_power', 
            'avg_timing_criticality', 'avg_congestion', 'avg_area', 'avg_power'
        ]
        self.is_trained = False
    
    def _extract_features(self, sample):
        """Extract numerical features from training sample"""
        features = []
        f = sample['features']
        
        # Extract core features
        features.extend([
            f.get('num_nodes', 0),
            f.get('num_edges', 0),
            f.get('total_area', 0),
            f.get('total_power', 0),
            f.get('avg_timing_criticality', 0),
            f.get('avg_congestion', 0),
            f.get('avg_area', 0),
            f.get('avg_power', 0)
        ])
        
        return features
    
    def _extract_targets(self, sample):
        """Extract target variables from training sample"""
        # For now, we'll use simple heuristics to create targets
        # In a real system, these would come from actual EDA tool results
        features = sample['features']
        
        # Create synthetic targets based on features (in real system, these come from EDA tools)
        num_nodes = features.get('num_nodes', 100)
        num_edges = features.get('num_edges', 100)
        
        targets = {
            'area': features.get('total_area', 0) + num_nodes * 0.1,
            'power': features.get('total_power', 0) + num_nodes * 0.001,
            'timing': 2.0 + (num_nodes / 1000),  # Base 2ns + complexity
            'drc_violations': max(0, num_nodes // 200)  # Estimate DRC violations
        }
        
        return targets
    
    def prepare_training_data(self, training_dataset):
        """Prepare training data from dataset"""
        X = []  # Features
        y_area = []  # Area targets
        y_power = []  # Power targets
        y_timing = []  # Timing targets
        y_drc = []  # DRC targets
        
        for sample in training_dataset['samples']:
            # Extract features
            features = self._extract_features(sample)
            X.append(features)
            
            # Extract targets
            targets = self._extract_targets(sample)
            y_area.append(targets['area'])
            y_power.append(targets['power'])
            y_timing.append(targets['timing'])
            y_drc.append(targets['drc_violations'])
        
        return np.array(X), np.array(y_area), np.array(y_power), np.array(y_timing), np.array(y_drc)
    
    def train(self, training_dataset_path):
        """Train the PPA prediction models"""
        print("Loading training dataset...")
        with open(training_dataset_path, 'r') as f:
            dataset = json.load(f)
        
        print(f"Loaded {len(dataset['samples'])} training samples")
        
        # Prepare data
        X, y_area, y_power, y_timing, y_drc = self.prepare_training_data(dataset)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data for training and validation
        X_train, X_val, y_area_train, y_area_val = train_test_split(
            X_scaled, y_area, test_size=0.2, random_state=42
        )
        _, _, y_power_train, y_power_val = train_test_split(
            X_scaled, y_power, test_size=0.2, random_state=42
        )
        _, _, y_timing_train, y_timing_val = train_test_split(
            X_scaled, y_timing, test_size=0.2, random_state=42
        )
        _, _, y_drc_train, y_drc_val = train_test_split(
            X_scaled, y_drc, test_size=0.2, random_state=42
        )
        
        print("Training area prediction model...")
        self.area_model.fit(X_train, y_area_train)
        
        print("Training power prediction model...")
        self.power_model.fit(X_train, y_power_train)
        
        print("Training timing prediction model...")
        self.timing_model.fit(X_train, y_timing_train)
        
        print("Training DRC prediction model...")
        self.drc_model.fit(X_train, y_drc_train)
        
        # Validate models
        self._validate_models(
            X_val, y_area_val, y_power_val, y_timing_val, y_drc_val
        )
        
        self.is_trained = True
        print("All models trained successfully!")
    
    def _validate_models(self, X_val, y_area_val, y_power_val, y_timing_val, y_drc_val):
        """Validate trained models"""
        print("\nValidating models...")
        
        # Predict on validation set
        area_pred = self.area_model.predict(X_val)
        power_pred = self.power_model.predict(X_val)
        timing_pred = self.timing_model.predict(X_val)
        drc_pred = self.drc_model.predict(X_val)
        
        # Calculate metrics
        area_mae = mean_absolute_error(y_area_val, area_pred)
        area_r2 = r2_score(y_area_val, area_pred)
        
        power_mae = mean_absolute_error(y_power_val, power_pred)
        power_r2 = r2_score(y_power_val, power_pred)
        
        timing_mae = mean_absolute_error(y_timing_val, timing_pred)
        timing_r2 = r2_score(y_timing_val, timing_pred)
        
        drc_mae = mean_absolute_error(y_drc_val, drc_pred)
        drc_r2 = r2_score(y_drc_val, drc_pred)
        
        print(f"Area Model - MAE: {area_mae:.2f}, R²: {area_r2:.3f}")
        print(f"Power Model - MAE: {power_mae:.4f}, R²: {power_r2:.3f}")
        print(f"Timing Model - MAE: {timing_mae:.3f}, R²: {timing_r2:.3f}")
        print(f"DRC Model - MAE: {drc_mae:.2f}, R²: {drc_r2:.3f}")
    
    def predict(self, features):
        """Predict PPA metrics for given features"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert features to the expected format
        if isinstance(features, dict):
            # Extract features in the same order as training
            feature_vector = [
                features.get('num_nodes', 0),
                features.get('num_edges', 0),
                features.get('total_area', 0),
                features.get('total_power', 0),
                features.get('avg_timing_criticality', 0),
                features.get('avg_congestion', 0),
                features.get('avg_area', 0),
                features.get('avg_power', 0)
            ]
        else:
            feature_vector = features
        
        # Scale features
        feature_scaled = self.scaler.transform([feature_vector])
        
        # Make predictions
        area_pred = self.area_model.predict(feature_scaled)[0]
        power_pred = self.power_model.predict(feature_scaled)[0]
        timing_pred = self.timing_model.predict(feature_scaled)[0]
        drc_pred = self.drc_model.predict(feature_scaled)[0]
        
        return {
            'area': float(area_pred),
            'power': float(power_pred),
            'timing': float(timing_pred),
            'drc_violations': int(max(0, round(drc_pred)))
        }
    
    def save_model(self, model_path):
        """Save trained models to disk"""
        model_data = {
            'area_model': self.area_model,
            'power_model': self.power_model,
            'timing_model': self.timing_model,
            'drc_model': self.drc_model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, model_path)
        print(f"Models saved to {model_path}")
    
    def load_model(self, model_path):
        """Load trained models from disk"""
        model_data = joblib.load(model_path)
        
        self.area_model = model_data['area_model']
        self.power_model = model_data['power_model']
        self.timing_model = model_data['timing_model']
        self.drc_model = model_data['drc_model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = model_data['is_trained']
        
        print(f"Models loaded from {model_path}")


def main():
    print("Training ML Models for Silicon Intelligence System")
    print("=" * 60)
    
    # Initialize predictor
    predictor = DesignPPAPredictor()
    
    # Train the models
    training_data_path = "./training_dataset.json"
    
    if os.path.exists(training_data_path):
        print(f"Training models with data from {training_data_path}")
        predictor.train(training_data_path)
        
        # Save the trained models
        model_save_path = "./trained_ppa_predictor.joblib"
        predictor.save_model(model_save_path)
        
        # Test the models with a sample
        print(f"\nTesting trained models...")
        sample_features = {
            'num_nodes': 1000,
            'num_edges': 2000,
            'total_area': 10000,
            'total_power': 5,
            'avg_timing_criticality': 0.3,
            'avg_congestion': 0.2,
            'avg_area': 10,
            'avg_power': 0.005
        }
        
        predictions = predictor.predict(sample_features)
        print(f"Sample predictions for design with {sample_features['num_nodes']} nodes:")
        print(f"  Area: {predictions['area']:.2f}")
        print(f"  Power: {predictions['power']:.4f}")
        print(f"  Timing: {predictions['timing']:.3f}")
        print(f"  DRC Violations: {predictions['drc_violations']}")
        
        print(f"\n{'='*60}")
        print("ML MODEL TRAINING COMPLETE!")
        print(f"{'='*60}")
        print("Models are now ready for:")
        print("- PPA prediction for new designs")
        print("- Integration with design flow")
        print("- Validation against EDA tools")
        print("- Continuous learning from real data")
    else:
        print(f"❌ Training data not found: {training_data_path}")
        print("Please run prepare_training_data.py first")


if __name__ == "__main__":
    main()