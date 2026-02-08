#!/usr/bin/env python3
"""
Advanced ML Training System for Silicon Intelligence System
Enhanced training with cross-validation, hyperparameter tuning, and model evaluation
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import joblib
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(__file__))

class AdvancedDesignPPATrainer:
    """
    Advanced ML Trainer for PPA prediction with multiple algorithms and optimization
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = [
            'num_nodes', 'num_edges', 'total_area', 'total_power', 
            'avg_timing_criticality', 'avg_congestion', 'avg_area', 'avg_power'
        ]
        self.target_metrics = ['area', 'power', 'timing', 'drc_violations']
        self.training_history = {}
        self.best_models = {}
        
    def _extract_features_and_targets(self, training_dataset):
        """Extract features and targets from training dataset"""
        samples = training_dataset['samples']
        
        # Create feature matrix
        X = []
        y_dict = {metric: [] for metric in self.target_metrics}
        
        for sample in samples:
            features = sample['features']
            
            # Extract features in consistent order
            x_row = [
                features.get('num_nodes', 0),
                features.get('num_edges', 0),
                features.get('total_area', 0),
                features.get('total_power', 0),
                features.get('avg_timing_criticality', 0),
                features.get('avg_congestion', 0),
                features.get('avg_area', 0),
                features.get('avg_power', 0)
            ]
            X.append(x_row)
            
            # Create synthetic targets based on features (in real system, from EDA tools)
            num_nodes = features.get('num_nodes', 100)
            num_edges = features.get('num_edges', 100)
            
            # More sophisticated target generation
            y_dict['area'].append(features.get('total_area', 0) + num_nodes * 0.1)
            y_dict['power'].append(features.get('total_power', 0) + num_nodes * 0.001)
            y_dict['timing'].append(2.0 + (num_nodes / 1000) + (features.get('avg_timing_criticality', 0) * 0.5))
            y_dict['drc_violations'].append(max(0, num_nodes // 200 + int(features.get('avg_congestion', 0) * 100)))
        
        return np.array(X), {k: np.array(v) for k, v in y_dict.items()}
    
    def _create_model_pipelines(self):
        """Create multiple model pipelines for comparison"""
        pipelines = {}
        
        # Random Forest Pipeline
        rf_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        ])
        
        # Gradient Boosting Pipeline
        gb_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ])
        
        # Neural Network Pipeline
        nn_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42))
        ])
        
        # Linear Regression Pipeline
        lr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        
        pipelines = {
            'RandomForest': rf_pipeline,
            'GradientBoosting': gb_pipeline,
            'NeuralNetwork': nn_pipeline,
            'LinearRegression': lr_pipeline
        }
        
        return pipelines
    
    def _hyperparameter_tuning(self, X_train, y_train, model_name, pipeline):
        """Perform hyperparameter tuning for a specific model"""
        print(f"Tuning hyperparameters for {model_name}...")
        
        if model_name == 'RandomForest':
            param_grid = {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__max_depth': [None, 10, 20],
                'regressor__min_samples_split': [2, 5, 10]
            }
        elif model_name == 'GradientBoosting':
            param_grid = {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__learning_rate': [0.01, 0.1, 0.2],
                'regressor__max_depth': [3, 5, 7]
            }
        elif model_name == 'NeuralNetwork':
            param_grid = {
                'regressor__hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100)],
                'regressor__alpha': [0.0001, 0.001, 0.01]
            }
        else:
            # For simple models, return original pipeline
            return pipeline
        
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        return grid_search.best_estimator_
    
    def train_models(self, training_dataset_path, tune_hyperparameters=True):
        """Train multiple models with cross-validation"""
        print("Loading and preparing training data...")
        with open(training_dataset_path, 'r') as f:
            dataset = json.load(f)
        
        print(f"Loaded {len(dataset['samples'])} training samples")
        
        # Extract features and targets
        X, y_dict = self._extract_features_and_targets(dataset)
        print(f"Feature matrix shape: {X.shape}")
        
        # Create model pipelines
        pipelines = self._create_model_pipelines()
        
        # Train models for each target metric
        for metric in self.target_metrics:
            print(f"\nTraining models for {metric} prediction...")
            y = y_dict[metric]
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            metric_models = {}
            
            for model_name, pipeline in pipelines.items():
                print(f"  Training {model_name} for {metric}...")
                
                # Hyperparameter tuning if requested
                if tune_hyperparameters:
                    trained_model = self._hyperparameter_tuning(
                        X_train, y_train, model_name, pipeline
                    )
                else:
                    pipeline.fit(X_train, y_train)
                    trained_model = pipeline
                
                # Cross-validation
                cv_scores = cross_val_score(
                    trained_model, X_train, y_train, cv=3, 
                    scoring='neg_mean_absolute_error'
                )
                
                # Validation metrics
                trained_model.fit(X_train, y_train)
                y_pred = trained_model.predict(X_val)
                
                mae = mean_absolute_error(y_val, y_pred)
                mse = mean_squared_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                
                print(f"    CV MAE: {-cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                print(f"    Val MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")
                
                metric_models[model_name] = {
                    'model': trained_model,
                    'cv_score': -cv_scores.mean(),
                    'val_mae': mae,
                    'val_r2': r2,
                    'predictions': y_pred,
                    'actual': y_val
                }
            
            # Select best model for this metric based on validation R²
            best_model_name = max(metric_models.keys(), 
                                key=lambda x: metric_models[x]['val_r2'])
            self.best_models[metric] = {
                'model_name': best_model_name,
                'model': metric_models[best_model_name]['model'],
                'performance': metric_models[best_model_name]
            }
            
            print(f"  Best model for {metric}: {best_model_name} (R² = {self.best_models[metric]['performance']['val_r2']:.4f})")
        
        print("\n✅ All models trained successfully!")
        return self.best_models
    
    def evaluate_models(self):
        """Comprehensive model evaluation"""
        print("\n" + "="*60)
        print("MODEL EVALUATION REPORT")
        print("="*60)
        
        evaluation_results = {}
        
        for metric, model_info in self.best_models.items():
            model_name = model_info['model_name']
            performance = model_info['performance']
            
            print(f"\n{metric.upper()} Prediction:")
            print(f"  Best Model: {model_name}")
            print(f"  Validation MAE: {performance['val_mae']:.4f}")
            print(f"  Validation R²: {performance['val_r2']:.4f}")
            print(f"  Cross-Validation Score: {performance['cv_score']:.4f}")
            
            evaluation_results[metric] = {
                'model_name': model_name,
                'val_mae': performance['val_mae'],
                'val_r2': performance['val_r2'],
                'cv_score': performance['cv_score']
            }
        
        return evaluation_results
    
    def save_models(self, model_dir="./advanced_models"):
        """Save all trained models"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save best models
        for metric, model_info in self.best_models.items():
            model_path = os.path.join(model_dir, f"best_{metric}_model.joblib")
            joblib.dump(model_info['model'], model_path)
            print(f"Saved {metric} model to {model_path}")
        
        # Save model info
        model_info_path = os.path.join(model_dir, "model_info.json")
        model_info = {}
        for metric, info in self.best_models.items():
            model_info[metric] = {
                'model_name': info['model_name'],
                'val_mae': float(info['performance']['val_mae']),
                'val_r2': float(info['performance']['val_r2']),
                'cv_score': float(info['performance']['cv_score'])
            }
        
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Model information saved to {model_info_path}")
        return model_dir
    
    def plot_training_results(self):
        """Plot training results and model comparisons"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Advanced ML Training Results', fontsize=16)
            
            for idx, (metric, model_info) in enumerate(self.best_models.items()):
                ax = axes[idx//2, idx%2]
                
                y_pred = model_info['performance']['predictions']
                y_true = model_info['performance']['actual']
                
                # Scatter plot of predictions vs actual
                ax.scatter(y_true, y_pred, alpha=0.6)
                ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Predicted Values')
                ax.set_title(f'{metric.capitalize()} Prediction\n(R² = {model_info["performance"]["val_r2"]:.3f})')
                
                # Add perfect prediction line
                lims = [
                    np.min([ax.get_xlim(), ax.get_ylim()]),
                    np.max([ax.get_xlim(), ax.get_ylim()])
                ]
                ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
                ax.set_aspect('equal')
            
            plt.tight_layout()
            plt.savefig('./advanced_training_results.png', dpi=300, bbox_inches='tight')
            print("Training results plot saved to ./advanced_training_results.png")
            
        except ImportError:
            print("Matplotlib not available, skipping plots")
    
    def predict(self, features, metric='area'):
        """Make prediction using the best model for a specific metric"""
        if metric not in self.best_models:
            raise ValueError(f"Metric {metric} not available. Available: {list(self.best_models.keys())}")
        
        model = self.best_models[metric]['model']
        
        # Convert features to the expected format
        if isinstance(features, dict):
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
        
        prediction = model.predict([feature_vector])[0]
        return float(prediction)


def main():
    print("Advanced ML Training for Silicon Intelligence System")
    print("=" * 70)
    
    # Initialize trainer
    trainer = AdvancedDesignPPATrainer()
    
    # Train models
    training_data_path = "./training_dataset.json"
    
    if os.path.exists(training_data_path):
        print(f"Training advanced models with data from {training_data_path}")
        best_models = trainer.train_models(training_data_path, tune_hyperparameters=True)
        
        # Evaluate models
        eval_results = trainer.evaluate_models()
        
        # Save models
        model_dir = trainer.save_models()
        
        # Plot results
        trainer.plot_training_results()
        
        # Test predictions
        print(f"\nTesting predictions with sample features...")
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
        
        for metric in ['area', 'power', 'timing', 'drc_violations']:
            pred = trainer.predict(sample_features, metric)
            print(f"  {metric}: {pred:.4f}")
        
        print(f"\n{'='*70}")
        print("✅ ADVANCED ML TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"Models saved to: {model_dir}")
        print("Next steps:")
        print("- Integrate with EDA tool flows")
        print("- Validate with real silicon data")
        print("- Deploy for production predictions")
        print("- Implement continuous learning")
    else:
        print(f"❌ Training data not found: {training_data_path}")
        print("Please run prepare_training_data.py first")


if __name__ == "__main__":
    main()