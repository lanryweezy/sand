#!/usr/bin/env python3
"""
Advanced ML Training with Neural Networks for Silicon Intelligence System
Using scikit-learn's MLPRegressor for neural network capabilities
"""

import json
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import pandas as pd

sys.path.append(os.path.dirname(__file__))

class AdvancedMLTrainer:
    """
    Advanced ML Trainer with Neural Networks and Ensemble Methods
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
        self.cv_results = {}
        
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
            total_area = features.get('total_area', 1000)
            total_power = features.get('total_power', 1)
            
            # More sophisticated target generation
            y_dict['area'].append(total_area + num_nodes * 0.1 + np.random.normal(0, 10))
            y_dict['power'].append(total_power + num_nodes * 0.001 + np.random.normal(0, 0.01))
            y_dict['timing'].append(2.0 + (num_nodes / 1000) + (features.get('avg_timing_criticality', 0) * 0.5) + np.random.normal(0, 0.05))
            y_dict['drc_violations'].append(max(0, num_nodes // 200 + int(features.get('avg_congestion', 0) * 100) + np.random.poisson(1)))
        
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
        
        # Neural Network (MLP) Pipeline
        nn_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10
            ))
        ])
        
        # Linear Regression Pipeline
        lr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        
        # Ridge Regression Pipeline
        ridge_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=1.0, random_state=42))
        ])
        
        # Support Vector Regression Pipeline
        svr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', SVR(kernel='rbf', C=1.0, gamma='scale'))
        ])
        
        pipelines = {
            'RandomForest': rf_pipeline,
            'GradientBoosting': gb_pipeline,
            'NeuralNetwork': nn_pipeline,
            'LinearRegression': lr_pipeline,
            'RidgeRegression': ridge_pipeline,
            'SVR': svr_pipeline
        }
        
        return pipelines
    
    def train_models(self, training_dataset_path: str, use_cv: bool = True, cv_folds: int = 5):
        """Train models with cross-validation and hyperparameter tuning"""
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
                
                # Cross-validation if requested
                if use_cv:
                    cv_scores = cross_val_score(
                        pipeline, X_train, y_train, cv=cv_folds, 
                        scoring='neg_mean_absolute_error', n_jobs=-1
                    )
                    cv_mae = -cv_scores.mean()
                    cv_std = cv_scores.std()
                else:
                    cv_mae = 0.0
                    cv_std = 0.0
                
                # Train and validate
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_val)
                
                mae = mean_absolute_error(y_val, y_pred)
                mse = mean_squared_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                
                print(f"    CV MAE: {cv_mae:.4f} (+/- {cv_std * 2:.4f})")
                print(f"    Val MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")
                
                metric_models[model_name] = {
                    'model': pipeline,
                    'cv_score': cv_mae,
                    'cv_std': cv_std,
                    'val_mae': mae,
                    'val_mse': mse,
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
        
        print("\nAll models trained successfully!")
        return self.best_models
    
    def hyperparameter_tuning(self, training_dataset_path: str):
        """Perform hyperparameter tuning for the best models"""
        print("\nPerforming hyperparameter tuning...")
        
        with open(training_dataset_path, 'r') as f:
            dataset = json.load(f)
        
        X, y_dict = self._extract_features_and_targets(dataset)
        
        # Define parameter grids for key models
        param_grids = {
            'RandomForest': {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__max_depth': [None, 10, 20, 30],
                'regressor__min_samples_split': [2, 5, 10],
                'regressor__min_samples_leaf': [1, 2, 4]
            },
            'NeuralNetwork': {
                'regressor__hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100), (100, 50, 25)],
                'regressor__alpha': [0.0001, 0.001, 0.01, 0.1],
                'regressor__learning_rate_init': [0.001, 0.01, 0.1]
            },
            'GradientBoosting': {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__learning_rate': [0.01, 0.1, 0.2],
                'regressor__max_depth': [3, 5, 7],
                'regressor__subsample': [0.8, 0.9, 1.0]
            }
        }
        
        for metric in self.target_metrics:
            y = y_dict[metric]
            
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            best_score = float('-inf')
            best_model = None
            best_params = None
            
            for model_name, param_grid in param_grids.items():
                print(f"  Tuning {model_name} for {metric}...")

                # Create base regressor
                if model_name == 'RandomForest':
                    regressor = RandomForestRegressor(random_state=42, n_jobs=-1)
                elif model_name == 'NeuralNetwork':
                    regressor = MLPRegressor(max_iter=1000, random_state=42)
                elif model_name == 'GradientBoosting':
                    regressor = GradientBoostingRegressor(random_state=42)
                else:
                    continue  # Skip models not in param_grids

                # Create pipeline for grid search
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', regressor)
                ])

                # Update param_grid to use pipeline parameter names
                pipeline_param_grid = {}
                for param, values in param_grid.items():
                    pipeline_param_grid[f'regressor__{param}'] = values

                # Grid search
                grid_search = GridSearchCV(
                    pipeline, pipeline_param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1
                )

                grid_search.fit(X_train, y_train)

                # Evaluate on validation set
                y_pred = grid_search.predict(X_val)
                val_r2 = r2_score(y_val, y_pred)

                print(f"    Best params: {grid_search.best_params_}")
                print(f"    Validation R²: {val_r2:.4f}")

                if val_r2 > best_score:
                    best_score = val_r2
                    best_model = grid_search.best_estimator_
            
            # Update best model for this metric
            if best_model is not None:
                # Evaluate the tuned model
                y_pred_tuned = best_model.predict(X_val)
                mae_tuned = mean_absolute_error(y_val, y_pred_tuned)
                r2_tuned = r2_score(y_val, y_pred_tuned)
                
                print(f"  Tuned model for {metric}: R² = {r2_tuned:.4f}, MAE = {mae_tuned:.4f}")
                
                # Replace the model in best_models
                self.best_models[metric] = {
                    'model_name': f"Tuned_{self.best_models[metric]['model_name']}",
                    'model': best_model,
                    'performance': {
                        'val_mae': mae_tuned,
                        'val_r2': r2_tuned,
                        'predictions': y_pred_tuned,
                        'actual': y_val
                    }
                }
        
        print("✅ Hyperparameter tuning completed!")
    
    def evaluate_models(self):
        """Comprehensive model evaluation"""
        print("\n" + "="*70)
        print("ADVANCED ML MODEL EVALUATION REPORT")
        print("="*70)
        
        evaluation_results = {}
        
        for metric, model_info in self.best_models.items():
            model_name = model_info['model_name']
            performance = model_info['performance']
            
            print(f"\n{metric.upper()} Prediction:")
            print(f"  Best Model: {model_name}")
            print(f"  Validation MAE: {performance['val_mae']:.4f}")
            print(f"  Validation R²: {performance['val_r2']:.4f}")
            
            # Additional metrics
            y_pred = performance['predictions']
            y_actual = performance['actual']
            
            mse = mean_squared_error(y_actual, y_pred)
            rmse = np.sqrt(mse)
            
            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((y_actual - y_pred) / (y_actual + 1e-8))) * 100
            
            print(f"  Validation MSE: {mse:.4f}")
            print(f"  Validation RMSE: {rmse:.4f}")
            print(f"  Validation MAPE: {mape:.2f}%")
            
            evaluation_results[metric] = {
                'model_name': model_name,
                'val_mae': float(performance['val_mae']),
                'val_r2': float(performance['val_r2']),
                'val_mse': float(mse),
                'val_rmse': float(rmse),
                'val_mape': float(mape)
            }
        
        return evaluation_results
    
    def save_models(self, model_dir="./advanced_ml_models"):
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
                'val_mse': float(info['performance'].get('val_mse', 0)),
                'val_rmse': float(info['performance'].get('val_rmse', 0)),
                'val_mape': float(info['performance'].get('val_mape', 0))
            }
        
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Model information saved to {model_info_path}")
        return model_dir
    
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
    
    def plot_model_comparison(self):
        """Plot model comparison and performance metrics"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Advanced ML Model Performance Comparison', fontsize=16)
            
            for idx, (metric, model_info) in enumerate(self.best_models.items()):
                ax = axes[idx//2, idx%2]
                
                y_pred = model_info['performance']['predictions']
                y_actual = model_info['performance']['actual']
                
                # Scatter plot of predictions vs actual
                ax.scatter(y_actual, y_pred, alpha=0.6, s=50)
                ax.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2, label='Perfect Prediction')
                
                # Add R² value to plot
                r2 = model_info['performance']['val_r2']
                ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Predicted Values')
                ax.set_title(f'{metric.capitalize()} Prediction\n({model_info["model_name"]})')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('./advanced_ml_model_comparison.png', dpi=300, bbox_inches='tight')
            print("Model comparison plot saved to ./advanced_ml_model_comparison.png")
            
        except ImportError:
            print("Matplotlib not available, skipping plots")
    
    def get_feature_importance(self, metric='area'):
        """Get feature importance for a specific model (if available)"""
        model_info = self.best_models.get(metric)
        if not model_info:
            return None
        
        model = model_info['model']
        
        # Extract the regressor from the pipeline
        regressor = model.named_steps['regressor']
        
        # Get feature importance if available
        if hasattr(regressor, 'feature_importances_'):
            # For tree-based models
            importances = regressor.feature_importances_
            return dict(zip(self.feature_columns, importances))
        elif hasattr(regressor, 'coef_'):
            # For linear models
            coef = np.abs(regressor.coef_)
            if len(coef) == len(self.feature_columns):
                return dict(zip(self.feature_columns, coef))
        
        return None


def main():
    print("Advanced ML Training with Neural Networks for Silicon Intelligence System")
    print("=" * 80)
    
    # Initialize trainer
    trainer = AdvancedMLTrainer()
    
    # Train models
    training_data_path = "./training_dataset.json"

    if os.path.exists(training_data_path):
        print(f"Training advanced models with data from {training_data_path}")

        # Train initial models
        best_models = trainer.train_models(training_data_path, use_cv=True)

        # Perform hyperparameter tuning
        print(f"\nPerforming hyperparameter tuning...")
        trainer.hyperparameter_tuning(training_data_path)

        # Evaluate models
        eval_results = trainer.evaluate_models()

        # Save models
        model_dir = trainer.save_models()

        # Plot model comparison
        trainer.plot_model_comparison()

        # Test predictions
        print(f"\nTesting advanced models with sample features...")
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

        # Show feature importance for area model
        print(f"\nFeature Importance for Area Prediction:")
        importance = trainer.get_feature_importance('area')
        if importance:
            for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                print(f"  {feature}: {imp:.4f}")
        else:
            print("  Feature importance not available for this model type")
        
        print(f"\n{'='*80}")
        print("ADVANCED ML TRAINING WITH NEURAL NETWORKS COMPLETE!")
        print(f"{'='*80}")
        print(f"Models saved to: {model_dir}")
        print("Advanced capabilities include:")
        print("- Multiple ML algorithms (RF, GB, NN, LR, Ridge, SVR)")
        print("- Cross-validation and hyperparameter tuning")
        print("- Neural networks with multiple hidden layers")
        print("- Ensemble methods and model comparison")
        print("- Comprehensive evaluation metrics")
        print("- Production-ready model deployment")
    else:
        print(f"❌ Training data not found: {training_data_path}")
        print("Please run prepare_training_data.py first")


if __name__ == "__main__":
    main()