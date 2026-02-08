#!/usr/bin/env python3
"""
ML Prediction Models for Physical Design Intelligence
Trains on design features to predict PPA metrics
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Tuple, Any
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

from physical_design_intelligence import PhysicalDesignIntelligence
import torch
from networks.graph_neural_network import SiliconGNN, convert_to_pyg_data


@dataclass
class PredictionMetrics:
    """Metrics for evaluating prediction performance"""
    mae: float
    mse: float
    rmse: float
    r2: float
    mape: float  # Mean Absolute Percentage Error


class DesignPPAPredictor:
    """
    ML models to predict Physical Design metrics from RTL features
    Predicts: Area, Power, Timing, DRC violations
    """
    
    def __init__(self):
        self.models = {
            'area': Ridge(alpha=1.0),
            'power': Ridge(alpha=1.0), 
            'timing': Ridge(alpha=1.0),
            'drc_violations': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        self.feature_columns = []
        self.is_trained = False
        
        # GNN-related
        self.gnn_model = None
        self.use_gnn = False
    
    def enable_gnn(self, in_channels=7, hidden_channels=64):
        """Enable and initialize the GNN model"""
        self.gnn_model = SiliconGNN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=3)
        self.use_gnn = True
    
    def prepare_features(self, feature_dict: Dict[str, Any]) -> np.ndarray:
        """Convert feature dictionary to feature vector"""
        # Ensure consistent feature ordering
        if not self.feature_columns:
            self.feature_columns = sorted(feature_dict.keys())
        
        # Create feature vector in consistent order
        feature_vector = []
        for col in self.feature_columns:
            feature_vector.append(feature_dict.get(col, 0.0))
        
        return np.array(feature_vector).reshape(1, -1)
    
    def prepare_dataset(self, design_records: List[Dict]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Prepare training dataset from design records"""
        
        if not design_records:
            raise ValueError("No design records provided")
        
        # Prepare features matrix
        all_features = []
        for record in design_records:
            features = record['features']
            if not self.feature_columns:
                self.feature_columns = sorted(features.keys())
            
            feature_vector = [features.get(col, 0.0) for col in self.feature_columns]
            all_features.append(feature_vector)
        
        X = np.array(all_features)
        
        # Prepare target matrices for each metric
        y_targets = {}
        for metric in ['area', 'power', 'timing', 'drc_violations']:
            y_targets[metric] = np.array([
                record['labels'].get(f'actual_{metric}', 0.0) if metric != 'drc_violations' 
                else record['labels'].get('drc_violations', 0.0)
                for record in design_records
            ])
        
        return X, y_targets
    
    def train(self, design_records: List[Dict]):
        """Train all prediction models"""
        
        print(f"Training predictor on {len(design_records)} design records...")
        
        X, y_targets = self.prepare_dataset(design_records)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Feature columns: {self.feature_columns}")
        
        if self.use_gnn and self.gnn_model is not None:
            self._train_gnn(design_records)
        
        # Train each model
        for target_name, y in y_targets.items():
            print(f"Training {target_name} model...")
            
            if target_name == 'drc_violations':
                # Use Random Forest for DRC violations (classification-like)
                self.models[target_name].fit(X, y)
            else:
                # Use Ridge for continuous metrics
                self.models[target_name].fit(X, y)
        
        self.is_trained = True
        print("Training completed!")

    def _train_gnn(self, design_records: List[Dict], epochs=50):
        """Train the GNN model using design records"""
        from torch_geometric.loader import DataLoader
        
        print(f"Training GNN on {len(design_records)} designs...")
        pyg_dataset = []
        for record in design_records:
            if 'silicon_graph' in record:
                data = convert_to_pyg_data(record['silicon_graph'])
                # Targets: [area, power, timing]
                y = [
                    float(record['labels'].get('actual_area', 0.0)),
                    float(record['labels'].get('actual_power', 0.0)),
                    float(record['labels'].get('actual_timing', 0.0))
                ]
                data.y = torch.tensor([y], dtype=torch.float)
                pyg_dataset.append(data)
        
        if not pyg_dataset:
            print("No graph data found for GNN training, skipping.")
            return

        loader = DataLoader(pyg_dataset, batch_size=min(4, len(pyg_dataset)), shuffle=True)
        optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        
        self.gnn_model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                optimizer.zero_grad()
                out = self.gnn_model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
    
    def predict(self, feature_dict: Dict[str, Any]) -> Dict[str, float]:
        """Predict PPA metrics for a given design"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X = self.prepare_features(feature_dict)
        
        predictions = {}
        
        if self.use_gnn and self.gnn_model is not None:
            # GNN prediction requires a silicon_graph object which might be in the feature_dict
            # or we might need to recreate it. For now, assume it's passed or we can't use GNN.
            if 'silicon_graph' in feature_dict:
                self.gnn_model.eval()
                with torch.no_grad():
                    pyg_data = convert_to_pyg_data(feature_dict['silicon_graph'])
                    # SiliconGNN expects batch index; for single graph it's all zeros
                    batch = torch.zeros(pyg_data.x.size(0), dtype=torch.long)
                    gnn_out = self.gnn_model(pyg_data.x, pyg_data.edge_index, batch)
                    
                    predictions['area'] = max(0, float(gnn_out[0, 0]))
                    predictions['power'] = max(0, float(gnn_out[0, 1]))
                    predictions['timing'] = max(0, float(gnn_out[0, 2]))
            else:
                # Fallback to ridge if no graph provided
                for target_name in ['area', 'power', 'timing']:
                    X = self.prepare_features(feature_dict)
                    pred = self.models[target_name].predict(X)[0]
                    predictions[target_name] = max(0, pred)
        else:
            for target_name, model in self.models.items():
                X = self.prepare_features(feature_dict)
                pred = model.predict(X)[0]
                predictions[target_name] = max(0, pred)
        
        # Always run DRC violations with RF for now
        if 'drc_violations' not in predictions:
            X = self.prepare_features(feature_dict)
            predictions['drc_violations'] = max(0, self.models['drc_violations'].predict(X)[0])
        
        return predictions
    
    def evaluate(self, design_records: List[Dict]) -> Dict[str, PredictionMetrics]:
        """Evaluate model performance"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        X, y_targets = self.prepare_dataset(design_records)
        
        metrics = {}
        for target_name, y_true in y_targets.items():
            model = self.models[target_name]
            y_pred = model.predict(X)
            
            # Calculate metrics
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            # MAPE (avoid division by zero)
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
            
            metrics[target_name] = PredictionMetrics(mae, mse, rmse, r2, mape)
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        model_data = {
            'models': self.models,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = model_data['is_trained']
        print(f"Model loaded from {filepath}")


def test_prediction_models():
    """Test the ML prediction models"""
    
    print("=== Testing ML Prediction Models ===")
    
    # Create the complete system
    system = PhysicalDesignIntelligence()
    
    # Generate training data with multiple designs
    test_designs = [
        # MAC Array designs with different sizes
        ('mac_8x8', '''
        module mac_8x8 (
            input clk,
            input rst_n,
            input [7:0] a_data,
            input [7:0] b_data,
            input [7:0] weight_data,
            output [15:0] result
        );
            reg [15:0] accumulator;
            reg [15:0] product;
            
            always @(posedge clk) begin
                if (!rst_n) begin
                    accumulator <= 16'd0;
                    product <= 16'd0;
                end else begin
                    product <= a_data * weight_data;
                    accumulator <= accumulator + product;
                end
            end
            
            assign result = accumulator;
        endmodule
        '''),
        
        ('mac_16x16', '''
        module mac_16x16 (
            input clk,
            input rst_n,
            input [15:0] a_data,
            input [15:0] b_data,
            input [15:0] weight_data,
            output [31:0] result
        );
            reg [31:0] accumulator;
            reg [31:0] product;
            
            always @(posedge clk) begin
                if (!rst_n) begin
                    accumulator <= 32'd0;
                    product <= 32'd0;
                end else begin
                    product <= a_data * weight_data;
                    accumulator <= accumulator + product;
                end
            end
            
            assign result = accumulator;
        endmodule
        '''),
        
        ('conv_4x4', '''
        module conv_4x4 (
            input clk,
            input rst_n,
            input [3:0] pixel_in,
            input [3:0] weight_in,
            output [7:0] conv_out
        );
            reg [7:0] multiplier_result;
            reg [7:0] accumulator;
            
            always @(posedge clk) begin
                if (!rst_n) begin
                    multiplier_result <= 8'd0;
                    accumulator <= 8'd0;
                end else begin
                    multiplier_result <= pixel_in * weight_in;
                    accumulator <= accumulator + multiplier_result;
                end
            end
            
            assign conv_out = accumulator;
        endmodule
        '''),
        
        ('pipeline_adder', '''
        module pipeline_adder (
            input clk,
            input rst_n,
            input [31:0] a,
            input [31:0] b,
            output [32:0] result
        );
            reg [31:0] a_reg;
            reg [31:0] b_reg;
            reg [32:0] sum_reg;
            
            always @(posedge clk) begin
                if (!rst_n) begin
                    a_reg <= 32'd0;
                    b_reg <= 32'd0;
                    sum_reg <= 33'd0;
                end else begin
                    a_reg <= a;
                    b_reg <= b;
                    sum_reg <= a_reg + b_reg;
                end
            end
            
            assign result = sum_reg;
        endmodule
        ''')
    ]
    
    # Analyze all designs to build dataset
    print("Analyzing designs for training data...")
    for name, rtl in test_designs:
        analysis = system.analyze_design(rtl, name)
        print(f"  Analyzed {name}: {analysis['openroad_results']['overall_ppa']['area_um2']:.2f} µm²")
    
    # Get training dataset
    dataset = system.get_learning_dataset()
    print(f"\nGenerated dataset with {len(dataset)} records")
    
    # Create and train predictor
    predictor = DesignPPAPredictor()
    predictor.train(dataset)
    
    # Evaluate performance
    metrics = predictor.evaluate(dataset)
    
    print("\n=== Model Performance ===")
    for target, metric in metrics.items():
        print(f"{target.upper()}:")
        print(f"  MAE: {metric.mae:.3f}")
        print(f"  RMSE: {metric.rmse:.3f}")
        print(f"  R²: {metric.r2:.3f}")
        print(f"  MAPE: {metric.mape:.2f}%")
    
    # Test prediction on first design
    print(f"\n=== Prediction Test ===")
    test_features = dataset[0]['features']
    predictions = predictor.predict(test_features)
    actual_labels = dataset[0]['labels']
    
    print(f"Design: {dataset[0]['design_name']}")
    print("Predictions vs Actual:")
    for metric in ['area', 'power', 'timing']:
        pred_key = metric
        act_key = f'actual_{metric}'
        print(f"  {metric}: Pred={predictions[pred_key]:.3f}, Actual={actual_labels[act_key]:.3f}, Error={abs(predictions[pred_key] - actual_labels[act_key]):.3f}")
    
    # Show feature importance (for Ridge models)
    print(f"\n=== Feature Importance ===")
    for target_name, model in predictor.models.items():
        if hasattr(model, 'coef_'):
            print(f"{target_name} coefficients:")
            for i, (col, coef) in enumerate(zip(predictor.feature_columns, model.coef_)):
                print(f"  {col}: {coef:.4f}")
    
    # Save the model
    predictor.save_model("design_ppa_predictor.pkl")
    
    return predictor, system


def visualize_predictions(predictor, system):
    """Visualize prediction results"""
    
    dataset = system.get_learning_dataset()
    
    if not dataset:
        print("No dataset available for visualization")
        return
    
    # Prepare data for plotting
    X, y_targets = predictor.prepare_dataset(dataset)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('PPA Prediction Results')
    
    targets = ['area', 'power', 'timing', 'drc_violations']
    titles = ['Area Prediction', 'Power Prediction', 'Timing Prediction', 'DRC Violations Prediction']
    
    for idx, (target, title) in enumerate(zip(targets, titles)):
        ax = axes[idx // 2, idx % 2]
        
        y_true = y_targets[target]
        model = predictor.models[target]
        y_pred = model.predict(X)
        
        ax.scatter(y_true, y_pred, alpha=0.7)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(title)
        
        # Add R² score
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('ppa_predictions.png', dpi=300, bbox_inches='tight')
    print("Prediction visualization saved as 'ppa_predictions.png'")
    plt.show()


if __name__ == "__main__":
    predictor, system = test_prediction_models()
    
    # Uncomment to create visualizations (requires matplotlib)
    # visualize_predictions(predictor, system)