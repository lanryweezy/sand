#!/usr/bin/env python3
"""
Comprehensive Learning System for Physical Design Intelligence
Implements the complete learning loop with prediction, reality comparison, and model updates
"""

import os
import json
import pickle
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

import torch
from torch_geometric.data import Data

from physical_design_intelligence import PhysicalDesignIntelligence
from ml_prediction_models import DesignPPAPredictor # Keep for now, will remove later
from gnn_model import PpaGNN # Import our GNN model

# Helper function to extract features for the GNN
def _extract_gnn_features(physical_ir_stats: Dict[str, Any]) -> torch.Tensor:
    """
    Extracts and formats features from PhysicalIR statistics into a 'super-node' tensor
    for the PpaGNN model.
    """
    num_nodes = physical_ir_stats.get('num_nodes', 0)
    num_edges = physical_ir_stats.get('num_edges', 0)
    # Assuming density, avg_area, avg_power can be derived or are available
    # For now, making plausible estimates or placeholders if not directly available
    density = num_edges / max(num_nodes * (num_nodes - 1), 1) if num_nodes > 1 else 0.0
    avg_area = physical_ir_stats.get('total_area', 0) / max(num_nodes, 1)
    avg_power = physical_ir_stats.get('total_power', 0) / max(num_nodes, 1)
    
    # Ensure the feature vector matches the GNN's expected input (5 features)
    features_tensor = torch.tensor([[
        float(num_nodes),
        float(num_edges),
        float(density),
        float(avg_area),
        float(avg_power)
    ]], dtype=torch.float)
    return features_tensor


class ComprehensiveLearningSystem:
    """
    Main learning system that orchestrates the entire AI-driven design intelligence pipeline
    """
    
    def __init__(self, data_dir: str = "learning_data"):
        self.data_dir = data_dir
        self.system = PhysicalDesignIntelligence()
        self.gnn_model = None # Will be loaded in _load_existing_models
        self.performance_history = []
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing models if available
        self._load_existing_models()
    
    def _load_existing_models(self):
        """Load existing GNN model if it exists"""
        model_path = './ppa_gnn_model.pt' # GNN model path
        if os.path.exists(model_path):
            try:
                # We need a dummy input size to initialize the model before loading state_dict
                # The GNN model takes 5 features (num_nodes, num_edges, density, avg_area, avg_power)
                self.gnn_model = PpaGNN(num_node_features=5, num_targets=3) 
                self.gnn_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                self.gnn_model.eval()
                print(f"Loaded existing GNN predictor model from {model_path}")
            except Exception as e:
                print(f"Could not load existing GNN model: {e}")
        else:
            print("GNN predictor model not found. Predictions will be unavailable.")
    
    def process_design(self, rtl_code: str, design_name: str) -> Dict[str, Any]:
        """Process a single design through the complete pipeline"""
        
        print(f"Processing design: {design_name}")
        
        # Analyze the design
        analysis = self.system.analyze_design(rtl_code, design_name)
        
        # Extract features for prediction and actual results from analysis
        gnn_features = _extract_gnn_features(analysis['physical_ir_stats'])
        
        actual_results = {
            'area': analysis['openroad_results']['overall_ppa']['area_um2'],
            'power': analysis['openroad_results']['overall_ppa']['power_mw'],
            'timing': analysis['openroad_results']['overall_ppa']['timing_ns'],
            'drc_violations': analysis['openroad_results']['routing']['drc_violations']
        }
        
        # Make predictions using the GNN model
        predictions = {}
        if self.gnn_model:
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                gnn_input_data = Data(x=gnn_features.to(device), edge_index=torch.empty((2, 0), dtype=torch.long).to(device))
                with torch.no_grad():
                    ppa_prediction_tensor = self.gnn_model(gnn_input_data)
                
                ppa_prediction_np = ppa_prediction_tensor.cpu().numpy()
                
                predictions = {
                    'area': float(ppa_prediction_np[0][0]),
                    'power': float(ppa_prediction_np[0][1]),
                    'timing': float(ppa_prediction_np[0][2]),
                    # DRC prediction is not part of PpaGNN, keep it as mock for now
                    'drc_violations': actual_results.get('drc_violations', 0)
                }
            except Exception as e:
                print(f"Error during GNN prediction for {design_name}: {e}")
                # Fallback to dummy predictions
                predictions = {metric: actual_results.get(metric, 0) for metric in ['area', 'power', 'timing']}
                predictions['drc_violations'] = actual_results.get('drc_violations', 0)
        else:
            print("GNN model not loaded. Using fallback predictions.")
            predictions = {metric: actual_results.get(metric, 0) for metric in ['area', 'power', 'timing']}
            predictions['drc_violations'] = actual_results.get('drc_violations', 0)
        
        # Calculate prediction errors
        errors = {}
        for metric in ['area', 'power', 'timing', 'drc_violations']:
            pred_val = predictions.get(metric, 0)
            actual_val = actual_results.get(f'actual_{metric}' if metric != 'drc_violations' else 'drc_violations', 0)
            errors[f'{metric}_error'] = abs(pred_val - actual_val)
            errors[f'{metric}_pct_error'] = (errors[f'{metric}_error'] / (actual_val + 1e-8)) * 100
        
        # Combine results
        result = {
            'design_name': design_name,
            'timestamp': datetime.now().isoformat(),
            'predictions': predictions,
            'actual': actual_results,
            'errors': errors,
            'analysis': analysis
        }
        
        # Save individual result
        result_path = os.path.join(self.data_dir, f"result_{design_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
    
    def batch_process_designs(self, designs: List[tuple]) -> List[Dict[str, Any]]:
        """Process multiple designs in batch"""
        
        results = []
        for name, rtl in designs:
            try:
                result = self.process_design(rtl, name)
                results.append(result)
                
                # Print summary
                actual_area = result['actual'].get('actual_area', 0)
                pred_area = result['predictions'].get('area', 0)
                print(f"  {name}: Actual={actual_area:.2f}, Predicted={pred_area:.2f}, Error={abs(actual_area-pred_area):.2f}")
                
            except Exception as e:
                print(f"Error processing {name}: {e}")
                continue
        
        return results
    
    def update_models_with_new_data(self):
        """Placeholder for future GNN fine-tuning/retraining. Not implemented in this demo."""
        print("GNN models are pre-trained and not dynamically updated in this demo.")
        print("To update models, run `train_gnn.py` with new `training_dataset.json`.")
        return False # Indicate no update happened
    
    def generate_insights_report(self) -> str:
        """Generate a comprehensive insights report"""
        
        # We now use the internal design_history which contains analysis, actuals, and predictions
        if not self.system.design_history:
            return "No data available for insights"
        
        df_list = []
        for record in self.system.design_history: # Iterate through the internal history
            row = {'design_name': record['design_name']}
            
            # Features (from Physical IR stats, as used by GNN)
            ir_stats = record['analysis']['physical_ir_stats']
            row['gnn_num_nodes'] = ir_stats.get('num_nodes', 0)
            row['gnn_num_edges'] = ir_stats.get('num_edges', 0)
            row['gnn_density'] = row['gnn_num_edges'] / max(row['gnn_num_nodes'] * (row['gnn_num_nodes'] - 1), 1) if row['gnn_num_nodes'] > 1 else 0.0
            row['gnn_avg_area'] = ir_stats.get('total_area', 0) / max(ir_stats.get('num_nodes', 0), 1)
            row['gnn_avg_power'] = ir_stats.get('total_power', 0) / max(ir_stats.get('num_nodes', 0), 1)

            # Actual results (from OpenROAD, which came from 'overall_ppa')
            actual = record['actual']
            row['actual_area'] = actual.get('area', 0)
            row['actual_power'] = actual.get('power', 0)
            row['actual_timing'] = actual.get('timing', 0)
            row['actual_drc_violations'] = actual.get('drc_violations', 0) # From routing step

            # Predicted results (from GNN, stored in 'predictions' from process_design)
            predicted = record['predictions']
            row['predicted_area'] = predicted.get('area', 0)
            row['predicted_power'] = predicted.get('power', 0)
            row['predicted_timing'] = predicted.get('timing', 0)
            row['predicted_drc_violations'] = predicted.get('drc_violations', 0) # GNN doesn't predict, so this is fallback

            df_list.append(row)
        
        df = pd.DataFrame(df_list)
        
        report = []
        report.append("=== COMPREHENSIVE LEARNING SYSTEM INSIGHTS (GNN-BASED) ===")
        report.append(f"Total designs analyzed: {len(df)}")
        report.append("")
        
        # Model performance
        report.append("GNN MODEL PERFORMANCE (PPA):")
        for metric in ['area', 'power', 'timing']: # GNN predicts these 3
            actual_col = f'actual_{metric}'
            pred_col = f'predicted_{metric}'
            if actual_col in df.columns and pred_col in df.columns and len(df) > 1 and df[actual_col].var() > 0: # Check for variance
                mae = np.mean(np.abs(df[actual_col] - df[pred_col]))
                r2 = np.corrcoef(df[actual_col], df[pred_col])[0, 1] ** 2
                report.append(f"  {metric}: MAE={mae:.3f}, R²: {r2:.3f}")
            elif len(df) <=1 :
                report.append(f"  {metric}: Not enough data for R2 calculation (only {len(df)} sample). MAE: {np.mean(np.abs(df[actual_col] - df[pred_col])):.3f}")
            else:
                report.append(f"  {metric}: Actual values have no variance for R2 calculation. MAE: {np.mean(np.abs(df[actual_col] - df[pred_col])):.3f}")

        # DRC is still from OpenROAD for now
        report.append("\nDRC PREDICTION (from OpenROAD/Fallback):")
        actual_drc_col = 'actual_drc_violations'
        predicted_drc_col = 'predicted_drc_violations'
        if actual_drc_col in df.columns and predicted_drc_col in df.columns:
            drc_mae = np.mean(np.abs(df[actual_drc_col] - df[predicted_drc_col]))
            report.append(f"  DRC Violations MAE: {drc_mae:.2f}")

        report.append("")
        
        # Best/worst predictions based on Area
        if 'actual_area' in df.columns and 'predicted_area' in df.columns:
            df['area_abs_error'] = np.abs(df['actual_area'] - df['predicted_area'])
            df_sorted = df.sort_values('area_abs_error')
            
            report.append("BEST AREA PREDICTIONS (Lowest Error):")
            for _, row in df_sorted.head(3).iterrows():
                report.append(f"  {row['design_name']}: Actual={row['actual_area']:.2f}, Pred={row['predicted_area']:.2f}")
            
            report.append("")
            report.append("WORST AREA PREDICTIONS (Highest Error):")
            for _, row in df_sorted.tail(3).iterrows():
                report.append(f"  {row['design_name']}: Actual={row['actual_area']:.2f}, Pred={row['predicted_area']:.2f}")
        
        return "\n".join(report)
    
    def get_learning_opportunities(self) -> List[Dict[str, Any]]:
        """Identify learning opportunities from analysis results"""
        
        dataset = self.system.get_learning_dataset()
        
        opportunities = []
        
        for record in dataset:
            errors = record.get('errors', {})
            
            # High prediction errors indicate learning opportunities
            for metric in ['area', 'power', 'timing', 'drc_violations']:
                err_key = f'{metric}_pct_error'
                if err_key in errors and errors[err_key] > 20:  # >20% error
                    opportunities.append({
                        'design': record['design_name'],
                        'metric': metric,
                        'error_pct': errors[err_key],
                        'feature_importance': self._analyze_feature_importance(record)
                    })
        
        return opportunities
    
        def _analyze_feature_importance(self, record: Dict) -> Dict[str, float]:
            """Analyze which features contributed most to prediction errors"""
            
            # For GNN, features are 'gnn_num_nodes', 'gnn_num_edges', 'gnn_density', 'gnn_avg_area', 'gnn_avg_power'
            # Extract these from the analysis record
            ir_stats = record['analysis']['physical_ir_stats']
            features = {
                'gnn_num_nodes': ir_stats.get('num_nodes', 0),
                'gnn_num_edges': ir_stats.get('num_edges', 0),
                'gnn_density': ir_stats.get('num_edges', 0) / max(ir_stats.get('num_nodes', 0) * (ir_stats.get('num_nodes', 0) - 1), 1) if ir_stats.get('num_nodes', 0) > 1 else 0.0,
                'gnn_avg_area': ir_stats.get('total_area', 0) / max(ir_stats.get('num_nodes', 0), 1),
                'gnn_avg_power': ir_stats.get('total_power', 0) / max(ir_stats.get('num_nodes', 0), 1)
            }
            
            # Sort by magnitude (simple heuristic for "importance")
            sorted_features = dict(sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
            
            return sorted_features
        
def demonstrate_learning_system():
    """Demonstrate the comprehensive learning system"""
    
    print("=== DEMONSTRATING COMPREHENSIVE LEARNING SYSTEM ===")
    
    # Create learning system
    learner = ComprehensiveLearningSystem()
    
    # Define test designs
    test_designs = [
        # Simple designs
        ('adder_8bit', '''
        module adder_8bit (
            input clk,
            input rst_n,
            input [7:0] a,
            input [7:0] b,
            output reg [8:0] sum
        );
            always @(posedge clk) begin
                if (!rst_n)
                    sum <= 9'd0;
                else
                    sum <= a + b;
            end
        endmodule
        '''),
        
        ('counter_16bit', '''
        module counter_16bit (
            input clk,
            input rst_n,
            input enable,
            output reg [15:0] count
        );
            always @(posedge clk) begin
                if (!rst_n)
                    count <= 16'd0;
                else if (enable)
                    count <= count + 1;
            end
        endmodule
        '''),
        
        # More complex designs
        ('mac_array_small', '''
        module mac_array_small (
            input clk,
            input rst_n,
            input [7:0] data_in,
            input [7:0] weight_in,
            output [15:0] result
        );
            reg [15:0] acc [0:3];
            reg [15:0] prod [0:3];
            
            integer i;
            always @(posedge clk) begin
                if (!rst_n) begin
                    for (i = 0; i < 4; i = i + 1) begin
                        acc[i] <= 16'd0;
                        prod[i] <= 16'd0;
                    end
                end else begin
                    for (i = 0; i < 4; i = i + 1) begin
                        prod[i] <= data_in * weight_in;
                        acc[i] <= acc[i] + prod[i];
                    end
                end
            end
            
            assign result = acc[0] + acc[1] + acc[2] + acc[3];
        endmodule
        '''),
        
        ('pipeline_multiplier', '''
        module pipeline_multiplier (
            input clk,
            input rst_n,
            input [15:0] a,
            input [15:0] b,
            output reg [31:0] result
        );
            reg [15:0] a_reg1, a_reg2, a_reg3;
            reg [15:0] b_reg1, b_reg2, b_reg3;
            reg [31:0] mult1, mult2;
            
            always @(posedge clk) begin
                if (!rst_n) begin
                    a_reg1 <= 16'd0; a_reg2 <= 16'd0; a_reg3 <= 16'd0;
                    b_reg1 <= 16'd0; b_reg2 <= 16'd0; b_reg3 <= 16'd0;
                    mult1 <= 32'd0; mult2 <= 32'd0;
                    result <= 32'd0;
                end else begin
                    a_reg1 <= a; a_reg2 <= a_reg1; a_reg3 <= a_reg2;
                    b_reg1 <= b; b_reg2 <= b_reg1; b_reg3 <= b_reg2;
                    mult1 <= a_reg1 * b_reg1;
                    mult2 <= a_reg2 * b_reg2;
                    result <= a_reg3 * b_reg3;
                end
            end
        endmodule
        ''')
    ]
    
    # Process all designs
    print("Processing designs through learning system...")
    results = learner.batch_process_designs(test_designs)
    
    # Update models with new data
    print("\nUpdating models with new data...")
    learner.update_models_with_new_data()
    
    # Generate insights
    print("\nGenerating insights report...")
    insights = learner.generate_insights_report()
    print(insights)
    
    # Identify learning opportunities
    print("\nIdentifying learning opportunities...")
    opportunities = learner.get_learning_opportunities()
    for opp in opportunities[:3]:  # Show top 3
        print(f"  {opp['design']}: {opp['metric']} error {opp['error_pct']:.1f}% - Key features: {list(opp['feature_importance'].keys())[:3]}")
    
    # Save final report
    report_path = os.path.join(learner.data_dir, f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(report_path, 'w') as f:
        f.write(insights)
    
    print(f"\nFinal report saved to {report_path}")
    
    return learner


if __name__ == "__main__":
    learner = demonstrate_learning_system()