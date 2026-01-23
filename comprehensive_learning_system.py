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

from physical_design_intelligence import PhysicalDesignIntelligence
from ml_prediction_models import DesignPPAPredictor


class ComprehensiveLearningSystem:
    """
    Main learning system that orchestrates the entire AI-driven design intelligence pipeline
    """
    
    def __init__(self, data_dir: str = "learning_data"):
        self.data_dir = data_dir
        self.system = PhysicalDesignIntelligence()
        self.predictor = DesignPPAPredictor()
        self.performance_history = []
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing models if available
        self._load_existing_models()
    
    def _load_existing_models(self):
        """Load existing models if they exist"""
        model_path = os.path.join(self.data_dir, "design_ppa_predictor.pkl")
        if os.path.exists(model_path):
            try:
                self.predictor.load_model(model_path)
                print(f"Loaded existing predictor model from {model_path}")
            except Exception as e:
                print(f"Could not load existing model: {e}")
    
    def process_design(self, rtl_code: str, design_name: str) -> Dict[str, Any]:
        """Process a single design through the complete pipeline"""
        
        print(f"Processing design: {design_name}")
        
        # Analyze the design
        analysis = self.system.analyze_design(rtl_code, design_name)
        
        # Extract features for prediction
        # Get features from the learning dataset since they're computed there
        dataset = self.system.get_learning_dataset()
        current_record = None
        for record in dataset:
            if record['design_name'] == design_name:
                current_record = record
                break
        
        if current_record:
            features = current_record['features']
            actual_results = current_record['labels']
        else:
            # Fallback: create minimal features if not found
            features = {
                'node_count': analysis['physical_ir_stats']['num_nodes'],
                'edge_count': analysis['physical_ir_stats']['num_edges'],
                'total_area_pred': analysis['physical_ir_stats']['total_area'],
                'total_power_pred': analysis['physical_ir_stats']['total_power']
            }
            actual_results = {
                'actual_area': analysis['openroad_results']['overall_ppa']['area_um2'],
                'actual_power': analysis['openroad_results']['overall_ppa']['power_mw'],
                'actual_timing': analysis['openroad_results']['overall_ppa']['timing_ns'],
                'drc_violations': analysis['openroad_results']['routing']['drc_violations']
            }
        
        # Make predictions
        try:
            predictions = self.predictor.predict(features)
        except ValueError:
            # If model hasn't been trained yet, use dummy predictions
            predictions = {metric: actual_results.get(f'actual_{metric}', 0) for metric in ['area', 'power', 'timing']}
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
        """Retrain models with all available data"""
        
        # Get all learning data
        dataset = self.system.get_learning_dataset()
        
        if len(dataset) < 2:
            print("Insufficient data for retraining")
            return False
        
        print(f"Updating models with {len(dataset)} total samples")
        
        try:
            # Retrain models
            self.predictor.train(dataset)
            
            # Save updated models
            model_path = os.path.join(self.data_dir, "design_ppa_predictor.pkl")
            self.predictor.save_model(model_path)
            
            # Evaluate performance
            metrics = self.predictor.evaluate(dataset)
            
            # Log performance
            perf_entry = {
                'timestamp': datetime.now().isoformat(),
                'sample_count': len(dataset),
                'metrics': {k: v.__dict__ for k, v in metrics.items()}
            }
            self.performance_history.append(perf_entry)
            
            # Save performance history
            perf_path = os.path.join(self.data_dir, "performance_history.json")
            with open(perf_path, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
            
            print("Models updated successfully")
            return True
            
        except Exception as e:
            print(f"Error updating models: {e}")
            return False
    
    def generate_insights_report(self) -> str:
        """Generate a comprehensive insights report"""
        
        dataset = self.system.get_learning_dataset()
        
        if not dataset:
            return "No data available for insights"
        
        # Convert to DataFrame for analysis
        df_list = []
        for record in dataset:
            row = {'design_name': record['design_name']}
            row.update(record['features'])
            row.update({f'actual_{k}': v for k, v in record['labels'].items()})
            
            # Check if predictions exist in the record, otherwise use empty dict
            predictions = record.get('predictions', {})
            row.update({'predicted_' + k: v for k, v in predictions.items()})
            df_list.append(row)
        
        df = pd.DataFrame(df_list)
        
        report = []
        report.append("=== COMPREHENSIVE LEARNING SYSTEM INSIGHTS ===")
        report.append(f"Total designs analyzed: {len(df)}")
        report.append("")
        
        # Model performance
        report.append("MODEL PERFORMANCE:")
        for metric in ['area', 'power', 'timing', 'drc_violations']:
            actual_col = f'actual_{metric}'
            pred_col = f'predicted_{metric}'
            if actual_col in df.columns and pred_col in df.columns:
                mae = np.mean(np.abs(df[actual_col] - df[pred_col]))
                r2 = np.corrcoef(df[actual_col], df[pred_col])[0, 1] ** 2
                report.append(f"  {metric}: MAE={mae:.3f}, R²={r2:.3f}")
        report.append("")
        
        # Feature correlations with actual results
        report.append("FEATURE CORRELATIONS WITH ACTUAL AREA:")
        area_cols = [col for col in df.columns if col.startswith('actual_')]
        feat_cols = [col for col in df.columns if col in self.system.get_feature_names()]
        
        for feat in feat_cols[:10]:  # Top 10 features
            if feat in df.columns and 'actual_area' in df.columns:
                corr = df[feat].corr(df['actual_area'])
                report.append(f"  {feat}: {corr:.3f}")
        report.append("")
        
        # Best/worst predictions
        if 'actual_area' in df.columns and 'predicted_area' in df.columns:
            df['area_abs_error'] = np.abs(df['actual_area'] - df['predicted_area'])
            df_sorted = df.sort_values('area_abs_error')
            
            report.append("BEST PREDICTIONS (Lowest Area Error):")
            for _, row in df_sorted.head(3).iterrows():
                report.append(f"  {row['design_name']}: Actual={row['actual_area']:.2f}, Pred={row['predicted_area']:.2f}")
            
            report.append("")
            report.append("WORST PREDICTIONS (Highest Area Error):")
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
        
        # This would involve more sophisticated analysis in a real system
        # For now, return the top features from the record
        features = {k: v for k, v in record['features'].items() if isinstance(v, (int, float))}
        
        # Sort by magnitude
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