#!/usr/bin/env python3
"""
Complete Silicon Intelligence System Entry Point
Demonstrates the full pipeline from RTL to predictions to learning
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from comprehensive_learning_system import ComprehensiveLearningSystem
from physical_design_intelligence import PhysicalDesignIntelligence
from ml_prediction_models import DesignPPAPredictor


def main():
    """Main entry point for the Silicon Intelligence System"""
    
    print("🚀 Starting Silicon Intelligence System")
    print("=" * 60)
    
    # Create the complete learning system
    learner = ComprehensiveLearningSystem()
    
    # Example RTL designs to process
    example_designs = [
        ('simple_adder', '''
        module simple_adder (
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
        
        ('complex_mac', '''
        module complex_mac (
            input clk,
            input rst_n,
            input [15:0] data_in,
            input [15:0] weight_in,
            input valid,
            output reg [31:0] result,
            output reg ready
        );
            reg [31:0] accumulator;
            reg [31:0] product;
            
            always @(posedge clk) begin
                if (!rst_n) begin
                    accumulator <= 32'd0;
                    product <= 32'd0;
                    result <= 32'd0;
                    ready <= 1'b0;
                end else if (valid) begin
                    product <= data_in * weight_in;
                    accumulator <= accumulator + product;
                    result <= accumulator;
                    ready <= 1'b1;
                end else begin
                    ready <= 1'b0;
                end
            end
        endmodule
        ''')
    ]
    
    print("Processing example designs...")
    results = learner.batch_process_designs(example_designs)
    
    print("\n" + "=" * 60)
    print("📊 SYSTEM ANALYSIS REPORT")
    print("=" * 60)
    
    # Generate comprehensive insights
    insights = learner.generate_insights_report()
    print(insights)
    
    print("\n" + "=" * 60)
    print("🎯 LEARNING OPPORTUNITIES IDENTIFIED")
    print("=" * 60)
    
    # Show learning opportunities
    opportunities = learner.get_learning_opportunities()
    if opportunities:
        for i, opp in enumerate(opportunities[:5], 1):  # Top 5
            print(f"{i}. {opp['design']}: {opp['metric']} prediction off by {opp['error_pct']:.1f}%")
            print(f"   Key features: {list(opp['feature_importance'].keys())[:3]}")
    else:
        print("No significant learning opportunities identified (prediction errors <20%)")
    
    print("\n" + "=" * 60)
    print("📈 PREDICTION ACCURACY SUMMARY")
    print("=" * 60)
    
    # Show final accuracy metrics
    dataset = learner.system.get_learning_dataset()
    if dataset:
        avg_errors = {'area': [], 'power': [], 'timing': [], 'drc_violations': []}
        
        for record in dataset:
            errors = record.get('errors', {})
            for metric in avg_errors:
                err_key = f'{metric}_pct_error'
                if err_key in errors:
                    avg_errors[metric].append(errors[err_key])
        
        for metric, errors in avg_errors.items():
            if errors:
                avg_err = sum(errors) / len(errors)
                print(f"{metric.upper()}: Average prediction error {avg_err:.2f}%")
    
    print("\n" + "=" * 60)
    print("✅ SYSTEM STATUS: OPERATIONAL")
    print("The Silicon Intelligence System is now capable of:")
    print("- Parsing RTL designs with professional tools")
    print("- Building Physical IR for physical reasoning")
    print("- Predicting PPA metrics with ML models")
    print("- Learning from prediction errors")
    print("- Identifying optimization opportunities")
    print("- Generating actionable insights")
    print("=" * 60)
    
    return learner


def demonstrate_capability():
    """Demonstrate the key capability: turning prediction accuracy into authority"""
    
    print("\n🔍 DEMONSTRATING CORE CAPABILITY")
    print("Turning prediction accuracy into authority...")
    print("-" * 50)
    
    # Create a fresh system
    system = PhysicalDesignIntelligence()
    
    # Test prediction accuracy on known design patterns
    test_rtls = [
        ('small_adder', '''
        module small_adder(input [7:0] a, input [7:0] b, output [8:0] sum);
            assign sum = a + b;
        endmodule
        '''),
        ('medium_multiplier', '''
        module medium_multiplier(
            input clk, input rst_n,
            input [15:0] a, input [15:0] b,
            output reg [31:0] product
        );
            always @(posedge clk) begin
                if (!rst_n) product <= 0;
                else product <= a * b;
            end
        endmodule
        ''')
    ]
    
    print("Testing prediction accuracy...")
    for name, rtl in test_rtls:
        analysis = system.analyze_design(rtl, name)
        
        # Get the latest record from the system's learning dataset to access predictions
        dataset = system.get_learning_dataset()
        current_record = None
        for record in dataset:
            if record['design_name'] == name:
                current_record = record
                break
        
        if current_record:
            predictions = current_record.get('predictions', {})
            actuals = current_record.get('labels', {})
            
            print(f"\n{name}:")
            for metric in ['area', 'power', 'timing']:
                pred = predictions.get(metric, 0)
                actual = actuals.get(f'actual_{metric}', 0)
                error_pct = abs(pred - actual) / (actual + 1e-8) * 100
                print(f"  {metric}: Pred={pred:.2f}, Actual={actual:.2f}, Error={error_pct:.2f}%")
        else:
            print(f"\n{name}: Could not find analysis record")
    
    print("\n🎯 AUTHORITY ESTABLISHED")
    print("As prediction accuracy improves, the system gains authority")
    print("to make autonomous design decisions and optimizations.")


if __name__ == "__main__":
    learner = main()
    
    # Demonstrate the core capability
    demonstrate_capability()
    
    print(f"\n💾 All results saved to {learner.data_dir}/ directory")
    print("Ready for next phase: autonomous design optimization")