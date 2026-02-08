#!/usr/bin/env python3
"""
Complete Physical Design Intelligence System
Integrates RTL parsing, Physical IR, and OpenROAD flow
"""

import sys
import os
from typing import Dict, Any, List
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rtl_to_physical_bridge import RTLToPhysicalIRBridge
from real_openroad_interface import RealOpenROADInterface
from physical_ir import PhysicalIR
import tempfile
import json


class PhysicalDesignIntelligence:
    """
    Complete physical design intelligence system
    Integrates:
    - RTL parsing
    - Physical IR generation
    - OpenROAD flow execution
    - Result analysis and learning
    """
    
    def __init__(self):
        self.rtl_bridge = RTLToPhysicalIRBridge()
        self.openroad_interface = RealOpenROADInterface()
        self.design_history = []
    
    def analyze_design(self, rtl_content: str, design_name: str = "unnamed") -> Dict[str, Any]:
        """
        Complete design analysis flow
        
        Args:
            rtl_content: Verilog source code
            design_name: Name for the design
            
        Returns:
            Complete analysis results
        """
        print(f"=== Analyzing Design: {design_name} ===")
        
        physical_ir = PhysicalIR() # Initialize a dummy PhysicalIR
        physical_ir_stats = {
            'num_nodes': 0, 'num_edges': 0, 'total_area': 0.0, 'total_power': 0.0,
            'max_fanout': 0, 'critical_path_nodes': [], 'congestion_hotspots': [],
            'node_types': {}
        }
        openroad_results = {
            'overall_ppa': {'area_um2': 0.0, 'power_mw': 0.0, 'timing_ns': 0.0, 'drc_violations': 0},
            'routing': {'drc_violations': 0},
            'placement': {'timing_slack_ps': 0.0, 'congestion_map': []},
            'floorplan': {'utilization': 0.0},
            'success': False
        }
        comparison = {'area': {}, 'power': {}, 'timing': {}}
        learning_insights = {'complexity_factors': {}, 'bottleneck_identification': {}, 
                             'improvement_opportunities': [], 'feature_correlations': {}}
        
        # Step 1: Convert RTL to Physical IR
        print("Step 1: Converting RTL to Physical IR...")
        try:
            physical_ir = self.rtl_bridge.convert_rtl_to_physical_ir(rtl_content)
            physical_ir_stats = physical_ir.get_statistics()
        except Exception as e:
            print(f"  Warning: RTL to Physical IR conversion failed for {design_name}: {e}")
            # physical_ir is already a dummy, physical_ir_stats set to default
        
        # Step 2: Run OpenROAD flow
        print("Step 2: Running OpenROAD flow...")
        try:
            openroad_results = self.openroad_interface.run_full_flow(rtl_content)
            # Ensure essential keys exist, even if with default values from initial openroad_results
            openroad_results.setdefault('overall_ppa', {'area_um2': 0.0, 'power_mw': 0.0, 'timing_ns': 0.0, 'drc_violations': 0})
            openroad_results.setdefault('routing', {'drc_violations': 0})
            openroad_results.setdefault('placement', {'timing_slack_ps': 0.0, 'congestion_map': []})
            openroad_results.setdefault('floorplan', {'utilization': 0.0})

        except Exception as e:
            print(f"  Warning: OpenROAD flow execution failed for {design_name}: {e}")
            # openroad_results is already a default dict
        
        # Step 3: Compare predictions with actual results (only if enough data is present)
        print("Step 3: Comparing predictions with actual results...")
        try:
            comparison = self._compare_predictions_with_results(physical_ir, openroad_results)
        except Exception as e:
            print(f"  Warning: Prediction comparison failed for {design_name}: {e}")
            # comparison is already a default dict
        
        # Step 4: Extract learning insights
        try:
            learning_insights = self._extract_learning_insights(physical_ir, openroad_results)
        except Exception as e:
            print(f"  Warning: Extracting learning insights failed for {design_name}: {e}")
            # learning_insights is already a default dict
            
        analysis_report = {
            'design_name': design_name,
            'physical_ir_stats': physical_ir_stats,
            'openroad_results': openroad_results,
            'prediction_comparison': comparison,
            'learning_insights': learning_insights
        }
        
        # Store in history for learning
        self.design_history.append(analysis_report)
        
        return analysis_report
    def _compare_predictions_with_results(self, physical_ir: PhysicalIR, openroad_results: Dict) -> Dict:
        """Compare Physical IR predictions with OpenROAD results"""
        
        predicted_area = physical_ir.estimate_area()
        predicted_power = physical_ir.estimate_power()
        predicted_max_delay = physical_ir.estimate_max_delay()
        
        actual_area = openroad_results['overall_ppa']['area_um2']
        actual_power = openroad_results['overall_ppa']['power_mw']
        actual_timing = openroad_results['overall_ppa']['timing_ns']
        
        return {
            'area': {
                'predicted': predicted_area,
                'actual': actual_area,
                'error': abs(predicted_area - actual_area) / actual_area if actual_area > 0 else 0,
                'accuracy': 1.0 - (abs(predicted_area - actual_area) / actual_area) if actual_area > 0 else 0
            },
            'power': {
                'predicted': predicted_power,
                'actual': actual_power,
                'error': abs(predicted_power - actual_power) / actual_power if actual_power > 0 else 0,
                'accuracy': 1.0 - (abs(predicted_power - actual_power) / actual_power) if actual_power > 0 else 0
            },
            'timing': {
                'predicted': predicted_max_delay,
                'actual': actual_timing,
                'error': abs(predicted_max_delay - actual_timing) / actual_timing if actual_timing > 0 else 0,
                'accuracy': 1.0 - (abs(predicted_max_delay - actual_timing) / actual_timing) if actual_timing > 0 else 0
            }
        }
    
    def _extract_learning_insights(self, physical_ir: PhysicalIR, openroad_results: Dict) -> Dict:
        """Extract insights for learning and improvement"""
        
        insights = {
            'complexity_factors': self._analyze_complexity_factors(physical_ir),
            'bottleneck_identification': self._identify_bottlenecks(openroad_results),
            'improvement_opportunities': self._identify_improvements(physical_ir, openroad_results),
            'feature_correlations': self._analyze_feature_correlations(physical_ir, openroad_results)
        }
        
        return insights
    
    def _analyze_complexity_factors(self, physical_ir: PhysicalIR) -> Dict:
        """Analyze factors contributing to design complexity"""
        
        stats = physical_ir.get_statistics()
        
        return {
            'node_count': stats['num_nodes'],
            'edge_count': stats['num_edges'],
            'max_fanout': stats['max_fanout'],
            'critical_path_length': len(physical_ir.get_critical_path_nodes()),
            'congestion_hotspots': stats['congestion_hotspots'],
            'register_ratio': stats['node_types'].get('register', 0) / max(stats['num_nodes'], 1)
        }
    
    def _identify_bottlenecks(self, openroad_results: Dict) -> Dict:
        """Identify bottlenecks from OpenROAD results"""
        
        return {
            'timing_violations': openroad_results['placement']['timing_slack_ps'] < 0,
            'drc_violations': openroad_results['routing']['drc_violations'],
            'congestion_issues': openroad_results['placement']['congestion_map'],
            'utilization': openroad_results['floorplan']['utilization']
        }
    
    def _identify_improvements(self, physical_ir: PhysicalIR, openroad_results: Dict) -> List[str]:
        """Identify potential improvements"""
        
        improvements = []
        
        # Check if timing can be improved
        if openroad_results['placement']['timing_slack_ps'] < 0:
            improvements.append("Pipeline critical paths")
        
        # Check if congestion can be reduced
        if openroad_results['placement']['congestion_map']:
            improvements.append("Cluster related modules to reduce congestion")
        
        # Check if area utilization is low
        if openroad_results['floorplan']['utilization'] < 0.3:
            improvements.append("Optimize floorplan for better utilization")
        
        # Check if DRC violations exist
        if openroad_results['routing']['drc_violations'] > 0:
            improvements.append("Fix DRC violations")
        
        return improvements
    
    def _analyze_feature_correlations(self, physical_ir: PhysicalIR, openroad_results: Dict) -> Dict:
        """Analyze correlations between IR features and PPA results"""
        
        # This would be more sophisticated in a real system
        # For now, simple correlation analysis
        
        stats = physical_ir.get_statistics()
        
        return {
            'node_count_vs_area': stats['num_nodes'] / openroad_results['overall_ppa']['area_um2'] if openroad_results['overall_ppa']['area_um2'] > 0 else 0,
            'edge_count_vs_congestion': stats['num_edges'] / (openroad_results['placement']['congestion_map'][0]['congestion_level'] if openroad_results['placement']['congestion_map'] else 1),
            'critical_path_vs_timing': len(physical_ir.get_critical_path_nodes()) / openroad_results['overall_ppa']['timing_ns'] if openroad_results['overall_ppa']['timing_ns'] > 0 else 0
        }
    
    def generate_report(self, analysis: Dict) -> str:
        """Generate human-readable analysis report"""
        
        report = f"""
PHYSICAL DESIGN INTELLIGENCE REPORT
===================================

Design: {analysis['design_name']}

PHYSICAL IR SUMMARY:
- Nodes: {analysis['physical_ir_stats']['num_nodes']}
- Edges: {analysis['physical_ir_stats']['num_edges']}
- Node Types: {analysis['physical_ir_stats']['node_types']}
- Total Area Est: {analysis['physical_ir_stats']['total_area']:.2f} µm²
- Total Power Est: {analysis['physical_ir_stats']['total_power']:.3f} mW
- Critical Paths: {analysis['physical_ir_stats']['critical_path_nodes']}

OPENROAD RESULTS:
- Actual Area: {analysis['openroad_results']['overall_ppa']['area_um2']:.2f} µm²
- Actual Power: {analysis['openroad_results']['overall_ppa']['power_mw']:.3f} mW
- Actual Timing: {analysis['openroad_results']['overall_ppa']['timing_ns']:.3f} ns
- DRC Violations: {analysis['openroad_results']['routing']['drc_violations']}

PREDICTION ACCURACY:
- Area: {analysis['prediction_comparison']['area']['accuracy']:.2%} (Error: {analysis['prediction_comparison']['area']['error']:.2%})
- Power: {analysis['prediction_comparison']['power']['accuracy']:.2%} (Error: {analysis['prediction_comparison']['power']['error']:.2%})
- Timing: {analysis['prediction_comparison']['timing']['accuracy']:.2%} (Error: {analysis['prediction_comparison']['timing']['error']:.2%})

LEARNING INSIGHTS:
- Complexity Factors: {analysis['learning_insights']['complexity_factors']}
- Bottlenecks: {analysis['learning_insights']['bottleneck_identification']}
- Improvement Opportunities: {analysis['learning_insights']['improvement_opportunities']}
        """
        
        return report
    
    def get_learning_dataset(self) -> List[Dict]:
        """Return dataset for ML model training"""
        
        dataset = []
        for record in self.design_history:
            # Create feature vector from Physical IR
            features = {
                'node_count': record['physical_ir_stats']['num_nodes'],
                'edge_count': record['physical_ir_stats']['num_edges'],
                'max_fanout': record['physical_ir_stats']['max_fanout'],
                'critical_path_nodes': record['physical_ir_stats']['critical_path_nodes'],
                'register_count': record['physical_ir_stats'].get('node_types', {}).get('register', 0),
                'combinational_count': record['physical_ir_stats'].get('node_types', {}).get('combinational', 0),
                'port_count': record['physical_ir_stats'].get('node_types', {}).get('port', 0),
                'total_area_pred': record['physical_ir_stats']['total_area'],
                'total_power_pred': record['physical_ir_stats']['total_power']
            }
            
            # Labels from OpenROAD results
            labels = {
                'actual_area': record['openroad_results']['overall_ppa']['area_um2'],
                'actual_power': record['openroad_results']['overall_ppa']['power_mw'],
                'actual_timing': record['openroad_results']['overall_ppa']['timing_ns'],
                'drc_violations': record['openroad_results']['routing']['drc_violations'],
                'timing_violations': record['openroad_results']['placement']['timing_slack_ps'] < 0
            }
            
            dataset.append({
                'features': features,
                'labels': labels,
                'design_name': record['design_name']
            })
        
        return dataset

    def get_feature_names(self) -> List[str]:
        """Return the names of features used in the learning dataset"""
        if self.design_history:
            # Get features from the first record as template
            dataset = self.get_learning_dataset()
            if dataset:
                return list(dataset[0]['features'].keys())
        # Return default feature names
        return [
            'node_count', 'edge_count', 'max_fanout', 'register_count',
            'combinational_count', 'port_count', 'total_area_pred', 'total_power_pred'
        ]


def test_complete_system():
    """Test the complete physical design intelligence system"""
    
    print("=== Testing Complete Physical Design Intelligence System ===")
    
    # Create the system
    system = PhysicalDesignIntelligence()
    
    # Test design 1: MAC array
    mac_array_rtl = '''
    module mac_array_32x32 (
        input clk,
        input rst_n,
        input [31:0] a_data,
        input [31:0] b_data,
        input [31:0] weight_data,
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
    '''
    
    # Analyze the design
    analysis = system.analyze_design(mac_array_rtl, "MAC_Array_32x32")
    
    # Generate report
    report = system.generate_report(analysis)
    print(report)
    
    # Test design 2: Convolution core
    conv_core_rtl = '''
    module conv_core (
        input clk,
        input rst_n,
        input [7:0] pixel_in,
        input [7:0] weight_in,
        output [15:0] conv_out
    );
        reg [15:0] multiplier_result;
        reg [15:0] accumulator;
        
        always @(posedge clk) begin
            if (!rst_n) begin
                multiplier_result <= 16'd0;
                accumulator <= 16'd0;
            end else begin
                multiplier_result <= pixel_in * weight_in;
                accumulator <= accumulator + multiplier_result;
            end
        end
        
        assign conv_out = accumulator;
    endmodule
    '''
    
    # Analyze second design
    analysis2 = system.analyze_design(conv_core_rtl, "Conv_Core_8x8")
    report2 = system.generate_report(analysis2)
    print("\n" + "="*50)
    print(report2)
    
    # Show learning dataset
    dataset = system.get_learning_dataset()
    print(f"\nLearning Dataset Size: {len(dataset)} designs")
    if dataset:
        print("First record features:", dataset[0]['features'])
        print("First record labels:", dataset[0]['labels'])
    
    return system


if __name__ == "__main__":
    test_complete_system()