#!/usr/bin/env python3
"""
Mock OpenROAD Interface
Simulates OpenROAD flow for testing and development
"""

import os
import tempfile
import subprocess
from typing import Dict, Any, List, Optional
import json
import random


class MockOpenROADInterface:
    """
    Mock OpenROAD interface for development and testing
    Simulates the real OpenROAD flow without requiring actual installation
    """
    
    def __init__(self):
        self.flow_metrics = {}
    
    def run_synthesis(self, rtl_file: str, top_module: str = None) -> Dict[str, Any]:
        """
        Mock synthesis step using Yosys
        """
        print(f"Mock synthesis: {rtl_file}")
        
        # Simulate synthesis metrics based on RTL complexity
        rtl_size = os.path.getsize(rtl_file)
        num_lines = sum(1 for line in open(rtl_file))
        
        # Estimate metrics based on RTL size
        cell_count = max(100, min(10000, int(num_lines * 10)))  # Rough estimate
        area_um2 = cell_count * 2.0  # 2.0 um2 per cell estimate
        power_mw = cell_count * 0.001  # 0.001 mW per cell estimate
        
        return {
            'tool': 'yosys',
            'step': 'synthesis',
            'cell_count': cell_count,
            'estimated_area_um2': area_um2,
            'estimated_power_mw': power_mw,
            'timing_paths': [],
            'utilization': 0.3,  # 30% utilization
            'success': True
        }
    
    def run_floorplan(self, netlist_file: str) -> Dict[str, Any]:
        """
        Mock floorplanning step
        """
        print(f"Mock floorplan: {netlist_file}")
        
        # Simulate floorplan metrics
        return {
            'tool': 'openroad',
            'step': 'floorplan',
            'die_area_um2': 10000.0,
            'core_area_um2': 8000.0,
            'utilization': 0.4,
            'aspect_ratio': 1.0,
            'row_height': 2.0,
            'track_pitch': 0.2,
            'macros_placed': 0,
            'success': True
        }
    
    def run_placement(self, def_file: str, sdc_file: str = None) -> Dict[str, Any]:
        """
        Mock placement step
        """
        print(f"Mock placement: {def_file}")
        
        # Simulate placement metrics
        return {
            'tool': 'openroad',
            'step': 'placement',
            'wirelength_um': 15000.0,
            'congestion_map': self._generate_congestion_map(),
            'timing_slack_ps': -100.0,  # Negative means violation
            'cell_density': 0.6,
            'placement_cost': 1200.0,
            'success': True
        }
    
    def run_cts(self, def_file: str, sdc_file: str = None) -> Dict[str, Any]:
        """
        Mock clock tree synthesis
        """
        print(f"Mock CTS: {def_file}")
        
        return {
            'tool': 'openroad',
            'step': 'cts',
            'clock_skew_ps': 50.0,
            'clock_latency_ns': 2.0,
            'buffer_count': 50,
            'clock_utilization': 0.8,
            'success': True
        }
    
    def run_routing(self, def_file: str) -> Dict[str, Any]:
        """
        Mock routing step
        """
        print(f"Mock routing: {def_file}")
        
        return {
            'tool': 'openroad',
            'step': 'routing',
            'routeability': 0.95,
            'drc_violations': 5,
            'wirelength_um': 25000.0,
            'via_count': 1000,
            'layer_usage': {'met1': 0.3, 'met2': 0.4, 'met3': 0.2},
            'success': True
        }
    
    def run_full_flow(self, rtl_content: str, top_module: str = None) -> Dict[str, Any]:
        """
        Run the complete OpenROAD flow
        """
        print("Running mock OpenROAD flow...")
        
        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write RTL to file
            rtl_file = os.path.join(temp_dir, "design.v")
            with open(rtl_file, 'w') as f:
                f.write(rtl_content)
            
            # Run flow steps
            synth_result = self.run_synthesis(rtl_file, top_module)
            floorplan_result = self.run_floorplan(rtl_file.replace('.v', '.nl'))  # Mock netlist
            placement_result = self.run_placement(rtl_file.replace('.v', '.def'))
            cts_result = self.run_cts(rtl_file.replace('.v', '.def'))
            routing_result = self.run_routing(rtl_file.replace('.v', '.def'))
            
            # Aggregate results
            flow_result = {
                'synthesis': synth_result,
                'floorplan': floorplan_result,
                'placement': placement_result,
                'cts': cts_result,
                'routing': routing_result,
                'overall_ppa': {
                    'area_um2': synth_result['estimated_area_um2'] * 1.5,  # Add 50% for P&R overhead
                    'power_mw': synth_result['estimated_power_mw'] * 1.2,  # Add 20% for P&R overhead
                    'timing_ns': abs(placement_result['timing_slack_ps']) / 1000.0  # Convert ps to ns
                },
                'success': all([
                    synth_result['success'],
                    floorplan_result['success'],
                    placement_result['success'],
                    cts_result['success'],
                    routing_result['success']
                ])
            }
            
            return flow_result
    
    def _generate_congestion_map(self) -> List[Dict[str, Any]]:
        """Generate mock congestion map"""
        congestion_map = []
        for x in range(10):
            for y in range(10):
                congestion_map.append({
                    'x': x,
                    'y': y,
                    'congestion_level': random.uniform(0.0, 1.0),
                    'metal_layer': 'met1'
                })
        return congestion_map


def test_mock_openroad():
    """Test the mock OpenROAD interface"""
    
    print("=== Testing Mock OpenROAD Interface ===")
    
    # Test MAC array RTL
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
    
    mock_or = MockOpenROADInterface()
    
    # Run the flow
    result = mock_or.run_full_flow(mac_array_rtl, "mac_array_32x32")
    
    print(f"✅ Mock OpenROAD flow completed: {result['success']}")
    print(f"Area: {result['overall_ppa']['area_um2']:.2f} µm²")
    print(f"Power: {result['overall_ppa']['power_mw']:.3f} mW")
    print(f"Timing: {result['overall_ppa']['timing_ns']:.3f} ns")
    print(f"Synthesis cells: {result['synthesis']['cell_count']}")
    print(f"DRC violations: {result['routing']['drc_violations']}")
    
    return result


if __name__ == "__main__":
    test_mock_openroad()