#!/usr/bin/env python3
"""
Bridge between professional RTL extractor and CanonicalSiliconGraph
This connects the PyVerilog extraction to the core system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.canonical_silicon_graph import CanonicalSiliconGraph, NodeType, NodeAttributes
from professional_rtl_extractor import ProfessionalRTLExtractor
import tempfile


class RTLParsingBridge:
    """
    Bridge between professional RTL extraction and CanonicalSiliconGraph
    Converts PyVerilog output to graph-compatible format
    """
    
    def __init__(self):
        self.extractor = ProfessionalRTLExtractor()
    
    def build_graph_from_rtl(self, rtl_content: str) -> CanonicalSiliconGraph:
        """
        Build CanonicalSiliconGraph from RTL content using professional extraction
        
        Args:
            rtl_content: Verilog source code string
            
        Returns:
            CanonicalSiliconGraph instance populated with RTL structure
        """
        # Extract design information using professional extractor
        rtl_data = self.extractor.extract_from_string(rtl_content)
        
        # Create canonical graph
        graph = CanonicalSiliconGraph()
        
        # Convert extracted data to graph format
        self._convert_rtl_to_graph(rtl_data, graph)
        
        return graph
    
    def _convert_rtl_to_graph(self, rtl_data: dict, graph: CanonicalSiliconGraph):
        """
        Convert RTL extraction data to CanonicalSiliconGraph format
        """
        # Add modules as high-level nodes
        for module in rtl_data.get('modules', []):
            module_name = module.get('name', 'unnamed_module')
            
            # Add module node
            module_attrs = NodeAttributes(
                node_type=NodeType.CELL,
                cell_type='module',
                area=10.0,  # Placeholder area
                power=0.05,  # Placeholder power
                timing_criticality=0.0,
                estimated_congestion=0.0
            )
            
            graph.graph.add_node(module_name, **module_attrs.__dict__)
        
        # Add ports
        for port in rtl_data.get('ports', []):
            port_name = port.get('name', 'unnamed_port')
            direction = port.get('direction', 'unknown')
            
            port_attrs = NodeAttributes(
                node_type=NodeType.PORT,
                cell_type=f'PORT_{direction.upper()}',
                area=0.0,
                power=0.0,
                timing_criticality=0.0,
                estimated_congestion=0.0
            )
            
            graph.graph.add_node(port_name, **port_attrs.__dict__)
            
            # Connect port to its module (simplified)
            # In real implementation, we'd know which module contains which ports
            if rtl_data.get('modules'):
                module_name = rtl_data['modules'][0].get('name', 'unnamed_module')
                if graph.graph.has_node(module_name):
                    graph.graph.add_edge(port_name, module_name)
        
        # Add registers
        for reg in rtl_data.get('registers', []):
            reg_name = reg.get('name', 'unnamed_register')
            width = reg.get('width', 1)  # Default to 1-bit
            
            # Estimate area and power based on width
            area = width * 2.0  # 2.0 units per bit for registers
            power = width * 0.02  # 0.02W per bit
            
            reg_attrs = NodeAttributes(
                node_type=NodeType.CELL,
                cell_type='register',
                area=area,
                power=power,
                timing_criticality=0.8,  # Registers are typically timing critical
                estimated_congestion=0.0
            )
            
            graph.graph.add_node(reg_name, **reg_attrs.__dict__)
        
        # Add wires
        for wire in rtl_data.get('wires', []):
            wire_name = wire.get('name', 'unnamed_wire')
            width = wire.get('width', 1)
            
            wire_attrs = NodeAttributes(
                node_type=NodeType.SIGNAL,
                cell_type='wire',
                area=0.0,
                power=0.001,  # Small power for signal
                timing_criticality=0.1,
                estimated_congestion=0.0
            )
            
            graph.graph.add_node(wire_name, **wire_attrs.__dict__)
        
        # Add connections based on assignments and always blocks
        self._add_connections(rtl_data, graph)
        
        # Apply timing analysis based on always blocks
        self._apply_timing_analysis(rtl_data, graph)
    
    def _add_connections(self, rtl_data: dict, graph: CanonicalSiliconGraph):
        """Add connections between nodes based on assignments and logic"""
        
        # Add connections from assignments
        for assign in rtl_data.get('assignments', []):
            lhs = assign.get('lhs', '')
            rhs = assign.get('rhs', '')
            
            # Connect LHS to RHS if both exist as nodes
            if lhs and rhs and graph.graph.has_node(lhs) and graph.graph.has_node(rhs):
                graph.graph.add_edge(lhs, rhs)
        
        # Handle always blocks (the count might be stored as an int in fallback extraction)
        always_blocks_count = rtl_data.get('always_blocks', 0)
        if isinstance(always_blocks_count, int):
            # If it's just a count, we can't iterate over it
            # In real PyVerilog extraction, this would be a list of actual blocks
            pass
        else:
            # If it's a list of actual blocks, process them
            for always_block in always_blocks_count:
                sensitivity = always_block.get('sensitivity', '')
                statement = always_block.get('statement', '')
                
                # Extract signals from sensitivity list
                # This is a simplified approach - real extraction would be more detailed
                if sensitivity:
                    # Assume sensitivity list contains signal names
                    # In real PyVerilog, this would be properly parsed
                    pass
    
    def _apply_timing_analysis(self, rtl_data: dict, graph: CanonicalSiliconGraph):
        """Apply basic timing analysis to estimate critical paths"""
        
        # Get always blocks - handle both count (int) and list formats
        always_blocks_data = rtl_data.get('always_blocks', [])
        
        # If it's an integer, it's just a count, not actual blocks
        if isinstance(always_blocks_data, int):
            # If we have always blocks (count > 0), increase criticality for clock-related nodes
            if always_blocks_data > 0:
                for node, attrs in graph.graph.nodes(data=True):
                    if 'clk' in node.lower() or 'clock' in node.lower():
                        graph.graph.nodes[node]['timing_criticality'] = max(
                            graph.graph.nodes[node]['timing_criticality'], 0.9
                        )
                    elif attrs.get('cell_type') == 'register':
                        graph.graph.nodes[node]['timing_criticality'] = max(
                            graph.graph.nodes[node]['timing_criticality'], 0.8
                        )
        else:
            # It's a list of actual always blocks
            if always_blocks_data:
                # Find all registers and increase their criticality
                for node, attrs in graph.graph.nodes(data=True):
                    if attrs.get('cell_type') == 'register':
                        graph.graph.nodes[node]['timing_criticality'] = max(
                            graph.graph.nodes[node]['timing_criticality'], 0.8
                        )
                    
                    # Also increase criticality for signals connected to always blocks
                    # This is a simplified approach
                    if 'clk' in node.lower() or 'clock' in node.lower():
                        graph.graph.nodes[node]['timing_criticality'] = max(
                            graph.graph.nodes[node]['timing_criticality'], 0.9
                        )


def test_bridge():
    """Test the bridge with AI accelerator pattern"""
    
    # Test MAC array pattern
    mac_array_code = '''
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
    
    print("=== Testing RTL Parsing Bridge ===")
    
    bridge = RTLParsingBridge()
    
    # Build graph from RTL
    graph = bridge.build_graph_from_rtl(mac_array_code)
    
    # Check results
    stats = graph.get_graph_statistics()
    
    print(f"✅ Bridge created graph successfully!")
    print(f"Nodes: {stats['num_nodes']}")
    print(f"Edges: {stats['num_edges']}")
    print(f"Node types: {stats['node_types']}")
    print(f"Total area: {stats['total_area']:.2f}")
    print(f"Total power: {stats['total_power']:.3f}")
    
    # Verify key nodes exist
    nodes = list(graph.graph.nodes())
    print(f"\nExtracted nodes: {nodes}")
    
    expected_nodes = ['clk', 'rst_n', 'a_data', 'b_data', 'weight_data', 'result', 'accumulator', 'product']
    found_nodes = [node for node in expected_nodes if node in nodes]
    
    print(f"\nExpected nodes found: {len(found_nodes)}/{len(expected_nodes)}")
    print(f"Found: {found_nodes}")
    
    # Check for critical timing nodes
    critical_nodes = graph.get_timing_critical_nodes(threshold=0.5)
    print(f"Timing critical nodes: {critical_nodes}")
    
    return graph


if __name__ == "__main__":
    test_bridge()