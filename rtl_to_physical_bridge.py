#!/usr/bin/env python3
"""
Bridge from RTL extraction to Physical IR
Connects the professional RTL parser to the physical reasoning layer
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from physical_ir import PhysicalIR, PhysicalNode, PhysicalEdge, NodeType
from rtl_bridge import RTLParsingBridge
import re


class RTLToPhysicalIRBridge:
    """
    Bridge from RTL extraction to Physical IR
    Converts RTL structure to physical reasoning representation
    """
    
    def __init__(self):
        self.rtl_bridge = RTLParsingBridge()
    
    def convert_rtl_to_physical_ir(self, rtl_content: str) -> PhysicalIR:
        """
        Convert RTL content to Physical IR representation
        
        Args:
            rtl_content: Verilog source code string
            
        Returns:
            PhysicalIR instance with physical reasoning structure
        """
        # Use the RTL bridge to extract structure
        canonical_graph = self.rtl_bridge.build_graph_from_rtl(rtl_content)
        
        # Create Physical IR
        physical_ir = PhysicalIR()
        physical_ir.metadata['source_rtl'] = rtl_content[:200] + "..."  # Truncate for metadata
        
        # Convert canonical graph nodes to Physical IR nodes
        self._convert_graph_nodes(canonical_graph, physical_ir)
        
        # Convert edges
        self._convert_graph_edges(canonical_graph, physical_ir)
        
        # Apply physical analysis
        self._analyze_physical_properties(physical_ir)
        
        return physical_ir
    
    def _convert_graph_nodes(self, canonical_graph, physical_ir: PhysicalIR):
        """Convert canonical graph nodes to Physical IR nodes"""
        
        for node_id in canonical_graph.graph.nodes():
            node_attrs = canonical_graph.graph.nodes[node_id]
            
            # Determine node type based on canonical graph attributes
            node_type = self._map_node_type(node_attrs, node_id)
            
            # Extract physical characteristics
            bit_width = self._extract_bit_width(node_id, node_attrs)
            area_estimate = self._estimate_area(node_type, bit_width)
            power_estimate = self._estimate_power(node_type, bit_width)
            delay_estimate = self._estimate_delay(node_type, bit_width)
            timing_criticality = self._calculate_timing_criticality(node_id, node_attrs)
            
            # Create PhysicalNode
            physical_node = PhysicalNode(
                id=node_id,
                node_type=node_type,
                bit_width=bit_width,
                area_estimate=area_estimate,
                power_estimate=power_estimate,
                delay_estimate=delay_estimate,
                timing_criticality=timing_criticality,
                is_clock_sensitive='clk' in node_id.lower() or 'clock' in node_id.lower(),
                clock_domain=self._extract_clock_domain(node_id, node_attrs),
                congestion_score=self._calculate_congestion_score(node_attrs)
            )
            
            physical_ir.add_node(physical_node)
    
    def _convert_graph_edges(self, canonical_graph, physical_ir: PhysicalIR):
        """Convert canonical graph edges to Physical IR edges"""
        
        for source, target in canonical_graph.graph.edges():
            # Get edge attributes from canonical graph
            edge_attrs = canonical_graph.graph[source][target]
            
            # Determine edge type
            edge_type = self._infer_edge_type(source, target, edge_attrs)
            bit_width = self._infer_edge_bit_width(source, target)
            estimated_delay = self._estimate_interconnect_delay(bit_width)
            
            # Create PhysicalEdge
            physical_edge = PhysicalEdge(
                source=source,
                target=target,
                edge_type=edge_type,
                bit_width=bit_width,
                estimated_delay=estimated_delay
            )
            
            physical_ir.add_edge(physical_edge)
    
    def _map_node_type(self, node_attrs: dict, node_id: str) -> NodeType:
        """Map canonical graph node attributes to Physical IR node type"""
        
        # Check for specific cell types
        cell_type = node_attrs.get('cell_type', '').lower()
        
        # Map based on name patterns and attributes
        if 'clk' in node_id.lower() or 'clock' in node_id.lower():
            return NodeType.CLOCK_DOMAIN
        elif 'rst' in node_id.lower() or 'reset' in node_id.lower():
            return NodeType.CONTROL
        elif 'port' in cell_type:
            return NodeType.PORT
        elif any(keyword in cell_type for keyword in ['dff', 'reg', 'ff', 'flipflop']):
            return NodeType.REGISTER
        elif any(keyword in cell_type for keyword in ['ram', 'rom', 'mem', 'fifo']):
            return NodeType.MEMORY
        elif any(keyword in cell_type for keyword in ['mux', 'demux']):
            return NodeType.MUX
        elif any(keyword in cell_type for keyword in ['comp', 'cmp', 'equal', 'greater', 'less']):
            return NodeType.COMPARE
        elif any(keyword in cell_type for keyword in ['add', 'sub', 'mul', 'div', 'mult', 'sum']):
            return NodeType.ARITHMETIC
        elif any(keyword in cell_type for keyword in ['and', 'or', 'xor', 'nand', 'nor', 'xnor', 'not']):
            return NodeType.COMBINATIONAL
        elif 'register' in cell_type or node_attrs.get('node_type') == 'cell':
            # If it's marked as a cell and has register-like properties
            return NodeType.REGISTER
        else:
            # Default to combinational for unknown cells
            return NodeType.COMBINATIONAL
    
    def _extract_bit_width(self, node_id: str, node_attrs: dict) -> int:
        """Extract bit width from node attributes or name"""
        
        # Look for width information in attributes
        if 'width' in node_attrs:
            # This might be a complex width specification, extract numeric part
            width_val = node_attrs['width']
            if isinstance(width_val, str):
                # Try to extract numbers from width specification like "[31:0]"
                nums = re.findall(r'\d+', width_val)
                if nums:
                    # Calculate width: if range is [high:low], width = high-low+1
                    if len(nums) >= 2:
                        high = int(nums[0])
                        low = int(nums[1]) if len(nums) > 1 else 0
                        return max(high, low) - min(high, low) + 1
                    else:
                        return int(nums[0])
            elif isinstance(width_val, (int, float)):
                return int(width_val)
        
        # Look for bit width in node name (e.g., [31:0])
        name_match = re.search(r'\[(\d+):(\d+)\]', node_id)
        if name_match:
            high = int(name_match.group(1))
            low = int(name_match.group(2))
            return high - low + 1
        
        # Default to 1 bit
        return 1
    
    def _estimate_area(self, node_type: NodeType, bit_width: int) -> float:
        """Estimate area based on node type and bit width"""
        
        base_areas = {
            NodeType.PORT: 0.0,
            NodeType.REGISTER: 2.0,  # 2.0 units per bit
            NodeType.COMBINATIONAL: 0.5,  # 0.5 units per gate
            NodeType.ARITHMETIC: 1.5,  # More complex operations
            NodeType.MUX: 0.8,
            NodeType.COMPARE: 1.0,
            NodeType.MEMORY: 10.0,  # Memories are large
            NodeType.CONTROL: 0.5,
            NodeType.CLOCK_DOMAIN: 0.0
        }
        
        base_area = base_areas.get(node_type, 0.5)
        
        # Scale by bit width for multi-bit operations
        if node_type in [NodeType.REGISTER, NodeType.ARITHMETIC, NodeType.MEMORY]:
            return base_area * bit_width
        else:
            return base_area
    
    def _estimate_power(self, node_type: NodeType, bit_width: int) -> float:
        """Estimate power consumption based on node type and bit width"""
        
        base_powers = {
            NodeType.PORT: 0.0,
            NodeType.REGISTER: 0.01,  # 0.01W per bit
            NodeType.COMBINATIONAL: 0.005,
            NodeType.ARITHMETIC: 0.02,
            NodeType.MUX: 0.007,
            NodeType.COMPARE: 0.008,
            NodeType.MEMORY: 0.1,
            NodeType.CONTROL: 0.006,
            NodeType.CLOCK_DOMAIN: 0.0
        }
        
        base_power = base_powers.get(node_type, 0.005)
        
        # Scale by bit width for multi-bit operations
        if node_type in [NodeType.REGISTER, NodeType.ARITHMETIC, NodeType.MEMORY]:
            return base_power * bit_width
        else:
            return base_power
    
    def _estimate_delay(self, node_type: NodeType, bit_width: int) -> float:
        """Estimate delay based on node type and bit width"""
        
        base_delays = {
            NodeType.PORT: 0.0,
            NodeType.REGISTER: 0.1,  # 0.1ns for register access
            NodeType.COMBINATIONAL: 0.05,  # 0.05ns per gate
            NodeType.ARITHMETIC: 0.5,  # Variable for arithmetic operations
            NodeType.MUX: 0.1,
            NodeType.COMPARE: 0.2,
            NodeType.MEMORY: 1.0,  # Memories have higher latency
            NodeType.CONTROL: 0.05,
            NodeType.CLOCK_DOMAIN: 0.0
        }
        
        base_delay = base_delays.get(node_type, 0.05)
        
        # For arithmetic operations, delay might scale with bit width (logarithmic)
        if node_type == NodeType.ARITHMETIC and bit_width > 1:
            import math
            return base_delay * (math.log2(bit_width) if bit_width > 1 else 1)
        elif node_type in [NodeType.REGISTER, NodeType.MEMORY]:
            return base_delay * bit_width  # Scale linearly for storage
        else:
            return base_delay
    
    def _calculate_timing_criticality(self, node_id: str, node_attrs: dict) -> float:
        """Calculate timing criticality based on node attributes"""
        
        criticality = 0.0
        
        # Increase criticality for clock-related nodes
        if 'clk' in node_id.lower() or 'clock' in node_id.lower():
            criticality = max(criticality, 0.9)
        
        # Increase for register nodes
        if node_attrs.get('cell_type', '').lower() in ['register', 'dff', 'ff']:
            criticality = max(criticality, 0.8)
        
        # Increase for arithmetic operations
        if any(keyword in node_id.lower() for keyword in ['add', 'mult', 'multiply', 'sum']):
            criticality = max(criticality, 0.6)
        
        # Use existing timing criticality if available
        existing_criticality = node_attrs.get('timing_criticality', 0.0)
        criticality = max(criticality, existing_criticality)
        
        return min(criticality, 1.0)  # Cap at 1.0
    
    def _extract_clock_domain(self, node_id: str, node_attrs: dict) -> str:
        """Extract clock domain information"""
        # Default to a domain based on the presence of clock in name
        if 'clk' in node_id.lower() or 'clock' in node_id.lower():
            return node_id.lower().replace('clk', '').replace('clock', '').strip('_') or 'default_clk'
        else:
            return 'default_clk'
    
    def _calculate_congestion_score(self, node_attrs: dict) -> float:
        """Calculate initial congestion score"""
        # Start with base score, may be updated after placement
        return node_attrs.get('estimated_congestion', 0.0)
    
    def _infer_edge_type(self, source: str, target: str, edge_attrs: dict) -> str:
        """Infer edge type based on source and target nodes"""
        
        if 'clk' in source.lower() or 'clock' in source.lower():
            return 'clock'
        elif 'rst' in source.lower() or 'reset' in source.lower():
            return 'reset'
        elif any(keyword in source.lower() for keyword in ['ctrl', 'control']):
            return 'control'
        else:
            return 'dataflow'
    
    def _infer_edge_bit_width(self, source: str, target: str) -> int:
        """Infer edge bit width based on source/target"""
        # For now, return 1 - in a real system, this would be inferred from signal types
        return 1
    
    def _estimate_interconnect_delay(self, bit_width: int) -> float:
        """Estimate interconnect delay based on bit width"""
        # Simple estimate: 0.01ns per bit width
        return 0.01 * bit_width
    
    def _analyze_physical_properties(self, physical_ir: PhysicalIR):
        """Analyze and update physical properties after conversion"""
        
        # Update global estimates
        physical_ir.metadata['estimated_area'] = physical_ir.estimate_area()
        physical_ir.metadata['estimated_power'] = physical_ir.estimate_power()
        physical_ir.metadata['estimated_max_delay'] = physical_ir.estimate_max_delay()


def test_rtl_to_physical_bridge():
    """Test the RTL to Physical IR bridge"""
    
    print("=== Testing RTL to Physical IR Bridge ===")
    
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
    
    bridge = RTLToPhysicalIRBridge()
    
    # Convert RTL to Physical IR
    physical_ir = bridge.convert_rtl_to_physical_ir(mac_array_code)
    
    # Check results
    stats = physical_ir.get_statistics()
    
    print(f"✅ Conversion successful!")
    print(f"Nodes: {stats['num_nodes']}")
    print(f"Edges: {stats['num_edges']}")
    print(f"Node types: {stats['node_types']}")
    print(f"Total area: {stats['total_area']:.2f} µm²")
    print(f"Total power: {stats['total_power']:.3f} mW")
    print(f"Critical path nodes: {stats['critical_path_nodes']}")
    print(f"Congestion hotspots: {stats['congestion_hotspots']}")
    
    # Show some physical nodes
    print(f"\nPhysical nodes:")
    for node_id, node in physical_ir.nodes.items():
        if node.node_type != NodeType.PORT:  # Skip ports for brevity
            print(f"  {node_id}: {node.node_type.value}, {node.bit_width}b, {node.area_estimate:.1f}µm²")
    
    return physical_ir


if __name__ == "__main__":
    test_rtl_to_physical_bridge()