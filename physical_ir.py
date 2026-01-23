#!/usr/bin/env python3
"""
Physical Design Intermediate Representation (IR)
Structured representation for physical reasoning and AI analysis
"""

import networkx as nx
import json
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
import numpy as np


class NodeType(Enum):
    """Types of nodes in the physical IR"""
    REGISTER = "register"           # Sequential elements
    COMBINATIONAL = "combinational" # Logic gates, arithmetic
    MEMORY = "memory"              # Memories, FIFOs
    MUX = "mux"                    # Multiplexers
    COMPARE = "compare"            # Comparators
    ARITHMETIC = "arithmetic"      # Adders, multipliers
    CONTROL = "control"            # FSM, control logic
    PORT = "port"                  # Interface ports
    CLOCK_DOMAIN = "clock_domain"  # Clock boundaries


@dataclass
class PhysicalNode:
    """Node in the physical IR with physical characteristics"""
    id: str
    node_type: NodeType
    bit_width: int = 1
    depth: int = 1  # For memories, pipelines
    fanout: int = 0  # Output connections
    fanin: int = 0   # Input connections
    clock_domain: str = "default_clk"
    is_clock_sensitive: bool = False
    area_estimate: float = 0.0  # µm² estimate
    power_estimate: float = 0.0  # mW estimate
    delay_estimate: float = 0.0  # ns estimate
    timing_criticality: float = 0.0  # 0.0-1.0
    congestion_score: float = 0.0  # 0.0-1.0
    position_hint: Optional[Tuple[float, float]] = None
    region_hint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhysicalEdge:
    """Edge in the physical IR"""
    source: str
    target: str
    edge_type: str = "dataflow"  # dataflow, clock, reset, control
    bit_width: int = 1
    estimated_delay: float = 0.0  # ns
    congestion_score: float = 0.0  # 0.0-1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PhysicalIR:
    """
    Physical Design Intermediate Representation
    Purpose: Enable physical reasoning and AI analysis
    """
    
    def __init__(self):
        self.nodes: Dict[str, PhysicalNode] = {}
        self.edges: List[PhysicalEdge] = []
        self.graph = nx.DiGraph()  # For topology analysis
        self.metadata = {
            'created_at': None,
            'source_rtl': None,
            'target_process': 'sky130',
            'estimated_area': 0.0,
            'estimated_power': 0.0,
            'estimated_max_delay': 0.0
        }
    
    def add_node(self, node: PhysicalNode):
        """Add a physical node to the IR"""
        self.nodes[node.id] = node
        # Add to networkx graph for analysis
        self.graph.add_node(node.id, **node.__dict__)
    
    def add_edge(self, edge: PhysicalEdge):
        """Add a physical edge to the IR"""
        self.edges.append(edge)
        # Add to networkx graph for analysis
        self.graph.add_edge(edge.source, edge.target, **edge.__dict__)
    
    def connect_nodes(self, source_id: str, target_id: str, 
                     edge_type: str = "dataflow", bit_width: int = 1,
                     estimated_delay: float = 0.0) -> PhysicalEdge:
        """Convenience method to connect two nodes"""
        edge = PhysicalEdge(
            source=source_id,
            target=target_id,
            edge_type=edge_type,
            bit_width=bit_width,
            estimated_delay=estimated_delay
        )
        self.add_edge(edge)
        
        # Update fanout/fanin counts
        if source_id in self.nodes:
            self.nodes[source_id].fanout += 1
        if target_id in self.nodes:
            self.nodes[target_id].fanin += 1
        
        return edge
    
    def get_fanout_tree(self, node_id: str) -> List[str]:
        """Get all nodes reachable from this node (fanout tree)"""
        if node_id not in self.graph:
            return []
        return list(nx.descendants(self.graph, node_id))
    
    def get_fanin_tree(self, node_id: str) -> List[str]:
        """Get all nodes that reach this node (fanin tree)"""
        if node_id not in self.graph:
            return []
        return list(nx.ancestors(self.graph, node_id))
    
    def calculate_depth(self, node_id: str) -> int:
        """Calculate depth of node in the dataflow graph"""
        if node_id not in self.graph:
            return 0
        # Calculate longest path from any input node
        input_nodes = [n for n, d in self.graph.in_degree() if d == 0]
        max_depth = 0
        for input_node in input_nodes:
            try:
                path_length = nx.shortest_path_length(self.graph, input_node, node_id)
                max_depth = max(max_depth, path_length)
            except nx.NetworkXNoPath:
                continue
        return max_depth
    
    def get_critical_path_nodes(self) -> List[str]:
        """Get nodes on the critical path"""
        # For now, return nodes with highest timing criticality
        critical_nodes = []
        for node_id, node in self.nodes.items():
            if node.timing_criticality > 0.7:  # Threshold for "critical"
                critical_nodes.append(node_id)
        return critical_nodes
    
    def get_congestion_hotspots(self, threshold: float = 0.7) -> List[str]:
        """Get nodes with high congestion scores"""
        hotspots = []
        for node_id, node in self.nodes.items():
            if node.congestion_score > threshold:
                hotspots.append(node_id)
        return hotspots
    
    def estimate_area(self) -> float:
        """Estimate total area of the design"""
        return sum(node.area_estimate for node in self.nodes.values())
    
    def estimate_power(self) -> float:
        """Estimate total power of the design"""
        return sum(node.power_estimate for node in self.nodes.values())
    
    def estimate_max_delay(self) -> float:
        """Estimate maximum delay in the design"""
        if not self.nodes:
            return 0.0
        return max(node.delay_estimate for node in self.nodes.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the IR"""
        node_types = {}
        total_area = 0.0
        total_power = 0.0
        max_delay = 0.0
        
        for node in self.nodes.values():
            node_types[node.node_type.value] = node_types.get(node.node_type.value, 0) + 1
            total_area += node.area_estimate
            total_power += node.power_estimate
            max_delay = max(max_delay, node.delay_estimate)
        
        return {
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'node_types': node_types,
            'total_area': total_area,
            'total_power': total_power,
            'max_delay': max_delay,
            'avg_fanout': np.mean([n.fanout for n in self.nodes.values()]) if self.nodes else 0,
            'max_fanout': max([n.fanout for n in self.nodes.values()], default=0),
            'critical_path_nodes': len(self.get_critical_path_nodes()),
            'congestion_hotspots': len(self.get_congestion_hotspots())
        }
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize the IR to dictionary"""
        return {
            'nodes': {
                nid: {
                    'id': node.id,
                    'node_type': node.node_type.value,
                    'bit_width': node.bit_width,
                    'depth': node.depth,
                    'fanout': node.fanout,
                    'fanin': node.fanin,
                    'clock_domain': node.clock_domain,
                    'is_clock_sensitive': node.is_clock_sensitive,
                    'area_estimate': node.area_estimate,
                    'power_estimate': node.power_estimate,
                    'delay_estimate': node.delay_estimate,
                    'timing_criticality': node.timing_criticality,
                    'congestion_score': node.congestion_score,
                    'position_hint': node.position_hint,
                    'region_hint': node.region_hint,
                    'metadata': node.metadata
                } for nid, node in self.nodes.items()
            },
            'edges': [
                {
                    'source': edge.source,
                    'target': edge.target,
                    'edge_type': edge.edge_type,
                    'bit_width': edge.bit_width,
                    'estimated_delay': edge.estimated_delay,
                    'congestion_score': edge.congestion_score,
                    'metadata': edge.metadata
                } for edge in self.edges
            ],
            'metadata': self.metadata
        }
    
    def deserialize(self, data: Dict[str, Any]):
        """Deserialize the IR from dictionary"""
        self.nodes = {}
        self.edges = []
        self.graph = nx.DiGraph()
        
        # Rebuild nodes
        for node_id, node_data in data['nodes'].items():
            node = PhysicalNode(
                id=node_data['id'],
                node_type=NodeType(node_data['node_type']),
                bit_width=node_data['bit_width'],
                depth=node_data['depth'],
                fanout=node_data['fanout'],
                fanin=node_data['fanin'],
                clock_domain=node_data['clock_domain'],
                is_clock_sensitive=node_data['is_clock_sensitive'],
                area_estimate=node_data['area_estimate'],
                power_estimate=node_data['power_estimate'],
                delay_estimate=node_data['delay_estimate'],
                timing_criticality=node_data['timing_criticality'],
                congestion_score=node_data['congestion_score'],
                position_hint=node_data['position_hint'],
                region_hint=node_data['region_hint'],
                metadata=node_data['metadata']
            )
            self.add_node(node)
        
        # Rebuild edges
        for edge_data in data['edges']:
            edge = PhysicalEdge(
                source=edge_data['source'],
                target=edge_data['target'],
                edge_type=edge_data['edge_type'],
                bit_width=edge_data['bit_width'],
                estimated_delay=edge_data['estimated_delay'],
                congestion_score=edge_data['congestion_score'],
                metadata=edge_data['metadata']
            )
            self.add_edge(edge)
        
        self.metadata = data['metadata']


def test_physical_ir():
    """Test the Physical IR with an AI accelerator pattern"""
    
    print("=== Testing Physical IR ===")
    
    ir = PhysicalIR()
    
    # Create a simple MAC unit: accumulator <- accumulator + (a * b)
    
    # Input ports
    clk_node = PhysicalNode(
        id="clk",
        node_type=NodeType.PORT,
        bit_width=1,
        is_clock_sensitive=True,
        area_estimate=0.0,
        power_estimate=0.0,
        delay_estimate=0.0
    )
    
    a_node = PhysicalNode(
        id="a_data",
        node_type=NodeType.PORT,
        bit_width=32,
        area_estimate=0.0,
        power_estimate=0.001,
        delay_estimate=0.0
    )
    
    b_node = PhysicalNode(
        id="b_data", 
        node_type=NodeType.PORT,
        bit_width=32,
        area_estimate=0.0,
        power_estimate=0.001,
        delay_estimate=0.0
    )
    
    # Multiplier
    mult_node = PhysicalNode(
        id="multiplier",
        node_type=NodeType.ARITHMETIC,
        bit_width=32,
        area_estimate=50.0,  # Large multiplier
        power_estimate=0.1,
        delay_estimate=1.5,  # Multiplier delay
        timing_criticality=0.8
    )
    
    # Adder
    adder_node = PhysicalNode(
        id="adder",
        node_type=NodeType.ARITHMETIC,
        bit_width=32,
        area_estimate=25.0,
        power_estimate=0.05,
        delay_estimate=1.0,
        timing_criticality=0.9
    )
    
    # Accumulator register
    accum_reg = PhysicalNode(
        id="accumulator",
        node_type=NodeType.REGISTER,
        bit_width=32,
        area_estimate=20.0,  # 32-bit register
        power_estimate=0.02,
        delay_estimate=0.2,
        timing_criticality=0.95,  # Critical register
        is_clock_sensitive=True
    )
    
    # Output port
    result_node = PhysicalNode(
        id="result",
        node_type=NodeType.PORT,
        bit_width=32,
        area_estimate=0.0,
        power_estimate=0.001,
        delay_estimate=0.0
    )
    
    # Add all nodes
    for node in [clk_node, a_node, b_node, mult_node, adder_node, accum_reg, result_node]:
        ir.add_node(node)
    
    # Connect them: a * b -> adder, accumulator -> adder, result
    ir.connect_nodes("a_data", "multiplier", "dataflow", 32)
    ir.connect_nodes("b_data", "multiplier", "dataflow", 32)
    ir.connect_nodes("multiplier", "adder", "dataflow", 32)
    ir.connect_nodes("accumulator", "adder", "dataflow", 32)
    ir.connect_nodes("adder", "accumulator", "dataflow", 32)  # Feedback
    ir.connect_nodes("accumulator", "result", "dataflow", 32)
    
    # Add clock connection
    ir.connect_nodes("clk", "accumulator", "clock", 1)
    
    # Print statistics
    stats = ir.get_statistics()
    print(f"✅ Physical IR created successfully!")
    print(f"Nodes: {stats['num_nodes']}")
    print(f"Edges: {stats['num_edges']}")
    print(f"Node types: {stats['node_types']}")
    print(f"Total area: {stats['total_area']:.2f} µm²")
    print(f"Total power: {stats['total_power']:.3f} mW")
    print(f"Critical path nodes: {stats['critical_path_nodes']}")
    print(f"Max fanout: {stats['max_fanout']}")
    
    # Show critical path
    critical_nodes = ir.get_critical_path_nodes()
    print(f"Critical nodes: {critical_nodes}")
    
    # Test serialization
    serialized = ir.serialize()
    print(f"Serialization: {len(json.dumps(serialized))} bytes")
    
    # Test deserialization
    new_ir = PhysicalIR()
    new_ir.deserialize(serialized)
    new_stats = new_ir.get_statistics()
    print(f"Deserialized stats match: {stats['num_nodes'] == new_stats['num_nodes']}")
    
    return ir


if __name__ == "__main__":
    test_physical_ir()