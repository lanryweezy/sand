"""
Canonical Silicon Graph - Unified representation for all physical design aspects

This module implements a common mental model that all agents and components
can use to communicate and share information about the chip design.

Features:
- Unified graph representation for all design aspects
- Deep copy support for independent graph instances
- Consistency validation
- JSON serialization/deserialization
- Transaction support for atomic updates
"""

import networkx as nx
import copy
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import contextmanager
import numpy as np
from utils.logger import get_logger

try:
    import silicon_intelligence_cpp as sic
    HAS_CPP_CORE = True
except ImportError:
    HAS_CPP_CORE = False


class NodeType(Enum):
    """Types of nodes in the silicon graph"""
    CELL = "cell"           # Standard cells
    MACRO = "macro"         # IP blocks, memories
    PORT = "port"           # IO ports
    CLOCK = "clock"         # Clock sources and buffers
    POWER = "power"         # Power supply nodes
    SIGNAL = "signal"       # Signal nets


class EdgeType(Enum):
    """Types of edges in the silicon graph"""
    CONNECTION = "connection"     # Electrical connection
    PHYSICAL_PROXIMITY = "proximity"  # Physical adjacency
    TIMING_DEPENDENCY = "timing"      # Timing relationship
    POWER_FEED = "power_feed"         # Power distribution


@dataclass
class NodeAttributes:
    """Standard attributes for nodes in the silicon graph"""
    node_type: NodeType
    cell_type: Optional[str] = None      # Specific cell type (AND2, DFF, etc.)
    area: Optional[float] = None         # Physical area
    power: Optional[float] = None        # Power consumption
    delay: Optional[float] = None        # Propagation delay
    capacitance: Optional[float] = None  # Capacitance
    position: Optional[Tuple[float, float]] = None  # Physical coordinates
    region: Optional[str] = None         # Physical region/floorplan
    voltage_domain: Optional[str] = None # Voltage domain
    clock_domain: Optional[str] = None   # Clock domain
    is_clock_root: bool = False          # For clock nodes
    is_macro: bool = False               # For macro identification
    estimated_congestion: float = 0.0    # Estimated routing congestion
    timing_criticality: float = 0.0      # Timing criticality (0-1)


@dataclass
class EdgeAttributes:
    """Standard attributes for edges in the silicon graph"""
    edge_type: EdgeType
    resistance: Optional[float] = None   # Resistance for connections
    capacitance: Optional[float] = None  # Capacitance for connections
    delay: Optional[float] = None        # Delay for signal propagation
    length: Optional[float] = None       # Physical length
    layers_used: Optional[List[str]] = None  # Metal layers used
    congestion: float = 0.0              # Current congestion level
    capacity: float = 1.0                # Capacity (for routing)


class CanonicalSiliconGraph:
    """
    Canonical Silicon Graph - Unified representation for chip design
    
    Everything becomes a graph:
    - Nodes = cells, macros, clock elements
    - Edges = timing, power, spatial affinity
    - Fields = density, delay sensitivity, variability
    
    All agents read and write to this graph for communication.
    
    Features:
    - Deep copy support for independent instances
    - Consistency validation
    - JSON serialization/deserialization
    - Transaction support for atomic updates
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.graph = nx.MultiDiGraph()  # MultiDiGraph to allow multiple edges of different types
        self.metadata = {
            'created_at': None,
            'target_node': None,
            'design_hierarchy': [],
            'constraint_sets': [],
            'power_domains': [],
            'clock_domains': []
        }
        self._transaction_stack = []  # Stack for nested transactions
        
        # Initialize C++ Accelerator if available
        self.cpp_engine = sic.GraphEngine() if HAS_CPP_CORE else None
        
        # Enums and attribute maps for C++ conversion
        self._node_type_map = {
            NodeType.CELL: sic.NodeType.CELL,
            NodeType.MACRO: sic.NodeType.MACRO,
            NodeType.PORT: sic.NodeType.PORT,
            NodeType.CLOCK: sic.NodeType.CLOCK,
            NodeType.POWER: sic.NodeType.POWER,
            NodeType.SIGNAL: sic.NodeType.SIGNAL
        } if HAS_CPP_CORE else {}
        
        self._edge_type_map = {
            EdgeType.CONNECTION: sic.EdgeType.CONNECTION,
            EdgeType.PHYSICAL_PROXIMITY: sic.EdgeType.PHYSICAL_PROXIMITY,
            EdgeType.TIMING_DEPENDENCY: sic.EdgeType.TIMING_DEPENDENCY,
            EdgeType.POWER_FEED: sic.EdgeType.POWER_FEED
        } if HAS_CPP_CORE else {}
    
    def __deepcopy__(self, memo):
        """
        Create a deep copy of the graph
        
        Args:
            memo: Dictionary for tracking copied objects (used by copy module)
            
        Returns:
            New CanonicalSiliconGraph instance with independent copy of data
        """
        # Create new instance
        new_graph = CanonicalSiliconGraph()
        
        # Deep copy the NetworkX graph
        new_graph.graph = copy.deepcopy(self.graph, memo)
        
        # Deep copy metadata
        new_graph.metadata = copy.deepcopy(self.metadata, memo)
        
        # Deep copy transaction stack
        new_graph._transaction_stack = copy.deepcopy(self._transaction_stack, memo)
        
        return new_graph
    
    def copy(self):
        """Create a shallow copy of the graph"""
        return copy.copy(self)
    
    def validate_graph_consistency(self) -> Tuple[bool, List[str]]:
        """
        Validate graph consistency and integrity
        
        Checks:
        - All nodes have required attributes
        - All edges have required attributes
        - No orphaned nodes (except ports)
        - Attribute values in valid ranges
        - No duplicate node names
        - Edge endpoints exist
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check 1: All nodes have required attributes
        required_node_attrs = ['node_type']
        for node, attrs in self.graph.nodes(data=True):
            for required_attr in required_node_attrs:
                if required_attr not in attrs:
                    errors.append(f"Node '{node}' missing required attribute: {required_attr}")
            
            # Check 2: Timing criticality in valid range
            crit = attrs.get('timing_criticality', 0.0)
            if not (0.0 <= crit <= 1.0):
                errors.append(f"Node '{node}' has invalid timing_criticality: {crit} (must be 0-1)")
            
            # Check 3: Estimated congestion in valid range
            cong = attrs.get('estimated_congestion', 0.0)
            if not (0.0 <= cong <= 1.0):
                errors.append(f"Node '{node}' has invalid estimated_congestion: {cong} (must be 0-1)")
            
            # Check 4: Area must be non-negative
            area = attrs.get('area', 0.0)
            if area < 0:
                errors.append(f"Node '{node}' has negative area: {area}")
            
            # Check 5: Power must be non-negative
            power = attrs.get('power', 0.0)
            if power < 0:
                errors.append(f"Node '{node}' has negative power: {power}")
        
        # Check 6: All edges have required attributes
        required_edge_attrs = ['edge_type']
        for src, dst, key, attrs in self.graph.edges(keys=True, data=True):
            for required_attr in required_edge_attrs:
                if required_attr not in attrs:
                    errors.append(f"Edge '{src}'->{dst}' missing required attribute: {required_attr}")
        
        # Check 7: No orphaned nodes (except ports)
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get('node_type') != 'port':
                if self.graph.degree(node) == 0:
                    errors.append(f"Node '{node}' is orphaned (no connections)")
        
        # Check 8: Edge endpoints exist
        for src, dst, key, attrs in self.graph.edges(keys=True, data=True):
            if src not in self.graph.nodes():
                errors.append(f"Edge source node '{src}' does not exist")
            if dst not in self.graph.nodes():
                errors.append(f"Edge target node '{dst}' does not exist")
        
        # Check 9: No duplicate node names
        node_names = list(self.graph.nodes())
        if len(node_names) != len(set(node_names)):
            errors.append("Duplicate node names detected")
        
        # Check 10: Metadata consistency
        if not isinstance(self.metadata, dict):
            errors.append("Metadata is not a dictionary")
        
        return len(errors) == 0, errors
    
    @contextmanager
    def transaction(self):
        """
        Context manager for atomic graph updates
        
        Usage:
            with graph.transaction():
                graph.update_node_attributes('node1', attr='value')
                graph.update_edge_attributes('node1', 'node2', EdgeType.CONNECTION, attr='value')
                # If exception occurs, changes are rolled back
        
        Yields:
            Self for chaining
        """
        # Save current state
        saved_graph = copy.deepcopy(self.graph)
        saved_metadata = copy.deepcopy(self.metadata)
        
        try:
            yield self
        except Exception as e:
            # Rollback on error
            self.logger.warning(f"Transaction failed, rolling back: {str(e)}")
            self.graph = saved_graph
            self.metadata = saved_metadata
            raise e
    
    def serialize_to_json(self, filepath: str) -> None:
        """
        Serialize graph to JSON file
        
        Args:
            filepath: Path to output JSON file
            
        Raises:
            IOError: If file cannot be written
        """
        self.logger.info(f"Serializing graph to {filepath}")
        
        try:
            data = {
                'nodes': {},
                'edges': [],
                'metadata': self.metadata
            }
            
            # Serialize nodes
            for node, attrs in self.graph.nodes(data=True):
                node_attrs = {}
                for key, value in attrs.items():
                    # Convert enums to strings
                    if isinstance(value, Enum):
                        node_attrs[key] = value.value
                    # Convert numpy types to Python types
                    elif isinstance(value, (np.integer, np.floating)):
                        node_attrs[key] = float(value) if isinstance(value, np.floating) else int(value)
                    # Convert lists/tuples
                    elif isinstance(value, (list, tuple)):
                        node_attrs[key] = list(value)
                    else:
                        node_attrs[key] = value
                data['nodes'][str(node)] = node_attrs
            
            # Serialize edges
            for src, dst, key, attrs in self.graph.edges(keys=True, data=True):
                edge_attrs = {}
                for attr_key, attr_value in attrs.items():
                    # Convert enums to strings
                    if isinstance(attr_value, Enum):
                        edge_attrs[attr_key] = attr_value.value
                    # Convert numpy types
                    elif isinstance(attr_value, (np.integer, np.floating)):
                        edge_attrs[attr_key] = float(attr_value) if isinstance(attr_value, np.floating) else int(attr_value)
                    # Convert lists/tuples
                    elif isinstance(attr_value, (list, tuple)):
                        edge_attrs[attr_key] = list(attr_value)
                    else:
                        edge_attrs[attr_key] = attr_value
                
                data['edges'].append({
                    'source': str(src),
                    'target': str(dst),
                    'key': str(key),
                    'attributes': edge_attrs
                })
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Successfully serialized graph to {filepath}")
        
        except Exception as e:
            self.logger.error(f"Failed to serialize graph: {str(e)}")
            raise IOError(f"Failed to serialize graph to {filepath}: {str(e)}")
    
    def deserialize_from_json(self, filepath: str) -> None:
        """
        Deserialize graph from JSON file
        
        Args:
            filepath: Path to input JSON file
            
        Raises:
            IOError: If file cannot be read or is invalid
        """
        self.logger.info(f"Deserializing graph from {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Clear current graph
            self.graph.clear()
            
            # Deserialize nodes
            for node_name, node_attrs in data['nodes'].items():
                # Convert string enums back to Enum objects
                if 'node_type' in node_attrs:
                    try:
                        node_attrs['node_type'] = NodeType(node_attrs['node_type'])
                    except (ValueError, KeyError):
                        pass  # Keep as string if not a valid enum
                
                self.graph.add_node(node_name, **node_attrs)
            
            # Deserialize edges
            for edge_data in data['edges']:
                src = edge_data['source']
                dst = edge_data['target']
                key = edge_data['key']
                attrs = edge_data['attributes']
                
                # Convert string enums back to Enum objects
                if 'edge_type' in attrs:
                    try:
                        attrs['edge_type'] = EdgeType(attrs['edge_type'])
                    except (ValueError, KeyError):
                        pass  # Keep as string if not a valid enum
                
                self.graph.add_edge(src, dst, key=key, **attrs)
            
            # Restore metadata
            self.metadata = data['metadata']
            
            self.logger.info(f"Successfully deserialized graph from {filepath}")
        
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON file: {str(e)}")
            raise IOError(f"Invalid JSON file {filepath}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Failed to deserialize graph: {str(e)}")
            raise IOError(f"Failed to deserialize graph from {filepath}: {str(e)}")
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the graph
        
        Returns:
            Dictionary with graph statistics
        """
        node_types = {}
        edge_types = {}
        total_area = 0.0
        total_power = 0.0
        
        for node, attrs in self.graph.nodes(data=True):
            node_type = attrs.get('node_type', 'unknown')
            node_types[str(node_type)] = node_types.get(str(node_type), 0) + 1
            total_area += attrs.get('area', 0.0)
            total_power += attrs.get('power', 0.0)
        
        for src, dst, key, attrs in self.graph.edges(keys=True, data=True):
            edge_type = attrs.get('edge_type', 'unknown')
            edge_types[str(edge_type)] = edge_types.get(str(edge_type), 0) + 1
        
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'node_types': node_types,
            'edge_types': edge_types,
            'total_area': total_area,
            'total_power': total_power,
            'avg_area': total_area / max(self.graph.number_of_nodes(), 1),
            'avg_power': total_power / max(self.graph.number_of_nodes(), 1),
            'density': self.graph.number_of_edges() / max(self.graph.number_of_nodes(), 1)
        }
    
    def build_from_rtl(self, rtl_data: Dict, constraints: Dict = None, 
                       floorplan_hints: Dict = None, power_info: Dict = None) -> 'CanonicalSiliconGraph':
        """
        Build the canonical silicon graph from RTL and constraints
        
        Args:
            rtl_data: Parsed RTL data
            constraints: Design constraints (SDC, etc.)
            floorplan_hints: Optional floorplan guidance
            power_info: Power information (UPF, etc.)
            
        Returns:
            Self for chaining
        """
        self.logger.info("Building canonical silicon graph from RTL")
        
        # Add nodes from RTL
        self._add_nodes_from_rtl(rtl_data)
        
        # Add edges representing connections
        self._add_edges_from_connections(rtl_data)
        
        # Apply constraints to graph
        if constraints:
            self._apply_constraints(constraints)
        
        # Apply power information
        if power_info:
            self._apply_power_info(power_info)
        
        # Apply floorplan hints
        if floorplan_hints:
            self._apply_floorplan_hints(floorplan_hints)
        
        # Initialize physical properties
        self._initialize_physical_properties()
        
        return self
    
    def _add_nodes_from_rtl(self, rtl_data: Dict):
        """Add nodes to the graph based on RTL data"""
        # Add cells from RTL
        for cell_instance in rtl_data.get('instances', []):
            cell_name = cell_instance['name']
            cell_type = cell_instance['type']
            
            # Determine node type based on cell type
            if cell_type.startswith('sky130'):
                node_type = NodeType.CELL
            elif cell_type in ['RAM', 'ROM', 'SRAM', 'DRAM', 'FLASH']:
                node_type = NodeType.MACRO
            elif cell_type in ['BUF', 'CLKBUF']:
                node_type = NodeType.CLOCK
            elif cell_type in ['VDD', 'VSS', 'POWER']:
                node_type = NodeType.POWER
            elif cell_type in ['INPUT', 'OUTPUT', 'INOUT']:
                node_type = NodeType.PORT
            else:
                node_type = NodeType.SIGNAL
            
            # Get cell properties from library (simplified)
            cell_props = self._get_cell_properties(cell_type)
            
            # Create node attributes
            attrs = NodeAttributes(
                node_type=node_type,
                cell_type=cell_type,
                area=cell_props.get('area', 1.0),
                power=cell_props.get('power', 0.01),
                delay=cell_props.get('delay', 0.1),
                capacitance=cell_props.get('capacitance', 0.001),
                is_macro=(node_type == NodeType.MACRO),
                timing_criticality=0.0  # Will be calculated later
            )
            
            # Add node to graph
            self.graph.add_node(cell_name, **attrs.__dict__)
            
            # Update C++ Accelerator
            self._update_cpp_node(cell_name, attrs)
        
        # Add ports separately
        for port in rtl_data.get('ports', []):
            port_name = port['name']
            port_direction = port['direction']  # input, output, inout
            
            node_type = NodeType.PORT
            attrs = NodeAttributes(
                node_type=node_type,
                cell_type=f"PORT_{port_direction.upper()}",
                area=0.0,  # Ports don't have area
                power=0.0,  # Ports don't consume power directly
                delay=0.0,
                capacitance=0.001,
                timing_criticality=0.0
            )
            
            self.graph.add_node(port_name, **attrs.__dict__)
            
            # Update C++ Accelerator
            self._update_cpp_node(port_name, attrs)
    
    def _add_edges_from_connections(self, rtl_data: Dict):
        """Add edges representing electrical connections between nodes"""
        for net in rtl_data.get('nets', []):
            net_name = net['name']
            # Handle nets with or without connections
            connections = net.get('connections', [])  # Default to empty list if no connections

            # Create a net node to represent the electrical connection
            self.graph.add_node(net_name,
                              node_type=NodeType.SIGNAL,
                              cell_type='NET',
                              area=0.0,
                              power=0.0,
                              delay=0.0,
                              capacitance=net.get('capacitance', 0.001))

            # Connect all instances/ports to the net (if connections exist)
            for instance_port in connections:
                if isinstance(instance_port, tuple) and len(instance_port) == 2:
                    instance, port = instance_port
                    # Add connection edge
                    edge_attrs = EdgeAttributes(
                        edge_type=EdgeType.CONNECTION,
                        capacitance=0.0001,
                        delay=0.01,
                        length=0.0  # Will be determined by placement
                    )

                    # Connect instance port to net
                    self.graph.add_edge(instance, net_name, **edge_attrs.__dict__)
                    # Connect net to instance port (reverse direction for bidirectional)
                    self.graph.add_edge(net_name, instance, **edge_attrs.__dict__)
                    
                    # Update C++ Accelerator
                    self._update_cpp_edge(instance, net_name, edge_attrs)
                    self._update_cpp_edge(net_name, instance, edge_attrs)
    
    def _apply_constraints(self, constraints: Dict):
        """Apply design constraints to the graph"""
        # Apply clock constraints
        for clock_def in constraints.get('clocks', []):
            clock_name = clock_def.get('name')
            period = clock_def.get('period')
            uncertainty = clock_def.get('uncertainty', 0.1)
            
            # Find clock-related nodes and update their attributes
            for node, attrs in self.graph.nodes(data=True):
                if attrs.get('cell_type', '').startswith('CLK') or clock_name in node:
                    self.graph.nodes[node]['timing_criticality'] = max(
                        self.graph.nodes[node]['timing_criticality'],
                        0.8  # High criticality for clock nodes
                    )
        
        # Apply timing path constraints
        for path_constraint in constraints.get('timing_paths', []):
            from_node = path_constraint.get('from')
            to_node = path_constraint.get('to')
            constraint_value = path_constraint.get('constraint')
            
            # Update criticality for nodes on constrained paths
            if from_node and to_node:
                try:
                    # Find path between nodes and increase criticality
                    if self.graph.has_node(from_node) and self.graph.has_node(to_node):
                        # Mark nodes on this path as timing critical
                        path_nodes = self._find_path_nodes(from_node, to_node)
                        for node in path_nodes:
                            if node in self.graph:
                                current_crit = self.graph.nodes[node].get('timing_criticality', 0.0)
                                self.graph.nodes[node]['timing_criticality'] = min(current_crit + 0.3, 1.0)
                except:
                    # If path not found, just mark the specific nodes
                    for node in [from_node, to_node]:
                        if node in self.graph:
                            current_crit = self.graph.nodes[node].get('timing_criticality', 0.0)
                            self.graph.nodes[node]['timing_criticality'] = min(current_crit + 0.2, 1.0)
    
    def _apply_floorplan_hints(self, floorplan_hints: Dict):
        """Apply floorplan hints to the graph"""
        self.logger.debug("Applying floorplan hints.")
        for region_name, region_info in floorplan_hints.get('regions', {}).items():
            region_nodes = region_info.get('nodes', [])
            region_position = region_info.get('position')
            region_size = region_info.get('size')
            
            for node_name in region_nodes:
                if self.graph.has_node(node_name):
                    self.graph.nodes[node_name]['region'] = region_name
                    if region_position:
                        self.graph.nodes[node_name]['position'] = region_position
                        self.logger.debug(f"  Node '{node_name}' assigned to region '{region_name}' with position {region_position}.")

    def _apply_power_info(self, power_info: Dict):
        """Apply power information (from UPF) to graph nodes"""
        self.logger.debug("Applying power information from UPF.")
        # Apply power domain information
        for pd in power_info.get('power_domains', []):
            domain_name = pd['name']
            supply = pd.get('supply', 'VDD')
            ground = pd.get('ground', 'VSS')
            
            # Heuristic: apply to nodes in that domain (if node has 'power_domain' attribute)
            for node, attrs in self.graph.nodes(data=True):
                if attrs.get('power_domain') == domain_name:
                    self.graph.nodes[node]['voltage_domain'] = domain_name
                    self.graph.nodes[node]['supply_net'] = supply
                    self.graph.nodes[node]['ground_net'] = ground
                    self.logger.debug(f"  Node '{node}' assigned to power domain '{domain_name}'.")

        # Apply voltage domain information
        for vd in power_info.get('voltage_domains', []):
            domain_name = vd['name']
            voltage = vd.get('voltage', 1.0)
            
            # Heuristic: apply to nodes in that voltage domain
            for node, attrs in self.graph.nodes(data=True):
                if attrs.get('voltage_domain') == domain_name:
                    self.graph.nodes[node]['voltage'] = voltage
                    self.logger.debug(f"  Node '{node}' assigned voltage '{voltage}' in domain '{domain_name}'.")

        # Apply power switch information
        for ps in power_info.get('power_switches', []):
            switch_name = ps['name']
            if self.graph.has_node(switch_name):
                self.graph.nodes[switch_name]['is_power_switch'] = True
                self.logger.debug(f"  Node '{switch_name}' marked as power switch.")

        # Apply isolation cell information
        for iso_cell in power_info.get('isolation_cells', []):
            # Heuristic: find cells in domain and mark them for isolation
            domain = iso_cell['domain']
            for node, attrs in self.graph.nodes(data=True):
                if attrs.get('power_domain') == domain:
                    self.graph.nodes[node]['needs_isolation'] = True
                    self.logger.debug(f"  Node '{node}' in domain '{domain}' marked for isolation.")


    
    def _initialize_physical_properties(self):
        """Initialize physical properties for all nodes"""
        for node, attrs in self.graph.nodes(data=True):
            # Initialize congestion estimates
            self.graph.nodes[node]['estimated_congestion'] = 0.0
            
            # Initialize power domain if not set
            if not attrs.get('voltage_domain'):
                self.graph.nodes[node]['voltage_domain'] = 'DEFAULT_VDD'
            
            # Initialize clock domain if not set
            if not attrs.get('clock_domain'):
                self.graph.nodes[node]['clock_domain'] = 'DEFAULT_CLK'
    
    def _get_cell_properties(self, cell_type: str) -> Dict[str, float]:
        """Get cell properties from library (simulated lookup with variations)"""
        # This would normally come from a technology library (e.g., Liberty file)
        # For now, return default values based on cell type with some variations.
        self.logger.debug(f"Retrieving properties for cell type: {cell_type}")
        
        # Base defaults for common cell categories
        base_defaults = {
            'DFF':    {'area': 2.0,  'power': 0.02,  'delay': 0.15, 'capacitance': 0.002},
            'INV':    {'area': 1.0,  'power': 0.005, 'delay': 0.05, 'capacitance': 0.001},
            'BUF':    {'area': 1.5,  'power': 0.01,  'delay': 0.08, 'capacitance': 0.0015},
            'AND':    {'area': 2.0,  'power': 0.015, 'delay': 0.1,  'capacitance': 0.002},
            'OR':     {'area': 2.0,  'power': 0.015, 'delay': 0.1,  'capacitance': 0.002},
            'XOR':    {'area': 3.0,  'power': 0.025, 'delay': 0.15, 'capacitance': 0.003},
            'MUX':    {'area': 2.5,  'power': 0.02,  'delay': 0.12, 'capacitance': 0.0025},
            'NAND':   {'area': 1.2,  'power': 0.007, 'delay': 0.06, 'capacitance': 0.0012},
            'NOR':    {'area': 1.2,  'power': 0.007, 'delay': 0.06, 'capacitance': 0.0012},
            'RAM':    {'area': 50.0, 'power': 0.5,   'delay': 0.5,  'capacitance': 0.05}, # Memory macro
            'ROM':    {'area': 40.0, 'power': 0.4,   'delay': 0.4,  'capacitance': 0.04}, # Memory macro
            'IO':     {'area': 10.0, 'power': 0.1,   'delay': 0.2,  'capacitance': 0.01}, # IO cell
            'UNKNOWN':{'area': 1.0,  'power': 0.01,  'delay': 0.1,  'capacitance': 0.001},
        }
        
        # Determine base cell type for lookup
        base_cell_type = 'UNKNOWN'
        for key in base_defaults.keys():
            if cell_type.upper().startswith(key):
                base_cell_type = key
                break
        
        props = base_defaults[base_cell_type].copy()
        
        # Introduce variations based on drive strength (e.g., _X1, _X2, _X4, _X8)
        drive_strength_match = re.search(r'_X(\d+)$', cell_type, re.IGNORECASE)
        if drive_strength_match:
            drive_strength = int(drive_strength_match.group(1))
            if drive_strength > 1:
                props['area'] *= (1.0 + (drive_strength - 1) * 0.2) # Larger area for stronger drive
                props['power'] *= (1.0 + (drive_strength - 1) * 0.3) # More power
                props['delay'] /= (1.0 + (drive_strength - 1) * 0.1) # Faster delay
                props['capacitance'] *= (1.0 + (drive_strength - 1) * 0.1) # More capacitance
                self.logger.debug(f"  Adjusted properties for drive strength _X{drive_strength}.")

        self.logger.debug(f"  Returning properties for '{cell_type}': {props}")
        return props
    
    def _find_path_nodes(self, start_node: str, end_node: str, max_path_length: Optional[int] = None) -> List[str]:
        """Find nodes on a path between start and end (simplified)"""
        self.logger.debug(f"Searching for path from {start_node} to {end_node}.")
        if not self.graph.has_node(start_node) or not self.graph.has_node(end_node):
            self.logger.warning(f"Start node '{start_node}' or end node '{end_node}' not found in graph.")
            return []

        try:
            # Simple shortest path for demonstration
            if max_path_length:
                # Use a generator to find paths and stop after max_path_length
                all_paths = nx.all_shortest_paths(self.graph, start_node, end_node)
                path = next(all_paths) # Get the first shortest path
                if len(path) > max_path_length:
                    self.logger.warning(f"Path from {start_node} to {end_node} exceeds max_path_length {max_path_length}.")
                    return []
            else:
                path = nx.shortest_path(self.graph, start_node, end_node)
            
            self.logger.debug(f"Path found from {start_node} to {end_node}: {path}")
            return path
        except nx.NetworkXNoPath:
            self.logger.warning(f"No path found between {start_node} and {end_node}.")
            return []
        except Exception as e:
            self.logger.error(f"Error finding path between {start_node} and {end_node}: {e}")
            return []    
    def add_physical_proximity_edges(self, distance_threshold: float = 10.0):
        """
        Add physical proximity edges based on estimated positions
        
        Args:
            distance_threshold: Distance threshold for adding proximity edges
        """
        # This would normally use actual placement information
        # For now, we'll create a simplified grid-based proximity
        for node1 in self.graph.nodes():
            for node2 in self.graph.nodes():
                if node1 != node2:
                    # Calculate "distance" based on some heuristic
                    # In a real implementation, this would use actual coordinates
                    distance = self._estimate_distance(node1, node2)
                    
                    if distance <= distance_threshold:
                        edge_attrs = EdgeAttributes(
                            edge_type=EdgeType.PHYSICAL_PROXIMITY,
                            length=distance,
                            congestion=0.0
                        )
                        
                        self.graph.add_edge(node1, node2, **edge_attrs.__dict__)
    
    def _estimate_distance(self, node1: str, node2: str) -> float:
        """Estimate distance between two nodes"""
        # Prioritize using actual placement coordinates if available
        pos1 = self.graph.nodes[node1].get('position')
        pos2 = self.graph.nodes[node2].get('position')
        
        if pos1 and pos2:
            # Calculate Manhattan distance using available coordinates
            distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
            self.logger.debug(f"Estimating distance between {node1} and {node2} using positions: {distance:.2f}")
            return distance
        else:
            # Fallback to hash-based approach if placement coordinates are not available
            self.logger.debug(f"Estimating distance between {node1} and {node2} using hash-based fallback.")
            import hashlib
            
            # Create a pseudo-coordinate based on node names
            hash_pos1 = int(hashlib.md5(node1.encode()).hexdigest()[:8], 16) % 1000
            hash_pos2 = int(hashlib.md5(node2.encode()).hexdigest()[:8], 16) % 1000
            
            # Return Manhattan distance
            return abs(hash_pos1 - hash_pos2) / 100.0  # Normalize to a smaller range
    
    def get_timing_critical_nodes(self, threshold: float = 0.5) -> List[str]:
        """Get nodes with timing criticality above threshold"""
        if self.cpp_engine:
            return self.cpp_engine.get_timing_critical_nodes(threshold)
            
        critical_nodes = []
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get('timing_criticality', 0.0) >= threshold:
                critical_nodes.append(node)
        return critical_nodes
    
    def get_high_power_nodes(self, threshold: float = 0.05) -> List[str]:
        """Get nodes with power consumption above threshold"""
        high_power_nodes = []
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get('power', 0.0) >= threshold:
                high_power_nodes.append(node)
        return high_power_nodes
    
    def get_macros(self) -> List[str]:
        """Get all macro nodes"""
        macros = []
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get('is_macro', False):
                macros.append(node)
        return macros
    
    def get_clock_roots(self) -> List[str]:
        """Get all clock root nodes"""
        clock_roots = []
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get('is_clock_root', False):
                clock_roots.append(node)
        return clock_roots
    
    def update_node_attributes(self, node: str, **kwargs):
        """Update attributes for a specific node"""
        if self.graph.has_node(node):
            for key, value in kwargs.items():
                self.graph.nodes[node][key] = value
                
            # Sync to C++
            if self.cpp_engine:
                # Get the full updated attributes from Python side
                py_attrs = self.graph.nodes[node]
                self._update_cpp_node_from_dict(node, py_attrs)
    
    def update_edge_attributes(self, node1: str, node2: str, edge_type: EdgeType, **kwargs):
        """Update attributes for a specific edge"""
        # Find the edge with the specified type
        for key, edge_attrs in self.graph.edges(node1, data=True, keys=True):
            if edge_attrs.get('edge_type') == edge_type:
                for attr_key, attr_value in kwargs.items():
                    self.graph.edges[node1, node2, key][attr_key] = attr_value
                break
    
    def get_subgraph_by_region(self, region: str) -> 'CanonicalSiliconGraph':
        """Get a subgraph containing only nodes from a specific region"""
        region_nodes = [n for n, attrs in self.graph.nodes(data=True) 
                       if attrs.get('region') == region]
        
        subgraph = self.graph.subgraph(region_nodes).copy()
        
        # Create a new CanonicalSiliconGraph with the subgraph
        result = CanonicalSiliconGraph()
        result.graph = subgraph
        result.metadata = self.metadata.copy()
        
        return result
    
    def get_density_map(self, grid_size: int = 10) -> np.ndarray:
        """Get a density map of the design"""
        # This would normally use actual placement information
        # For now, return a uniform density
        return np.ones((grid_size, grid_size)) * 0.5
    
    def serialize_to_dict(self) -> Dict:
        """Serialize the graph to a dictionary format"""
        # Convert NetworkX graph to a serializable format
        nodes_data = {}
        for node, attrs in self.graph.nodes(data=True):
            nodes_data[node] = attrs
        
        edges_data = []
        for src, dst, attrs in self.graph.edges(data=True):
            edges_data.append({
                'source': src,
                'target': dst,
                'attributes': attrs
            })
        
        return {
            'nodes': nodes_data,
            'edges': edges_data,
            'metadata': self.metadata
        }
    
    def deserialize_from_dict(self, data: Dict):
        """Deserialize the graph from a dictionary format"""
        # Clear current graph
        self.graph.clear()
        
        # Add nodes
        for node_name, node_attrs in data['nodes'].items():
            self.graph.add_node(node_name, **node_attrs)
        
        # Add edges
        for edge_data in data['edges']:
            src = edge_data['source']
            dst = edge_data['target']
            attrs = edge_data['attributes']
            self.graph.add_edge(src, dst, **attrs)
        
        # Restore metadata
        self.metadata = data['metadata']

    # --- C++ Accelerator Helpers ---

    def _update_cpp_node(self, node_name: str, attrs: NodeAttributes):
        """Sync a NodeAttributes object to the C++ engine"""
        if not self.cpp_engine:
            return
            
        cpp_attrs = sic.NodeAttributes()
        cpp_attrs.node_type = self._node_type_map.get(attrs.node_type, sic.NodeType.CELL)
        cpp_attrs.cell_type = attrs.cell_type or ""
        cpp_attrs.area = float(attrs.area or 0.0)
        cpp_attrs.power = float(attrs.power or 0.01)
        cpp_attrs.delay = float(attrs.delay or 0.1)
        cpp_attrs.capacitance = float(attrs.capacitance or 0.001)
        cpp_attrs.is_clock_root = bool(attrs.is_clock_root)
        cpp_attrs.is_macro = bool(attrs.is_macro)
        cpp_attrs.estimated_congestion = float(attrs.estimated_congestion or 0.0)
        cpp_attrs.timing_criticality = float(attrs.timing_criticality or 0.0)
        
        if attrs.position:
            cpp_attrs.position = (float(attrs.position[0]), float(attrs.position[1]))
        
        cpp_attrs.region = attrs.region or ""
        cpp_attrs.voltage_domain = attrs.voltage_domain or ""
        cpp_attrs.clock_domain = attrs.clock_domain or ""
        
        self.cpp_engine.add_node(node_name, cpp_attrs)

    def _update_cpp_node_from_dict(self, node_name: str, attrs: Dict):
        """Sync a raw dictionary of attributes to the C++ engine"""
        if not self.cpp_engine:
            return
            
        cpp_attrs = sic.NodeAttributes()
        node_type = attrs.get('node_type')
        if isinstance(node_type, str):
            try:
                node_type = NodeType(node_type)
            except ValueError:
                node_type = NodeType.CELL
        
        cpp_attrs.node_type = self._node_type_map.get(node_type, sic.NodeType.CELL)
        cpp_attrs.cell_type = attrs.get('cell_type', "")
        cpp_attrs.area = float(attrs.get('area', 0.0))
        cpp_attrs.power = float(attrs.get('power', 0.01))
        cpp_attrs.delay = float(attrs.get('delay', 0.1))
        cpp_attrs.capacitance = float(attrs.get('capacitance', 0.001))
        cpp_attrs.is_clock_root = bool(attrs.get('is_clock_root', False))
        cpp_attrs.is_macro = bool(attrs.get('is_macro', False))
        cpp_attrs.estimated_congestion = float(attrs.get('estimated_congestion', 0.0))
        cpp_attrs.timing_criticality = float(attrs.get('timing_criticality', 0.0))
        
        pos = attrs.get('position')
        if pos:
            cpp_attrs.position = (float(pos[0]), float(pos[1]))
            
        cpp_attrs.region = attrs.get('region', "")
        cpp_attrs.voltage_domain = attrs.get('voltage_domain', "")
        cpp_attrs.clock_domain = attrs.get('clock_domain', "")
        
        self.cpp_engine.add_node(node_name, cpp_attrs)

    def _update_cpp_edge(self, src: str, dst: str, attrs: EdgeAttributes):
        """Sync an EdgeAttributes object to the C++ engine"""
        if not self.cpp_engine:
            return
            
        cpp_attrs = sic.EdgeAttributes()
        cpp_attrs.edge_type = self._edge_type_map.get(attrs.edge_type, sic.EdgeType.CONNECTION)
        cpp_attrs.resistance = float(attrs.resistance or 0.0)
        cpp_attrs.capacitance = float(attrs.capacitance or 0.001)
        cpp_attrs.delay = float(attrs.delay or 0.01)
        cpp_attrs.length = float(attrs.length or 0.0)
        cpp_attrs.layers_used = attrs.layers_used or []
        cpp_attrs.congestion = float(attrs.congestion or 0.0)
        cpp_attrs.capacity = float(attrs.capacity or 1.0)
        
        self.cpp_engine.add_edge(src, dst, cpp_attrs)