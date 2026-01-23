# Silicon Intelligence System - Implementation Details

## Tier 1: Critical Foundation

### 1.1 RTL Parser Implementation

#### Current State
- `silicon-intelligence/data/rtl_parser.py` exists but is mostly placeholder
- No actual Verilog/VHDL parsing
- No constraint parsing (SDC/UPF)

#### Implementation Plan

**Step 1: Choose Parsing Libraries**
```python
# Recommended libraries:
# - pyverilog: Verilog parsing (https://github.com/PyHDL/pyverilog)
# - pyhdl: VHDL parsing (https://github.com/PyHDL/pyhdl)
# - pyyaml: For constraint files
# - lark: For custom grammar parsing if needed

# Add to requirements.txt:
pyverilog>=1.3.0
pyhdl>=0.11.0
pyyaml>=6.0
```

**Step 2: Implement Core Parser Methods**

```python
# In rtl_parser.py

class RTLParser:
    def parse_verilog(self, verilog_file: str) -> Dict:
        """
        Parse Verilog file and extract design information
        
        Returns:
        {
            'instances': [
                {'name': 'u_cpu', 'type': 'CPU', 'parameters': {...}},
                ...
            ],
            'nets': [
                {'name': 'clk', 'connections': [('u_cpu', 'clk'), ...]},
                ...
            ],
            'ports': [
                {'name': 'clk', 'direction': 'input', 'width': 1},
                ...
            ],
            'hierarchy': {...},
            'parameters': {...}
        }
        """
        # Use pyverilog to parse
        # Extract instances, nets, ports
        # Build hierarchy
        pass
    
    def parse_sdc(self, sdc_file: str) -> Dict:
        """
        Parse SDC (Synopsys Design Constraints) file
        
        Returns:
        {
            'clocks': [
                {'name': 'clk', 'period': 10.0, 'uncertainty': 0.5},
                ...
            ],
            'timing_paths': [
                {'from': 'input_port', 'to': 'output_port', 'constraint': 8.0},
                ...
            ],
            'input_delays': [...],
            'output_delays': [...],
            'false_paths': [...]
        }
        """
        # Parse SDC commands
        # Extract clock definitions
        # Extract timing constraints
        pass
    
    def parse_upf(self, upf_file: str) -> Dict:
        """
        Parse UPF (Unified Power Format) file
        
        Returns:
        {
            'power_domains': [
                {'name': 'PD_CPU', 'supply': 'VDD', 'ground': 'VSS'},
                ...
            ],
            'voltage_domains': [...],
            'power_switches': [...],
            'isolation_cells': [...]
        }
        """
        # Parse UPF commands
        # Extract power domains
        # Extract voltage domains
        pass
    
    def build_rtl_data(self, verilog_file: str, sdc_file: str = None, 
                      upf_file: str = None) -> Dict:
        """
        Build complete RTL data structure
        """
        rtl_data = self.parse_verilog(verilog_file)
        
        if sdc_file:
            rtl_data['constraints'] = self.parse_sdc(sdc_file)
        
        if upf_file:
            rtl_data['power_info'] = self.parse_upf(upf_file)
        
        return rtl_data
```

**Step 3: Create Test Suite**

```python
# In tests/test_rtl_parser.py

def test_parse_simple_verilog():
    """Test parsing a simple Verilog file"""
    parser = RTLParser()
    rtl_data = parser.parse_verilog('tests/fixtures/simple.v')
    
    assert 'instances' in rtl_data
    assert 'nets' in rtl_data
    assert 'ports' in rtl_data
    assert len(rtl_data['instances']) > 0

def test_parse_sdc():
    """Test parsing SDC constraints"""
    parser = RTLParser()
    constraints = parser.parse_sdc('tests/fixtures/constraints.sdc')
    
    assert 'clocks' in constraints
    assert len(constraints['clocks']) > 0
    assert 'period' in constraints['clocks'][0]

def test_parse_upf():
    """Test parsing UPF power constraints"""
    parser = RTLParser()
    power_info = parser.parse_upf('tests/fixtures/power.upf')
    
    assert 'power_domains' in power_info
    assert len(power_info['power_domains']) > 0
```

**Step 4: Create Test Fixtures**

Create sample RTL files in `tests/fixtures/`:
- `simple.v` - Simple counter or adder
- `medium.v` - More complex design with hierarchy
- `constraints.sdc` - Sample timing constraints
- `power.upf` - Sample power constraints

#### Success Criteria
- [ ] Can parse Verilog files with 100+ instances
- [ ] Can parse SDC files with 10+ constraints
- [ ] Can parse UPF files with multiple power domains
- [ ] All test cases pass
- [ ] Handles error cases gracefully

---

### 1.2 CanonicalSiliconGraph Robustness

#### Current State
- Graph structure exists
- Basic node/edge operations work
- Missing: deepcopy, serialization, consistency checking

#### Implementation Plan

**Step 1: Implement Deepcopy**

```python
# In canonical_silicon_graph.py

import copy

class CanonicalSiliconGraph:
    def __deepcopy__(self, memo):
        """Create a deep copy of the graph"""
        # Create new instance
        new_graph = CanonicalSiliconGraph()
        
        # Deep copy the NetworkX graph
        new_graph.graph = copy.deepcopy(self.graph, memo)
        
        # Deep copy metadata
        new_graph.metadata = copy.deepcopy(self.metadata, memo)
        
        return new_graph
    
    def copy(self):
        """Create a shallow copy"""
        return copy.copy(self)
```

**Step 2: Add Consistency Validation**

```python
def validate_graph_consistency(self) -> Tuple[bool, List[str]]:
    """
    Validate graph consistency
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Check 1: All nodes have required attributes
    for node, attrs in self.graph.nodes(data=True):
        if 'node_type' not in attrs:
            errors.append(f"Node {node} missing node_type")
        if 'area' not in attrs:
            errors.append(f"Node {node} missing area")
    
    # Check 2: All edges have required attributes
    for src, dst, attrs in self.graph.edges(data=True):
        if 'edge_type' not in attrs:
            errors.append(f"Edge {src}->{dst} missing edge_type")
    
    # Check 3: No orphaned nodes (except ports)
    for node, attrs in self.graph.nodes(data=True):
        if attrs.get('node_type') != 'port':
            if self.graph.degree(node) == 0:
                errors.append(f"Node {node} is orphaned")
    
    # Check 4: Timing criticality in valid range
    for node, attrs in self.graph.nodes(data=True):
        crit = attrs.get('timing_criticality', 0.0)
        if not (0.0 <= crit <= 1.0):
            errors.append(f"Node {node} has invalid criticality: {crit}")
    
    return len(errors) == 0, errors
```

**Step 3: Implement Serialization**

```python
import json

def serialize_to_json(self, filepath: str):
    """Serialize graph to JSON file"""
    data = {
        'nodes': {},
        'edges': [],
        'metadata': self.metadata
    }
    
    # Serialize nodes
    for node, attrs in self.graph.nodes(data=True):
        # Convert enums to strings
        node_attrs = {}
        for key, value in attrs.items():
            if isinstance(value, Enum):
                node_attrs[key] = value.value
            else:
                node_attrs[key] = value
        data['nodes'][node] = node_attrs
    
    # Serialize edges
    for src, dst, key, attrs in self.graph.edges(keys=True, data=True):
        edge_attrs = {}
        for key, value in attrs.items():
            if isinstance(value, Enum):
                edge_attrs[key] = value.value
            else:
                edge_attrs[key] = value
        data['edges'].append({
            'source': src,
            'target': dst,
            'attributes': edge_attrs
        })
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def deserialize_from_json(self, filepath: str):
    """Deserialize graph from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Clear current graph
    self.graph.clear()
    
    # Deserialize nodes
    for node_name, node_attrs in data['nodes'].items():
        # Convert string enums back to Enum objects
        if 'node_type' in node_attrs:
            node_attrs['node_type'] = NodeType(node_attrs['node_type'])
        self.graph.add_node(node_name, **node_attrs)
    
    # Deserialize edges
    for edge_data in data['edges']:
        src = edge_data['source']
        dst = edge_data['target']
        attrs = edge_data['attributes']
        if 'edge_type' in attrs:
            attrs['edge_type'] = EdgeType(attrs['edge_type'])
        self.graph.add_edge(src, dst, **attrs)
    
    # Restore metadata
    self.metadata = data['metadata']
```

**Step 4: Add Transaction Support**

```python
from contextlib import contextmanager

class CanonicalSiliconGraph:
    def __init__(self):
        # ... existing code ...
        self._transaction_stack = []
    
    @contextmanager
    def transaction(self):
        """Context manager for atomic graph updates"""
        # Save current state
        import copy
        saved_graph = copy.deepcopy(self.graph)
        saved_metadata = copy.deepcopy(self.metadata)
        
        try:
            yield self
        except Exception as e:
            # Rollback on error
            self.graph = saved_graph
            self.metadata = saved_metadata
            raise e
```

#### Success Criteria
- [ ] Deepcopy creates independent copies
- [ ] Consistency validation catches all errors
- [ ] Serialization/deserialization round-trips correctly
- [ ] Transaction support works
- [ ] Performance acceptable for 100k+ node graphs

---

### 1.3 Basic Agent Proposal Generation

#### Current State
- Base agent class exists
- `propose_action()` is abstract
- No actual proposals generated

#### Implementation Plan

**Step 1: Implement FloorplanAgent Proposals**

```python
# In agents/floorplan_agent.py

class FloorplanAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentType.FLOORPLAN)
        self.strategies = [
            'hierarchical_clustering',
            'linear_arrangement',
            'grid_based',
            'thermal_aware',
            'power_aware'
        ]
    
    def propose_action(self, graph: CanonicalSiliconGraph) -> Optional[AgentProposal]:
        """Generate a floorplan proposal"""
        
        # Get macros from graph
        macros = graph.get_macros()
        if not macros:
            return None
        
        # Select strategy (for now, round-robin)
        strategy = self.strategies[len(self.proposal_history) % len(self.strategies)]
        
        # Generate proposal based on strategy
        if strategy == 'hierarchical_clustering':
            return self._propose_hierarchical_clustering(graph, macros)
        elif strategy == 'thermal_aware':
            return self._propose_thermal_aware(graph, macros)
        # ... other strategies
    
    def _propose_hierarchical_clustering(self, graph: CanonicalSiliconGraph, 
                                        macros: List[str]) -> AgentProposal:
        """Propose hierarchical clustering floorplan"""
        
        # Group macros by connectivity
        clusters = self._cluster_macros(graph, macros)
        
        # Generate positions for clusters
        positions = self._generate_cluster_positions(clusters)
        
        # Create proposal
        proposal = AgentProposal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            proposal_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            action_type='place_macros',
            targets=macros,
            parameters={
                'positions': positions,
                'strategy': 'hierarchical_clustering',
                'clusters': clusters
            },
            confidence_score=0.75,
            risk_profile={
                'timing_risk': 0.2,
                'power_risk': 0.1,
                'area_risk': 0.15
            },
            cost_vector={
                'power': 0.05,
                'performance': -0.1,  # Negative = improvement
                'area': 0.08,
                'yield': 0.0,
                'schedule': 0.0
            },
            predicted_outcome={
                'total_area': sum(graph.graph.nodes[m].get('area', 0) for m in macros),
                'estimated_congestion': 0.6,
                'timing_slack': 0.5
            },
            dependencies=[],
            conflicts_with=[]
        )
        
        return proposal
    
    def _cluster_macros(self, graph: CanonicalSiliconGraph, 
                       macros: List[str]) -> Dict[str, List[str]]:
        """Cluster macros by connectivity"""
        # Simple clustering: group by connectivity
        clusters = {}
        for i, macro in enumerate(macros):
            cluster_id = f"cluster_{i // 3}"  # 3 macros per cluster
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(macro)
        return clusters
    
    def _generate_cluster_positions(self, clusters: Dict) -> Dict[str, Tuple[float, float]]:
        """Generate positions for clusters"""
        positions = {}
        for i, (cluster_id, macros) in enumerate(clusters.items()):
            # Simple grid layout
            x = (i % 4) * 1000
            y = (i // 4) * 1000
            positions[cluster_id] = (x, y)
        return positions
    
    def evaluate_proposal_impact(self, proposal: AgentProposal, 
                                graph: CanonicalSiliconGraph) -> Dict[str, float]:
        """Evaluate impact of proposal"""
        return {
            'area_impact': proposal.cost_vector.get('area', 0),
            'power_impact': proposal.cost_vector.get('power', 0),
            'timing_impact': proposal.cost_vector.get('performance', 0),
            'congestion_impact': 0.1
        }
```

**Step 2: Implement PlacementAgent Proposals**

```python
# In agents/placement_agent.py

class PlacementAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentType.PLACEMENT)
        self.strategies = [
            'congestion_aware',
            'timing_driven',
            'power_aware',
            'density_aware',
            'mixed_mode'
        ]
    
    def propose_action(self, graph: CanonicalSiliconGraph) -> Optional[AgentProposal]:
        """Generate a placement proposal"""
        
        # Get cells to place
        cells = [n for n, attrs in graph.graph.nodes(data=True) 
                if attrs.get('node_type') == 'cell']
        
        if not cells:
            return None
        
        # Select strategy
        strategy = self.strategies[len(self.proposal_history) % len(self.strategies)]
        
        # Generate proposal
        if strategy == 'congestion_aware':
            return self._propose_congestion_aware(graph, cells)
        elif strategy == 'timing_driven':
            return self._propose_timing_driven(graph, cells)
        # ... other strategies
    
    def _propose_congestion_aware(self, graph: CanonicalSiliconGraph, 
                                 cells: List[str]) -> AgentProposal:
        """Propose congestion-aware placement"""
        
        # Identify congestion hotspots
        hotspots = self._identify_congestion_hotspots(graph)
        
        # Place cells away from hotspots
        positions = self._generate_positions_avoiding_hotspots(cells, hotspots)
        
        proposal = AgentProposal(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            proposal_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            action_type='place_cells',
            targets=cells,
            parameters={
                'positions': positions,
                'strategy': 'congestion_aware',
                'hotspots': hotspots
            },
            confidence_score=0.8,
            risk_profile={
                'timing_risk': 0.15,
                'power_risk': 0.05,
                'area_risk': 0.0
            },
            cost_vector={
                'power': 0.02,
                'performance': -0.05,
                'area': 0.0,
                'yield': 0.05,
                'schedule': 0.0
            },
            predicted_outcome={
                'estimated_congestion': 0.5,
                'timing_slack': 0.6,
                'wirelength': 1000
            },
            dependencies=[],
            conflicts_with=[]
        )
        
        return proposal
    
    def _identify_congestion_hotspots(self, graph: CanonicalSiliconGraph) -> List[str]:
        """Identify congestion hotspots"""
        # Find regions with high estimated congestion
        hotspots = []
        for node, attrs in graph.graph.nodes(data=True):
            if attrs.get('estimated_congestion', 0) > 0.7:
                hotspots.append(node)
        return hotspots
    
    def _generate_positions_avoiding_hotspots(self, cells: List[str], 
                                             hotspots: List[str]) -> Dict[str, Tuple[float, float]]:
        """Generate positions avoiding hotspots"""
        positions = {}
        for i, cell in enumerate(cells):
            # Simple grid layout
            x = (i % 10) * 100
            y = (i // 10) * 100
            positions[cell] = (x, y)
        return positions
    
    def evaluate_proposal_impact(self, proposal: AgentProposal, 
                                graph: CanonicalSiliconGraph) -> Dict[str, float]:
        """Evaluate impact of proposal"""
        return {
            'congestion_impact': -0.1,  # Improvement
            'timing_impact': 0.05,
            'power_impact': 0.02,
            'area_impact': 0.0
        }
```

**Step 3: Implement Other Agents Similarly**

- ClockAgent: CTS proposals
- PowerAgent: Power grid proposals
- RoutingAgent: Routing proposals
- ThermalAgent: Thermal management proposals
- YieldAgent: Yield optimization proposals

**Step 4: Create Test Suite**

```python
# In tests/test_agents.py

def test_floorplan_agent_proposal():
    """Test floorplan agent generates proposals"""
    agent = FloorplanAgent()
    graph = create_test_graph_with_macros()
    
    proposal = agent.propose_action(graph)
    
    assert proposal is not None
    assert proposal.action_type == 'place_macros'
    assert len(proposal.targets) > 0
    assert 0.0 <= proposal.confidence_score <= 1.0

def test_placement_agent_proposal():
    """Test placement agent generates proposals"""
    agent = PlacementAgent()
    graph = create_test_graph_with_cells()
    
    proposal = agent.propose_action(graph)
    
    assert proposal is not None
    assert proposal.action_type == 'place_cells'
    assert len(proposal.targets) > 0
```

#### Success Criteria
- [ ] Each agent generates realistic proposals
- [ ] Proposals have valid cost vectors
- [ ] Confidence scores are reasonable
- [ ] Proposals can be applied to graph
- [ ] All test cases pass

---

## Tier 2: Predictive Models

### 2.1 Congestion Predictor

#### Implementation Approach

**Start with heuristic-based prediction, then add ML**

```python
# In models/congestion_predictor.py

class CongestionPredictor:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.model = None  # Will be ML model later
    
    def predict_congestion(self, graph: CanonicalSiliconGraph, 
                          process_node: str = '7nm') -> Dict[str, Any]:
        """
        Predict routing congestion
        
        Returns:
        {
            'global_congestion': 0.65,
            'local_congestion': {
                'region_1': 0.7,
                'region_2': 0.5,
                ...
            },
            'layer_congestion': {
                'metal1': 0.6,
                'metal2': 0.7,
                ...
            },
            'hotspots': [
                {'region': 'region_1', 'severity': 0.9, 'cause': 'high_fanout'},
                ...
            ],
            'confidence': 0.85
        }
        """
        
        # Calculate heuristic-based congestion
        global_cong = self._calculate_global_congestion(graph)
        local_cong = self._calculate_local_congestion(graph)
        layer_cong = self._calculate_layer_congestion(graph)
        hotspots = self._identify_hotspots(graph, local_cong)
        
        return {
            'global_congestion': global_cong,
            'local_congestion': local_cong,
            'layer_congestion': layer_cong,
            'hotspots': hotspots,
            'confidence': 0.75  # Heuristic confidence
        }
    
    def _calculate_global_congestion(self, graph: CanonicalSiliconGraph) -> float:
        """Calculate global congestion estimate"""
        # Heuristic: based on cell density and fanout
        
        total_cells = len([n for n, attrs in graph.graph.nodes(data=True) 
                          if attrs.get('node_type') == 'cell'])
        
        total_nets = len([n for n, attrs in graph.graph.nodes(data=True) 
                         if attrs.get('node_type') == 'signal'])
        
        avg_fanout = sum(len(list(graph.graph.successors(n))) 
                        for n in graph.graph.nodes()) / max(total_nets, 1)
        
        # Normalize to 0-1 range
        congestion = min((total_cells / 1000.0) * (avg_fanout / 10.0), 1.0)
        
        return congestion
    
    def _calculate_local_congestion(self, graph: CanonicalSiliconGraph) -> Dict[str, float]:
        """Calculate congestion by region"""
        region_congestion = {}
        
        for node, attrs in graph.graph.nodes(data=True):
            region = attrs.get('region', 'default')
            
            if region not in region_congestion:
                region_congestion[region] = []
            
            # Congestion based on fanout
            fanout = len(list(graph.graph.successors(node)))
            region_congestion[region].append(fanout)
        
        # Average fanout per region
        result = {}
        for region, fanouts in region_congestion.items():
            avg_fanout = sum(fanouts) / len(fanouts) if fanouts else 0
            result[region] = min(avg_fanout / 20.0, 1.0)  # Normalize
        
        return result
    
    def _calculate_layer_congestion(self, graph: CanonicalSiliconGraph) -> Dict[str, float]:
        """Calculate congestion by metal layer"""
        # Simplified: assume even distribution across layers
        return {
            'metal1': 0.5,
            'metal2': 0.6,
            'metal3': 0.55,
            'metal4': 0.5,
            'metal5': 0.45
        }
    
    def _identify_hotspots(self, graph: CanonicalSiliconGraph, 
                          local_cong: Dict[str, float]) -> List[Dict]:
        """Identify congestion hotspots"""
        hotspots = []
        
        for region, congestion in local_cong.items():
            if congestion > 0.7:
                # Find cause
                cause = self._identify_congestion_cause(graph, region)
                hotspots.append({
                    'region': region,
                    'severity': congestion,
                    'cause': cause
                })
        
        return hotspots
    
    def _identify_congestion_cause(self, graph: CanonicalSiliconGraph, 
                                  region: str) -> str:
        """Identify cause of congestion in region"""
        # Find nodes in region
        region_nodes = [n for n, attrs in graph.graph.nodes(data=True) 
                       if attrs.get('region') == region]
        
        # Check for high fanout
        high_fanout_nodes = [n for n in region_nodes 
                            if len(list(graph.graph.successors(n))) > 20]
        
        if high_fanout_nodes:
            return 'high_fanout'
        
        # Check for density
        if len(region_nodes) > 100:
            return 'high_density'
        
        return 'unknown'
    
    def train(self, training_data: List[Dict]):
        """Train ML model on historical data"""
        # This would implement actual ML training
        # For now, just log
        self.logger.info(f"Training congestion predictor on {len(training_data)} samples")
```

#### Success Criteria
- [ ] Heuristic predictor works
- [ ] Identifies congestion hotspots
- [ ] Confidence scores reasonable
- [ ] Can be integrated with agents

---

## Implementation Checklist

### Week 1: RTL Parser
- [ ] Set up pyverilog/pyhdl
- [ ] Implement Verilog parser
- [ ] Implement SDC parser
- [ ] Implement UPF parser
- [ ] Create test fixtures
- [ ] Write comprehensive tests

### Week 2: Graph Robustness
- [ ] Implement deepcopy
- [ ] Add consistency validation
- [ ] Implement serialization
- [ ] Add transaction support
- [ ] Performance testing
- [ ] Write tests

### Week 3-4: Agent Proposals
- [ ] Implement FloorplanAgent proposals
- [ ] Implement PlacementAgent proposals
- [ ] Implement other agent proposals
- [ ] Create proposal evaluation
- [ ] Write comprehensive tests
- [ ] Integration testing

### Week 5-6: Predictive Models
- [ ] Implement heuristic congestion predictor
- [ ] Implement timing analyzer
- [ ] Enhance DRC predictor
- [ ] Create training data pipelines
- [ ] Write tests
- [ ] Validate accuracy

---

## Code Quality Standards

### Testing
- Minimum 80% code coverage
- Unit tests for all public methods
- Integration tests for agent interactions
- Performance tests for large graphs

### Documentation
- Docstrings for all classes and methods
- Type hints for all parameters
- Usage examples in docstrings
- README for each module

### Performance
- Graph operations < 100ms for 100k nodes
- Proposal generation < 1s per agent
- Negotiation round < 5s
- Serialization < 500ms

---

## Next Steps

1. **Start with RTL Parser** - This is the foundation
2. **Enhance CanonicalSiliconGraph** - Needed for robustness
3. **Implement Agent Proposals** - Enables testing
4. **Add Predictive Models** - Enables intelligence
5. **Integrate with EDA Tools** - Enables real flows

See `IMPLEMENTATION_ROADMAP.md` for overall strategy and timeline.
