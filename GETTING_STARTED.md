# Silicon Intelligence System - Getting Started Guide

## Quick Overview

The Silicon Intelligence System is a sophisticated AI-powered chip design automation platform. The codebase has a solid architectural foundation but needs implementation work to move from blueprint to production.

**Current Status**: ~30% complete
**Target**: Production-ready system with real ML models and EDA integration

---

## What's Already Done

✅ **Architecture & Design**
- Canonical Silicon Graph (unified data structure)
- Agent negotiation protocol
- Parallel Reality Engine framework
- Learning loop controller structure
- Base agent classes

✅ **Scaffolding**
- Project structure
- Module organization
- Logging infrastructure
- Basic utilities

---

## What Needs to Be Done

❌ **Critical (Blocking)**
1. RTL Parser - Parse real Verilog/VHDL designs
2. Graph Robustness - Deepcopy, serialization, validation
3. Agent Proposals - Actual proposal generation logic
4. Predictive Models - Congestion, timing, DRC prediction

❌ **Important (Enables Features)**
5. EDA Tool Integration - OpenROAD, Innovus, Fusion Compiler
6. Strategy Selection - Learning-based agent strategies
7. Parameter Generation - Data-driven parameter optimization
8. Conflict Resolution - Partial acceptance algorithms

❌ **Nice to Have (Polish)**
9. Advanced ML Models - Graph Neural Networks
10. Silicon Data Integration - Learning from real chips
11. Performance Optimization - Large design handling
12. Visualization - Design metrics dashboards

---

## Implementation Priority

### Phase 1: Foundation (Weeks 1-4)
**Goal**: Get end-to-end flow working with real RTL

1. **RTL Parser** (Week 1)
   - Parse Verilog/VHDL
   - Parse SDC constraints
   - Parse UPF power specs
   - Build CanonicalSiliconGraph

2. **Graph Robustness** (Week 2)
   - Implement deepcopy
   - Add consistency validation
   - Implement serialization
   - Add transaction support

3. **Agent Proposals** (Weeks 3-4)
   - FloorplanAgent proposals
   - PlacementAgent proposals
   - Other agent proposals
   - Proposal evaluation

### Phase 2: Intelligence (Weeks 5-8)
**Goal**: Add predictive models and agent intelligence

1. **Predictive Models** (Weeks 5-6)
   - Congestion predictor
   - Timing analyzer
   - DRC predictor
   - Training pipelines

2. **Agent Intelligence** (Weeks 7-8)
   - Strategy selection logic
   - Parameter generation
   - Risk assessment
   - Cost vectors

### Phase 3: Integration (Weeks 9-12)
**Goal**: Integrate with real EDA tools

1. **EDA Integration** (Weeks 9-10)
   - OpenROAD integration
   - EDA script generation
   - Output parsing
   - Feedback loops

2. **Learning Loop** (Weeks 11-12)
   - Silicon data integration
   - Model updates
   - Continuous improvement

---

## How to Start

### Step 1: Understand the Architecture

Read these files in order:
1. `README.md` - System overview
2. `PROJECT_STRUCTURE.md` - File organization
3. `silicon-intelligence/core/canonical_silicon_graph.py` - Core data structure
4. `silicon-intelligence/agents/base_agent.py` - Agent framework

### Step 2: Set Up Development Environment

```bash
# Clone repository
git clone <repo-url>
cd silicon-intelligence

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy

# Run existing tests
pytest tests/ -v
```

### Step 3: Start with RTL Parser

This is the highest-priority item. It's the foundation for everything else.

```bash
# 1. Add dependencies to requirements.txt
pyverilog>=1.3.0
pyhdl>=0.11.0
pyyaml>=6.0

# 2. Implement RTLParser in silicon-intelligence/data/rtl_parser.py
# See IMPLEMENTATION_DETAILS.md for detailed guidance

# 3. Create test fixtures in tests/fixtures/
# - simple.v (10 instances)
# - medium.v (100 instances)
# - constraints.sdc (timing constraints)
# - power.upf (power constraints)

# 4. Write tests in tests/test_rtl_parser.py

# 5. Run tests
pytest tests/test_rtl_parser.py -v
```

### Step 4: Enhance CanonicalSiliconGraph

Once RTL parser works, enhance the graph:

```bash
# 1. Implement __deepcopy__ in CanonicalSiliconGraph
# 2. Add validate_graph_consistency() method
# 3. Implement serialize_to_json() and deserialize_from_json()
# 4. Add transaction support with @contextmanager
# 5. Write comprehensive tests
# 6. Performance test with 100k+ nodes
```

### Step 5: Implement Agent Proposals

Once graph is robust, implement agent proposals:

```bash
# 1. Implement FloorplanAgent.propose_action()
# 2. Implement PlacementAgent.propose_action()
# 3. Implement other agent proposals
# 4. Write tests for each agent
# 5. Test negotiation round
```

---

## Key Files to Understand

### Core Architecture
- `silicon-intelligence/core/canonical_silicon_graph.py` - Unified data structure
- `silicon-intelligence/agents/base_agent.py` - Agent framework
- `silicon-intelligence/core/parallel_reality_engine.py` - Parallel exploration
- `silicon-intelligence/core/learning_loop.py` - Learning from silicon

### Agents
- `silicon-intelligence/agents/floorplan_agent.py` - Macro placement
- `silicon-intelligence/agents/placement_agent.py` - Cell placement
- `silicon-intelligence/agents/clock_agent.py` - Clock tree synthesis
- `silicon-intelligence/agents/power_agent.py` - Power optimization
- `silicon-intelligence/agents/routing_agent.py` - Routing
- `silicon-intelligence/agents/thermal_agent.py` - Thermal management
- `silicon-intelligence/agents/yield_agent.py` - Yield optimization

### Models
- `silicon-intelligence/models/congestion_predictor.py` - Congestion prediction
- `silicon-intelligence/models/timing_analyzer.py` - Timing analysis
- `silicon-intelligence/models/drc_predictor.py` - DRC violation prediction
- `silicon-intelligence/models/advanced_ml_models.py` - Advanced ML models

### Data Processing
- `silicon-intelligence/data/rtl_parser.py` - RTL parsing (NEEDS IMPLEMENTATION)
- `silicon-intelligence/core/openroad_interface.py` - OpenROAD integration

---

## Common Tasks

### Task: Add a New Agent

1. Create new file in `silicon-intelligence/agents/`
2. Inherit from `BaseAgent`
3. Implement `propose_action()` method
4. Implement `evaluate_proposal_impact()` method
5. Register with `AgentNegotiator`
6. Write tests

Example:
```python
from silicon_intelligence.agents.base_agent import BaseAgent, AgentType, AgentProposal

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentType.CUSTOM)
    
    def propose_action(self, graph):
        # Generate proposal
        return AgentProposal(...)
    
    def evaluate_proposal_impact(self, proposal, graph):
        # Evaluate impact
        return {...}
```

### Task: Add a New Predictor

1. Create new file in `silicon-intelligence/models/`
2. Implement prediction method
3. Add training method
4. Write tests

Example:
```python
class MyPredictor:
    def predict(self, graph):
        # Make prediction
        return {...}
    
    def train(self, training_data):
        # Train model
        pass
```

### Task: Add a New Test

1. Create test file in `tests/`
2. Use pytest framework
3. Follow naming convention: `test_*.py`
4. Write descriptive test names

Example:
```python
import pytest
from silicon_intelligence.core.canonical_silicon_graph import CanonicalSiliconGraph

def test_graph_creation():
    graph = CanonicalSiliconGraph()
    assert graph is not None
    assert len(graph.graph.nodes()) == 0

def test_add_node():
    graph = CanonicalSiliconGraph()
    graph.graph.add_node('test_node', node_type='cell')
    assert 'test_node' in graph.graph.nodes()
```

---

## Testing Strategy

### Unit Tests
- Test individual functions/methods
- Mock external dependencies
- Aim for >80% coverage

### Integration Tests
- Test component interactions
- Use real data where possible
- Test end-to-end flows

### Performance Tests
- Test with large designs (100k+ nodes)
- Measure execution time
- Identify bottlenecks

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_rtl_parser.py -v

# Run with coverage
pytest tests/ --cov=silicon_intelligence --cov-report=html

# Run specific test
pytest tests/test_rtl_parser.py::test_parse_simple_verilog -v
```

---

## Code Quality Standards

### Style
- Follow PEP 8
- Use black for formatting
- Use flake8 for linting
- Use mypy for type checking

```bash
# Format code
black silicon_intelligence/

# Check style
flake8 silicon_intelligence/

# Type check
mypy silicon_intelligence/
```

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

## Debugging Tips

### Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Inspect Graph Structure
```python
from silicon_intelligence.core.canonical_silicon_graph import CanonicalSiliconGraph

graph = CanonicalSiliconGraph()
# ... build graph ...

# Print nodes
for node, attrs in graph.graph.nodes(data=True):
    print(f"{node}: {attrs}")

# Print edges
for src, dst, attrs in graph.graph.edges(data=True):
    print(f"{src} -> {dst}: {attrs}")

# Validate consistency
is_valid, errors = graph.validate_graph_consistency()
if not is_valid:
    for error in errors:
        print(f"ERROR: {error}")
```

### Profile Performance
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# ... code to profile ...

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

---

## Common Issues and Solutions

### Issue: Import Errors
**Solution**: Make sure all dependencies are installed
```bash
pip install -r requirements.txt
```

### Issue: Graph Operations Slow
**Solution**: Profile and optimize hot paths
```bash
# Use cProfile to identify bottlenecks
python -m cProfile -s cumulative script.py
```

### Issue: Tests Failing
**Solution**: Check test fixtures and dependencies
```bash
# Run with verbose output
pytest tests/ -vv -s

# Run specific test
pytest tests/test_file.py::test_name -vv
```

### Issue: Memory Usage High
**Solution**: Use generators instead of lists where possible
```python
# Bad: Creates entire list in memory
nodes = [n for n in graph.graph.nodes()]

# Good: Generator, lazy evaluation
nodes = (n for n in graph.graph.nodes())
```

---

## Next Steps

1. **Read the documentation**
   - README.md
   - PROJECT_STRUCTURE.md
   - IMPLEMENTATION_ROADMAP.md

2. **Understand the architecture**
   - Study CanonicalSiliconGraph
   - Study BaseAgent
   - Study AgentNegotiator

3. **Start implementing**
   - Begin with RTL Parser (Week 1)
   - Follow SPRINT_TASKS.md for detailed tasks
   - Use IMPLEMENTATION_DETAILS.md for guidance

4. **Test thoroughly**
   - Write unit tests
   - Write integration tests
   - Performance test

5. **Iterate and improve**
   - Get feedback
   - Refactor as needed
   - Optimize performance

---

## Resources

### Documentation
- `README.md` - System overview
- `PROJECT_STRUCTURE.md` - File organization
- `IMPLEMENTATION_ROADMAP.md` - Implementation strategy
- `IMPLEMENTATION_DETAILS.md` - Detailed implementation guidance
- `SPRINT_TASKS.md` - Sprint-level tasks

### External Resources
- [NetworkX Documentation](https://networkx.org/)
- [PyVerilog Documentation](https://github.com/PyHDL/pyverilog)
- [Pytest Documentation](https://docs.pytest.org/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)

### Tools
- Git - Version control
- pytest - Testing framework
- black - Code formatter
- flake8 - Linter
- mypy - Type checker

---

## Getting Help

1. **Check existing code** - Look for similar implementations
2. **Read documentation** - Check docstrings and comments
3. **Run tests** - Tests show expected behavior
4. **Debug** - Use print statements and debugger
5. **Ask team** - Discuss with other developers

---

## Success Criteria

### Week 1 (RTL Parser)
- [ ] Can parse real Verilog files
- [ ] Can parse SDC constraints
- [ ] Can parse UPF power specs
- [ ] All tests pass
- [ ] >80% code coverage

### Week 2 (Graph Robustness)
- [ ] Deepcopy works correctly
- [ ] Consistency validation works
- [ ] Serialization works
- [ ] Performance acceptable
- [ ] All tests pass

### Week 3-4 (Agent Proposals)
- [ ] All agents generate proposals
- [ ] Proposals are realistic
- [ ] Negotiation works
- [ ] All tests pass

### Overall (Phase 1)
- [ ] End-to-end flow works
- [ ] Can parse RTL
- [ ] Can generate proposals
- [ ] Can run negotiation
- [ ] Can serialize/deserialize

---

## Questions?

Refer to:
1. `IMPLEMENTATION_ROADMAP.md` - For overall strategy
2. `IMPLEMENTATION_DETAILS.md` - For detailed guidance
3. `SPRINT_TASKS.md` - For specific tasks
4. Code comments and docstrings - For implementation details
5. Test files - For usage examples

Good luck! 🚀
