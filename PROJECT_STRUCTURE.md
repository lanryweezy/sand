silicon-intelligence/
├── agents/                 # Specialist AI agents (Placement, Clock, Routing, Power, Yield)
│   ├── __init__.py
│   ├── base_agent.py      # Base agent interface and negotiation protocol
│   ├── placement_agent.py # Placement optimization agent
│   ├── clock_agent.py     # Clock tree synthesis agent
│   ├── routing_agent.py   # Routing optimization agent
│   ├── power_agent.py     # Power optimization agent
│   └── yield_agent.py     # Yield and manufacturability agent
├── cognitive/             # GenAI brain components
│   ├── __init__.py
│   ├── physical_risk_oracle.py  # Predicts physical implementation risks
│   ├── design_intent_interpreter.py  # Interprets design intent
│   ├── silicon_knowledge_model.py    # Trained model for silicon understanding
│   └── reasoning_engine.py           # Chain-of-thought reasoning
├── core/                  # Core system components
│   ├── __init__.py
│   ├── canonical_silicon_graph.py    # Unified graph representation
│   ├── constraint_manager.py         # Manages design constraints
│   ├── pdk_interface.py              # Process design kit interface
│   └── gdsii_generator.py            # GDSII output generation
├── data/                  # Data processing and training utilities
│   ├── __init__.py
│   ├── dataset_builder.py            # Builds training datasets
│   ├── rtl_parser.py                 # Parses RTL for graph representation
│   ├── def_parser.py                 # Parses DEF files
│   └── gds_parser.py                 # Parses GDSII files
├── models/                # ML models and training
│   ├── __init__.py
│   ├── silicon_language_model.py     # Domain-specific language model
│   ├── congestion_predictor.py       # Predicts routing congestion
│   ├── timing_analyzer.py            # Timing analysis model
│   └── drc_predictor.py              # Design rule violation prediction
├── networks/              # Neural network architectures
│   ├── __init__.py
│   ├── graph_neural_network.py       # GNN for silicon graph processing
│   ├── transformer_architecture.py   # Transformer for design intent
│   └── reinforcement_learning.py     # RL for optimization
├── utils/                 # Utility functions
│   ├── __init__.py
│   ├── logger.py                     # Logging utilities
│   ├── config.py                     # Configuration management
│   ├── metrics.py                    # PPA metric calculations
│   └── visualization.py              # Visualization tools
├── experiments/           # Experiment tracking and management
│   ├── __init__.py
│   └── experiment_runner.py          # Runs experimental flows
├── tests/                 # Unit and integration tests
│   ├── __init__.py
│   ├── test_agents.py               # Tests for agents
│   ├── test_cognitive.py            # Tests for cognitive components
│   └── test_core.py                 # Tests for core components
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project metadata
└── main.py                # Main entry point