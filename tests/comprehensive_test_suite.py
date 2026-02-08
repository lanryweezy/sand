"""
Comprehensive Test Suite for Silicon Intelligence System

This module provides comprehensive testing of all system components and their integration.
"""

import unittest
import tempfile
import os
from pathlib import Path
import numpy as np
from silicon_intelligence.core.canonical_silicon_graph import CanonicalSiliconGraph
from silicon_intelligence.cognitive.advanced_cognitive_system import PhysicalRiskOracle
from silicon_intelligence.agents.floorplan_agent import FloorplanAgent
from silicon_intelligence.agents.placement_agent import PlacementAgent
from silicon_intelligence.agents.clock_agent import ClockAgent
from silicon_intelligence.agents.power_agent import PowerAgent
from silicon_intelligence.agents.yield_agent import YieldAgent
from silicon_intelligence.agents.routing_agent import RoutingAgent
from silicon_intelligence.agents.thermal_agent import ThermalAgent
from silicon_intelligence.agents.advanced_conflict_resolution import EnhancedAgentNegotiator
from silicon_intelligence.core.parallel_reality_engine import ParallelRealityEngine
from silicon_intelligence.models.advanced_ml_models import (
    AdvancedCongestionPredictor, AdvancedTimingAnalyzer, AdvancedDRCPredictor
)
from silicon_intelligence.core.comprehensive_learning_loop import LearningLoopController
from silicon_intelligence.core.flow_orchestrator import FlowOrchestrator
from silicon_intelligence.utils.logger import get_logger


class TestCanonicalSiliconGraph(unittest.TestCase):
    """Test the Canonical Silicon Graph implementation"""
    
    def setUp(self):
        self.graph = CanonicalSiliconGraph()
    
    def test_graph_initialization(self):
        """Test graph initialization"""
        self.assertIsNotNone(self.graph.graph)
        self.assertEqual(len(self.graph.graph.nodes()), 0)
        self.assertEqual(len(self.graph.graph.edges()), 0)
    
    def test_node_addition(self):
        """Test adding nodes to the graph"""
        self.graph.graph.add_node('cell1', 
                                node_type='cell',
                                power=0.1,
                                area=2.0,
                                timing_criticality=0.3)
        
        self.assertEqual(len(self.graph.graph.nodes()), 1)
        self.assertIn('cell1', self.graph.graph.nodes())
        
        attrs = self.graph.graph.nodes['cell1']
        self.assertEqual(attrs['node_type'], 'cell')
        self.assertEqual(attrs['power'], 0.1)
        self.assertEqual(attrs['area'], 2.0)
        self.assertEqual(attrs['timing_criticality'], 0.3)
    
    def test_edge_addition(self):
        """Test adding edges to the graph"""
        self.graph.graph.add_node('cell1', node_type='cell')
        self.graph.graph.add_node('cell2', node_type='cell')
        self.graph.graph.add_edge('cell1', 'cell2', edge_type='connection')
        
        self.assertEqual(len(self.graph.graph.edges()), 1)
        self.assertTrue(self.graph.graph.has_edge('cell1', 'cell2'))
    
    def test_graph_serialization(self):
        """Test graph serialization and deserialization"""
        # Add some nodes and edges
        self.graph.graph.add_node('cell1', node_type='cell', power=0.1, area=2.0)
        self.graph.graph.add_node('cell2', node_type='cell', power=0.2, area=3.0)
        self.graph.graph.add_edge('cell1', 'cell2', edge_type='connection')
        
        # Create temporary file for serialization
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Serialize
            self.graph.serialize_to_json(temp_path)
            
            # Create new graph and deserialize
            new_graph = CanonicalSiliconGraph()
            new_graph.deserialize_from_json(temp_path)
            
            # Check that the new graph has the same content
            self.assertEqual(len(new_graph.graph.nodes()), 2)
            self.assertEqual(len(new_graph.graph.edges()), 1)
            self.assertIn('cell1', new_graph.graph.nodes())
            self.assertIn('cell2', new_graph.graph.nodes())
            self.assertTrue(new_graph.graph.has_edge('cell1', 'cell2'))
            
        finally:
            os.unlink(temp_path)


class TestPhysicalRiskOracle(unittest.TestCase):
    """Test the Physical Risk Oracle"""
    
    def setUp(self):
        self.oracle = PhysicalRiskOracle()
    
    def test_oracle_initialization(self):
        """Test oracle initialization"""
        self.assertIsNotNone(self.oracle.congestion_predictor)
        self.assertIsNotNone(self.oracle.timing_analyzer)
        self.assertIsNotNone(self.oracle.drc_predictor)
        self.assertIsNotNone(self.oracle.design_intent_interpreter)
    
    def test_mock_risk_prediction(self):
        """Test risk prediction with mock data"""
        # Create temporary files for testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as rtl_f:
            rtl_f.write("""
module test();
  reg [31:0] data;
  wire out;
  assign out = data[0];
endmodule
""")
            rtl_path = rtl_f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sdc', delete=False) as sdc_f:
            sdc_f.write("""
create_clock -name clk -period 3.333 [get_ports clk]
set_input_delay -clock clk -max 1.0 [all_inputs]
set_output_delay -clock clk -max 1.0 [all_outputs]
""")
            sdc_path = sdc_f.name
        
        try:
            # Test risk prediction
            results = self.oracle.predict_physical_risks(rtl_path, sdc_path, "7nm")
            
            # Check that results have expected structure
            self.assertIsInstance(results, dict)
            self.assertIn('overall_confidence', results)
            self.assertIn('congestion_heatmap', results)
            self.assertIn('timing_risk_zones', results)
            self.assertIn('recommendations', results)
            
            # Check confidence is in valid range
            self.assertGreaterEqual(results['overall_confidence'], 0.0)
            self.assertLessEqual(results['overall_confidence'], 1.0)
            
        finally:
            os.unlink(rtl_path)
            os.unlink(sdc_path)


class TestAgents(unittest.TestCase):
    """Test the AI agents"""
    
    def setUp(self):
        self.agents = [
            FloorplanAgent(),
            PlacementAgent(),
            ClockAgent(),
            PowerAgent(),
            YieldAgent(),
            RoutingAgent(),
            ThermalAgent()
        ]
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        for agent in self.agents:
            self.assertIsNotNone(agent.agent_id)
            self.assertIsNotNone(agent.agent_type)
            self.assertGreaterEqual(agent.authority_level, 0.0)
            self.assertLessEqual(agent.authority_level, 1.0)
    
    def test_agent_propose_action(self):
        """Test agent proposal generation"""
        # Create a simple test graph
        graph = CanonicalSiliconGraph()
        graph.graph.add_node('test_cell', node_type='cell', power=0.1, area=2.0, timing_criticality=0.3)
        
        for agent in self.agents:
            # Test that agents can handle a basic graph without crashing
            try:
                proposal = agent.propose_action(graph)
                # Proposals can be None if no action needed
                if proposal is not None:
                    self.assertTrue(hasattr(proposal, 'agent_id'))
                    self.assertTrue(hasattr(proposal, 'proposal_id'))
                    self.assertTrue(hasattr(proposal, 'confidence_score'))
            except Exception as e:
                self.fail(f"Agent {agent.__class__.__name__} failed to propose action: {str(e)}")


class TestAgentNegotiator(unittest.TestCase):
    """Test the agent negotiator"""
    
    def setUp(self):
        self.negotiator = EnhancedAgentNegotiator()
        
        # Register test agents
        self.agents = [
            FloorplanAgent(),
            PlacementAgent(),
            ClockAgent()
        ]
        
        for agent in self.agents:
            self.negotiator.register_agent(agent)
    
    def test_negotiator_initialization(self):
        """Test negotiator initialization"""
        self.assertEqual(len(self.negotiator.agents), 3)
        self.assertEqual(len(self.negotiator.agent_registry), 3)
    
    def test_negotiation_round(self):
        """Test negotiation round execution"""
        # Create a simple test graph
        graph = CanonicalSiliconGraph()
        graph.graph.add_node('cell1', node_type='cell', power=0.1, area=2.0, timing_criticality=0.3)
        graph.graph.add_node('cell2', node_type='cell', power=0.2, area=3.0, timing_criticality=0.6)
        graph.graph.add_edge('cell1', 'cell2')
        
        # Run negotiation round
        result = self.negotiator.run_negotiation_round(graph)
        
        # Check result structure
        self.assertTrue(hasattr(result, 'accepted_proposals'))
        self.assertTrue(hasattr(result, 'rejected_proposals'))
        self.assertTrue(hasattr(result, 'partially_accepted_proposals'))


class TestParallelRealityEngine(unittest.TestCase):
    """Test the parallel reality engine"""
    
    def setUp(self):
        self.engine = ParallelRealityEngine(max_workers=2)
    
    def test_engine_initialization(self):
        """Test engine initialization"""
        self.assertEqual(self.engine.max_workers, 2)
        self.assertIsNotNone(self.engine.executor)
    
    def test_parallel_execution(self):
        """Test parallel execution functionality"""
        # Create a simple test graph
        graph = CanonicalSiliconGraph()
        graph.graph.add_node('cell1', node_type='cell', power=0.1, area=2.0)
        graph.graph.add_node('cell2', node_type='cell', power=0.2, area=3.0)
        
        # Define simple strategies
        def strategy1(graph_state, iteration):
            return [{'action': 'test_action_1', 'target': 'cell1'}]
        
        def strategy2(graph_state, iteration):
            return [{'action': 'test_action_2', 'target': 'cell2'}]
        
        # Run parallel execution
        strategies = [strategy1, strategy2]
        universes = self.engine.run_parallel_execution(graph, strategies, max_iterations=2)
        
        # Check results
        self.assertEqual(len(universes), 2)  # Two strategies
        for universe in universes:
            self.assertIsNotNone(universe.graph)
            self.assertGreaterEqual(universe.score, 0.0)


class TestMLModels(unittest.TestCase):
    """Test the ML models"""
    
    def setUp(self):
        self.congestion_predictor = AdvancedCongestionPredictor()
        self.timing_analyzer = AdvancedTimingAnalyzer()
        self.drc_predictor = AdvancedDRCPredictor()
    
    def test_model_initialization(self):
        """Test ML model initialization"""
        self.assertIsNotNone(self.congestion_predictor)
        self.assertIsNotNone(self.timing_analyzer)
        self.assertIsNotNone(self.drc_predictor)
    
    def test_model_prediction(self):
        """Test model prediction functionality"""
        # Create a simple test graph
        graph = CanonicalSiliconGraph()
        graph.graph.add_node('cell1', node_type='cell', power=0.1, area=2.0, timing_criticality=0.3)
        graph.graph.add_node('cell2', node_type='cell', power=0.2, area=3.0, timing_criticality=0.6)
        
        # Test congestion prediction
        congestion_result = self.congestion_predictor.predict(graph)
        self.assertIsInstance(congestion_result, dict)
        
        # Test timing analysis
        timing_result = self.timing_analyzer.analyze(graph, {})
        self.assertIsInstance(timing_result, list)


class TestLearningLoop(unittest.TestCase):
    """Test the learning loop"""
    
    def setUp(self):
        self.controller = LearningLoopController()
    
    def test_controller_initialization(self):
        """Test learning controller initialization"""
        self.assertIsNotNone(self.controller)
        self.assertFalse(self.controller.is_trained)
    
    def test_model_updates(self):
        """Test model update functionality"""
        # Create mock models and agents
        mock_congestion_model = lambda: None
        mock_timing_model = lambda: None
        mock_drc_model = lambda: None
        mock_intent_interpreter = lambda: None
        mock_knowledge_model = lambda: None
        mock_reasoning_engine = lambda: None
        mock_agents = [FloorplanAgent(), PlacementAgent()]
        
        # This should not raise an exception
        try:
            self.controller.update_all_models(
                mock_congestion_model,
                mock_timing_model,
                mock_drc_model,
                mock_intent_interpreter,
                mock_knowledge_model,
                mock_reasoning_engine,
                mock_agents
            )
        except Exception as e:
            self.fail(f"Learning loop update failed: {str(e)}")


class TestFlowOrchestrator(unittest.TestCase):
    """Test the flow orchestrator"""
    
    def setUp(self):
        self.orchestrator = FlowOrchestrator()
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization"""
        self.assertIsNotNone(self.orchestrator.physical_risk_oracle)
        self.assertIsNotNone(self.orchestrator.negotiator)
        self.assertIsNotNone(self.orchestrator.parallel_engine)


class TestIntegration(unittest.TestCase):
    """Test system integration"""
    
    def test_full_system_pipeline(self):
        """Test the full system pipeline"""
        # This is a high-level integration test
        oracle = PhysicalRiskOracle()
        graph_constructor = CanonicalSiliconGraph()
        negotiator = EnhancedAgentNegotiator()
        
        # Register some agents
        agents = [FloorplanAgent(), PlacementAgent(), ClockAgent()]
        for agent in agents:
            negotiator.register_agent(agent)
        
        # Create simple test files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as rtl_f:
            rtl_f.write("""
module integration_test(
    input clk,
    input rst_n,
    input [31:0] data_in,
    output [31:0] data_out
);
    reg [31:0] reg1;
    assign data_out = reg1;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) reg1 <= 32'b0;
        else reg1 <= data_in;
    end
endmodule
""")
            rtl_path = rtl_f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sdc', delete=False) as sdc_f:
            sdc_f.write("""
create_clock -name core_clk -period 3.333 [get_ports clk]
set_input_delay -clock core_clk -max 1.0 [remove_from_collection [all_inputs] [get_ports {clk rst_n}]]
set_output_delay -clock core_clk -max 1.2 [all_outputs]
""")
            sdc_path = sdc_f.name
        
        try:
            # Run a simplified version of the full flow
            risk_results = oracle.predict_physical_risks(rtl_path, sdc_path, "7nm")
            
            # Build graph (simplified)
            graph_constructor.graph.add_node('integration_test', node_type='module', area=100.0)
            graph_constructor.graph.add_node('reg1', node_type='cell', area=2.0, power=0.05)
            graph_constructor.graph.add_edge('integration_test', 'reg1')
            
            # Run negotiation
            negotiation_result = negotiator.run_negotiation_round(graph_constructor)
            
            # Assertions
            self.assertGreaterEqual(risk_results['overall_confidence'], 0.0)
            self.assertIsNotNone(negotiation_result)
            
        finally:
            os.unlink(rtl_path)
            os.unlink(sdc_path)


def run_comprehensive_test_suite():
    """Run the comprehensive test suite"""
    logger = get_logger(__name__)
    logger.info("Running comprehensive Silicon Intelligence System test suite")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    test_classes = [
        TestCanonicalSiliconGraph,
        TestPhysicalRiskOracle,
        TestAgents,
        TestAgentNegotiator,
        TestParallelRealityEngine,
        TestMLModels,
        TestLearningLoop,
        TestFlowOrchestrator,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("SILICON INTELLIGENCE SYSTEM - TEST SUITE RESULTS")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%" if result.testsRun > 0 else "0%")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed! System is functioning correctly.")
    else:
        print("\n✗ Some tests failed. Please review the errors above.")
    
    print("="*60)
    
    return result.wasSuccessful()


def run_performance_tests():
    """Run performance-specific tests"""
    logger = get_logger(__name__)
    logger.info("Running performance tests")
    
    import time
    
    # Test 1: Graph construction performance
    print("\nPerformance Test 1: Graph Construction Speed")
    start_time = time.time()
    
    large_graph = CanonicalSiliconGraph()
    for i in range(10000):  # Add 10k nodes
        large_graph.graph.add_node(f'cell_{i}', 
                                 node_type='cell',
                                 power=np.random.random() * 0.1,
                                 area=np.random.random() * 5.0,
                                 timing_criticality=np.random.random())
    
    # Add some edges
    for i in range(9999):
        if np.random.random() > 0.7:  # Sparse connectivity
            large_graph.graph.add_edge(f'cell_{i}', f'cell_{i+1}')
    
    construction_time = time.time() - start_time
    print(f"  ✓ Constructed graph with {len(large_graph.graph.nodes())} nodes and {len(large_graph.graph.edges())} edges in {construction_time:.3f}s")
    print(f"  Speed: {len(large_graph.graph.nodes()) / construction_time:.0f} nodes/sec")
    
    # Test 2: Risk prediction performance
    print("\nPerformance Test 2: Risk Prediction Speed")
    oracle = PhysicalRiskOracle()
    
    # Create temporary files for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as rtl_f:
        rtl_content = "module test();\n"
        for i in range(1000):
            rtl_content += f"  wire sig_{i};\n"
            rtl_content += f"  assign sig_{i} = sig_{max(0, i-1)};\n"
        rtl_content += "endmodule\n"
        rtl_f.write(rtl_content)
        rtl_path = rtl_f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sdc', delete=False) as sdc_f:
        sdc_content = "create_clock -name clk -period 3.333 [get_ports clk]\n"
        for i in range(50):
            sdc_content += f"set_input_delay -clock clk -max 1.0 [get_ports sig_{i}]\n"
        sdc_f.write(sdc_content)
        sdc_path = sdc_f.name
    
    try:
        prediction_start = time.time()
        risk_results = oracle.predict_physical_risks(rtl_path, sdc_path, "7nm")
        prediction_time = time.time() - prediction_start
        
        print(f"  ✓ Predicted risks for design in {prediction_time:.3f}s")
        print(f"  Confidence: {risk_results['overall_confidence']:.3f}")
        
    finally:
        os.unlink(rtl_path)
        os.unlink(sdc_path)
    
    # Test 3: Agent negotiation performance
    print("\nPerformance Test 3: Agent Negotiation Speed")
    negotiator = EnhancedAgentNegotiator()
    
    agents = [FloorplanAgent(), PlacementAgent(), ClockAgent(), PowerAgent()]
    for agent in agents:
        negotiator.register_agent(agent)
    
    # Create test graph
    test_graph = CanonicalSiliconGraph()
    for i in range(100):
        test_graph.graph.add_node(f'cell_{i}', 
                                node_type='cell',
                                power=0.01 * (i % 10),
                                area=2.0 + (i % 5),
                                timing_criticality=min(1.0, (i % 20) / 10.0))
    
    negotiation_start = time.time()
    negotiation_result = negotiator.run_negotiation_round(test_graph)
    negotiation_time = time.time() - negotiation_start
    
    print(f"  ✓ Negotiated with {len(agents)} agents on {len(test_graph.graph.nodes())} nodes in {negotiation_time:.3f}s")
    print(f"  Proposals: {len(negotiation_result.accepted_proposals)} accepted, {len(negotiation_result.rejected_proposals)} rejected")
    
    print("\nPerformance tests completed!")


if __name__ == "__main__":
    print("Silicon Intelligence System - Comprehensive Test Suite")
    print("="*55)
    
    # Run comprehensive test suite
    success = run_comprehensive_test_suite()
    
    # Run performance tests
    print("\n" + "="*55)
    run_performance_tests()
    
    print("\nTest suite execution completed.")
    if success:
        print("✓ System is ready for production use!")
    else:
        print("✗ Address test failures before production use.")
    
    print("="*55)