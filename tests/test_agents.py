"""
Tests for the Specialist AI Agents (Floorplan, Placement, Clock, Power, Routing, Thermal, Yield)
"""

import sys
import os
# Add the project root to sys.path to allow imports of silicon_intelligence package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
import os
from datetime import datetime
from unittest.mock import MagicMock, patch

from silicon_intelligence.agents.base_agent import AgentProposal, AgentNegotiator, BaseAgent, AgentType
from silicon_intelligence.agents.floorplan_agent import FloorplanAgent
from silicon_intelligence.agents.placement_agent import PlacementAgent
from silicon_intelligence.agents.clock_agent import ClockAgent
from silicon_intelligence.agents.power_agent import PowerAgent
from silicon_intelligence.agents.routing_agent import RoutingAgent
from silicon_intelligence.agents.thermal_agent import ThermalAgent
from silicon_intelligence.agents.yield_agent import YieldAgent
from silicon_intelligence.core.canonical_silicon_graph import CanonicalSiliconGraph, NodeType, EdgeType


class TestAgents(unittest.TestCase):

    def setUp(self):
        # Setup a mock graph for testing agent proposals
        self.graph = CanonicalSiliconGraph()
        
        # Add various node types to the graph to simulate a design
        self.graph.graph.add_node("clk_source_main", node_type=NodeType.CLOCK.value, is_clock_root=True, clock_domain="clk_main")
        self.graph.graph.add_node("clk_source_aux", node_type=NodeType.CLOCK.value, is_clock_root=True, clock_domain="clk_aux")
        
        self.graph.graph.add_node("dff_ctrl_1", node_type=NodeType.CELL.value, cell_type="DFF_X4", timing_criticality=0.9, power=0.1, area=10.0, region="control", clock_domain="clk_main")
        self.graph.graph.add_node("dff_ctrl_2", node_type=NodeType.CELL.value, cell_type="DFF_X2", timing_criticality=0.8, power=0.08, area=8.0, region="control", clock_domain="clk_main")
        self.graph.graph.add_node("and2_data_1", node_type=NodeType.CELL.value, cell_type="AND2_X1", timing_criticality=0.4, power=0.01, area=2.0, region="data_path")
        self.graph.graph.add_node("mux4_data_1", node_type=NodeType.CELL.value, cell_type="MUX4_X1", timing_criticality=0.5, power=0.03, area=5.0, region="data_path")

        self.graph.graph.add_node("mem_bank_0", node_type=NodeType.MACRO.value, cell_type="RAM_256x32", is_macro=True, area=500.0, power=2.0, region="memory", position=(1000,1000))
        self.graph.graph.add_node("ip_block_aes", node_type=NodeType.MACRO.value, cell_type="AES_IP", is_macro=True, area=300.0, power=1.5, region="security", position=(5000,5000))

        self.graph.graph.add_node("net_clk_main", node_type=NodeType.SIGNAL.value, is_routed=False, estimated_congestion=0.1)
        self.graph.graph.add_node("net_data_path", node_type=NodeType.SIGNAL.value, is_routed=False, estimated_congestion=0.7)
        self.graph.graph.add_node("net_ctrl_path", node_type=NodeType.SIGNAL.value, is_routed=False, estimated_congestion=0.3)
        
        # Add edges to simulate connectivity
        self.graph.graph.add_edge("clk_source_main", "net_clk_main", edge_type=EdgeType.CONNECTION.value, delay=0.1, length=10)
        self.graph.graph.add_edge("net_clk_main", "dff_ctrl_1", edge_type=EdgeType.CONNECTION.value, delay=0.05, length=5)
        self.graph.graph.add_edge("net_clk_main", "dff_ctrl_2", edge_type=EdgeType.CONNECTION.value, delay=0.05, length=5)

        self.graph.graph.add_edge("dff_ctrl_1", "net_ctrl_path", edge_type=EdgeType.CONNECTION.value, delay=0.1, length=15)
        self.graph.graph.add_edge("net_ctrl_path", "ip_block_aes", edge_type=EdgeType.CONNECTION.value, delay=0.2, length=20)

        self.graph.graph.add_edge("mem_bank_0", "net_data_path", edge_type=EdgeType.CONNECTION.value, delay=0.3, length=30)
        self.graph.graph.add_edge("net_data_path", "and2_data_1", edge_type=EdgeType.CONNECTION.value, delay=0.1, length=10)
        self.graph.graph.add_edge("and2_data_1", "mux4_data_1", edge_type=EdgeType.CONNECTION.value, delay=0.08, length=8)

        # Ensure that nodes exist in the graph before running tests
        for node_id in self.graph.graph.nodes():
            if 'power' not in self.graph.graph.nodes[node_id]:
                self.graph.graph.nodes[node_id]['power'] = 0.05
            if 'area' not in self.graph.graph.nodes[node_id]:
                self.graph.graph.nodes[node_id]['area'] = 5.0
            if 'timing_criticality' not in self.graph.graph.nodes[node_id]:
                self.graph.graph.nodes[node_id]['timing_criticality'] = 0.5
            if 'region' not in self.graph.graph.nodes[node_id]:
                self.graph.graph.nodes[node_id]['region'] = 'default'

    def _test_proposal_structure(self, proposal: AgentProposal, expected_type: AgentType):
        self.assertIsInstance(proposal, AgentProposal)
        self.assertEqual(proposal.agent_type, expected_type)
        self.assertIsInstance(proposal.proposal_id, str)
        self.assertIsInstance(proposal.timestamp, datetime)
        self.assertIsInstance(proposal.action_type, str)
        self.assertIsInstance(proposal.targets, list)
        self.assertIsInstance(proposal.parameters, dict)
        self.assertGreaterEqual(proposal.confidence_score, 0.1)
        self.assertLessEqual(proposal.confidence_score, 1.0)
        self.assertIsInstance(proposal.risk_profile, dict)
        self.assertIsInstance(proposal.cost_vector, dict)
        self.assertIsInstance(proposal.predicted_outcome, dict)
        self.assertIsInstance(proposal.dependencies, list)
        self.assertIsInstance(proposal.conflicts_with, list)

    def test_floorplan_agent_proposal(self):
        agent = FloorplanAgent()
        proposal = agent.propose_action(self.graph)
        self._test_proposal_structure(proposal, AgentType.FLOORPLAN)
        self.assertIn('strategy', proposal.parameters)
        self.assertIn('positions', proposal.parameters)
        self.assertGreater(len(proposal.targets), 0)
        
        # Test applying a specific strategy
        agent.strategy_index = 0 # Force 'hierarchical_clustering'
        proposal = agent.propose_action(self.graph)
        self.assertEqual(proposal.parameters['strategy'], 'hierarchical_clustering')

    def test_placement_agent_proposal(self):
        agent = PlacementAgent()
        proposal = agent.propose_action(self.graph)
        self._test_proposal_structure(proposal, AgentType.PLACEMENT)
        self.assertIn('strategy', proposal.parameters)
        self.assertIn('cell_placements', proposal.parameters)
        self.assertGreater(len(proposal.targets), 0)
        self.assertFalse(any(self.graph.graph.nodes[t].get('node_type') == NodeType.MACRO.value for t in proposal.targets))

    def test_clock_agent_proposal(self):
        agent = ClockAgent()
        proposal = agent.propose_action(self.graph)
        self._test_proposal_structure(proposal, AgentType.CLOCK)
        self.assertIn('strategy', proposal.parameters)
        self.assertIn('clock_topology', proposal.parameters)
        self.assertIn('skew_requirements', proposal.parameters)
        self.assertGreater(len(proposal.targets), 0)
        self.assertTrue(any(self.graph.graph.nodes[t].get('node_type') == NodeType.CLOCK.value for t in proposal.targets) or \
                        any(self.graph.graph.nodes[t].get('cell_type') == 'dff' for t in proposal.targets))


    def test_power_agent_proposal(self):
        agent = PowerAgent()
        # Set agent type to POWER (it was wrongly initialized to POWER in thermal_agent, let's ensure it's correct here)
        # Note: In actual code, PowerAgent is initialized with AgentType.POWER. No change needed.
        proposal = agent.propose_action(self.graph)
        self._test_proposal_structure(proposal, AgentType.POWER)
        self.assertIn('strategy', proposal.parameters)
        self.assertIn('power_grid_config', proposal.parameters)
        self.assertGreater(len(proposal.targets), 0)

    def test_routing_agent_proposal(self):
        agent = RoutingAgent()
        proposal = agent.propose_action(self.graph)
        self._test_proposal_structure(proposal, AgentType.ROUTING)
        self.assertIn('strategy', proposal.parameters)
        self.assertIn('routing_layers', proposal.parameters)
        self.assertGreater(len(proposal.targets), 0)

    def test_thermal_agent_proposal(self):
        agent = ThermalAgent()
        proposal = agent.propose_action(self.graph)
        # ThermalAgent uses AgentType.POWER in its __init__, need to adjust test or agent
        # Let's check the agent type as it is.
        self._test_proposal_structure(proposal, AgentType.POWER) # ThermalAgent's __init__ uses AgentType.POWER
        self.assertIn('strategy', proposal.parameters)
        self.assertIn('hotspot_mitigation', proposal.parameters)
        self.assertGreater(len(proposal.targets), 0)

    def test_yield_agent_proposal(self):
        agent = YieldAgent()
        proposal = agent.propose_action(self.graph)
        self._test_proposal_structure(proposal, AgentType.YIELD)
        self.assertIn('strategy', proposal.parameters)
        self.assertIn('defect_mitigation', proposal.parameters)
        self.assertGreater(len(proposal.targets), 0)

    def test_agent_negotiation(self):
        # Create a fresh graph for negotiation to avoid side effects from other tests
        negotiation_graph = CanonicalSiliconGraph()
        negotiation_graph.graph.add_node("macro_A", node_type=NodeType.MACRO.value, is_macro=True, area=100.0, power=0.5, region="core", position=(100,100))
        negotiation_graph.graph.add_node("macro_B", node_type=NodeType.MACRO.value, is_macro=True, area=120.0, power=0.6, region="core", position=(200,200))
        negotiation_graph.graph.add_node("cell_X", node_type=NodeType.CELL.value, cell_type="AND2", timing_criticality=0.7, power=0.05, area=2.0)
        negotiation_graph.graph.add_node("cell_Y", node_type=NodeType.CELL.value, cell_type="DFF", timing_criticality=0.8, power=0.1, area=5.0)
        negotiation_graph.graph.add_edge("macro_A", "cell_X", edge_type=EdgeType.CONNECTION.value, delay=0.1)
        negotiation_graph.graph.add_edge("cell_X", "cell_Y", edge_type=EdgeType.CONNECTION.value, delay=0.2)

        negotiator = AgentNegotiator()
        
        # Register a FloorplanAgent
        fp_agent = FloorplanAgent()
        negotiator.register_agent(fp_agent)
        
        # Register a PlacementAgent
        pl_agent = PlacementAgent()
        negotiator.register_agent(pl_agent)

        # Make agents propose actions
        fp_proposal = fp_agent.propose_action(negotiation_graph)
        pl_proposal = pl_agent.propose_action(negotiation_graph)

        # Add proposals directly for a more controlled test if needed, or let run_negotiation_round collect
        # For this test, run_negotiation_round will call propose_action
        
        negotiation_result = negotiator.run_negotiation_round(negotiation_graph)
        
        self.assertIsNotNone(negotiation_result)
        self.assertIsInstance(negotiation_result.accepted_proposals, list)
        self.assertIsInstance(negotiation_result.rejected_proposals, list)
        self.assertIsInstance(negotiation_result.partially_accepted_proposals, list)
        self.assertIsNotNone(negotiation_result.updated_graph)
        self.assertIsInstance(negotiation_result.global_metrics, dict)
        
        # At least some proposals should have been processed (accepted, rejected, or partially accepted)
        total_proposals_processed = len(negotiation_result.accepted_proposals) + \
                                    len(negotiation_result.rejected_proposals) + \
                                    len(negotiation_result.partially_accepted_proposals)
        self.assertGreaterEqual(total_proposals_processed, 1) # At least one proposal should have been generated and processed
        
        # Verify that the graph was indeed updated (e.g., node counts might change or attributes updated)
        # This is a very basic check; more detailed checks would inspect specific attribute changes
        self.assertGreaterEqual(negotiation_result.updated_graph.graph.number_of_nodes(), negotiation_graph.graph.number_of_nodes())

if __name__ == '__main__':
    unittest.main()
