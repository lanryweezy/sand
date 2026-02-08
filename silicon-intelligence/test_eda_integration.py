#!/usr/bin/env python3
"""
Test script to demonstrate EDA tool integration capabilities
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from integration.eda_integration import EDAIntegrationLayer
from core.canonical_silicon_graph import CanonicalSiliconGraph
from core.openroad_interface import OpenROADInterface, OpenROADConfig


def test_eda_integration():
    print("🚀 Testing EDA Tool Integration")
    print("=" * 60)
    
    # Create a sample design graph
    print("\n1. Creating sample design graph...")
    graph = CanonicalSiliconGraph()
    
    # Add some sample cells to the graph
    graph.graph.add_node('cpu_core', node_type='macro', area=50000, power=0.5, is_macro=True)
    graph.graph.add_node('memory_bank1', node_type='macro', area=30000, power=0.3, is_macro=True)
    graph.graph.add_node('memory_bank2', node_type='macro', area=30000, power=0.3, is_macro=True)
    graph.graph.add_node('peripheral1', node_type='cell', area=500, power=0.001)
    graph.graph.add_node('peripheral2', node_type='cell', area=300, power=0.0005)
    
    # Add some connections
    graph.graph.add_edge('cpu_core', 'memory_bank1')
    graph.graph.add_edge('cpu_core', 'memory_bank2')
    graph.graph.add_edge('cpu_core', 'peripheral1')
    graph.graph.add_edge('peripheral1', 'peripheral2')
    
    print(f"   ✓ Created graph with {graph.graph.number_of_nodes()} nodes and {graph.graph.number_of_edges()} edges")
    
    # Test OpenROAD interface directly
    print("\n2. Testing OpenROAD interface...")
    config = OpenROADConfig(design_name='test_soc', top_module='test_soc')
    openroad_interface = OpenROADInterface(config)
    
    # Run mock placement
    placement_results = openroad_interface.run_placement(graph)
    print(f"   ✓ Mock placement completed: {placement_results['success']}")
    
    # Run mock routing
    routing_results = openroad_interface.run_routing(graph)
    print(f"   ✓ Mock routing completed: {routing_results['success']}")
    
    # Run complete flow
    complete_results = openroad_interface.run_complete_flow(graph)
    print(f"   ✓ Complete flow completed")
    
    # Test EDA integration layer
    print("\n3. Testing EDA integration layer...")
    eda_layer = EDAIntegrationLayer()
    
    tool_results = eda_layer.run_tool_comparison(graph, 'test_soc')
    print(f"   ✓ Tool comparison completed")
    
    # Display results
    print("\n4. Results Summary:")
    print(f"   OpenROAD - Area: {tool_results['openroad']['overall_ppa']['area_um2']:.0f} um²")
    print(f"   OpenROAD - Power: {tool_results['openroad']['overall_ppa']['power_mw']:.3f} mW")
    print(f"   OpenROAD - Timing: {tool_results['openroad']['overall_ppa']['timing_ns']:.3f} ns")
    print(f"   OpenROAD - DRC Violations: {tool_results['openroad']['overall_ppa']['drc_violations']}")
    
    print("\n" + "=" * 60)
    print("✅ EDA Tool Integration Framework Ready!")
    print("\nNext Steps:")
    print("- Connect to real OpenROAD installation")
    print("- Implement Innovus and Fusion Compiler interfaces")
    print("- Integrate with real design flows")
    print("- Connect to open-source silicon data")
    print("=" * 60)


if __name__ == "__main__":
    test_eda_integration()