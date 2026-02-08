import silicon_intelligence_cpp as sic
import sys

def verify_cpp_core():
    print(f"--- Silicon Intelligence C++ Verification ---")
    print(f"Python version: {sys.version}")
    
    try:
        # 1. Test Module Presence
        print(f"Module found: {sic.__name__}")
        
        # 2. Test Enum
        print(f"Testing NodeType enum: {sic.NodeType.CELL}, {sic.NodeType.MACRO}")
        
        # 3. Test GraphEngine
        ge = sic.GraphEngine()
        print(f"Initial node count: {ge.num_nodes()}")
        
        # 4. Add nodes with full attributes
        attrs = sic.NodeAttributes()
        attrs.node_type = sic.NodeType.CELL
        attrs.cell_type = "AND2_X1"
        attrs.area = 1.2
        attrs.timing_criticality = 0.8
        
        ge.add_node("u1", attrs)
        
        attrs.node_type = sic.NodeType.MACRO
        attrs.cell_type = "RAM_256x8"
        attrs.area = 500.0
        attrs.timing_criticality = 0.2
        ge.add_node("u2", attrs)
        
        # 5. Add edges
        eattrs = sic.EdgeAttributes()
        eattrs.edge_type = sic.EdgeType.CONNECTION
        eattrs.delay = 0.5
        ge.add_edge("u1", "u2", eattrs)
        
        print(f"Node count: {ge.num_nodes()}")
        print(f"Edge count: {ge.num_edges()}")
        
        # 6. Test Algorithms
        critical = ge.get_timing_critical_nodes(0.5)
        print(f"Critical nodes (>=0.5): {critical}")
        
        if ge.num_nodes() == 2 and ge.num_edges() == 1 and "u1" in critical:
            print("\n[SUCCESS] C++ Phase 2 Engine is functional!")
        else:
            print(f"\n[FAILURE] Logic verification failed. Critical nodes: {critical}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n[ERROR] Verification failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_cpp_core()
