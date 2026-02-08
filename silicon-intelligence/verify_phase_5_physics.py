import silicon_intelligence_cpp as si
import math

def test_physics():
    print("Testing Phase 5: Force-Directed Physics...")
    engine = si.GraphEngine()
    
    # Create a small graph: u1 -- u2 -- u3
    # They should stabilize at some distance k
    engine.add_node("u1", si.NodeAttributes())
    engine.add_node("u2", si.NodeAttributes())
    engine.add_node("u3", si.NodeAttributes())
    
    engine.add_edge("u1", "u2", si.EdgeAttributes())
    engine.add_edge("u2", "u3", si.EdgeAttributes())
    
    # Random initial positions
    engine.set_node_position("u1", 50.0, 50.0)
    engine.set_node_position("u2", 51.0, 51.0) # Close to u1
    engine.set_node_position("u3", 500.0, 500.0) # Far from u1,u2
    
    kernels = si.OptimizationKernels(engine)
    
    print(f"Starting Pos u1: {engine.get_node_position('u1')}")
    print(f"Starting Pos u2: {engine.get_node_position('u2')}")
    print(f"Starting Pos u3: {engine.get_node_position('u3')}")
    
    # Run many iterations to see convergence
    for i in range(100):
        kernels.apply_forces(100.0, 100.0) # k=100.0
        
    p1 = engine.get_node_position("u1")
    p2 = engine.get_node_position("u2")
    p3 = engine.get_node_position("u3")
    
    print(f"Final Pos u1: {p1}")
    print(f"Final Pos u2: {p2}")
    print(f"Final Pos u3: {p3}")
    
    # Check if u3 (far) was pulled closer by u2
    # and u1,u2 (too close) were pushed apart or held by k
    
    dist12 = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    dist23 = math.sqrt((p2[0]-p3[0])**2 + (p2[1]-p3[1])**2)
    
    print(f"Distance 1-2: {dist12:.2f}")
    print(f"Distance 2-3: {dist23:.2f}")
    
    # With k=100, they should move towards having distance ~100
    if dist12 > 1.0 and dist23 < 600.0:
        print("✅ Success: Forces are moving nodes correctly.")
    else:
        print("❌ Failure: Positions did not converge as expected.")

if __name__ == "__main__":
    test_physics()
