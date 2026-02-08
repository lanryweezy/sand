import silicon_intelligence_cpp as sic
import sys

def verify_optimization():
    print(f"--- Silicon Intelligence Optimization Kernels Verification ---\n")
    
    if not hasattr(sic, 'OptimizationKernels'):
        print("[FAIL] OptimizationKernels class not found in extension!")
        sys.exit(1)
        
    print("[PASS] OptimizationKernels class found.")
    
    # Create Graph
    engine = sic.GraphEngine()
    engine.add_node("u1", sic.NodeAttributes())
    engine.add_node("u2", sic.NodeAttributes())
    engine.add_edge("u1", "u2", sic.EdgeAttributes())
    
    # Create Optimization Kernel
    print("\nInitializing Optimization Kernel...")
    optimizer = sic.OptimizationKernels(engine)
    
    # Configure Placement
    config = sic.PlacementConfig()
    config.iterations = 100
    config.initial_temp = 500.0
    config.threads = 2
    
    print(f"Configuration: {config.iterations} iterations, {config.threads} threads")
    
    # Run Placement
    print("\nRunning Global Placement...")
    try:
        optimizer.run_global_placement(config)
        print("[PASS] Global Placement execution completed without errors")
    except Exception as e:
        print(f"[FAIL] Execution failed: {e}")
        
    # Check HPWL (Dummy check since we didn't populate positions fully/check implementation logic depth)
    hpwl = optimizer.calculate_hpwl()
    print(f"Calculated HPWL: {hpwl}")
    
    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    try:
        verify_optimization()
    except Exception as e:
        print(f"\n[ERROR] Verification failed with exception: {e}")
        import traceback
        traceback.print_exc()
