
import sys
import os

# Ensure we can import core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autonomous_optimizer import AdvancedAutonomousOptimizer, OptimizationStrategy

def test_clustering_optimization():
    print("Testing Clustering Strategy (Logic Merging)...")
    
    optimizer = AdvancedAutonomousOptimizer()
    
    # Create a test RTL with redundant logic
    test_rtl = '''
module redundant_logic (
    input [7:0] a,
    input [7:0] b,
    output [7:0] out1,
    output [7:0] out2
);

    wire [7:0] tmp1;
    wire [7:0] tmp2;
    
    assign tmp1 = a + b;
    assign tmp2 = a + b; // Redundant!
    
    assign out1 = tmp1;
    assign out2 = tmp2;

endmodule
'''
    
    print("\n--- Original RTL ---")
    print(test_rtl)
    
    # triggering Clustering
    params = {
        'module_name': 'redundant_logic'
    }
    
    optimized_rtl = optimizer._apply_clustering(test_rtl, params)
    
    print("\n--- Optimized RTL ---")
    print(optimized_rtl)
    
    # Verification
    # We expect tmp2 assignment to be gone and out2 to use tmp1
    # Use compact version to ignore spaces
    compact_rtl = optimized_rtl.replace(" ", "").replace("\n", "").replace("\t", "")
    
    if "assigntmp2=a+b" not in compact_rtl:
        print("\nSUCCESS: Redundant assignment removed.")
    else:
        print("\nFAILURE: Redundant assignment still present.")
        sys.exit(1)
        
    if "assignout2=tmp1" in compact_rtl:
         print("SUCCESS: Usage redirected to primary signal.")
    else:
         print("FAILURE: Usage not redirected.")
         print("\nDEBUG: Compact RTL:")
         print(compact_rtl)
         sys.exit(1)

if __name__ == "__main__":
    test_clustering_optimization()
