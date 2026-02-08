"""
Power Mastery Verification - Demonstrates autonomous clock gating injection.
Branded by Street Heart Technologies.
"""

from autonomous_optimizer import AutonomousOptimizer, OptimizationStrategy
from core.canonical_silicon_graph import CanonicalSiliconGraph, NodeType

def test_power_mastery_flow():
    print("Testing Professional Power Mastery Flow...")
    
    optimizer = AutonomousOptimizer()
    
    # RTL with a candidate register for gating
    test_rtl = '''
module power_unit (
    input clk,
    input en,
    input [31:0] d,
    output reg [31:0] q
);
    always @(posedge clk) begin
        if (en)
            q <= d;
    end
endmodule
'''
    
    print("\n--- Original RTL ---")
    print(test_rtl)
    
    # Propose clock gating for 'q'
    params = {
        'module_name': 'power_unit',
        'target_signal': 'q',
        'enable_signal': 'en'
    }
    
    print(f"\nApplying Power Optimization: {OptimizationStrategy.CLOCK_GATING.value}")
    optimized_rtl = optimizer._apply_clock_gating(test_rtl, params)
    
    print("\n--- Power-Optimized RTL (AST-Generated) ---")
    print(optimized_rtl)
    
    if "sky130_fd_sc_hd__lpflow_is_1" in optimized_rtl and "q_gated_clk" in optimized_rtl:
        print("\n✅ Power Mastery: Clock Gate Injected Successfully!")
    else:
        print("\n❌ Power Mastery: Transformation Failed!")

if __name__ == "__main__":
    test_power_mastery_flow()
