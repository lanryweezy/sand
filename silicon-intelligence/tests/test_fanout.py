
import sys
import os

# Ensure we can import core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autonomous_optimizer import AdvancedAutonomousOptimizer, OptimizationStrategy

def test_fanout_optimization():
    print("Testing Fanout Optimization Strategy...")
    
    optimizer = AdvancedAutonomousOptimizer()
    
    # Create a test RTL with high fanout
    # A single 'clk' driving multiple registers
    test_rtl = '''
module high_fanout_module (
    input clk,
    input [3:0] d,
    output reg [3:0] q1,
    output reg [3:0] q2,
    output reg [3:0] q3,
    output reg [3:0] q4
);

    always @(posedge clk) begin
        q1 <= d;
    end
    
    always @(posedge clk) begin
        q2 <= d;
    end
    
    always @(posedge clk) begin
        q3 <= d;
    end
    
    always @(posedge clk) begin
        q4 <= d;
    end

endmodule
'''
    
    print("\n--- Original RTL ---")
    print(test_rtl)
    
    # Manually trigger the specific strategy
    # We pretend metrics showed high fanout
    params = {
        'target_signal': 'clk',
        'module_name': 'high_fanout_module',
        'current_avg': 16.0 # Will trigger degree=4 split
    }
    
    optimized_rtl = optimizer._optimize_fanout(test_rtl, params)
    
    print("\n--- Optimized RTL ---")
    print(optimized_rtl)
    
    # Verification
    if "clk_buf_0" in optimized_rtl and "clk_buf_1" in optimized_rtl:
        print("\nSUCCESS: Buffer signals created.")
    else:
        print("\nFAILURE: Buffer signals not found.")
        sys.exit(1)
        
    # Check if original 'posedge clk' usage is reduced/eliminated in favor of buffers
    # Note: The exact redistribution depends on visitor order, but we expect mix of buf_0, buf_1 etc.
    if "posedge clk_buf_" in optimized_rtl:
         print("SUCCESS: Sinks are using buffered clocks.")
    else:
         print("FAILURE: Sinks are still using original clock.")
         sys.exit(1)

if __name__ == "__main__":
    test_fanout_optimization()
