
import sys
import os

# Ensure we can import core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autonomous_optimizer import AdvancedAutonomousOptimizer, OptimizationStrategy

def test_area_power_optimization():
    print("Testing Area/Power Balance Strategy (Input Isolation)...")
    
    optimizer = AdvancedAutonomousOptimizer()
    
    # Create a test RTL with combinational logic that can be isolated
    test_rtl = '''
module power_hungry_logic (
    input clk,
    input en,
    input [7:0] data_in,
    output reg [15:0] result
);

    wire [15:0] complex_calculation;
    assign complex_calculation = data_in * data_in + 8'hA5;

    always @(posedge clk) begin
        if (en)
            result <= complex_calculation;
    end

endmodule
'''
    
    print("\n--- Original RTL ---")
    print(test_rtl)
    
    # triggering Area/Power balance
    params = {
        'enable_signal': 'en',
        'targets': ['data_in'],
        'module_name': 'power_hungry_logic'
    }
    
    optimized_rtl = optimizer._balance_area_power(test_rtl, params)
    
    print("\n--- Optimized RTL ---")
    print(optimized_rtl)
    
    # Verification
    # Check for isolated signal name and some form of ternary logic with 'en'
    if "data_in_isolated" in optimized_rtl and any(x in optimized_rtl for x in ["en ?", "(en)?", "en?"]):
        print("\nSUCCESS: Input isolation logic found.")
    else:
        print("\nFAILURE: Input isolation logic not found.")
        print("\nDEBUG: Optimized RTL output:")
        print(optimized_rtl)
        sys.exit(1)
        
    # Check if calculation uses the isolated signal
    # Use replace to ignore spaces in the check
    compact_rtl = optimized_rtl.replace(" ", "").replace("\n", "")
    if "data_in_isolated*data_in_isolated" in compact_rtl:
         print("SUCCESS: Downstream logic is using isolated input.")
    else:
         print("FAILURE: Downstream logic still using raw input.")
         print("\nDEBUG: Optimized RTL output:")
         print(optimized_rtl)
         sys.exit(1)

if __name__ == "__main__":
    test_area_power_optimization()
