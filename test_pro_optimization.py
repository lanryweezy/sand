from autonomous_optimizer import AdvancedAutonomousOptimizer, OptimizationStrategy

def test_pro_optimization_loop():
    print("Testing Professional Optimization Loop (AST-based)...")
    
    optimizer = AdvancedAutonomousOptimizer()
    
    # Professional RTL input (ready for pipelining)
    test_rtl = '''
module test_engine (
    input clk,
    input [31:0] data_in,
    output [31:0] data_out
);
    wire [31:0] processed_data;
    assign processed_data = data_in ^ 32'hAAAAAAAA;
    assign data_out = processed_data;
endmodule
    '''
    
    print("\n--- Original RTL ---")
    print(test_rtl)
    
    # Force a pipelining opportunity with specific target
    params = {'module_name': 'test_engine', 'target_signal': 'processed_data'}
    
    print(f"\nApplying Optimization: {OptimizationStrategy.PIPELINE_CRITICAL_PATHS.value}")
    optimized_rtl = optimizer._apply_pipelining(test_rtl, params)
    
    print("\n--- Optimized RTL (AST-Generated) ---")
    print(optimized_rtl)
    
    if "processed_data_pipe_reg" in optimized_rtl and "always @(posedge clk)" in optimized_rtl:
        print("\n✅ AST-based Optimization Successful!")
    else:
        print("\n❌ AST-based Optimization Failed!")

if __name__ == "__main__":
    test_pro_optimization_loop()
