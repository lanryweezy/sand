import silicon_intelligence_cpp as sic
import sys

def verify_rtl_transformer():
    print(f"--- Silicon Intelligence RTL Transformer Verification ---\n")
    
    print(f"[DEBUG] Module file: {getattr(sic, '__file__', 'unknown')}")
    print(f"[DEBUG] Module contents: {dir(sic)}")

    if not hasattr(sic, 'RTLTransformer'):
        print("[FAIL] RTLTransformer class not found in extension!")
        sys.exit(1)
        
    print("[PASS] RTLTransformer class found.")
    
    transformer = sic.RTLTransformer()
    
    # Test 1: Simple Parsing
    source_verilog = """
    module test_mod (
        input clk,
        input [7:0] data_in,
        output [7:0] data_out
    );
        wire [7:0] internal_sig;
        assign internal_sig = data_in;
        assign data_out = internal_sig;
    endmodule
    """
    
    print("\nTest 1: Parsing Verilog Source...")
    mod = transformer.parse_verilog(source_verilog)
    
    if mod.name == "test_mod":
        print(f"[PASS] Parsed module name: {mod.name}")
    else:
        print(f"[FAIL] Expected 'test_mod', got '{mod.name}'")
        
    generated = mod.to_verilog()
    if "module test_mod" in generated and "endmodule" in generated:
        print("[PASS] Verilog generation successful")
    else:
        print("[FAIL] Verilog generation output suspicious")
        print(generated)

    # Test 2: Pipelining
    print("\nTest 2: Pipelining Transformation...")
    print("Adding pipeline stage to 'internal_sig'...")
    transformer.add_pipeline_stage(mod, "internal_sig")
    
    gen_pipelined = mod.to_verilog()
    if "reg [7:0] internal_sig_pipe_reg;" in gen_pipelined and "always @(posedge clk)" in gen_pipelined:
        print("[PASS] Pipeline register and logic added")
    else:
        print("[FAIL] Pipeline transformation missing expected elements")
        print(gen_pipelined)
        
    # Test 3: Clock Gating
    print("\nTest 3: Clock Gating Transformation...")
    # Create new module for clean test
    mod_cg = transformer.parse_verilog("""
    module cg_test (input clk, input en, output reg [31:0] out);
        always @(posedge clk) begin
            if (en) out <= 32'hFFFF;
        end
    endmodule
    """)
    
    transformer.insert_clock_gate(mod_cg, "out", "en")
    gen_cg = mod_cg.to_verilog()
    
    if "sky130_fd_sc_hd__lpflow_is_1 icg_out" in gen_cg:
        print("[PASS] Clock gate instance added")
    else:
        print("[FAIL] Clock gate instance missing")
        print(gen_cg)
        
    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    try:
        verify_rtl_transformer()
    except Exception as e:
        print(f"\n[ERROR] Verification failed with exception: {e}")
        import traceback
        traceback.print_exc()
