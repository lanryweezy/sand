"""
Professional System Demonstration - Silicon Intelligence (v2.0 PRO)
Demonstrates the full stack: GNN, AST Refactoring, PDK Integration, and TCL Generation.
Branded by Street Heart Technologies.
"""

from core.rtl_transformer import RTLTransformer
from core.pdk_manager import PDKManager
from core.tcl_generator import TCLGeneratorFactory
from utils.authority_dashboard import AuthorityDashboard
import os

def run_pro_demonstration():
    dashboard = AuthorityDashboard()
    transformer = RTLTransformer()
    pdk_mgr = PDKManager()
    
    print("Initializing Silicon Intelligence Professional Flow...")
    
    # 1. Source Design
    test_rtl = "demo_design.v"
    with open(test_rtl, "w") as f:
        f.write('''
module pro_alu (
    input clk,
    input [31:0] a,
    input [31:0] b,
    output [31:0] res
);
    wire [31:0] internal_op;
    assign internal_op = a & b;
    assign res = internal_op;
endmodule
''')

    # 2. Step 1: AST Transformation (Pipelining)
    print("Step 1: Analyzing critical paths via AST...")
    ast = transformer.parse_rtl(test_rtl)
    ast, reg_name = transformer.add_pipeline_stage(ast, 'pro_alu', 'internal_op')
    ast = transformer.update_signal_sinks(ast, 'pro_alu', 'internal_op', reg_name)
    optimized_rtl = transformer.generate_verilog(ast)
    
    dashboard.log_optimization("PIPELINE_AST", "pro_alu.internal_op", "SUCCESS")
    
    # 3. Step 2: PDK Management
    print("Step 2: Calibrating for SkyWater 130nm PDK...")
    tech_info = pdk_mgr.get_tech_summary()
    dashboard.log_optimization("PDK_BIND", "sky130A_hd", "READY")
    
    # 4. Step 3: TCL Generation
    print("Step 3: Generating Professional EDA Scripts...")
    gen_config = {
        'design_name': 'pro_alu',
        'verilog': 'optimized_pro_alu.v',
        'sdc': 'constraints/pro.sdc',
        'pdk_root': '/usr/local/pdk',
        'node': 130
    }
    innovus_gen = TCLGeneratorFactory.get_generator('innovus')
    innovus_tcl = innovus_gen.generate_full_flow(gen_config)
    
    with open("innovus_flow.tcl", "w") as f:
        f.write(innovus_tcl)
    
    dashboard.log_optimization("TCL_GEN", "Cadence_Innovus", "SUCCESS")
    
    # 5. Final Telemetry Display
    print("\n" + "="*50)
    dashboard.display_full_report()
    print("="*50)
    
    print("\nDemonstration Complete. artifacts generated:")
    print(" - innovus_flow.tcl (PDK-aware)")
    print(" - Authority Dashboard Telemetry (Rich output)")

if __name__ == "__main__":
    run_pro_demonstration()
