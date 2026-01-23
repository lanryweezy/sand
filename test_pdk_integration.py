from core.tcl_generator import TCLGeneratorFactory
import os

def test_pdk_tcl_generation():
    print("Testing PDK-Aware TCL Generation...")
    
    config = {
        'design_name': 'sky_adder',
        'verilog': 'src/adder.v',
        'sdc': 'src/timing.sdc',
        'pdk_root': '/opt/pdks',
        'pdk_variant': 'hd',
        'node': 130
    }
    
    innovus_gen = TCLGeneratorFactory.get_generator('innovus')
    tcl_output = innovus_gen.generate_full_flow(config)
    
    print("\n--- Generated PDK-Aware Innovus Script ---")
    print(tcl_output[:1000]) # Print beginning for verification
    
    # Check if SkyWater paths are present
    if "/opt/pdks/sky130A/libs.ref/sky130_fd_sc_hd/lef/sky130_fd_sc_hd.merged.lef" in tcl_output:
        print("\n✅ SkyWater PDK Paths Correctly Injected!")
    else:
        print("\n❌ SkyWater PDK Paths Missing!")

    if "connect_global_net VDD" in tcl_output:
        print("✅ Power Nets (VDD/VSS) Correctly Configured!")
    else:
        print("❌ Power Net Configuration Missing!")

if __name__ == "__main__":
    test_pdk_tcl_generation()
