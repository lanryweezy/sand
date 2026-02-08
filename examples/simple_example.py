"""
Example usage of the Silicon Intelligence System
"""

import os
import tempfile
from silicon_intelligence.cognitive.physical_risk_oracle import PhysicalRiskOracle
from silicon_intelligence.agents.floorplan_agent import FloorplanAgent
from silicon_intelligence.agents.placement_agent import PlacementAgent
from silicon_intelligence.agents.clock_agent import ClockAgent
from silicon_intelligence.agents.base_agent import AgentNegotiator


def create_sample_rtl():
    """Create a sample RTL file for testing"""
    rtl_content = """
// Sample RTL for testing
module test_design (
    input clk,
    input rst_n,
    input [7:0] data_in,
    output [7:0] data_out,
    output reg valid
);

    // Internal signals
    wire [7:0] reg_in;
    wire [7:0] comb_out;
    reg [7:0] pipeline_reg1;
    reg [7:0] pipeline_reg2;
    
    // Input register
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            pipeline_reg1 <= 8'b0;
        else
            pipeline_reg1 <= data_in;
    end
    
    // Combinational logic
    assign comb_out = pipeline_reg1 + 8'h10;
    
    // Output register
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            pipeline_reg2 <= 8'b0;
        else
            pipeline_reg2 <= comb_out;
    end
    
    // Output assignment
    assign data_out = pipeline_reg2;
    
    // Valid signal generation
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            valid <= 1'b0;
        else
            valid <= 1'b1;
    end

endmodule
"""
    return rtl_content


def create_sample_constraints():
    """Create a sample SDC file for testing"""
    sdc_content = """
# Sample constraints for testing
create_clock -name clk -period 5.000 -waveform {0.000 2.500} [get_ports clk]
set_input_delay -clock clk -max 2.000 [remove_from_collection [all_inputs] [get_ports clk]]
set_input_delay -clock clk -min 1.000 [remove_from_collection [all_inputs] [get_ports clk]]
set_output_delay -clock clk -max 2.000 [all_outputs]
set_output_delay -clock clk -min 1.000 [all_outputs]
set_clock_uncertainty -setup 0.1 [get_clocks clk]
set_clock_uncertainty -hold 0.05 [get_clocks clk]
"""
    return sdc_content


def main():
    print("Silicon Intelligence System - Example Usage")
    print("=" * 50)
    
    # Create temporary files for our example
    with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as rtl_file:
        rtl_file.write(create_sample_rtl())
        rtl_filename = rtl_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sdc', delete=False) as sdc_file:
        sdc_file.write(create_sample_constraints())
        sdc_filename = sdc_file.name
    
    try:
        # Initialize the Physical Risk Oracle
        print("\n1. Initializing Physical Risk Oracle...")
        oracle = PhysicalRiskOracle()
        
        # Predict physical risks
        print(f"\n2. Analyzing RTL: {os.path.basename(rtl_filename)}")
        print(f"   Constraints: {os.path.basename(sdc_filename)}")
        
        assessment = oracle.predict_physical_risks(
            rtl_file=rtl_filename,
            constraints_file=sdc_filename,
            node="7nm"
        )
        
        # Display results
        print(f"\n3. Physical Risk Assessment Results:")
        print(f"   - Congestion risk areas: {len(assessment.congestion_heatmap)}")
        print(f"   - Timing risk zones: {len(assessment.timing_risk_zones)}")
        print(f"   - Clock sensitivity issues: {len(assessment.clock_skew_sensitivity)}")
        print(f"   - Power hotspots: {len(assessment.power_density_hotspots)}")
        print(f"   - DRC risk classes: {len(assessment.drc_risk_classes)}")
        print(f"   - Overall confidence: {assessment.overall_confidence:.2f}")
        
        print(f"\n4. Recommendations:")
        for i, rec in enumerate(assessment.recommendations[:5], 1):  # Show first 5
            print(f"   {i}. {rec}")
        
        # Demonstrate agent system
        print(f"\n5. Demonstrating Agent System...")
        
        # Create agents
        floorplan_agent = FloorplanAgent()
        placement_agent = PlacementAgent()
        clock_agent = ClockAgent()
        
        # Create negotiator and register agents
        negotiator = AgentNegotiator()
        negotiator.register_agent(floorplan_agent)
        negotiator.register_agent(placement_agent)
        negotiator.register_agent(clock_agent)
        
        # For this example, we'll create a minimal graph to demonstrate
        # In a real scenario, this would come from the oracle's analysis
        print("   Agents registered successfully")
        print("   NOTE: Full agent negotiation requires a populated Canonical Silicon Graph")
        
        print(f"\n6. System initialized successfully!")
        print("   The Silicon Intelligence System is ready for RTL-to-GDSII implementation.")
        
    finally:
        # Clean up temporary files
        os.unlink(rtl_filename)
        os.unlink(sdc_filename)


if __name__ == "__main__":
    main()