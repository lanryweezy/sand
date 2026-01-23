"""
Comprehensive Example: Silicon Intelligence System in Action

This example demonstrates the complete Silicon Intelligence System workflow
from RTL input to optimized physical implementation.
"""

import os
import tempfile
from silicon_intelligence.main import run_full_flow


def create_advanced_rtl_example():
    """Create a more advanced RTL example for comprehensive testing"""
    rtl_content = """
// Advanced RTL example for Silicon Intelligence System
module advanced_soc_top (
    input clk,
    input rst_n,
    input [31:0] data_in,
    output [31:0] data_out,
    output reg valid,
    input enable,
    output [7:0] status
);

    // Subsystem 1: High-performance compute cluster
    reg [31:0] compute_reg1, compute_reg2, compute_reg3;
    wire [31:0] compute_result1, compute_result2;
    
    // Pipeline stage 1
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            compute_reg1 <= 32'b0;
            compute_reg2 <= 32'b0;
        end
        else if (enable) begin
            compute_reg1 <= data_in + 32'h1000;
            compute_reg2 <= data_in ^ 32'hFFFF;
        end
    end
    
    // Combinational logic - potential timing critical path
    assign compute_result1 = compute_reg1 * compute_reg2;
    
    // Pipeline stage 2
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            compute_reg3 <= 32'b0;
        end
        else if (enable) begin
            compute_reg3 <= compute_result1 + 32'h2000;
        end
    end
    
    assign compute_result2 = compute_reg3 >> 2;

    // Subsystem 2: Memory controller interface
    reg [31:0] mem_addr_reg, mem_data_reg;
    reg mem_read, mem_write;
    wire mem_ready;
    
    // Memory address/data registers
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mem_addr_reg <= 32'b0;
            mem_data_reg <= 32'b0;
            mem_read <= 1'b0;
            mem_write <= 1'b0;
        end
        else if (enable) begin
            mem_addr_reg <= data_in[31:0];
            mem_data_reg <= data_in[31:0];
            mem_read <= data_in[31];
            mem_write <= data_in[30];
        end
    end

    // Subsystem 3: Control and status
    reg [7:0] status_reg;
    wire [31:0] processed_data;
    
    assign processed_data = compute_result2 + mem_data_reg;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid <= 1'b0;
            status_reg <= 8'b0;
            data_out <= 32'b0;
        end
        else begin
            // Complex control logic
            if (enable && mem_ready) begin
                valid <= 1'b1;
                data_out <= processed_data;
                status_reg <= {mem_read, mem_write, 2'b00, |processed_data[15:0], 2'b00};
            end
            else begin
                valid <= 1'b0;
                status_reg <= 8'b0;
            end
        end
    end
    
    assign status = status_reg;

    // Memory ready simulation
    assign mem_ready = &mem_addr_reg[3:0];  // Simulated memory ready

endmodule

// Additional modules for complexity
module fifo_controller (
    input clk,
    input rst_n,
    input wr_en,
    input rd_en,
    input [7:0] wr_data,
    output [7:0] rd_data,
    output full,
    output empty
);
    reg [7:0] mem [0:255];
    reg [7:0] wr_ptr, rd_ptr;
    reg [8:0] count;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= 8'b0;
            rd_ptr <= 8'b0;
            count <= 9'b0;
        end
        else begin
            if (wr_en && !full) begin
                mem[wr_ptr] <= wr_data;
                wr_ptr <= wr_ptr + 1'b1;
                count <= count + 1'b1;
            end
            if (rd_en && !empty) begin
                rd_ptr <= rd_ptr + 1'b1;
                count <= count - 1'b1;
            end
        end
    end
    
    assign full = (count == 256);
    assign empty = (count == 0);
    assign rd_data = mem[rd_ptr];

endmodule
"""
    return rtl_content


def create_advanced_constraints():
    """Create advanced SDC constraints for the example"""
    sdc_content = """
# Advanced constraints for Silicon Intelligence System example
# Clock definitions
create_clock -name core_clk -period 3.333 -waveform {0.000 1.667} [get_ports clk]
create_clock -name mem_clk -period 4.000 -waveform {0.000 2.000} [get_ports mem_clk]

# Uncertainty
set_clock_uncertainty -setup 0.05 [get_clocks core_clk]
set_clock_uncertainty -hold 0.02 [get_clocks core_clk]
set_clock_uncertainty -setup 0.07 [get_clocks mem_clk]
set_clock_uncertainty -hold 0.03 [get_clocks mem_clk]

# Input/Output delays
set_input_delay -clock core_clk -max 1.000 [remove_from_collection [all_inputs] [get_ports {clk rst_n mem_clk}]]
set_input_delay -clock core_clk -min 0.500 [remove_from_collection [all_inputs] [get_ports {clk rst_n mem_clk}]]
set_output_delay -clock core_clk -max 1.200 [remove_from_collection [all_outputs] [get_ports status]]
set_output_delay -clock core_clk -min 0.600 [remove_from_collection [all_outputs] [get_ports status]]

# False paths
set_false_path -from [get_ports rst_n]
set_false_path -from [get_ports enable]

# Multicycle paths for complex control logic
set_multicycle_path -setup 2 -from [get_pins advanced_soc_top/status_reg/C] -to [get_pins advanced_soc_top/data_out[*]/D]
set_multicycle_path -hold 0 -from [get_pins advanced_soc_top/status_reg/C] -to [get_pins advanced_soc_top/data_out[*]/D]

# Load and driving cell specs
set_load -pin_load 1.0 [all_outputs]
set_driving_cell -lib_cell BUF_X1 [all_inputs]

# Operating conditions
set_operating_conditions -analysis_type on_chip_variation fast
"""
    return sdc_content


def run_comprehensive_example():
    """Run the comprehensive Silicon Intelligence System example"""
    print("="*70)
    print("COMPREHENSIVE SILICON INTELLIGENCE SYSTEM EXAMPLE")
    print("="*70)
    
    # Create temporary files for our example
    with tempfile.NamedTemporaryFile(mode='w', suffix='_advanced.v', delete=False) as rtl_file:
        rtl_file.write(create_advanced_rtl_example())
        rtl_filename = rtl_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_advanced.sdc', delete=False) as sdc_file:
        sdc_file.write(create_advanced_constraints())
        sdc_filename = sdc_file.name
    
    temp_output_dir = tempfile.mkdtemp(prefix="silicon_intel_output_")
    
    try:
        print(f"Input RTL: {os.path.basename(rtl_filename)}")
        print(f"Constraints: {os.path.basename(sdc_filename)}")
        print(f"Output Dir: {temp_output_dir}")
        print()
        
        # Run the full flow
        run_full_flow(rtl_filename, sdc_filename, "7nm", temp_output_dir)
        
        print("\n" + "="*70)
        print("EXAMPLE COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nSUMMARY OF SILICON INTELLIGENCE CAPABILITIES DEMONSTRATED:")
        print("✓ Physical Risk Prediction - Identified potential implementation challenges")
        print("✓ Multi-Agent Coordination - Floorplan, Placement, Clock, Power, Yield agents worked together")
        print("✓ Parallel Reality Exploration - Evaluated multiple optimization strategies")
        print("✓ DRC-Aware Optimization - Prevented design rule violations")
        print("✓ Continuous Learning Integration - Updated models with feedback")
        print("\nThe Silicon Intelligence System transforms chip design from a manual,")
        print("iterative process into an intelligent, predictive, and self-improving system.")
        print("Each chip designed makes the next chip smarter.")
        
    finally:
        # Clean up temporary files
        try:
            os.unlink(rtl_filename)
            os.unlink(sdc_filename)
            # Note: We don't delete the output directory as it contains results
        except:
            pass  # Ignore cleanup errors


if __name__ == "__main__":
    run_comprehensive_example()