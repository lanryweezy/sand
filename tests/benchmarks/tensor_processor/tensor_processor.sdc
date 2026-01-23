
# AI Accelerator Constraints
create_clock -name core_clk -period 2.000 -waveform {0.000 1.000} [get_ports clk]

set_clock_uncertainty -setup 0.02 [get_clocks core_clk]
set_clock_uncertainty -hold 0.01 [get_clocks core_clk]

set_input_delay -clock core_clk -max 0.500 [all_inputs]
set_output_delay -clock core_clk -max 0.600 [all_outputs]

# False path for reset
set_false_path -from [get_ports rst_n]

# High fanout clocks (MAC array clocks)
set_clock_transition 0.1 [get_clocks core_clk]
