# Sample SDC constraints file for testing

# Create main clock
create_clock -period 10.0 -name clk clk

# Create secondary clock
create_clock -period 20.0 -name clk2 clk2

# Set input delays
set_input_delay -clock clk -max 2.0 data_a
set_input_delay -clock clk -max 2.0 data_b
set_input_delay -clock clk -max 1.0 enable

# Set output delays
set_output_delay -clock clk -max 3.0 counter_out
set_output_delay -clock clk -max 3.0 sum_out

# Set timing paths
set_max_delay -from data_a -to sum_out 8.0
set_max_delay -from data_b -to sum_out 8.0

# Set false paths
set_false_path -from rst -to counter_out
