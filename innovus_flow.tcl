# --- Innovus Physical Design Flow (SkyWater Aware) ---
set_multi_cpu_usage -local_cpu 4
set_db design_process_node 130

# 1. Import Design
set_db init_lef_file "/usr/local/pdk\sky130A/libs.ref\sky130_fd_sc_hd/lef/sky130_fd_sc_hd.merged.lef"
set_db init_verilog "optimized_pro_alu.v"
set_db init_top_cell "pro_alu"
init_design
read_sdc "constraints/pro.sdc"

# 2. Common Setup
set_db connect_global_net VDD -type pg_pin -pin_base_name VDD
set_db connect_global_net VSS -type pg_pin -pin_base_name VSS

# 3. Floorplan
create_floorplan -core_density 0.7 -aspect_ratio 1.0

# 4. Placement
set_db place_design_style indigenous
place_design
check_place

# 5. Clock Tree Synthesis (CTS)
create_ccopt_clock_tree_spec
ccopt_design

# 6. Routing
set_db route_design_detail_post_route_spread_wire false
route_design -global -detail

# 7. Reporting
report_timing > timing_report.txt
report_power > power_report.txt
report_area > area_report.txt
exit