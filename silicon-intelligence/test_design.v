
module test_engine (
    input clk,
    input [7:0] data_in,
    output [7:0] data_out
);
    wire [7:0] processed_data;
    assign processed_data = data_in ^ 8'hFF;
    assign data_out = processed_data;
endmodule
