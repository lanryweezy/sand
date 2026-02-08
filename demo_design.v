
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
