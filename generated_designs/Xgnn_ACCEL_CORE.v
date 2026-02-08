module Xgnn_ACCEL_CORE (
    input clk,
    input reset_n,
    input [127:0] input_a,
    input [127:0] input_b,
    output reg [256-1:0] mac_out
);

    wire [256-1:0] product = input_a * input_b;
    reg [256-1:0] p_stage_0;
    reg [256-1:0] p_stage_1;
    reg [256-1:0] p_stage_2;
    reg [256-1:0] p_stage_3;

    always @(posedge clk) begin
        p_stage_0 <= product;
        p_stage_1 <= p_stage_0;
        p_stage_2 <= p_stage_1;
        p_stage_3 <= p_stage_2;
        mac_out <= mac_out + p_stage_3;
    end

endmodule