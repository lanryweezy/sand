module ULTIMATE_AI_CORE (
    input clk,
    input reset_n,
    input [63:0] input_a,
    input [63:0] input_b,
    output reg [128-1:0] mac_out
);

    wire [128-1:0] product = input_a * input_b;
    reg [128-1:0] p_stage_0;
    reg [128-1:0] p_stage_1;
    reg [128-1:0] p_stage_2;
    reg [128-1:0] p_stage_3;

    always @(posedge clk) begin
        p_stage_0 <= product;
        p_stage_1 <= p_stage_0;
        p_stage_2 <= p_stage_1;
        p_stage_3 <= p_stage_2;
        mac_out <= mac_out + p_stage_3;
    end

endmodule