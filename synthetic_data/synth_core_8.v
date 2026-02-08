module synth_core_8 (
    input clk,
    input [31:0] in_a,
    input [31:0] in_b,
    input [31:0] in_c,
    input sel,
    output reg [31:0] result
);

    wire [31:0] w_0;
    assign w_0 = in_c * in_b;
    wire [31:0] w_1;
    assign w_1 = in_a ^ w_0;
    wire [31:0] w_2;
    assign w_2 = in_b + in_a;

    always @(posedge clk) begin
        result <= w_2;
    end

endmodule