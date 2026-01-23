module synth_core_0 (
    input clk,
    input [31:0] in_a,
    input [31:0] in_b,
    input [31:0] in_c,
    input sel,
    output reg [31:0] result
);

    wire [31:0] w_0;
    assign w_0 = in_a ^ in_b;
    wire [31:0] w_1;
    assign w_1 = in_a * in_c;
    wire [31:0] w_2;
    assign w_2 = in_c + in_b;
    wire [31:0] w_3;
    assign w_3 = w_0 + w_0;
    wire [31:0] w_4;
    assign w_4 = sel ? in_c : in_b;
    wire [31:0] w_5;
    assign w_5 = w_2 * w_0;
    wire [31:0] w_6;
    assign w_6 = w_2 ^ in_c;
    wire [31:0] w_7;
    assign w_7 = w_3 + w_1;

    always @(posedge clk) begin
        result <= w_7;
    end

endmodule