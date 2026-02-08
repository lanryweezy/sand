module synth_core_5 (
    input clk,
    input [31:0] in_a,
    input [31:0] in_b,
    input [31:0] in_c,
    input sel,
    output reg [31:0] result
);

    wire [31:0] w_0;
    assign w_0 = in_b * in_c;
    wire [31:0] w_1;
    assign w_1 = in_a * w_0;
    wire [31:0] w_2;
    assign w_2 = w_1 ^ in_a;
    wire [31:0] w_3;
    assign w_3 = in_b * w_0;
    wire [31:0] w_4;
    assign w_4 = in_c ^ in_c;
    wire [31:0] w_5;
    assign w_5 = w_2 + in_c;
    wire [31:0] w_6;
    assign w_6 = in_b ^ w_3;
    wire [31:0] w_7;
    assign w_7 = w_5 + in_b;
    wire [31:0] w_8;
    assign w_8 = w_0 + w_6;

    always @(posedge clk) begin
        result <= w_8;
    end

endmodule