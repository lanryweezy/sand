module empire_core (
    input clk,
    input [63:0] in_a,
    input [63:0] in_b,
    input [63:0] in_c,
    input sel,
    output reg [63:0] result
);

    wire [63:0] w_0;
    assign w_0 = in_c ^ in_b;
    wire [63:0] w_1;
    assign w_1 = in_c + w_0;
    wire [63:0] w_2;
    assign w_2 = in_a * in_a;
    wire [63:0] w_3;
    assign w_3 = sel ? in_b : w_0;
    wire [63:0] w_4;
    assign w_4 = sel ? w_2 : in_c;
    wire [63:0] w_5;
    assign w_5 = sel ? in_a : w_2;
    wire [63:0] w_6;
    assign w_6 = w_3 ^ w_4;
    wire [63:0] w_7;
    assign w_7 = w_2 + in_a;
    wire [63:0] w_8;
    assign w_8 = sel ? w_6 : w_0;
    wire [63:0] w_9;
    assign w_9 = in_c * in_c;
    wire [63:0] w_10;
    assign w_10 = w_7 + w_9;
    wire [63:0] w_11;
    assign w_11 = w_5 * w_6;

    always @(posedge clk) begin
        result <= w_11;
    end

endmodule