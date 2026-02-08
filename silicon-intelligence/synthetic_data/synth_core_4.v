module synth_core_4 (
    input clk,
    input [31:0] in_a,
    input [31:0] in_b,
    input [31:0] in_c,
    input sel,
    output reg [31:0] result
);

    wire [31:0] w_0;
    assign w_0 = sel ? in_b : in_a;
    wire [31:0] w_1;
    assign w_1 = in_b + w_0;
    wire [31:0] w_2;
    assign w_2 = in_c + w_1;
    wire [31:0] w_3;
    assign w_3 = sel ? w_0 : w_1;
    wire [31:0] w_4;
    assign w_4 = w_2 + w_0;
    wire [31:0] w_5;
    assign w_5 = in_b * w_3;
    wire [31:0] w_6;
    assign w_6 = sel ? w_4 : in_c;

    always @(posedge clk) begin
        result <= w_6;
    end

endmodule