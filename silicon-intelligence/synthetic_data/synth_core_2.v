module synth_core_2 (
    input clk,
    input [31:0] in_a,
    input [31:0] in_b,
    input [31:0] in_c,
    input sel,
    output reg [31:0] result
);

    wire [31:0] w_0;
    assign w_0 = sel ? in_c : in_c;
    wire [31:0] w_1;
    assign w_1 = in_c ^ in_c;
    wire [31:0] w_2;
    assign w_2 = in_a + in_c;
    wire [31:0] w_3;
    assign w_3 = in_c * w_1;

    always @(posedge clk) begin
        result <= w_3;
    end

endmodule