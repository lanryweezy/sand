
module pipelined_adder_14bit (
    input clk,
    input rst_n,
    input [13:0] a,
    input [13:0] b,
    output reg [14:0] sum
);
    reg [14:0] pipe_stage [ ];
    integer i;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            for (i = 0; i < 2; i = i + 1)
                pipe_stage[i] <= { 15{1'b0} };
        end else begin
            pipe_stage[0] <= a + b;
            for (i = 1; i < 2; i = i + 1)
                pipe_stage[i] <= pipe_stage[i-1];
        end
    end
    
    assign sum = pipe_stage[1];
endmodule
