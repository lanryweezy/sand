
module final_boss (
    input clk,
    input [63:0] data_in_a,
    input [63:0] data_in_b,
    output [63:0] result
);
    wire [63:0] high_latency_net;
    assign high_latency_net = data_in_a * data_in_b; // Mult-cycle logic
    assign result = high_latency_net;
endmodule
