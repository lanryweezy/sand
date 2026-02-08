

module final_boss
(
  input clk,
  input [63:0] data_in_a,
  input [63:0] data_in_b,
  output [63:0] result
);

  wire [63:0] high_latency_net;
  assign high_latency_net = data_in_a * data_in_b;
  assign result = high_latency_net_pipe_reg;
  reg [63:0] high_latency_net_pipe_reg;

  always @(posedge clk) begin
    high_latency_net_pipe_reg <= high_latency_net_pipe_reg;
  end


endmodule

