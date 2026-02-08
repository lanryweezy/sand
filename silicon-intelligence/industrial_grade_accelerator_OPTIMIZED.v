

module crypto_accelerator_pro
(
  input clk,
  input rst_n,
  input en,
  input [31:0] data_in_a,
  input [31:0] data_in_b,
  input [31:0] data_in_c,
  output reg [63:0] data_out
);

  wire en_internal;
  assign en_internal = en & rst_n;
  wire [63:0] mult_result_1;
  wire [63:0] mult_result_2;
  assign mult_result_1 = data_in_a_isolated * data_in_b_isolated;
  wire [63:0] stage1_result;
  assign stage1_result = mult_result_1 + { 32'b0, data_in_c };
  reg [63:0] pipe1;
  reg [63:0] pipe2;
  reg [63:0] pipe3;
  reg [63:0] pipe4;
  reg [63:0] pipe5;
  reg [63:0] pipe6;
  reg [63:0] pipe7;
  reg [63:0] pipe8;

  always @(posedge clk_buf_0 or negedge rst_n) begin
    if(!rst_n) begin
      pipe1 <= 64'b0;
      pipe2 <= 64'b0;
      pipe3 <= 64'b0;
      pipe4 <= 64'b0;
      pipe5 <= 64'b0;
      pipe6 <= 64'b0;
      pipe7 <= 64'b0;
      pipe8 <= 64'b0;
      data_out <= 64'b0;
    end else if(en_internal) begin
      pipe1 <= stage1_result;
      pipe2 <= mult_result_1 + pipe1;
      pipe3 <= pipe2 ^ { data_in_a_isolated, data_in_b_isolated };
      pipe4 <= pipe3 + pipe2;
      pipe5 <= pipe4 | pipe1;
      pipe6 <= pipe5 & pipe2;
      pipe7 <= pipe6 << 2;
      pipe8 <= pipe7 ^ pipe4;
      data_out <= pipe8;
    end 
  end

  wire [31:0] data_in_a_isolated;
  assign data_in_a_isolated = (en)? data_in_a : 32'b0;
  wire [31:0] data_in_b_isolated;
  assign data_in_b_isolated = (en)? data_in_b : 32'b0;
  wire clk_buf_0;
  assign clk_buf_0 = clk_buf_1;
  wire clk_buf_1;
  assign clk_buf_1 = clk_buf_2;
  wire clk_buf_2;
  assign clk_buf_2 = clk_buf_3;
  wire clk_buf_3;
  assign clk_buf_3 = clk_buf_4;
  wire clk_buf_4;
  assign clk_buf_4 = clk_buf_0;

endmodule

