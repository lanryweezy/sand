
// MAC Array Benchmark - 32x32
module mac_array (
    input clk,
    input rst_n,
    input [255:0] a_data,
    input [255:0] b_data,
    output [16383:0] result
);

    reg [15:0] mac_results [1023:0];
    
    genvar i, j;
    generate
        for (i = 0; i < 32; i = i + 1) begin : row_gen
            for (j = 0; j < 32; j = j + 1) begin : col_gen
                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        mac_results[i*32 + j] <= 16'b0;
                    end
                    else begin
                        mac_results[i*32 + j] <= a_data[j*8+:8] * b_data[i*8+:8];
                    end
                end
            end
        end
    endgenerate
    
    assign result = {mac_results};
    
endmodule
