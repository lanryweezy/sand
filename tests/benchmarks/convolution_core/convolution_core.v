
// Convolution Core Benchmark
module conv_core (
    input clk,
    input rst_n,
    input [7:0] pixel_data [0:8],  // 3x3 window
    input [7:0] kernel_weights [0:8],
    output [15:0] conv_result
);

    reg [15:0] products [0:8];
    reg [15:0] accumulator;
    
    integer i;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= 16'b0;
        end
        else begin
            // Compute products
            for (i = 0; i < 9; i = i + 1) begin
                products[i] <= pixel_data[i] * kernel_weights[i];
            end
            
            // Accumulate
            accumulator <= 0;
            for (i = 0; i < 9; i = i + 1) begin
                accumulator <= accumulator + products[i];
            end
        end
    end
    
    assign conv_result = accumulator;

endmodule
