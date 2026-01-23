
// Tensor Processor Benchmark
module tensor_proc (
    input clk,
    input rst_n,
    input start,
    output reg done,
    input [31:0] tensor_a [0:63],
    input [31:0] tensor_b [0:63],
    output [31:0] result [0:63]
);

    reg [31:0] temp_result [0:63];
    reg [5:0] counter;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            counter <= 0;
            done <= 0;
        end
        else if (start && counter < 64) begin
            temp_result[counter] <= tensor_a[counter] * tensor_b[counter];
            counter <= counter + 1;
            if (counter == 63) done <= 1;
        end
        else if (done && !start) begin
            done <= 0;
            counter <= 0;
        end
    end
    
    assign result = temp_result;

endmodule
