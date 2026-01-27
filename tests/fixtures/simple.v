// Simple counter module for testing RTL parser
module counter (
    input clk,
    input rst,
    input enable,
    output [7:0] count
);

    reg [7:0] count_reg;
    
    always @(posedge clk) begin
        if (rst)
            count_reg <= 8'b0;
        else if (enable)
            count_reg <= count_reg + 1;
    end
    
    assign count = count_reg;

endmodule

// Simple adder module
module adder (
    input [7:0] a,
    input [7:0] b,
    output [8:0] sum
);

    assign sum = a + b;

endmodule

// Top module that instantiates counter and adder
module top (
    input clk,
    input rst,
    input enable,
    input [7:0] data_a,
    input [7:0] data_b,
    output [7:0] counter_out,
    output [8:0] sum_out
);

    wire [7:0] count_internal;
    
    // Instantiate counter
    counter u_counter (
        .clk(clk),
        .rst(rst),
        .enable(enable),
        .count(count_internal)
    );
    
    // Instantiate adder
    adder u_adder (
        .a(data_a),
        .b(data_b),
        .sum(sum_out)
    );
    
    assign counter_out = count_internal;

endmodule
