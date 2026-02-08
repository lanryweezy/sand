module SH_RISCV_FETCH_V1 (
    input clk,
    input reset_n,
    input stall,
    input [31:0] pc_target,
    input pc_src_sel,
    output reg [31:0] pc_out,
    output [31:0] pc_plus_4
);

    assign pc_plus_4 = pc_out + 4;

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) pc_out <= 32'h8000_0000;
        else if (!stall) begin
            pc_out <= pc_src_sel ? pc_target : pc_plus_4;
        end
    end
endmodule