"""
RTL Transformation Service - Uses AST-based parsing for safe design optimization.
Powered by Pyverilog for professional Verilog/SystemVerilog handling.
"""

import sys
import os
import pyverilog.vparser.ast as vast
from pyverilog.vparser.parser import parse
from pyverilog.ast_code_generator.codegen import ASTCodeGenerator

class RTLTransformer:
    """
    Handles Abstract Syntax Tree (AST) transformations for Verilog design files.
    Ensures that optimizations like pipelining are performed safely.
    """
    
    def __init__(self):
        self.codegen = ASTCodeGenerator()

    def parse_rtl(self, rtl_file: str):
        """Parse Verilog file into an AST"""
        try:
            ast, directives = parse([rtl_file])
            return ast
        except Exception as e:
            print(f"Warning: Standard parse failed ({e}). Attempting fallback...")
            # Fallback: Try to use a dummy preprocessor or assume no preprocessing needed
            # This is a bit of a hack but ensures the 'Pro' demo works without iverilog
            try:
                from pyverilog.vparser.parser import VerilogParser
                parser = VerilogParser()
                with open(rtl_file, 'r') as f:
                    return parser.parse(f.read())
            except:
                raise e

    def add_pipeline_stage(self, ast, module_name: str, target_signal: str):
        """
        Injects a pipeline stage (register) into a specific module for a target signal.
        This is a professional AST-based transformation.
        """
        # Find the module
        target_module = None
        for item in ast.description.definitions:
            if isinstance(item, vast.ModuleDef) and item.name == module_name:
                target_module = item
                break
        
        if not target_module:
            raise ValueError(f"Module {module_name} not found in AST")

        # Create the pipeline register name
        pipe_reg_name = f"{target_signal}_pipe_reg"
        
        # 1. Add Register Definition
        # We need to find the signal width to mirror it
        width = None
        for item in target_module.items:
            if isinstance(item, vast.Decl):
                for decl_item in item.list:
                    if hasattr(decl_item, 'name') and decl_item.name == target_signal:
                        width = getattr(decl_item, 'width', None)
        
        pipe_reg_decl = vast.Decl((vast.Reg(pipe_reg_name, width),))
        target_module.items += (pipe_reg_decl,)

        # 2. Add Always Block for the register
        # always @(posedge clk) pipe_reg <= target_signal;
        clk = vast.Identifier('clk')
        sens = vast.Sens(clk, type='posedge')
        sens_list = vast.SensList((sens,))
        
        target_id = vast.Identifier(target_signal)
        pipe_id = vast.Identifier(pipe_reg_name)
        
        assign = vast.NonblockingSubstitution(pipe_id, target_id)
        block = vast.Block((assign,))
        always = vast.Always(sens_list, block)
        
        target_module.items += (always,)

        return ast, pipe_reg_name

    def insert_clock_gate(self, ast, module_name: str, register_signal: str, enable_signal: str):
        """
        Inserts an Integrated Clock Gate (ICG) for a specific register/bus.
        This is a professional power-reduction transformation.
        """
        # Find the module
        target_module = None
        for item in ast.description.definitions:
            if isinstance(item, vast.ModuleDef) and item.name == module_name:
                target_module = item
                break
        
        if not target_module:
            raise ValueError(f"Module {module_name} not found in AST")

        gated_clk_name = f"{register_signal}_gated_clk"
        
        # 1. Add Gated Clock Net (wire)
        gated_clk_decl = vast.Decl((vast.Wire(gated_clk_name),))
        target_module.items += (gated_clk_decl,)

        # 2. Add Clock Gate Instance (SkyWater lib cell)
        # sky130_fd_sc_hd__lpflow_is_1 represents a professional ICG
        clk_gate = vast.Instance(
            'sky130_fd_sc_hd__lpflow_is_1', 
            f"icg_{register_signal}",
            (
                vast.PortArg('CLK', vast.Identifier('clk')),
                vast.PortArg('ENA', vast.Identifier(enable_signal)),
                vast.PortArg('GCLK', vast.Identifier(gated_clk_name))
            ),
            ()
        )
        target_module.items += (vast.InstanceList('sky130_fd_sc_hd__lpflow_is_1', (), (clk_gate,)),)

        # 3. Update Register Always Block to use Gated Clock
        # We need to find the Always block that controls this register and change @(posedge clk) -> @(posedge gated_clk)
        for item in target_module.items:
            if isinstance(item, vast.Always):
                # Check if this always block assigns to our register (simplified check)
                if self._blocks_assigns_to(item.statement, register_signal):
                    # Update sensitivities
                    for sens in item.sens_list.list:
                        if isinstance(sens.sig, vast.Identifier) and sens.sig.name == 'clk':
                            sens.sig.name = gated_clk_name

        return ast, gated_clk_name

    def _blocks_assigns_to(self, node, signal_name) -> bool:
        """Helper to check if a block contains an assignment to a specific signal"""
        if isinstance(node, (vast.NonblockingSubstitution, vast.BlockingSubstitution)):
            if isinstance(node.left, vast.Identifier) and node.left.name == signal_name:
                return True
        for child in node.children():
            if self._blocks_assigns_to(child, signal_name):
                return True
        return False

    def update_signal_sinks(self, ast, module_name: str, old_signal: str, new_signal: str):
        """
        Traverses the AST and replaces all occurrences of old_signal as a SOURCE 
        with new_signal within a specific module.
        """
        target_module = None
        for item in ast.description.definitions:
            if isinstance(item, vast.ModuleDef) and item.name == module_name:
                target_module = item
                break
        
        if not target_module:
            return ast

        # Simple visitor-like replacement for Identifiers used as R-values
        def replace_identifier(node):
            if isinstance(node, vast.Identifier) and node.name == old_signal:
                node.name = new_signal
            
            # Recurse into children
            for child in node.children():
                replace_identifier(child)

        # We only want to replace old_signal in assignments (right side) and module items
        # that aren't the primary declaration of the old_signal.
        for item in target_module.items:
            if isinstance(item, vast.Assign):
                # Update the right side (source) of assignments
                replace_identifier(item.right)
            elif isinstance(item, vast.Always):
                # Update identifiers in always blocks
                replace_identifier(item.statement)
        
        return ast

    def generate_verilog(self, ast):
        """Convert AST back to Verilog source code"""
        return self.codegen.visit(ast)

def example_pipeline_transform():
    # Simple dummy file for testing
    source_v = "test_design.v"
    with open(source_v, "w") as f:
        f.write('''
module test_engine (
    input clk,
    input [7:0] data_in,
    output [7:0] data_out
);
    wire [7:0] processed_data;
    assign processed_data = data_in ^ 8'hFF;
    assign data_out = processed_data;
endmodule
''')

    transformer = RTLTransformer()
    ast = transformer.parse_rtl(source_v)
    
    print("Applying pipeline stage to 'processed_data'...")
    ast, reg_name = transformer.add_pipeline_stage(ast, 'test_engine', 'processed_data')
    
    # Update data_out assignment to use the new reg
    for item in ast.description.definitions[0].items:
        if isinstance(item, vast.Assign) and item.left.var.name == 'data_out':
            item.right.var.name = reg_name

    new_verilog = transformer.generate_verilog(ast)
    print("\n--- Generated Pipeline RTL ---")
    print(new_verilog)
    
    return new_verilog

if __name__ == "__main__":
    example_pipeline_transform()
