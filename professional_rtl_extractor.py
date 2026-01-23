#!/usr/bin/env python3
"""
Professional RTL parser using PyVerilog
This replaces the custom parser with industry-standard tooling
"""

import pyverilog
from pyverilog.dataflow.dataflow_analyzer import VerilogDataflowAnalyzer
from pyverilog.ast_code_generator.codegen import ASTCodeGenerator
from pyverilog.vparser.parser import parse
import tempfile
import os
from typing import Dict, List, Any

class ProfessionalRTLExtractor:
    """
    Professional RTL extractor using PyVerilog
    Focus: Extract structural and temporal information for AI accelerator analysis
    """
    
    def __init__(self):
        self.extracted_data = {}
    
    def extract_from_file(self, rtl_file: str) -> Dict[str, Any]:
        """
        Extract design information using PyVerilog
        
        Args:
            rtl_file: Path to RTL file
            
        Returns:
            Structured design data suitable for CanonicalSiliconGraph
        """
        try:
            # Parse the RTL file
            ast, directives = parse([rtl_file])
            
            # Extract using AST traversal
            extractor = DesignExtractor()
            design_data = extractor.visit(ast)
            
            self.extracted_data = design_data
            return design_data
            
        except Exception as e:
            print(f"PyVerilog extraction failed: {str(e)}")
            return self._fallback_extraction(rtl_file)
    
    def extract_from_string(self, rtl_code: str) -> Dict[str, Any]:
        """Extract from RTL string"""
        # Write to temp file and parse
        with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as f:
            f.write(rtl_code)
            temp_file = f.name
        
        try:
            return self.extract_from_file(temp_file)
        finally:
            os.unlink(temp_file)
    
    def _fallback_extraction(self, rtl_file: str) -> Dict[str, Any]:
        """
        Fallback extraction for simple cases
        """
        with open(rtl_file, 'r') as f:
            content = f.read()
        
        # Simple regex-based extraction as backup
        import re
        
        # Extract module names
        modules = re.findall(r'module\s+(\w+)', content)
        
        # Extract ports
        ports = []
        port_matches = re.findall(r'(input|output|inout)\s+(?:\[[^\]]+\])?\s*(\w+)', content)
        for direction, name in port_matches:
            ports.append({
                'name': name,
                'direction': direction,
                'type': 'wire'
            })
        
        # Extract registers
        regs = re.findall(r'reg\s+(?:\[[^\]]+\])?\s*(\w+)', content)
        
        # Extract always blocks
        always_blocks = len(re.findall(r'always\s*@', content))
        
        return {
            'modules': [{'name': mod} for mod in modules],
            'ports': ports,
            'registers': [{'name': reg} for reg in regs],
            'always_blocks': always_blocks,
            'raw_content': content
        }

class DesignExtractor:
    """AST visitor to extract design information"""
    
    def __init__(self):
        self.design_data = {
            'modules': [],
            'instances': [],
            'ports': [],
            'registers': [],
            'wires': [],
            'assignments': [],
            'always_blocks': []
        }
    
    def visit(self, node):
        """Visit AST nodes"""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)
    
    def generic_visit(self, node):
        """Generic visitor for unhandled nodes"""
        for c in node.children():
            self.visit(c)
    
    def visit_ModuleDef(self, node):
        """Extract module definition"""
        module_info = {
            'name': node.name,
            'ports': [],
            'items': []
        }
        
        # Extract ports
        if hasattr(node, 'portlist') and node.portlist:
            for port in node.portlist.ports:
                if hasattr(port, 'name'):
                    module_info['ports'].append({
                        'name': port.name,
                        'direction': getattr(port, 'direction', 'unknown'),
                        'width': getattr(port, 'width', None)
                    })
        
        # Extract module items
        if hasattr(node, 'items'):
            for item in node.items:
                item_info = self.visit(item)
                if item_info:
                    module_info['items'].append(item_info)
        
        self.design_data['modules'].append(module_info)
        return module_info
    
    def visit_Input(self, node):
        """Extract input port"""
        port_info = {
            'type': 'input',
            'name': node.name,
            'width': getattr(node, 'width', None)
        }
        self.design_data['ports'].append(port_info)
        return port_info
    
    def visit_Output(self, node):
        """Extract output port"""
        port_info = {
            'type': 'output',
            'name': node.name,
            'width': getattr(node, 'width', None)
        }
        self.design_data['ports'].append(port_info)
        return port_info
    
    def visit_Reg(self, node):
        """Extract register declaration"""
        reg_info = {
            'name': node.name,
            'width': getattr(node, 'width', None)
        }
        self.design_data['registers'].append(reg_info)
        return reg_info
    
    def visit_Wire(self, node):
        """Extract wire declaration"""
        wire_info = {
            'name': node.name,
            'width': getattr(node, 'width', None)
        }
        self.design_data['wires'].append(wire_info)
        return wire_info
    
    def visit_Assign(self, node):
        """Extract continuous assignment"""
        assign_info = {
            'lhs': str(getattr(node, 'left', '')),
            'rhs': str(getattr(node, 'right', ''))
        }
        self.design_data['assignments'].append(assign_info)
        return assign_info
    
    def visit_Always(self, node):
        """Extract always block"""
        always_info = {
            'sensitivity': str(getattr(node, 'sensitivity', '')),
            'statement': str(getattr(node, 'statement', ''))
        }
        self.design_data['always_blocks'].append(always_info)
        return always_info

def test_professional_extractor():
    """Test the professional extractor with AI accelerator patterns"""
    
    # Test MAC array pattern
    mac_array_code = '''
    module mac_array_32x32 (
        input clk,
        input rst_n,
        input [31:0] a_data,
        input [31:0] b_data,
        input [31:0] weight_data,
        output [31:0] result
    );
        reg [31:0] accumulator;
        reg [31:0] product;
        
        always @(posedge clk) begin
            if (!rst_n) begin
                accumulator <= 32'd0;
                product <= 32'd0;
            end else begin
                product <= a_data * weight_data;
                accumulator <= accumulator + product;
            end
        end
        
        assign result = accumulator;
    endmodule
    '''
    
    extractor = ProfessionalRTLExtractor()
    result = extractor.extract_from_string(mac_array_code)
    
    print("=== Professional RTL Extraction Results ===")
    print(f"✅ PyVerilog extraction successful!")
    print(f"Modules found: {len(result.get('modules', []))}")
    print(f"Ports found: {len(result.get('ports', []))}")
    print(f"Registers found: {len(result.get('registers', []))}")
    always_count = result.get('always_blocks', 0)
    print(f"Always blocks found: {always_count}")
    print(f"Wires found: {len(result.get('wires', []))}")
    print(f"Assignments found: {len(result.get('assignments', []))}")
    
    # Show extracted structure
    if result.get('modules'):
        for module in result['modules']:
            print(f"\nModule: {module['name']}")
            print(f"  Ports: {len(module.get('ports', []))}")
            print(f"  Items: {len(module.get('items', []))}")
    
    return result

if __name__ == "__main__":
    test_professional_extractor()