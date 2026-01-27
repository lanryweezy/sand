#!/usr/bin/env python3
"""Test the RTL parser with a simple Verilog module"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'silicon-intelligence'))

from silicon_intelligence.data.comprehensive_rtl_parser import RTLParser
import tempfile

def test_simple_verilog():
    """Test parser with simple Verilog"""
    
    # Simple counter module
    simple_verilog = '''
module simple_counter (
    input clk,
    input rst_n,
    output [3:0] count
);

    reg [3:0] count_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            count_reg <= 4'h0;
        else
            count_reg <= count_reg + 1;
    end

    assign count = count_reg;

endmodule
'''
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as f:
        f.write(simple_verilog)
        temp_file = f.name
    
    try:
        print("Testing RTL parser...")
        parser = RTLParser()
        result = parser.parse(temp_file)
        
        print("SUCCESS: Parser worked!")
        print(f"Modules found: {len(result.get('modules', {}))}")
        print(f"Instances found: {len(result.get('instances', []))}")
        print(f"Ports found: {len(result.get('ports', []))}")
        print(f"Nets found: {len(result.get('nets', []))}")
        
        # Show module details
        if result.get('modules'):
            for name, mod_info in result['modules'].items():
                print(f"\nModule: {name}")
                print(f"  Ports: {len(mod_info.get('ports', []))}")
                print(f"  Instances: {len(mod_info.get('instances', []))}")
                print(f"  Nets: {len(mod_info.get('nets', []))}")
        
        return True
        
    except Exception as e:
        print(f"FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.unlink(temp_file)

if __name__ == "__main__":
    success = test_simple_verilog()
    sys.exit(0 if success else 1)