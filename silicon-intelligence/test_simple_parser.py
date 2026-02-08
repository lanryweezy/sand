#!/usr/bin/env python3
"""Minimal test for RTL parser with simple module"""

import sys
import os
sys.path.append('.')

from data.comprehensive_rtl_parser import RTLParser
import tempfile

def main():
    # Very simple Verilog without ranges
    verilog_code = '''
module simple_module (
    input clk,
    input rst,
    output data
);
    reg data_reg;
    
    always @(posedge clk) begin
        if (rst)
            data_reg <= 1'b0;
        else
            data_reg <= ~data_reg;
    end
    
    assign data = data_reg;
endmodule
'''

    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as f:
        f.write(verilog_code)
        temp_file = f.name

    try:
        print("Testing simple RTL parser...")
        parser = RTLParser()
        result = parser.parse(temp_file)
        
        print("SUCCESS!")
        print("Modules:", len(result.get('modules', {})))
        print("Instances:", len(result.get('instances', [])))
        print("Ports:", len(result.get('ports', [])))
        print("Nets:", len(result.get('nets', [])))
        
        # Show module details
        if result.get('modules'):
            for name, mod_info in result['modules'].items():
                print(f"\nModule: {name}")
                print(f"  Ports: {len(mod_info.get('ports', []))}")
                print(f"  Instances: {len(mod_info.get('instances', []))}")
                print(f"  Nets: {len(mod_info.get('nets', []))}")
        
        return True
        
    except Exception as e:
        print("FAILED:", str(e))
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.unlink(temp_file)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)