#!/usr/bin/env python3
"""Simple test for RTL parser"""

import sys
import os
sys.path.append('.')

from data.comprehensive_rtl_parser import RTLParser
import tempfile

def main():
    # Simple Verilog
    verilog_code = '''
module test_module (
    input clk,
    input rst,
    output [7:0] data
);
    reg [7:0] data_reg;
    
    always @(posedge clk) begin
        if (rst)
            data_reg <= 8'h0;
        else
            data_reg <= data_reg + 1;
    end
    
    assign data = data_reg;
endmodule
'''

    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as f:
        f.write(verilog_code)
        temp_file = f.name

    try:
        print("Testing RTL parser...")
        parser = RTLParser()
        result = parser.parse(temp_file)
        
        print("SUCCESS!")
        print("Modules:", len(result.get('modules', {})))
        print("Instances:", len(result.get('instances', [])))
        print("Ports:", len(result.get('ports', [])))
        print("Nets:", len(result.get('nets', [])))
        
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