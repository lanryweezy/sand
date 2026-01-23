#!/usr/bin/env python3
"""Debug the minimal parser step by step"""

import sys
import os
sys.path.append('.')

from minimal_rtl_parser import MinimalRTLParser

def debug_step_by_step():
    """Debug parser step by step"""
    
    # Start with the simplest possible always block
    simple_always = '''
    module test ();
        reg q;
        always @(posedge clk) begin
            q <= 1'b0;
        end
    endmodule
    '''
    
    parser = MinimalRTLParser()
    
    print("=== Debugging Simple Always Block ===")
    result = parser.parse(simple_always)
    
    if result:
        print("✅ Parsing succeeded")
        print(f"Modules: {len(result['modules'])}")
        for name, mod in result['modules'].items():
            print(f"Module {name}: {len(mod['always_blocks'])} always blocks")
    else:
        print("❌ Parsing failed")

if __name__ == "__main__":
    debug_step_by_step()