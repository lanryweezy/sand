"""
Robust RTL Parser for Silicon Intelligence System

Handles common RTL constructs and provides fallback parsing
when encountering unknown syntax.
"""

import re
from typing import Dict, List, Any, Optional


class RobustRTLParser:
    """A more robust RTL parser that handles common constructs"""
    
    def __init__(self):
        # Common patterns for RTL constructs
        self.patterns = {
            'module': r'\bmodule\s+(\w+)\s*\(',
            'input': r'input\s+(?:\[[^\]]+\]\s*)?(\w+)',
            'output': r'output\s+(?:\[[^\]]+\]\s*)?(\w+)',
            'wire': r'wire\s+(?:\[[^\]]+\]\s*)?(\w+)',
            'reg': r'reg\s+(?:\[[^\]]+\]\s*)?(\w+)',
            'assign': r'assign\s+(\w+)\s*=',
            'always': r'always\s*@',
            'endmodule': r'\bendmodule\b'
        }
    
    def parse_module(self, content: str) -> Dict[str, Any]:
        """Parse a single module from RTL content"""
        module_info = {
            'name': '',
            'inputs': [],
            'outputs': [],
            'wires': [],
            'regs': [],
            'assignments': [],
            'always_blocks': 0,
            'endmodule_found': False
        }
        
        lines = content.split('\n')
        current_module = False
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('//'):
                continue
                
            # Clean the line of problematic characters
            clean_line = re.sub(r'[\\@/]', '', line)
            
            # Look for module declaration
            if not current_module:
                module_match = re.search(self.patterns['module'], clean_line, re.IGNORECASE)
                if module_match:
                    module_info['name'] = module_match.group(1)
                    current_module = True
                    continue
            
            if current_module:
                # Extract inputs
                input_matches = re.findall(self.patterns['input'], clean_line, re.IGNORECASE)
                module_info['inputs'].extend(input_matches)
                
                # Extract outputs  
                output_matches = re.findall(self.patterns['output'], clean_line, re.IGNORECASE)
                module_info['outputs'].extend(output_matches)
                
                # Extract wires
                wire_matches = re.findall(self.patterns['wire'], clean_line, re.IGNORECASE)
                module_info['wires'].extend(wire_matches)
                
                # Extract regs
                reg_matches = re.findall(self.patterns['reg'], clean_line, re.IGNORECASE)
                module_info['regs'].extend(reg_matches)
                
                # Count always blocks
                if re.search(self.patterns['always'], clean_line, re.IGNORECASE):
                    module_info['always_blocks'] += 1
                    
                # Check for endmodule
                if re.search(self.patterns['endmodule'], clean_line, re.IGNORECASE):
                    module_info['endmodule_found'] = True
                    break
        
        return module_info
    
    def parse(self, rtl_content: str) -> Dict[str, Any]:
        """Parse RTL content and return structured data"""
        modules = []
        
        # Split content into potential modules
        parts = re.split(r'\bmodule\s+', rtl_content, flags=re.IGNORECASE)
        
        # Process each part (skip first since it's before first module)
        for i, part in enumerate(parts[1:], 1):
            full_module = 'module ' + part  # Add back the 'module' we split on
            module_data = self.parse_module(full_module)
            if module_data['name']:  # Only add if we found a module name
                modules.append(module_data)
        
        return {
            'modules': modules,
            'module_count': len(modules),
            'total_inputs': sum(len(m['inputs']) for m in modules),
            'total_outputs': sum(len(m['outputs']) for m in modules),
            'total_wires': sum(len(m['wires']) for m in modules),
            'total_regs': sum(len(m['regs']) for m in modules)
        }


# Global instance for compatibility
parser_instance = RobustRTLParser()


def parse_rtl_content(content: str) -> Dict[str, Any]:
    """Global function for backward compatibility"""
    return parser_instance.parse(content)
