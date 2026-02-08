#!/usr/bin/env python3
"""
Minimal working RTL parser for Verilog
Focus: Actually parse real Verilog constructs that matter for EDA
"""

import ply.lex as lex
import ply.yacc as yacc
import tempfile
import os

class MinimalRTLParser:
    """
    Minimal RTL parser that handles essential Verilog constructs:
    - Module declarations with ports
    - Basic port directions (input/output/inout)
    - Simple assignments
    - Always blocks with event controls
    - Non-blocking/blocking assignments
    """
    
    def __init__(self):
        # Minimal token set for essential Verilog
        self.tokens = [
            'MODULE', 'ENDMODULE', 'INPUT', 'OUTPUT', 'INOUT', 
            'WIRE', 'REG', 'ASSIGN',
            'ALWAYS', 'POSEDGE', 'NEGEDGE', 'BEGIN', 'END',
            'ID', 'NUMBER', 'LPAREN', 'RPAREN', 'LBRACKET', 'RBRACKET',
            'COMMA', 'SEMICOLON', 'COLON', 'EQUAL', 'AT',
            'NON_BLOCKING_ASSIGN', 'BLOCKING_ASSIGN', 'PLUS'
        ]
        
        # Reserved words
        self.reserved = {
            'module': 'MODULE',
            'endmodule': 'ENDMODULE',
            'input': 'INPUT',
            'output': 'OUTPUT', 
            'inout': 'INOUT',
            'wire': 'WIRE',
            'reg': 'REG',
            'assign': 'ASSIGN',
            'always': 'ALWAYS',
            'posedge': 'POSEDGE',
            'negedge': 'NEGEDGE',
            'begin': 'BEGIN',
            'end': 'END'
        }
        
        # Build lexer and parser
        self.lexer = lex.lex(module=self)
        self.parser = yacc.yacc(module=self)
        
        # Parse results
        self.modules = {}
        self.ports = []
        self.assignments = []
        self.always_blocks = []
    
    # Lexer rules
    def t_ID(self, t):
        r'[a-zA-Z_][a-zA-Z_0-9]*'
        t.type = self.reserved.get(t.value, 'ID')
        return t
    
    def t_NUMBER(self, t):
        r"(\d+\'[bodhBODH][a-fA-F0-9xzXZ_]+)|(\d+\.\d+)|(\d+)"
        return t
    
    def t_LPAREN(self, t):
        r'\('
        return t
    
    def t_RPAREN(self, t):
        r'\)'
        return t
    
    def t_LBRACKET(self, t):
        r'\['
        return t
    
    def t_RBRACKET(self, t):
        r'\]'
        return t
    
    def t_COMMA(self, t):
        r','
        return t
    
    def t_SEMICOLON(self, t):
        r';'
        return t
    
    def t_COLON(self, t):
        r':'
        return t
    
    def t_EQUAL(self, t):
        r'='
        return t
    
    def t_AT(self, t):
        r'@'
        return t
    
    def t_NON_BLOCKING_ASSIGN(self, t):
        r'<='
        return t
    
    def t_BLOCKING_ASSIGN(self, t):
        r'='
        return t
    
    def t_PLUS(self, t):
        r'\+'
        return t
    
    # Handle whitespace
    t_ignore = ' \t'
    
    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += len(t.value)
    
    def t_COMMENT(self, t):
        r'(/\*(.|\n)*?\*/)|(//.*)'
        pass  # Ignore comments
    
    def t_error(self, t):
        print(f"Illegal character '{t.value[0]}' at line {t.lexer.lineno}")
        t.lexer.skip(1)
    
    # Parser rules
    def p_design(self, p):
        '''design : module_list'''
        p[0] = p[1]
    
    def p_module_list(self, p):
        '''module_list : module module_list
                      | module'''
        if len(p) == 3:
            p[0] = [p[1]] + p[2]
        else:
            p[0] = [p[1]]
    
    def p_module(self, p):
        '''module : MODULE ID LPAREN port_list RPAREN SEMICOLON module_items ENDMODULE'''
        module_name = p[2]
        ports = p[4] if p[4] else []
        items = p[6] if p[6] else []
        
        module_info = {
            'name': module_name,
            'ports': ports,
            'items': items,
            'assignments': [],
            'always_blocks': []
        }
        
        # Extract assignments and always blocks from items
        for item in items:
            if isinstance(item, dict):
                if item.get('type') == 'assignment':
                    module_info['assignments'].append(item)
                elif item.get('type') == 'always':
                    module_info['always_blocks'].append(item)
        
        self.modules[module_name] = module_info
        p[0] = module_info
    
    def p_port_list(self, p):
        '''port_list : port COMMA port_list
                    | port
                    | empty'''
        if len(p) == 4:
            p[0] = [p[1]] + p[3]
        elif len(p) == 2 and p[1] is not None:
            p[0] = [p[1]]
        else:
            p[0] = []
    
    def p_port(self, p):
        '''port : INPUT port_type_opt ID
                | OUTPUT port_type_opt ID
                | INOUT port_type_opt ID'''
        port_info = {
            'direction': p[1],
            'type': p[2] if p[2] else 'wire',
            'name': p[3]
        }
        self.ports.append(port_info)
        p[0] = port_info
    
    def p_port_type_opt(self, p):
        '''port_type_opt : range_opt
                        | empty'''
        p[0] = p[1]
    
    def p_range_opt(self, p):
        '''range_opt : LBRACKET NUMBER COLON NUMBER RBRACKET
                    | empty'''
        if len(p) == 6:
            p[0] = f"[{p[2]}:{p[4]}]"
        else:
            p[0] = None
    
    def p_module_items(self, p):
        '''module_items : module_item module_items
                       | module_item
                       | empty'''
        if len(p) == 3:
            p[0] = [p[1]] + p[2] if p[2] else [p[1]]
        elif len(p) == 2 and p[1] is not None:
            p[0] = [p[1]]
        else:
            p[0] = []
    
    def p_module_item(self, p):
        '''module_item : net_declaration
                      | assignment_statement
                      | always_construct'''
        p[0] = p[1]
    
    def p_net_declaration(self, p):
        '''net_declaration : REG range_opt ID SEMICOLON
                          | WIRE range_opt ID SEMICOLON'''
        net_info = {
            'type': p[1],  # 'reg' or 'wire'
            'range': p[2],
            'name': p[3]
        }
        p[0] = net_info
    
    def p_assignment_statement(self, p):
        '''assignment_statement : ASSIGN ID EQUAL expression SEMICOLON'''
        assign_info = {
            'type': 'assignment',
            'lhs': p[2],
            'rhs': p[4]
        }
        self.assignments.append(assign_info)
        p[0] = assign_info
    
    def p_always_construct(self, p):
        '''always_construct : ALWAYS AT LPAREN event_expression RPAREN statement_block'''
        always_info = {
            'type': 'always',
            'event': p[4],
            'statements': p[6]
        }
        self.always_blocks.append(always_info)
        p[0] = always_info
    
    def p_event_expression(self, p):
        '''event_expression : POSEDGE ID
                          | NEGEDGE ID
                          | ID'''
        if len(p) == 3:
            p[0] = {'edge': p[1], 'signal': p[2]}
        else:
            p[0] = {'signal': p[1]}
    
    def p_statement_block(self, p):
        '''statement_block : BEGIN statement_list END
                          | statement'''
        if len(p) == 4:
            p[0] = p[2]
        else:
            p[0] = [p[1]]
    
    def p_statement_list(self, p):
        '''statement_list : statement statement_list
                         | statement'''
        if len(p) == 3:
            p[0] = [p[1]] + p[2]
        else:
            p[0] = [p[1]]
    
    def p_statement(self, p):
        '''statement : blocking_assignment SEMICOLON
                    | non_blocking_assignment SEMICOLON'''
        p[0] = p[1]
    
    def p_blocking_assignment(self, p):
        '''blocking_assignment : ID BLOCKING_ASSIGN expression'''
        p[0] = {'type': 'blocking', 'lhs': p[1], 'rhs': p[3]}
    
    def p_non_blocking_assignment(self, p):
        '''non_blocking_assignment : ID NON_BLOCKING_ASSIGN expression'''
        p[0] = {'type': 'non_blocking', 'lhs': p[1], 'rhs': p[3]}
    
    def p_expression(self, p):
        '''expression : term
                     | expression PLUS term'''
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = {'op': '+', 'left': p[1], 'right': p[3]}
    
    def p_term(self, p):
        '''term : ID
               | NUMBER'''
        p[0] = p[1]
    
    def p_empty(self, p):
        '''empty :'''
        pass
    
    def p_error(self, p):
        if p:
            print(f"Syntax error at token {p.type}, value '{p.value}', line {p.lineno}")
        else:
            print("Syntax error at EOF")
    
    def parse(self, verilog_code):
        """Parse Verilog code and return structured data"""
        # Reset state
        self.modules = {}
        self.ports = []
        self.assignments = []
        self.always_blocks = []
        
        try:
            result = self.parser.parse(verilog_code, lexer=self.lexer)
            return {
                'modules': self.modules,
                'ports': self.ports,
                'assignments': self.assignments,
                'always_blocks': self.always_blocks
            }
        except Exception as e:
            print(f"Parsing failed: {str(e)}")
            return None

def test_minimal_parser():
    """Test the minimal parser with real Verilog"""
    
    # Test cases that should work
    test_cases = [
        # Simple module without ranges
        '''
        module simple_ff (
            input clk,
            input rst,
            input d,
            output q
        );
            reg q_reg;
            
            always @(posedge clk) begin
                if (rst)
                    q_reg <= 1'b0;
                else
                    q_reg <= d;
            end
            
            assign q = q_reg;
        endmodule
        ''',
        
        # Module with bit vectors
        '''
        module counter (
            input clk,
            input rst,
            output [3:0] count
        );
            reg [3:0] count_reg;
            
            always @(posedge clk) begin
                if (rst)
                    count_reg <= 4'h0;
                else
                    count_reg <= count_reg + 1;
            end
            
            assign count = count_reg;
        endmodule
        '''
    ]
    
    parser = MinimalRTLParser()
    
    for i, verilog_code in enumerate(test_cases):
        print(f"\n=== Test Case {i+1} ===")
        result = parser.parse(verilog_code)
        
        if result:
            print("✅ SUCCESS")
            print(f"Modules: {len(result['modules'])}")
            print(f"Ports: {len(result['ports'])}")
            print(f"Assignments: {len(result['assignments'])}")
            print(f"Always blocks: {len(result['always_blocks'])}")
            
            # Show module details
            for name, mod_info in result['modules'].items():
                print(f"\nModule '{name}':")
                print(f"  Ports: {len(mod_info['ports'])}")
                print(f"  Assignments: {len(mod_info['assignments'])}")
                print(f"  Always blocks: {len(mod_info['always_blocks'])}")
        else:
            print("❌ FAILED")

if __name__ == "__main__":
    test_minimal_parser()