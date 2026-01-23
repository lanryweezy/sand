#!/usr/bin/env python3
"""
Expanded RTL parser for AI accelerator designs
Building on the working minimal parser to handle:
- Port declarations with directions and bit widths
- More complex expressions
- MAC arrays and convolution cores patterns
"""

import ply.lex as lex
import ply.yacc as yacc

class ExpandedRTLParser:
    def __init__(self):
        self.tokens = [
            'MODULE', 'ENDMODULE', 'INPUT', 'OUTPUT', 'INOUT',
            'REG', 'WIRE', 'ASSIGN', 'ALWAYS', 'POSEDGE', 'NEGEDGE',
            'BEGIN', 'END', 'ID', 'NUMBER', 'LPAREN', 'RPAREN',
            'LBRACKET', 'RBRACKET', 'COMMA', 'SEMICOLON', 'COLON',
            'AT', 'NON_BLOCKING_ASSIGN', 'EQUAL', 'PLUS', 'MINUS', 'TIMES',
            'NOT', 'IF', 'ELSE'
        ]
        
        self.reserved = {
            'module': 'MODULE', 'endmodule': 'ENDMODULE',
            'input': 'INPUT', 'output': 'OUTPUT', 'inout': 'INOUT',
            'reg': 'REG', 'wire': 'WIRE', 'assign': 'ASSIGN',
            'always': 'ALWAYS', 'posedge': 'POSEDGE', 'negedge': 'NEGEDGE',
            'begin': 'BEGIN', 'end': 'END',
            'if': 'IF', 'else': 'ELSE'
        }
        
        # Build lexer and parser
        self.lexer = lex.lex(module=self)
        self.parser = yacc.yacc(module=self, debug=False, write_tables=False)
        
        self.parsed_modules = []
        self.current_module = None
    
    # Lexer rules
    def t_ID(self, t):
        r'[a-zA-Z_][a-zA-Z_0-9]*'
        t.type = self.reserved.get(t.value, 'ID')
        return t
    
    def t_NUMBER(self, t):
        r"(\d+\'[bodhBODH][a-fA-F0-9xzXZ_]+)|(\d+)"
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
    
    def t_AT(self, t):
        r'@'
        return t
    
    def t_NON_BLOCKING_ASSIGN(self, t):
        r'<='
        return t
    
    def t_EQUAL(self, t):
        r'='
        return t
    
    def t_PLUS(self, t):
        r'\+'
        return t
    
    def t_MINUS(self, t):
        r'-'
        return t
    
    def t_TIMES(self, t):
        r'\*'
        return t
    
    def t_NOT(self, t):
        r'!'
        return t
    
    t_ignore = ' \t'
    
    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += len(t.value)
    
    def t_COMMENT(self, t):
        r'(/\*(.|\n)*?\*/)|(//.*)'
        pass
    
    def t_error(self, t):
        print(f"Lexer error: '{t.value[0]}' at line {t.lexer.lineno}")
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
        '''module : MODULE ID LPAREN port_list RPAREN SEMICOLON module_body ENDMODULE'''
        module_info = {
            'name': p[2],
            'ports': p[4],
            'body': p[6]
        }
        self.parsed_modules.append(module_info)
        self.current_module = module_info
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
        p[0] = {
            'direction': p[1],
            'type': p[2] if p[2] else 'wire',
            'name': p[3]
        }
    
    def p_port_type_opt(self, p):
        '''port_type_opt : range_opt'''
        p[0] = p[1]
    
    def p_range_opt(self, p):
        '''range_opt : LBRACKET NUMBER COLON NUMBER RBRACKET
                    | empty'''
        if len(p) == 6:
            p[0] = f"[{p[2]}:{p[4]}]"
        else:
            p[0] = None
    
    def p_module_body(self, p):
        '''module_body : module_item module_body
                      | module_item'''
        if len(p) == 3:
            p[0] = [p[1]] + p[2]
        else:
            p[0] = [p[1]]
    
    def p_module_item(self, p):
        '''module_item : reg_declaration
                      | wire_declaration
                      | assignment_statement
                      | always_construct'''
        p[0] = p[1]
    
    def p_reg_declaration(self, p):
        '''reg_declaration : REG range_opt ID SEMICOLON'''
        p[0] = {
            'type': 'reg',
            'range': p[2],
            'name': p[3]
        }
    
    def p_wire_declaration(self, p):
        '''wire_declaration : WIRE range_opt ID SEMICOLON'''
        p[0] = {
            'type': 'wire', 
            'range': p[2],
            'name': p[3]
        }
    
    def p_assignment_statement(self, p):
        '''assignment_statement : ASSIGN ID EQUAL expression SEMICOLON'''
        p[0] = {
            'type': 'assign',
            'lhs': p[2],
            'rhs': p[4]
        }
    
    def p_always_construct(self, p):
        '''always_construct : ALWAYS AT LPAREN event_expr RPAREN statement_block'''
        p[0] = {
            'type': 'always',
            'event': p[4],
            'block': p[6]
        }
    
    def p_event_expr(self, p):
        '''event_expr : POSEDGE ID
                     | NEGEDGE ID
                     | ID'''
        if len(p) == 3:
            p[0] = {'type': p[1].lower(), 'signal': p[2]}
        else:
            p[0] = {'type': 'signal', 'signal': p[1]}
    
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
        '''statement : blocking_assign SEMICOLON
                    | non_blocking_assign SEMICOLON
                    | if_statement'''
        p[0] = p[1]
    
    def p_blocking_assign(self, p):
        '''blocking_assign : ID EQUAL expression'''
        p[0] = {
            'type': 'blocking_assign',
            'lhs': p[1],
            'rhs': p[3]
        }
    
    def p_non_blocking_assign(self, p):
        '''non_blocking_assign : ID NON_BLOCKING_ASSIGN expression'''
        p[0] = {
            'type': 'nonblocking_assign',
            'lhs': p[1],
            'rhs': p[3]
        }
    
    def p_if_statement(self, p):
        '''if_statement : IF LPAREN expression RPAREN statement_block
                       | IF LPAREN expression RPAREN statement_block ELSE statement_block'''
        if len(p) == 6:
            p[0] = {
                'type': 'if',
                'condition': p[3],
                'then_block': p[5]
            }
        else:
            p[0] = {
                'type': 'if_else',
                'condition': p[3],
                'then_block': p[5],
                'else_block': p[7]
            }
    
    def p_expression(self, p):
        '''expression : term
                     | expression PLUS term
                     | expression MINUS term
                     | expression TIMES term
                     | NOT term'''
        if len(p) == 2:
            p[0] = p[1]
        elif len(p) == 3:
            p[0] = {'op': '!', 'operand': p[2]}
        else:
            p[0] = {
                'op': p[2],
                'left': p[1],
                'right': p[3]
            }
    
    def p_term(self, p):
        '''term : ID
               | NUMBER
               | LPAREN expression RPAREN'''
        if len(p) == 4:
            p[0] = p[2]
        else:
            p[0] = p[1]
    
    def p_empty(self, p):
        '''empty :'''
        pass
    
    def p_error(self, p):
        if p:
            print(f"Parser error at token {p.type} line {p.lineno}")
        else:
            print("Parser error at EOF")
    
    def parse(self, code):
        self.parsed_modules = []
        self.current_module = None
        try:
            result = self.parser.parse(code, lexer=self.lexer)
            return result
        except Exception as e:
            print(f"Parsing error: {e}")
            return None

def test_expanded_parser():
    """Test expanded parser with AI accelerator patterns"""
    
    # Test MAC array pattern
    mac_array_test = '''
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
    
    # Test convolution pattern
    conv_test = '''
    module conv_core (
        input clk,
        input rst_n,
        input [7:0] input_pixel,
        input [7:0] kernel_weight,
        output [15:0] conv_result
    );
        reg [15:0] multiply_result;
        reg [15:0] accumulate_result;
        
        always @(posedge clk) begin
            if (!rst_n) begin
                multiply_result <= 16'd0;
                accumulate_result <= 16'd0;
            end else begin
                multiply_result <= input_pixel * kernel_weight;
                accumulate_result <= accumulate_result + multiply_result;
            end
        end
        
        assign conv_result = accumulate_result;
    endmodule
    '''
    
    parser = ExpandedRTLParser()
    
    print("=== Testing MAC Array Pattern ===")
    result1 = parser.parse(mac_array_test)
    if result1:
        print("✅ MAC array parsed successfully")
        mod = parser.parsed_modules[0]
        print(f"Module: {mod['name']}")
        print(f"Ports: {len(mod['ports'])}")
        print(f"Body items: {len(mod['body'])}")
        for item in mod['body']:
            if isinstance(item, dict) and 'type' in item:
                print(f"  - {item['type']}")
            else:
                print(f"  - {type(item).__name__}: {str(item)[:50]}...")
    else:
        print("❌ MAC array parsing failed")
    
    print("\n=== Testing Convolution Pattern ===") 
    result2 = parser.parse(conv_test)
    if result2:
        print("✅ Convolution core parsed successfully")
        # Get the last parsed module (should be conv_core)
        mod = parser.parsed_modules[-1]
        print(f"Module: {mod['name']}")
        print(f"Ports: {len(mod['ports'])}")
        print(f"Body items: {len(mod['body'])}")
    else:
        print("❌ Convolution parsing failed")

if __name__ == "__main__":
    test_expanded_parser()