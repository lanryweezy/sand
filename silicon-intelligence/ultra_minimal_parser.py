#!/usr/bin/env python3
"""
Ultra-minimal RTL parser - just get basic constructs working
"""

import ply.lex as lex
import ply.yacc as yacc

class UltraMinimalParser:
    def __init__(self):
        self.tokens = [
            'MODULE', 'ENDMODULE', 'INPUT', 'OUTPUT', 'INOUT',
            'REG', 'WIRE', 'ASSIGN', 'ALWAYS', 'POSEDGE', 'NEGEDGE',
            'BEGIN', 'END', 'ID', 'NUMBER', 'LPAREN', 'RPAREN',
            'SEMICOLON', 'AT', 'NON_BLOCKING_ASSIGN', 'EQUAL'
        ]
        
        self.reserved = {
            'module': 'MODULE', 'endmodule': 'ENDMODULE',
            'input': 'INPUT', 'output': 'OUTPUT', 'inout': 'INOUT',
            'reg': 'REG', 'wire': 'WIRE', 'assign': 'ASSIGN',
            'always': 'ALWAYS', 'posedge': 'POSEDGE', 'negedge': 'NEGEDGE',
            'begin': 'BEGIN', 'end': 'END'
        }
        
        # Build lexer
        self.lexer = lex.lex(module=self)
        # Build parser
        self.parser = yacc.yacc(module=self, debug=False, write_tables=False)
        
        self.parsed_modules = []
    
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
    
    def t_SEMICOLON(self, t):
        r';'
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
    
    # Parser rules - ultra minimal
    def p_design(self, p):
        '''design : module'''
        p[0] = [p[1]]
    
    def p_module(self, p):
        '''module : MODULE ID LPAREN RPAREN SEMICOLON module_body ENDMODULE'''
        p[0] = {
            'name': p[2],
            'body': p[6]
        }
        self.parsed_modules.append(p[0])
    
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
        '''reg_declaration : REG ID SEMICOLON'''
        p[0] = {'type': 'reg', 'name': p[2]}
    
    def p_wire_declaration(self, p):
        '''wire_declaration : WIRE ID SEMICOLON'''
        p[0] = {'type': 'wire', 'name': p[2]}
    
    def p_assignment_statement(self, p):
        '''assignment_statement : ASSIGN ID EQUAL expression SEMICOLON'''
        p[0] = {'type': 'assign', 'lhs': p[2], 'rhs': p[4]}
    
    def p_expression(self, p):
        '''expression : ID'''
        p[0] = p[1]
    
    def p_always_construct(self, p):
        '''always_construct : ALWAYS AT LPAREN event_expr RPAREN statement_block'''
        p[0] = {
            'type': 'always',
            'event': p[4],
            'block': p[6]
        }
    
    def p_event_expr(self, p):
        '''event_expr : POSEDGE ID'''
        p[0] = {'type': 'posedge', 'signal': p[2]}
    
    def p_statement_block(self, p):
        '''statement_block : BEGIN statement END'''
        p[0] = p[2]
    
    def p_statement(self, p):
        '''statement : non_blocking_assign SEMICOLON'''
        p[0] = p[1]
    
    def p_non_blocking_assign(self, p):
        '''non_blocking_assign : ID NON_BLOCKING_ASSIGN ID'''
        p[0] = {
            'type': 'nonblocking_assign',
            'lhs': p[1],
            'rhs': p[3]
        }
    
    def p_error(self, p):
        if p:
            print(f"Parser error at token {p.type} line {p.lineno}")
        else:
            print("Parser error at EOF")
    
    def parse(self, code):
        self.parsed_modules = []
        try:
            result = self.parser.parse(code, lexer=self.lexer)
            return result
        except Exception as e:
            print(f"Parsing error: {e}")
            return None

def test_ultra_minimal():
    test_code = '''
    module test ();
        reg q;
        wire clk;
        always @(posedge clk) begin
            q <= clk;
        end
        assign clk = q;
    endmodule
    '''
    
    parser = UltraMinimalParser()
    result = parser.parse(test_code)
    
    if result:
        print("✅ Ultra-minimal parser WORKS!")
        print(f"Parsed {len(parser.parsed_modules)} modules")
        for mod in parser.parsed_modules:
            print(f"Module: {mod['name']}")
            print(f"Body items: {len(mod['body'])}")
            for item in mod['body']:
                print(f"  - {item}")
    else:
        print("❌ Ultra-minimal parser FAILED")

if __name__ == "__main__":
    test_ultra_minimal()