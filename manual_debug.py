#!/usr/bin/env python3
"""Manual grammar debugging"""

# Let me manually trace through what should happen:
#
# Input: always @(posedge clk) begin q <= 1'b0; end
#
# Grammar should match:
# always_construct -> ALWAYS AT LPAREN event_expression RPAREN statement_block
# event_expression -> POSEDGE ID  (matches "posedge clk")
# statement_block -> BEGIN statement_list END
# statement_list -> statement
# statement -> non_blocking_assignment SEMICOLON
# non_blocking_assignment -> ID NON_BLOCKING_ASSIGN expression

manual_test = '''
module test ();
    reg q;
    always @(posedge clk) begin
        q <= 1'b0;
    end
endmodule
'''

print("Manual grammar trace:")
print("1. always_construct should match: ALWAYS AT LPAREN event_expression RPAREN statement_block")
print("2. event_expression should match: POSEDGE ID")
print("3. statement_block should match: BEGIN statement_list END")
print("4. statement_list should match: statement")
print("5. statement should match: non_blocking_assignment SEMICOLON")
print("6. non_blocking_assignment should match: ID NON_BLOCKING_ASSIGN expression")

# The issue might be that the lexer isn't producing the right tokens
# Let me check what tokens are actually produced

import ply.lex as lex

# Copy the lexer rules from minimal_rtl_parser
tokens = [
    'MODULE', 'ENDMODULE', 'INPUT', 'OUTPUT', 'INOUT', 
    'WIRE', 'REG', 'ASSIGN',
    'ALWAYS', 'POSEDGE', 'NEGEDGE', 'BEGIN', 'END',
    'ID', 'NUMBER', 'LPAREN', 'RPAREN', 'LBRACKET', 'RBRACKET',
    'COMMA', 'SEMICOLON', 'COLON', 'EQUAL', 'AT',
    'NON_BLOCKING_ASSIGN', 'BLOCKING_ASSIGN', 'PLUS'
]

reserved = {
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

def t_ID(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    t.type = reserved.get(t.value, 'ID')
    return t

def t_NUMBER(t):
    r"(\d+\'[bodhBODH][a-fA-F0-9xzXZ_]+)|(\d+\.\d+)|(\d+)"
    return t

def t_LPAREN(t):
    r'\('
    return t

def t_RPAREN(t):
    r'\)'
    return t

def t_SEMICOLON(t):
    r';'
    return t

def t_AT(t):
    r'@'
    return t

def t_NON_BLOCKING_ASSIGN(t):
    r'<='
    return t

def t_COMMENT(t):
    r'(/\*(.|\n)*?\*/)|(//.*)'
    pass

t_ignore = ' \t'

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

def t_error(t):
    print(f"Illegal character '{t.value[0]}' at line {t.lexer.lineno}")
    t.lexer.skip(1)

# Build lexer
lexer = lex.lex()

# Test the specific always construct
test_text = "always @(posedge clk) begin q <= 1'b0; end"

print(f"\nTesting tokenization of: {test_text}")

lexer.input(test_text)
tokens_found = []
while True:
    tok = lexer.token()
    if not tok:
        break
    tokens_found.append((tok.type, tok.value))
    print(f"  {tok.type}: '{tok.value}'")

print(f"\nTokens found: {tokens_found}")

# Expected: ALWAYS, AT, LPAREN, POSEDGE, ID, RPAREN, BEGIN, ID, NON_BLOCKING_ASSIGN, NUMBER, SEMICOLON, END
expected = ['ALWAYS', 'AT', 'LPAREN', 'POSEDGE', 'ID', 'RPAREN', 'BEGIN', 'ID', 'NON_BLOCKING_ASSIGN', 'NUMBER', 'SEMICOLON', 'END']
print(f"Expected: {expected}")

# Check if we have all expected tokens
missing = []
for exp_token in expected:
    if not any(tok_type == exp_token for tok_type, _ in tokens_found):
        missing.append(exp_token)

if missing:
    print(f"Missing tokens: {missing}")
else:
    print("All expected tokens present!")

# The issue is probably in the grammar, not the lexer
print("\nThe lexer seems fine. The issue is likely in the grammar rules.")
print("Specifically, the statement_block -> BEGIN statement_list END rule")
print("might not be matching properly with the actual token sequence.")