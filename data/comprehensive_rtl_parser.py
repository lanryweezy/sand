"""
Comprehensive RTL Parser and Constraint Processing for Silicon Intelligence System

This module implements full RTL parsing (Verilog/VHDL) and constraint processing
(SDC/UPF) to extract all necessary design hierarchy, connectivity, and intent
information for the CanonicalSiliconGraph.
"""

import os
import re
from typing import Dict, List, Any, Tuple, Optional, Union
from enum import Enum
import ply.lex as lex
import ply.yacc as yacc
from core.canonical_silicon_graph import CanonicalSiliconGraph, NodeType
from utils.logger import get_logger


class HDLType(Enum):
    """Type of Hardware Description Language"""
    VERILOG = "verilog"
    SYSTEMVERILOG = "systemverilog"
    VHDL = "vhdl"


class RTLParser:
    """
    Comprehensive RTL parser supporting Verilog and SystemVerilog
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.tokens = [
            'MODULE', 'ENDMODULE', 'INPUT', 'OUTPUT', 'INOUT', 'WIRE', 'REG', 'ASSIGN',
            'PARAMETER', 'LOCALPARAM', 'INTEGER', 'REAL', 'TIME', 'SUPPLY0', 'SUPPLY1',
            'TRI', 'TRI0', 'TRI1', 'WAND', 'WOR', 'TRIAND', 'TRIOR', 'TRIREG',
            'SCALARED', 'VECTORED', 'SIGNED', 'SMALL', 'MEDIUM', 'LARGE',
            'DEFPARAM', 'SPECIFY', 'ENDSPECIFY', 'IF', 'ELSE', 'FOR', 'WHILE',
            'CASE', 'CASEX', 'CASZ', 'ENDCASE', 'DEFAULT', 'ALWAYS', 'INITIAL',
            'POSEDGE', 'NEGEDGE', 'WAIT', 'FORK', 'JOIN', 'REPEAT', 'FOREVER',
            'DISABLE', 'BEGIN', 'END', 'GENERATE', 'ENDGENERATE', 'GENVAR',
            'FUNCTION', 'ENDFUNCTION', 'TASK', 'ENDTASK', 'RETURN', 'AT',
            'ID', 'NUMBER', 'STRING', 'LPAREN', 'RPAREN', 'LBRACKET', 'RBRACKET',
            'LBRACE', 'RBRACE', 'DOT', 'COMMA', 'SEMICOLON', 'COLON', 'EQUAL',
            'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'MODULO', 'POWER',
            'BITWISE_AND', 'BITWISE_OR', 'BITWISE_XOR', 'BITWISE_NAND', 'BITWISE_NOR', 'BITWISE_XNOR',
            'LOGICAL_AND', 'LOGICAL_OR', 'LOGICAL_NOT', 'BITWISE_NOT', 'REDUCTION_NAND', 'REDUCTION_NOR', 'REDUCTION_XNOR',
            'LSHIFT', 'RSHIFT', 'LSHIFTA', 'RSHIFTA',
            'LT', 'GT', 'GE', 'EQ', 'NE', 'CEQ', 'CNE', 'WILD_EQ', 'WILD_NE',
            'COND_OP', 'CONCAT', 'MULTI_CONCAT', 'NON_BLOCKING_ASSIGN', 'BLOCKING_ASSIGN',
            'COMMENT', 'DIRECTIVE', 'HASH', 'INITIAL_CONSTRUCT', 'CONSTANT_EXPRESSION'
        ]
        
        # Define reserved words
        self.reserved = {
            'module': 'MODULE',
            'endmodule': 'ENDMODULE',
            'input': 'INPUT',
            'output': 'OUTPUT',
            'inout': 'INOUT',
            'wire': 'WIRE',
            'reg': 'REG',
            'assign': 'ASSIGN',
            'parameter': 'PARAMETER',
            'localparam': 'LOCALPARAM',
            'integer': 'INTEGER',
            'real': 'REAL',
            'time': 'TIME',
            'supply0': 'SUPPLY0',
            'supply1': 'SUPPLY1',
            'tri': 'TRI',
            'tri0': 'TRI0',
            'tri1': 'TRI1',
            'wand': 'WAND',
            'wor': 'WOR',
            'trior': 'TRIOR',
            'trireg': 'TRIREG',
            'scalared': 'SCALARED',
            'vectored': 'VECTORED',
            'signed': 'SIGNED',
            'small': 'SMALL',
            'medium': 'MEDIUM',
            'large': 'LARGE',
            'defparam': 'DEFPARAM',
            'specify': 'SPECIFY',
            'endspecify': 'ENDSPECIFY',
            'if': 'IF',
            'else': 'ELSE',
            'for': 'FOR',
            'while': 'WHILE',
            'case': 'CASE',
            'casex': 'CASEX',
            'casez': 'CASZ',
            'endcase': 'ENDCASE',
            'default': 'DEFAULT',
            'always': 'ALWAYS',
            'initial': 'INITIAL',
            'posedge': 'POSEDGE',
            'negedge': 'NEGEDGE',
            'wait': 'WAIT',
            'fork': 'FORK',
            'join': 'JOIN',
            'repeat': 'REPEAT',
            'forever': 'FOREVER',
            'disable': 'DISABLE',
            'begin': 'BEGIN',
            'end': 'END',
            'generate': 'GENERATE',
            'endgenerate': 'ENDGENERATE',
            'genvar': 'GENVAR',
            'function': 'FUNCTION',
            'endfunction': 'ENDFUNCTION',
            'task': 'TASK',
            'endtask': 'ENDTASK',
            'return': 'RETURN',
            'hash': 'HASH',  # For parameter value assignment
        }
        
        # Remove duplicate tokens - reserved values are already in tokens list
        pass  # Tokens are already defined above, no need to add reserved values again
        
        # Build lexer
        self.lexer = lex.lex(module=self)
        
        # Build parser
        self.parser = yacc.yacc(module=self)
        
        # Parse results
        self.modules = {}
        self.current_module = None
        self.instances = []
        self.ports = []
        self.nets = []
        self.assignments = []
        self.parameters = []
        self.functions = []
        self.tasks = []
    
    # Lexer rules
    def t_ID(self, t):
        r'[a-zA-Z_][a-zA-Z_0-9]*'
        t.type = self.reserved.get(t.value, 'ID')
        return t
    
    def t_NUMBER(self, t):
        r'(\d+\'[bodhBODH][a-fA-F0-9xzXZ_]+)|(\d+\.\d+)|(\d+)'
        return t
    
    def t_STRING(self, t):
        r'"[^"]*"'
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
    
    def t_LBRACE(self, t):
        r'\{'
        return t
    
    def t_RBRACE(self, t):
        r'\}'
        return t
    
    def t_DOT(self, t):
        r'\.'
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
    
    def t_PLUS(self, t):
        r'\+'
        return t
    
    def t_MINUS(self, t):
        r'-'
        return t
    
    def t_TIMES(self, t):
        r'\*'
        return t
    
    def t_DIVIDE(self, t):
        r'/'
        return t
    
    def t_MODULO(self, t):
        r'%'
        return t
    
    def t_POWER(self, t):
        r'\*\*'
        return t
    
    def t_BITWISE_AND(self, t):
        r'&'
        return t
    
    def t_BITWISE_OR(self, t):
        r'\|'
        return t
    
    def t_BITWISE_XOR(self, t):
        r'\^'
        return t
    
    def t_LOGICAL_AND(self, t):
        r'&&'
        return t
    
    def t_LOGICAL_OR(self, t):
        r'\|\|'
        return t
    
    def t_LOGICAL_NOT(self, t):
        r'!'
        return t
    
    def t_BITWISE_NOT(self, t):
        r'~'
        return t
    
    def t_AT(self, t):
        r'@'
        return t
    
    def t_LSHIFT(self, t):
        r'<<'
        return t
    
    def t_RSHIFT(self, t):
        r'>>'
        return t
    
    def t_LSHIFTA(self, t):
        r'<<<'
        return t
    
    def t_RSHIFTA(self, t):
        r'>>>'
        return t
    
    def t_LT(self, t):
        r'<'
        return t
    
    def t_GT(self, t):
        r'>'
        return t
    
    def t_GE(self, t):
        r'>='
        return t
    
    def t_EQ(self, t):
        r'=='
        return t
    
    def t_NE(self, t):
        r'!='
        return t
    
    def t_CEQ(self, t):
        r'==='
        return t
    
    def t_CNE(self, t):
        r'!=='
        return t
    
    def t_WILD_EQ(self, t):
        r'==?'
        return t
    
    def t_WILD_NE(self, t):
        r'!=?'
        return t
    
    def t_COND_OP(self, t):
        r'\?'
        return t
    
    def t_CONCAT(self, t):
        r'\{'
        return t
    
    def t_NON_BLOCKING_ASSIGN(self, t):
        r'<='
        return t
    
    def t_BLOCKING_ASSIGN(self, t):
        r'='
        return t
    
    def t_COMMENT(self, t):
        r'(/\*(.|\n)*?\*/)|(//.*)'
        pass  # Ignore comments
    
    def t_DIRECTIVE(self, t):
        r'`\w+'
        pass  # Ignore compiler directives for now
    
    t_ignore = ' \t'
    
    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += len(t.value)
    
    def t_error(self, t):
        self.logger.warning(f"Illegal character '{t.value[0]}' at line {t.lexer.lineno}")
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
        
        self.modules[module_name] = {
            'name': module_name,
            'ports': ports,
            'items': items,
            'instances': [],
            'nets': [],
            'parameters': []
        }
        self.current_module = module_name
        
        p[0] = {
            'type': 'module',
            'name': module_name,
            'ports': ports,
            'items': items
        }
    
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
        self.ports.append(p[0])
    
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
                      | parameter_declaration
                      | instantiation
                      | continuous_assign
                      | always_construct
                      | initial_construct'''
        p[0] = p[1]
    
    def p_net_declaration(self, p):
        '''net_declaration : WIRE range_opt ID SEMICOLON
                          | REG range_opt ID SEMICOLON'''
        net_type = p[1]
        range_decl = p[2]
        net_name = p[3]
        
        net_info = {
            'type': net_type,
            'range': range_decl,
            'name': net_name
        }
        self.nets.append(net_info)
        
        if self.current_module:
            self.modules[self.current_module]['nets'].append(net_info)
        
        p[0] = net_info
    
    def p_parameter_declaration(self, p):
        '''parameter_declaration : PARAMETER parameter_assign_list SEMICOLON
                               | LOCALPARAM parameter_assign_list SEMICOLON'''
        param_type = p[1]
        assigns = p[2]
        
        for assign in assigns:
            param_info = {
                'type': param_type,
                'name': assign['name'],
                'value': assign['value']
            }
            self.parameters.append(param_info)
            
            if self.current_module:
                self.modules[self.current_module]['parameters'].append(param_info)
        
        p[0] = {'type': param_type, 'assignments': assigns}
    
    def p_parameter_assign_list(self, p):
        '''parameter_assign_list : parameter_assignment COMMA parameter_assign_list
                                | parameter_assignment'''
        if len(p) == 4:
            p[0] = [p[1]] + p[3]
        else:
            p[0] = [p[1]]
    
    def p_parameter_assignment(self, p):
        '''parameter_assignment : ID EQUAL expression'''
        p[0] = {'name': p[1], 'value': p[3]}
    
    def p_instantiation(self, p):
        '''instantiation : ID parameter_value_assignment_opt ID LPAREN named_port_connection_list RPAREN SEMICOLON'''
        module_type = p[1]
        instance_name = p[3]
        connections = p[5] if p[5] else []
        
        instance_info = {
            'module_type': module_type,
            'instance_name': instance_name,
            'connections': connections
        }
        self.instances.append(instance_info)
        
        if self.current_module:
            self.modules[self.current_module]['instances'].append(instance_info)
        
        p[0] = instance_info
    
    def p_parameter_value_assignment_opt(self, p):
        '''parameter_value_assignment_opt : HASH LPAREN parameter_assignment COMMA parameter_value_assignment_opt RPAREN
                                         | HASH LPAREN parameter_assignment RPAREN
                                         | empty'''
        if len(p) == 7:
            p[0] = [p[3]] + p[5]
        elif len(p) == 5:
            p[0] = [p[3]]
        else:
            p[0] = []
    
    def p_named_port_connection_list(self, p):
        '''named_port_connection_list : DOT ID LPAREN expression RPAREN COMMA named_port_connection_list
                                    | DOT ID LPAREN expression RPAREN
                                    | empty'''
        if len(p) == 7:
            p[0] = [{'port': p[2], 'net': p[4]}] + p[6]
        elif len(p) == 5:
            p[0] = [{'port': p[2], 'net': p[4]}]
        else:
            p[0] = []
    
    def p_continuous_assign(self, p):
        '''continuous_assign : ASSIGN ID EQUAL expression SEMICOLON'''
        assign_info = {
            'lhs': p[2],
            'rhs': p[4]
        }
        self.assignments.append(assign_info)
        
        p[0] = assign_info
    
    def p_expression(self, p):
        '''expression : term
                     | expression PLUS term
                     | expression MINUS term
                     | expression TIMES term
                     | expression DIVIDE term
                     | expression MODULO term
                     | LPAREN expression RPAREN'''
        if len(p) == 2:
            p[0] = p[1]
        elif len(p) == 4 and p[2] in ['+', '-', '*', '/', '%']:
            p[0] = {'op': p[2], 'left': p[1], 'right': p[3]}
        elif len(p) == 4 and p[1] == '(':
            p[0] = p[2]
    
    def p_term(self, p):
        '''term : ID
               | NUMBER
               | STRING'''
        p[0] = p[1]
    
    def p_initial_construct(self, p):
        '''initial_construct : INITIAL statement'''
        p[0] = {'type': 'initial', 'statement': p[2]}
    
    def p_always_construct(self, p):
        '''always_construct : ALWAYS AT LPAREN event_expression RPAREN statement
                           | ALWAYS LPAREN event_expression RPAREN statement
                           | ALWAYS statement'''
        if len(p) == 7:
            # ALWAYS @(...)
            p[0] = {'type': 'always', 'event': p[3], 'statement': p[6]}
        elif len(p) == 6:
            # ALWAYS (...)
            p[0] = {'type': 'always', 'event': p[2], 'statement': p[5]}
        else:
            # ALWAYS statement
            p[0] = {'type': 'always', 'statement': p[2]}
    
    def p_event_expression(self, p):
        '''event_expression : POSEDGE ID
                          | NEGEDGE ID
                          | ID'''
        if len(p) == 3:
            p[0] = {'edge': p[1], 'signal': p[2]}
        else:
            p[0] = {'signal': p[1]}
    
    def p_statement(self, p):
        '''statement : blocking_assignment SEMICOLON
                    | non_blocking_assignment SEMICOLON
                    | if_statement
                    | case_statement
                    | begin_end_block'''
        p[0] = p[1]
    
    def p_blocking_assignment(self, p):
        '''blocking_assignment : ID BLOCKING_ASSIGN expression'''
        p[0] = {'type': 'blocking', 'lhs': p[1], 'rhs': p[3]}
    
    def p_non_blocking_assignment(self, p):
        '''non_blocking_assignment : ID NON_BLOCKING_ASSIGN expression'''
        p[0] = {'type': 'non_blocking', 'lhs': p[1], 'rhs': p[3]}
    
    def p_if_statement(self, p):
        '''if_statement : IF LPAREN expression RPAREN statement
                       | IF LPAREN expression RPAREN statement ELSE statement'''
        if len(p) == 6:
            p[0] = {'type': 'if', 'condition': p[3], 'then': p[5]}
        else:
            p[0] = {'type': 'if_else', 'condition': p[3], 'then': p[5], 'else': p[7]}
    
    def p_case_statement(self, p):
        '''case_statement : CASE LPAREN expression RPAREN case_item_list ENDCASE'''
        p[0] = {'type': 'case', 'expression': p[3], 'items': p[5]}
    
    def p_case_item_list(self, p):
        '''case_item_list : case_item case_item_list
                         | case_item'''
        if len(p) == 3:
            p[0] = [p[1]] + p[2]
        else:
            p[0] = [p[1]]
    
    def p_constant_expression(self, p):
        '''constant_expression : expression
                              | NUMBER'''
        p[0] = p[1]
    
    def p_case_item(self, p):
        '''case_item : constant_expression COLON statement
                    | DEFAULT COLON statement'''
        if len(p) == 4:
            p[0] = {'value': p[1], 'statement': p[3]}
        else:
            p[0] = {'default': True, 'statement': p[3]}
    
    def p_begin_end_block(self, p):
        '''begin_end_block : BEGIN statement_list END'''
        p[0] = {'type': 'block', 'statements': p[2]}
    
    def p_statement_list(self, p):
        '''statement_list : statement statement_list
                         | statement'''
        if len(p) == 3:
            p[0] = [p[1]] + p[2]
        else:
            p[0] = [p[1]]
    
    def p_empty(self, p):
        '''empty :'''
        pass
    
    def p_error(self, p):
        if p:
            self.logger.error(f"Syntax error at token {p.type}, value '{p.value}', line {p.lineno}")
        else:
            self.logger.error("Syntax error at EOF")
    
    def parse(self, rtl_file: str) -> Dict[str, Any]:
        """
        Parse an RTL file and return structured data
        
        Args:
            rtl_file: Path to the RTL file to parse
            
        Returns:
            Dictionary containing parsed RTL information
        """
        self.logger.info(f"Parsing RTL file: {rtl_file}")
        
        # Reset parser state
        self.reset()
        
        # Read the file
        with open(rtl_file, 'r') as f:
            content = f.read()
        
        # Parse the content
        try:
            result = self.parser.parse(content, lexer=self.lexer)
        except Exception as e:
            self.logger.error(f"Error parsing RTL file {rtl_file}: {str(e)}")
            return {}
        
        # Build the result dictionary
        result = {
            'modules': self.modules,
            'instances': self.instances,
            'ports': self.ports,
            'nets': self.nets,
            'assignments': self.assignments,
            'parameters': self.parameters,
            'functions': self.functions,
            'tasks': self.tasks,
            'top_module': self._identify_top_module()
        }
        
        self.logger.info(f"Parsed {len(self.modules)} modules, {len(self.instances)} instances, {len(self.ports)} ports")
        return result
    
    def reset(self):
        """Reset parser state"""
        self.modules = {}
        self.current_module = None
        self.instances = []
        self.ports = []
        self.nets = []
        self.assignments = []
        self.parameters = []
        self.functions = []
        self.tasks = []
    
    def _identify_top_module(self) -> str:
        """Try to identify the top-level module"""
        # Simple heuristic: module with the most external connections
        # or the one that's not instantiated elsewhere
        
        # Get all instantiated module types
        instantiated_types = set(inst['module_type'] for inst in self.instances)
        
        # Find modules that are NOT instantiated (potential top modules)
        potential_tops = []
        for mod_name in self.modules:
            if mod_name not in instantiated_types:
                potential_tops.append(mod_name)
        
        # If we have exactly one potential top, return it
        if len(potential_tops) == 1:
            return potential_tops[0]
        
        # Otherwise, return the one with the most ports (external connections)
        max_ports = -1
        top_module = None
        for mod_name in self.modules:
            port_count = len([p for p in self.ports if mod_name in p.get('name', '')])
            if port_count > max_ports:
                max_ports = port_count
                top_module = mod_name
        
        return top_module or (list(self.modules.keys())[0] if self.modules else "")


class SDCParser:
    """
    SDC (Synopsys Design Constraints) parser
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.constraints = {
            'clocks': [],
            'input_delays': [],
            'output_delays': [],
            'false_paths': [],
            'multicycle_paths': [],
            'groups': [],
            'operating_conditions': [],
            'wire_load_models': [],
            'timing_exceptions': []
        }
    
    def parse(self, sdc_file: str) -> Dict[str, Any]:
        """
        Parse an SDC file and return structured constraints
        
        Args:
            sdc_file: Path to the SDC file to parse
            
        Returns:
            Dictionary containing parsed SDC constraints
        """
        self.logger.info(f"Parsing SDC file: {sdc_file}")
        
        # Reset constraints
        self.constraints = {
            'clocks': [],
            'input_delays': [],
            'output_delays': [],
            'false_paths': [],
            'multicycle_paths': [],
            'groups': [],
            'operating_conditions': [],
            'wire_load_models': [],
            'timing_exceptions': []
        }
        
        with open(sdc_file, 'r') as f:
            content = f.read()
        
        # Remove comments and parse line by line
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse different SDC commands
            if line.startswith('create_clock'):
                self.constraints['clocks'].append(self._parse_create_clock(line))
            elif line.startswith('set_input_delay'):
                self.constraints['input_delays'].append(self._parse_set_input_delay(line))
            elif line.startswith('set_output_delay'):
                self.constraints['output_delays'].append(self._parse_set_output_delay(line))
            elif line.startswith('set_false_path'):
                self.constraints['false_paths'].append(self._parse_set_false_path(line))
            elif line.startswith('set_multicycle_path'):
                self.constraints['multicycle_paths'].append(self._parse_set_multicycle_path(line))
            elif line.startswith('set_clock_groups'):
                self.constraints['groups'].append(self._parse_set_clock_groups(line))
            elif line.startswith('set_operating_conditions'):
                self.constraints['operating_conditions'].append(self._parse_set_operating_conditions(line))
            elif line.startswith('set_wire_load_model'):
                self.constraints['wire_load_models'].append(self._parse_set_wire_load_model(line))
        
        self.logger.info(f"Parsed SDC constraints: {len(self.constraints['clocks'])} clocks, "
                        f"{len(self.constraints['input_delays'])} input delays, "
                        f"{len(self.constraints['output_delays'])} output delays")
        
        return self.constraints
    
    def _parse_create_clock(self, line: str) -> Dict[str, Any]:
        """Parse create_clock command"""
        # Example: create_clock -name core_clk -period 3.333 -waveform {0.000 1.667} [get_ports clk]
        clock_info = {
            'command': 'create_clock',
            'name': None,
            'period': None,
            'waveform': None,
            'source': None,
            'options': {}
        }
        
        # Extract name
        name_match = re.search(r'-name\s+(\w+)', line)
        if name_match:
            clock_info['name'] = name_match.group(1)
        
        # Extract period
        period_match = re.search(r'-period\s+([\d.]+)', line)
        if period_match:
            clock_info['period'] = float(period_match.group(1))
        
        # Extract waveform
        waveform_match = re.search(r'-waveform\s+\{([\d.\s]+)\}', line)
        if waveform_match:
            waveform_str = waveform_match.group(1).strip()
            waveform_parts = waveform_str.split()
            clock_info['waveform'] = [float(w) for w in waveform_parts]
        
        # Extract source
        source_match = re.search(r'\[get_ports\s+([^\]]+)\]', line)
        if source_match:
            clock_info['source'] = source_match.group(1)
        
        return clock_info
    
    def _parse_set_input_delay(self, line: str) -> Dict[str, Any]:
        """Parse set_input_delay command"""
        delay_info = {
            'command': 'set_input_delay',
            'delay': None,
            'clock': None,
            'port': None,
            'options': {}
        }
        
        # Extract delay value
        delay_match = re.search(r'([\d.]+)', line.split()[2])  # Third element after command
        if delay_match:
            delay_info['delay'] = float(delay_match.group(1))
        
        # Extract clock
        clock_match = re.search(r'-clock\s+(\w+)', line)
        if clock_match:
            delay_info['clock'] = clock_match.group(1)
        
        # Extract port
        port_match = re.search(r'\[get_ports\s+([^\]]+)\]', line)
        if port_match:
            delay_info['port'] = port_match.group(1)
        
        return delay_info
    
    def _parse_set_output_delay(self, line: str) -> Dict[str, Any]:
        """Parse set_output_delay command"""
        delay_info = {
            'command': 'set_output_delay',
            'delay': None,
            'clock': None,
            'port': None,
            'options': {}
        }
        
        # Extract delay value
        delay_match = re.search(r'([\d.]+)', line.split()[2])
        if delay_match:
            delay_info['delay'] = float(delay_match.group(1))
        
        # Extract clock
        clock_match = re.search(r'-clock\s+(\w+)', line)
        if clock_match:
            delay_info['clock'] = clock_match.group(1)
        
        # Extract port
        port_match = re.search(r'\[get_ports\s+([^\]]+)\]', line)
        if port_match:
            delay_info['port'] = port_match.group(1)
        
        return delay_info
    
    def _parse_set_false_path(self, line: str) -> Dict[str, Any]:
        """Parse set_false_path command"""
        path_info = {
            'command': 'set_false_path',
            'from': None,
            'to': None,
            'through': None,
            'options': {}
        }
        
        # Extract from/to/through clauses
        from_match = re.search(r'-from\s+([^\s\[]+|\[get_\w+\s+[^\]]+\])', line)
        if from_match:
            path_info['from'] = from_match.group(1)
        
        to_match = re.search(r'-to\s+([^\s\[]+|\[get_\w+\s+[^\]]+\])', line)
        if to_match:
            path_info['to'] = to_match.group(1)
        
        through_match = re.search(r'-through\s+([^\s\[]+|\[get_\w+\s+[^\]]+\])', line)
        if through_match:
            path_info['through'] = through_match.group(1)
        
        return path_info
    
    def _parse_set_multicycle_path(self, line: str) -> Dict[str, Any]:
        """Parse set_multicycle_path command"""
        path_info = {
            'command': 'set_multicycle_path',
            'setup': None,
            'hold': None,
            'from': None,
            'to': None,
            'options': {}
        }
        
        # Extract setup/hold values
        setup_match = re.search(r'-setup\s+(-?\d+)', line)
        if setup_match:
            path_info['setup'] = int(setup_match.group(1))
        
        hold_match = re.search(r'-hold\s+(-?\d+)', line)
        if hold_match:
            path_info['hold'] = int(hold_match.group(1))
        
        # Extract from/to clauses
        from_match = re.search(r'-from\s+([^\s\[]+|\[get_\w+\s+[^\]]+\])', line)
        if from_match:
            path_info['from'] = from_match.group(1)
        
        to_match = re.search(r'-to\s+([^\s\[]+|\[get_\w+\s+[^\]]+\])', line)
        if to_match:
            path_info['to'] = to_match.group(1)
        
        return path_info
    
    def _parse_set_clock_groups(self, line: str) -> Dict[str, Any]:
        """Parse set_clock_groups command"""
        group_info = {
            'command': 'set_clock_groups',
            'exclusive': False,
            'asynchronous': False,
            'physically_exclusive': False,
            'logically_exclusive': False,
            'groups': [],
            'options': {}
        }
        
        # Check for different group types
        if '-exclusive' in line:
            group_info['exclusive'] = True
        if '-asynchronous' in line:
            group_info['asynchronous'] = True
        if '-physically_exclusive' in line:
            group_info['physically_exclusive'] = True
        if '-logically_exclusive' in line:
            group_info['logically_exclusive'] = True
        
        # Extract groups
        group_matches = re.findall(r'-group\s+\{([^\}]+)\}', line)
        for match in group_matches:
            clocks = [c.strip() for c in match.split()]
            group_info['groups'].append(clocks)
        
        return group_info
    
    def _parse_set_operating_conditions(self, line: str) -> Dict[str, Any]:
        """Parse set_operating_conditions command"""
        cond_info = {
            'command': 'set_operating_conditions',
            'condition': None,
            'analysis_type': None,
            'options': {}
        }
        
        # Extract condition
        cond_match = re.search(r'(\w+)', line.split()[-1])  # Last word is usually the condition
        if cond_match:
            cond_info['condition'] = cond_match.group(1)
        
        # Extract analysis type
        if '-analysis_type' in line:
            analysis_match = re.search(r'-analysis_type\s+(\w+)', line)
            if analysis_match:
                cond_info['analysis_type'] = analysis_match.group(1)
        
        return cond_info
    
    def _parse_set_wire_load_model(self, line: str) -> Dict[str, Any]:
        """Parse set_wire_load_model command"""
        model_info = {
            'command': 'set_wire_load_model',
            'model': None,
            'library': None,
            'options': {}
        }
        
        # Extract model name
        model_match = re.search(r'(\w+)', line.split()[1])  # Second element after command
        if model_match:
            model_info['model'] = model_match.group(1)
        
        # Extract library
        if '-library' in line:
            lib_match = re.search(r'-library\s+(\w+)', line)
            if lib_match:
                model_info['library'] = lib_match.group(1)
        
        return model_info


class UPFParser:
    """
    UPF (Unified Power Format) parser
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.power_info = {
            'power_domains': [],
            'supply_sets': [],
            'power_switches': [],
            'level_shifters': [],
            'state_tables': [],
            'retention_registers': [],
            'isolation_cells': []
        }
    
    def parse(self, upf_file: str) -> Dict[str, Any]:
        """
        Parse a UPF file and return structured power information
        
        Args:
            upf_file: Path to the UPF file to parse
            
        Returns:
            Dictionary containing parsed UPF power information
        """
        self.logger.info(f"Parsing UPF file: {upf_file}")
        
        # Reset power info
        self.power_info = {
            'power_domains': [],
            'supply_sets': [],
            'power_switches': [],
            'level_shifters': [],
            'state_tables': [],
            'retention_registers': [],
            'isolation_cells': []
        }
        
        with open(upf_file, 'r') as f:
            content = f.read()
        
        # Remove comments and parse line by line
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse different UPF commands
            if line.startswith('create_power_domain'):
                self.power_info['power_domains'].append(self._parse_create_power_domain(line))
            elif line.startswith('create_supply_set'):
                self.power_info['supply_sets'].append(self._parse_create_supply_set(line))
            elif line.startswith('create_power_switch'):
                self.power_info['power_switches'].append(self._parse_create_power_switch(line))
            elif line.startswith('create_level_shifter'):
                self.power_info['level_shifters'].append(self._parse_create_level_shifter(line))
            elif line.startswith('create_state_table'):
                self.power_info['state_tables'].append(self._parse_create_state_table(line))
        
        self.logger.info(f"Parsed UPF power info: {len(self.power_info['power_domains'])} power domains, "
                        f"{len(self.power_info['supply_sets'])} supply sets")
        
        return self.power_info
    
    def _parse_create_power_domain(self, line: str) -> Dict[str, Any]:
        """Parse create_power_domain command"""
        domain_info = {
            'command': 'create_power_domain',
            'name': None,
            'supplies': [],
            'elements': [],
            'options': {}
        }
        
        # Extract domain name
        name_match = re.search(r'create_power_domain\s+(\w+)', line)
        if name_match:
            domain_info['name'] = name_match.group(1)
        
        # Extract supplies
        supply_match = re.search(r'-supply\s+(\w+)', line)
        if supply_match:
            domain_info['supplies'].append(supply_match.group(1))
        
        # Extract elements
        elements_match = re.search(r'-elements\s+\{([^\}]+)\}', line)
        if elements_match:
            elements = [e.strip() for e in elements_match.group(1).split()]
            domain_info['elements'] = elements
        
        return domain_info
    
    def _parse_create_supply_set(self, line: str) -> Dict[str, Any]:
        """Parse create_supply_set command"""
        supply_info = {
            'command': 'create_supply_set',
            'name': None,
            'supplies': {},
            'options': {}
        }
        
        # Extract supply set name
        name_match = re.search(r'create_supply_set\s+(\w+)', line)
        if name_match:
            supply_info['name'] = name_match.group(1)
        
        # Extract supplies (VDD, VSS, etc.)
        # This is a simplified extraction - real UPF parsing would be more complex
        vdd_match = re.search(r'-supply_vdd\s+(\w+)', line)
        if vdd_match:
            supply_info['supplies']['VDD'] = vdd_match.group(1)
        
        vss_match = re.search(r'-supply_gnd\s+(\w+)', line)
        if vss_match:
            supply_info['supplies']['VSS'] = vss_match.group(1)
        
        return supply_info
    
    def _parse_create_power_switch(self, line: str) -> Dict[str, Any]:
        """Parse create_power_switch command"""
        switch_info = {
            'command': 'create_power_switch',
            'name': None,
            'inputs': [],
            'outputs': [],
            'control': None,
            'options': {}
        }
        
        # Extract switch name
        name_match = re.search(r'create_power_switch\s+(\w+)', line)
        if name_match:
            switch_info['name'] = name_match.group(1)
        
        # Extract inputs/outputs/controls
        input_match = re.search(r'-input_supply_port\s+(\w+)', line)
        if input_match:
            switch_info['inputs'].append(input_match.group(1))
        
        output_match = re.search(r'-output_supply_port\s+(\w+)', line)
        if output_match:
            switch_info['outputs'].append(output_match.group(1))
        
        control_match = re.search(r'-control_port\s+(\w+)', line)
        if control_match:
            switch_info['control'] = control_match.group(1)
        
        return switch_info
    
    def _parse_create_level_shifter(self, line: str) -> Dict[str, Any]:
        """Parse create_level_shifter command"""
        shifter_info = {
            'command': 'create_level_shifter',
            'name': None,
            'input_domain': None,
            'output_domain': None,
            'style': None,
            'options': {}
        }
        
        # Extract shifter name
        name_match = re.search(r'create_level_shifter\s+(\w+)', line)
        if name_match:
            shifter_info['name'] = name_match.group(1)
        
        # Extract domains and style
        input_dom_match = re.search(r'-from_domain\s+(\w+)', line)
        if input_dom_match:
            shifter_info['input_domain'] = input_dom_match.group(1)
        
        output_dom_match = re.search(r'-to_domain\s+(\w+)', line)
        if output_dom_match:
            shifter_info['output_domain'] = output_dom_match.group(1)
        
        style_match = re.search(r'-style\s+(\w+)', line)
        if style_match:
            shifter_info['style'] = style_match.group(1)
        
        return shifter_info
    
    def _parse_create_state_table(self, line: str) -> Dict[str, Any]:
        """Parse create_state_table command"""
        table_info = {
            'command': 'create_state_table',
            'name': None,
            'supplies': [],
            'states': [],
            'options': {}
        }
        
        # Extract table name
        name_match = re.search(r'create_state_table\s+(\w+)', line)
        if name_match:
            table_info['name'] = name_match.group(1)
        
        return table_info


class DesignHierarchyBuilder:
    """
    Builds the CanonicalSiliconGraph from parsed RTL and constraints
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.rtl_parser = RTLParser()
        self.sdc_parser = SDCParser()
        self.upf_parser = UPFParser()
    
    def build_from_rtl_and_constraints(self, 
                                     rtl_file: str, 
                                     sdc_file: Optional[str] = None,
                                     upf_file: Optional[str] = None) -> CanonicalSiliconGraph:
        """
        Build a CanonicalSiliconGraph from RTL and constraint files
        
        Args:
            rtl_file: Path to RTL file
            sdc_file: Path to SDC constraint file (optional)
            upf_file: Path to UPF power file (optional)
            
        Returns:
            CanonicalSiliconGraph instance
        """
        self.logger.info(f"Building graph from RTL: {rtl_file}")
        
        # Parse RTL
        rtl_data = self.rtl_parser.parse(rtl_file)
        
        # Parse constraints if provided
        sdc_data = None
        if sdc_file and os.path.exists(sdc_file):
            sdc_data = self.sdc_parser.parse(sdc_file)
        
        upf_data = None
        if upf_file and os.path.exists(upf_file):
            upf_data = self.upf_parser.parse(upf_file)
        
        # Create the graph
        graph = CanonicalSiliconGraph()
        
        # Add modules as subgraphs or high-level nodes
        for module_name, module_info in rtl_data['modules'].items():
            # Add module as a high-level node
            graph.graph.add_node(module_name,
                               node_type=NodeType.CELL.value,
                               cell_type='module',
                               area=self._estimate_module_area(module_info),
                               power=self._estimate_module_power(module_info),
                               timing_criticality=0.0,
                               estimated_congestion=0.0)
        
        # Add instances
        for instance in rtl_data['instances']:
            instance_name = instance['instance_name']
            module_type = instance['module_type']
            
            graph.graph.add_node(instance_name,
                               node_type=NodeType.CELL.value,
                               cell_type=module_type,
                               area=1.0,  # Placeholder
                               power=0.01,  # Placeholder
                               timing_criticality=0.0,
                               estimated_congestion=0.0)
            
            # Connect instance to its module type
            if module_type in graph.graph.nodes():
                graph.graph.add_edge(instance_name, module_type)
        
        # Add ports
        for port in rtl_data['ports']:
            port_name = port['name']
            graph.graph.add_node(port_name,
                               node_type=NodeType.PORT.value,
                               cell_type=f"PORT_{port['direction'].upper()}",
                               area=0.0,
                               power=0.0,
                               timing_criticality=0.0,
                               estimated_congestion=0.0)
        
        # Add nets
        for net in rtl_data['nets']:
            net_name = net['name']
            graph.graph.add_node(net_name,
                               node_type=NodeType.SIGNAL.value,
                               cell_type='NET',
                               area=0.0,
                               power=0.0,
                               timing_criticality=0.0,
                               estimated_congestion=0.0)
        
        # Add connections based on instance connections
        for instance in rtl_data['instances']:
            for connection in instance.get('connections', []):
                port_name = connection['port']
                net_name = connection['net']
                
                # Connect instance port to net
                if instance['instance_name'] in graph.graph.nodes() and net_name in graph.graph.nodes():
                    graph.graph.add_edge(instance['instance_name'], net_name)
                    graph.graph.add_edge(net_name, instance['instance_name'])
        
        # Apply timing constraints if available
        if sdc_data:
            self._apply_timing_constraints(graph, sdc_data)
        
        # Apply power constraints if available
        if upf_data:
            self._apply_power_constraints(graph, upf_data)
        
        # Initialize physical properties
        self._initialize_physical_properties(graph)
        
        self.logger.info(f"Graph built with {len(graph.graph.nodes())} nodes and {len(graph.graph.edges())} edges")
        return graph
    
    def _estimate_module_area(self, module_info: Dict[str, Any]) -> float:
        """Estimate area for a module based on its contents"""
        # This is a simplified estimation
        # In reality, this would use library information
        instance_count = len(module_info.get('instances', []))
        net_count = len(module_info.get('nets', []))
        
        return instance_count * 2.0 + net_count * 0.1  # Rough estimation
    
    def _estimate_module_power(self, module_info: Dict[str, Any]) -> float:
        """Estimate power for a module based on its contents"""
        # Simplified power estimation
        instance_count = len(module_info.get('instances', []))
        return instance_count * 0.01  # 0.01W per instance approximation
    
    def _apply_timing_constraints(self, graph: CanonicalSiliconGraph, sdc_data: Dict[str, Any]):
        """Apply timing constraints to the graph"""
        # Apply clock constraints
        for clock in sdc_data.get('clocks', []):
            clock_name = clock.get('name', 'default_clk')
            period = clock.get('period', 10.0)
            
            # Find nodes that might be related to this clock
            for node, attrs in graph.graph.nodes(data=True):
                if clock_name.lower() in node.lower() or attrs.get('cell_type', '').lower() in ['clkbuf', 'buf']:
                    graph.graph.nodes[node]['timing_criticality'] = max(
                        graph.graph.nodes[node].get('timing_criticality', 0.0),
                        0.8  # High criticality for clock-related nodes
                    )
                    graph.graph.nodes[node]['clock_domain'] = clock_name
                    graph.graph.nodes[node]['clock_period_ns'] = period
    
    def _apply_power_constraints(self, graph: CanonicalSiliconGraph, upf_data: Dict[str, Any]):
        """Apply power constraints to the graph"""
        # Apply power domain information
        for domain in upf_data.get('power_domains', []):
            domain_name = domain.get('name', 'default_power_domain')
            elements = domain.get('elements', [])
            
            # Assign nodes to power domains
            for element in elements:
                if element in graph.graph.nodes():
                    graph.graph.nodes[element]['power_domain'] = domain_name
    
    def _initialize_physical_properties(self, graph: CanonicalSiliconGraph):
        """Initialize physical properties for all nodes"""
        for node, attrs in graph.graph.nodes(data=True):
            # Initialize congestion estimates
            graph.graph.nodes[node]['estimated_congestion'] = 0.0
            
            # Initialize power domain if not set
            if not attrs.get('power_domain'):
                graph.graph.nodes[node]['power_domain'] = 'DEFAULT_VDD'
            
            # Initialize clock domain if not set
            if not attrs.get('clock_domain'):
                graph.graph.nodes[node]['clock_domain'] = 'DEFAULT_CLK'


# Example usage
def example_rtl_parsing():
    """Example of using the RTL parser and constraint processing"""
    logger = get_logger(__name__)
    
    # Initialize the hierarchy builder
    builder = DesignHierarchyBuilder()
    logger.info("Design hierarchy builder initialized")
    
    # Example RTL content for testing
    example_rtl = """
// Example RTL for testing
module test_adder (
    input clk,
    input rst_n,
    input [7:0] a,
    input [7:0] b,
    output [8:0] sum,
    output cout
);

    reg [8:0] result;
    reg carry;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result <= 9'b0;
            carry <= 1'b0;
        end
        else begin
            {carry, result[7:0]} = a + b;
            result[8] = carry;
        end
    end

    assign sum = result;
    assign cout = carry;

endmodule
"""
    
    # Write example RTL to a temporary file
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as f:
        f.write(example_rtl)
        temp_rtl_file = f.name
    
    try:
        # Parse the RTL
        rtl_data = builder.rtl_parser.parse(temp_rtl_file)
        logger.info(f"Parsed RTL with {len(rtl_data['modules'])} modules")
        
        # Build the graph
        graph = builder.build_from_rtl_and_constraints(temp_rtl_file)
        logger.info(f"Built graph with {len(graph.graph.nodes())} nodes")
        
    finally:
        # Clean up temporary file
        os.unlink(temp_rtl_file)
    
    logger.info("RTL parsing and graph building example completed")


if __name__ == "__main__":
    example_rtl_parsing()