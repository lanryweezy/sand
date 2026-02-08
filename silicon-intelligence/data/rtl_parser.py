"""
RTL Parser - Parses RTL files for graph construction

This module parses Verilog/VHDL RTL files and extracts structural information
needed for the canonical silicon graph construction.

Supports:
- Verilog parsing (using pyverilog or regex-based)
- SDC constraint parsing
- UPF power constraint parsing
- Design hierarchy extraction
"""

import re
import os
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from utils.logger import get_logger

try:
    from pyverilog.verilog_parser import parse as verilog_parse
    PYVERILOG_AVAILABLE = True
except ImportError:
    PYVERILOG_AVAILABLE = False


class RTLParser:
    """
    RTL Parser - extracts structural information from RTL files
    
    Supports:
    - Verilog file parsing
    - SDC constraint parsing
    - UPF power constraint parsing
    - Design hierarchy extraction
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.reset()
    
    def reset(self):
        """Reset parser state"""
        self.modules = {}
        self.current_module = None
        self.instances = []
        self.ports = []
        self.nets = []
        self.assignments = []
        self.parameters = {}
        self.hierarchy = {}
    
    def parse_verilog(self, rtl_file: str) -> Dict[str, Any]:
        """
        Parse a Verilog file and return structured data
        
        Args:
            rtl_file: Path to the Verilog file to parse
            
        Returns:
            Dictionary containing parsed RTL information:
            {
                'instances': [{'name': str, 'type': str, 'parameters': dict}, ...],
                'ports': [{'name': str, 'direction': str, 'width': int}, ...],
                'nets': [{'name': str, 'type': str, 'width': int}, ...],
                'modules': {module_name: {...}},
                'hierarchy': {...},
                'top_module': str
            }
        """
        self.logger.info(f"Parsing Verilog file: {rtl_file}")
        self.reset()
        
        if not os.path.exists(rtl_file):
            self.logger.error(f"File not found: {rtl_file}")
            raise FileNotFoundError(f"RTL file not found: {rtl_file}")
        
        with open(rtl_file, 'r') as f:
            content = f.read()
        
        # Use regex-based parsing (pyverilog is optional)
        self._parse_verilog_with_regex(content)
        
        # Identify the top module
        top_module_name = self._identify_top_module()
        if top_module_name and top_module_name in self.modules:
            # Assign globally parsed elements to the top module for a simplified hierarchy
            self.modules[top_module_name]['ports'] = self.ports
            self.modules[top_module_name]['instances'] = self.instances
            self.modules[top_module_name]['nets'] = self.nets
        else:
            self.logger.warning(f"Could not identify a clear top module or it's missing from parsed modules. Global elements not assigned to a specific module.")

        # Build the result dictionary
        result = {
            'instances': self.instances,
            'ports': self.ports,
            'nets': self.nets,
            'modules': self.modules,
            'hierarchy': self.hierarchy, # This is still empty with current regex parsing
            'top_module': top_module_name,
            'parameters': self.parameters
        }
        
        self.logger.info(f"Parsed {len(self.instances)} instances, {len(self.ports)} ports, {len(self.nets)} nets for top module '{top_module_name}'.")
        return result
    
    def _parse_verilog_with_regex(self, content: str):
        """Parse Verilog using regex-based approach"""
        # Preprocess content to remove comments
        content = self._remove_comments(content)
        
        # Parse the content
        self._parse_modules(content)
        self._parse_instances(content)
        self._parse_ports(content)
        self._parse_nets(content)
        self._parse_assignments(content)
        self._parse_parameters(content)
    
    def parse_sdc(self, sdc_file: str) -> Dict[str, Any]:
        """
        Parse SDC (Synopsys Design Constraints) file
        
        Args:
            sdc_file: Path to the SDC file
            
        Returns:
            Dictionary containing timing constraints:
            {
                'clocks': [{'name': str, 'period': float, 'uncertainty': float}, ...],
                'timing_paths': [{'from': str, 'to': str, 'constraint': float}, ...],
                'input_delays': [...],
                'output_delays': [...],
                'false_paths': [...]
            }
        """
        self.logger.info(f"Parsing SDC file: {sdc_file}")
        
        if not os.path.exists(sdc_file):
            self.logger.warning(f"SDC file not found: {sdc_file}")
            return {
                'clocks': [],
                'timing_paths': [],
                'input_delays': [],
                'output_delays': [],
                'false_paths': []
            }
        
        with open(sdc_file, 'r') as f:
            content = f.read()
        
        # Remove comments
        content = self._remove_comments(content)
        
        constraints = {
            'clocks': self._parse_sdc_clocks(content),
            'timing_paths': self._parse_sdc_timing_paths(content),
            'input_delays': self._parse_sdc_input_delays(content),
            'output_delays': self._parse_sdc_output_delays(content),
            'false_paths': self._parse_sdc_false_paths(content)
        }
        
        self.logger.info(f"Parsed {len(constraints['clocks'])} clocks, "
                        f"{len(constraints['timing_paths'])} timing paths")
        return constraints
    
    def _parse_sdc_clocks(self, content: str) -> List[Dict[str, Any]]:
        """Parse clock definitions from SDC"""
        clocks = []
        
        # Match: create_clock -period <period> -name <name> [port]
        clock_pattern = r'create_clock\s+(?:-period\s+([\d.]+))?(?:\s+-name\s+(\w+))?(?:\s+(\w+))?'
        matches = re.findall(clock_pattern, content, re.IGNORECASE)
        
        for match in matches:
            period = float(match[0]) if match[0] else 10.0
            name = match[1] if match[1] else f"clk_{len(clocks)}"
            port = match[2] if match[2] else "clk"
            
            clocks.append({
                'name': name,
                'period': period,
                'port': port,
                'uncertainty': period * 0.05  # Default 5% uncertainty
            })
        
        return clocks
    
    def _parse_sdc_timing_paths(self, content: str) -> List[Dict[str, Any]]:
        """Parse timing path constraints from SDC"""
        paths = []
        
        # Match: set_max_delay -from <from> -to <to> <delay>
        path_pattern = r'set_max_delay\s+(?:-from\s+(\w+))?(?:\s+-to\s+(\w+))?(?:\s+([\d.]+))?'
        matches = re.findall(path_pattern, content, re.IGNORECASE)
        
        for match in matches:
            from_node = match[0] if match[0] else "input"
            to_node = match[1] if match[1] else "output"
            constraint = float(match[2]) if match[2] else 10.0
            
            paths.append({
                'from': from_node,
                'to': to_node,
                'constraint': constraint,
                'type': 'max_delay'
            })
        
        return paths
    
    def _parse_sdc_input_delays(self, content: str) -> List[Dict[str, Any]]:
        """Parse input delay constraints from SDC"""
        delays = []
        
        # Match: set_input_delay -clock <clock> -max <delay> [port]
        delay_pattern = r'set_input_delay\s+(?:-clock\s+(\w+))?(?:\s+-max\s+([\d.]+))?(?:\s+(\w+))?'
        matches = re.findall(delay_pattern, content, re.IGNORECASE)
        
        for match in matches:
            clock = match[0] if match[0] else "clk"
            delay = float(match[1]) if match[1] else 1.0
            port = match[2] if match[2] else "input"
            
            delays.append({
                'port': port,
                'clock': clock,
                'delay': delay
            })
        
        return delays
    
    def _parse_sdc_output_delays(self, content: str) -> List[Dict[str, Any]]:
        """Parse output delay constraints from SDC"""
        delays = []
        
        # Match: set_output_delay -clock <clock> -max <delay> [port]
        delay_pattern = r'set_output_delay\s+(?:-clock\s+(\w+))?(?:\s+-max\s+([\d.]+))?(?:\s+(\w+))?'
        matches = re.findall(delay_pattern, content, re.IGNORECASE)
        
        for match in matches:
            clock = match[0] if match[0] else "clk"
            delay = float(match[1]) if match[1] else 1.0
            port = match[2] if match[2] else "output"
            
            delays.append({
                'port': port,
                'clock': clock,
                'delay': delay
            })
        
        return delays
    
    def _parse_sdc_false_paths(self, content: str) -> List[Dict[str, Any]]:
        """Parse false path declarations from SDC"""
        paths = []
        
        # Match: set_false_path -from <from> -to <to>
        path_pattern = r'set_false_path\s+(?:-from\s+(\w+))?(?:\s+-to\s+(\w+))?'
        matches = re.findall(path_pattern, content, re.IGNORECASE)
        
        for match in matches:
            from_node = match[0] if match[0] else "*"
            to_node = match[1] if match[1] else "*"
            
            paths.append({
                'from': from_node,
                'to': to_node
            })
        
        return paths
    
    def parse_upf(self, upf_file: str) -> Dict[str, Any]:
        """
        Parse UPF (Unified Power Format) file
        
        Args:
            upf_file: Path to the UPF file
            
        Returns:
            Dictionary containing power information:
            {
                'power_domains': [{'name': str, 'supply': str, 'ground': str}, ...],
                'voltage_domains': [...],
                'power_switches': [...],
                'isolation_cells': [...]
            }
        """
        self.logger.info(f"Parsing UPF file: {upf_file}")
        
        if not os.path.exists(upf_file):
            self.logger.warning(f"UPF file not found: {upf_file}")
            return {
                'power_domains': [],
                'voltage_domains': [],
                'power_switches': [],
                'isolation_cells': []
            }
        
        with open(upf_file, 'r') as f:
            content = f.read()
        
        # Remove comments
        content = self._remove_comments(content)
        
        power_info = {
            'power_domains': self._parse_upf_power_domains(content),
            'voltage_domains': self._parse_upf_voltage_domains(content),
            'power_switches': self._parse_upf_power_switches(content),
            'isolation_cells': self._parse_upf_isolation_cells(content)
        }
        
        self.logger.info(f"Parsed {len(power_info['power_domains'])} power domains")
        return power_info
    
    def _parse_upf_power_domains(self, content: str) -> List[Dict[str, Any]]:
        """Parse power domain definitions from UPF"""
        domains = []
        
        # Match: create_power_domain -name <name> -supply {<supply>} -ground {<ground>}
        domain_pattern = r'create_power_domain\s+(?:-name\s+(\w+))?(?:\s+-supply\s+\{([^}]+)\})?(?:\s+-ground\s+\{([^}]+)\})?'
        matches = re.findall(domain_pattern, content, re.IGNORECASE)
        
        for match in matches:
            name = match[0] if match[0] else f"PD_{len(domains)}"
            supply = match[1].strip() if match[1] else "VDD"
            ground = match[2].strip() if match[2] else "VSS"
            
            domains.append({
                'name': name,
                'supply': supply,
                'ground': ground
            })
        
        return domains
    
    def _parse_upf_voltage_domains(self, content: str) -> List[Dict[str, Any]]:
        """Parse voltage domain definitions from UPF"""
        domains = []
        
        # Match: create_voltage_domain -name <name> -supply {<supply>}
        domain_pattern = r'create_voltage_domain\s+(?:-name\s+(\w+))?(?:\s+-supply\s+\{([^}]+)\})?'
        matches = re.findall(domain_pattern, content, re.IGNORECASE)
        
        for match in matches:
            name = match[0] if match[0] else f"VD_{len(domains)}"
            supply = match[1].strip() if match[1] else "VDD"
            
            domains.append({
                'name': name,
                'supply': supply,
                'voltage': 1.2  # Default voltage
            })
        
        return domains
    
    def _parse_upf_power_switches(self, content: str) -> List[Dict[str, Any]]:
        """Parse power switch definitions from UPF"""
        switches = []
        
        # Match: create_power_switch -name <name> -domain <domain>
        switch_pattern = r'create_power_switch\s+(?:-name\s+(\w+))?(?:\s+-domain\s+(\w+))?'
        matches = re.findall(switch_pattern, content, re.IGNORECASE)
        
        for match in matches:
            name = match[0] if match[0] else f"PS_{len(switches)}"
            domain = match[1] if match[1] else "PD_0"
            
            switches.append({
                'name': name,
                'domain': domain
            })
        
        return switches
    
    def _parse_upf_isolation_cells(self, content: str) -> List[Dict[str, Any]]:
        """Parse isolation cell definitions from UPF"""
        cells = []
        
        # Match: set_isolation -domain <domain> -isolation_power_net <net>
        iso_pattern = r'set_isolation\s+(?:-domain\s+(\w+))?(?:\s+-isolation_power_net\s+(\w+))?'
        matches = re.findall(iso_pattern, content, re.IGNORECASE)
        
        for match in matches:
            domain = match[0] if match[0] else "PD_0"
            power_net = match[1] if match[1] else "VDD"
            
            cells.append({
                'domain': domain,
                'power_net': power_net
            })
        
        return cells
    
    def build_rtl_data(self, verilog_file: str, sdc_file: Optional[str] = None,
                      upf_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Build complete RTL data structure from all input files
        
        Args:
            verilog_file: Path to Verilog file
            sdc_file: Optional path to SDC constraints file
            upf_file: Optional path to UPF power file
            
        Returns:
            Complete RTL data structure
        """
        self.logger.info("Building complete RTL data structure")
        
        # Parse Verilog
        rtl_data = self.parse_verilog(verilog_file)
        
        # Parse constraints if provided
        if sdc_file:
            rtl_data['constraints'] = self.parse_sdc(sdc_file)
        else:
            rtl_data['constraints'] = {
                'clocks': [],
                'timing_paths': [],
                'input_delays': [],
                'output_delays': [],
                'false_paths': []
            }
        
        # Parse power info if provided
        if upf_file:
            rtl_data['power_info'] = self.parse_upf(upf_file)
        else:
            rtl_data['power_info'] = {
                'power_domains': [],
                'voltage_domains': [],
                'power_switches': [],
                'isolation_cells': []
            }
        
        return rtl_data
    
    def _remove_comments(self, content: str) -> str:
        """Remove comments from Verilog content"""
        # Remove line comments (// ...)
        content = re.sub(r'//.*?\n', '\n', content)
        # Remove block comments (/* ... */)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        return content
    
    def _parse_modules(self, content: str):
        """Parse module declarations"""
        self.logger.debug("Parsing module declarations.")
        # Match module declarations: module name(...);
        # Also capture optional parameter declarations if present (e.g., module TOP #(parameter A=1))
        module_pattern = r'module\s+(\w+)\s*(?:#\s*\((.*?)\))?\s*\('
        matches = re.finditer(module_pattern, content, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            module_name = match.group(1).strip()
            param_str = match.group(2) # Raw parameter string

            # Initialize module info
            module_info = {
                'name': module_name,
                'ports': [], # Will be populated by _parse_ports
                'instances': [], # Will be populated by _parse_instances
                'nets': [], # Will be populated by _parse_nets
                'parameters': self._parse_module_parameters(param_str) if param_str else {}
            }
            self.modules[module_name] = module_info
            self.logger.debug(f"  Parsed module: {module_name} with parameters: {module_info['parameters']}")

    def _parse_module_parameters(self, param_str: str) -> Dict[str, Any]:
        """Parses module parameters from a parameter declaration block."""
        params = {}
        # Match "parameter NAME = VALUE" or "NAME = VALUE"
        param_pattern = r'(?:parameter\s+)?(\w+)\s*=\s*([^,;]+)'
        matches = re.finditer(param_pattern, param_str)
        for match in matches:
            name = match.group(1).strip()
            value = match.group(2).strip()
            # Attempt to convert to int/float
            try:
                params[name] = int(value)
            except ValueError:
                try:
                    params[name] = float(value)
                except ValueError:
                    params[name] = value
        return params

    
    def _parse_instances(self, content: str):
        """Parse module instantiations, including parameters"""
        self.logger.debug("Parsing module instances.")
        
        # Split by semicolons to handle multi-line instances
        statements = content.split(';')
        
        for statement in statements:
            # Skip if it looks like a module/port/net declaration
            if any(kw in statement.lower() for kw in ['module ', 'input ', 'output ', 'inout ', 'wire ', 'reg ', 'assign ']):
                continue
            
            # Match instance pattern: module_type #(params) instance_name (.port(...), ...)
            # Captures module_type, optional parameters, instance_name, and connections
            instance_pattern = r'(\w+)\s*(?:#\s*\((.*?)\)\s*)?(\w+)\s*\(\s*(.*?)\s*\)'
            match = re.search(instance_pattern, statement, re.DOTALL)
            
            if match:
                module_type = match.group(1).strip()
                param_str = match.group(2) # Raw parameter string if present
                instance_name = match.group(3).strip()
                connections_str = match.group(4)
                
                parsed_params = self._parse_instance_parameters(param_str) if param_str else {}
                connections = self._parse_connections(connections_str)
                
                instance = {
                    'name': instance_name,
                    'type': module_type,
                    'parameters': parsed_params,
                    'connections': connections
                }
                
                # Avoid duplicates and empty instance names
                if instance_name and not any(inst['name'] == instance_name for inst in self.instances):
                    self.instances.append(instance)
                    self.logger.debug(f"  Parsed instance: {instance_name} of type {module_type}")

    def _parse_instance_parameters(self, param_str: str) -> Dict[str, Any]:
        """Parses parameters from an instance's #(param_name=value, ...) block"""
        params = {}
        # Match pattern: .PARAM_NAME(VALUE) or PARAM_NAME=VALUE
        param_pattern = r'(?:\.(\w+)\s*\(\s*([^)]+)\s*\)|\s*(\w+)\s*=\s*([^,]+)(?:,\s*|\s*$))'
        
        matches = re.finditer(param_pattern, param_str)
        for match in matches:
            if match.group(1): # Matched .PARAM(VALUE)
                param_name = match.group(1)
                param_value = match.group(2).strip()
            elif match.group(3): # Matched PARAM=VALUE
                param_name = match.group(3)
                param_value = match.group(4).strip()
            else:
                continue
            
            # Attempt to convert to int/float if possible
            try:
                params[param_name] = int(param_value)
            except ValueError:
                try:
                    params[param_name] = float(param_value)
                except ValueError:
                    params[param_name] = param_value # Keep as string
        return params

    
    def _parse_connections(self, connections_str: str) -> List[Tuple[str, str]]:
        """Parse port connections from instance instantiation"""
        connections = []
        
        # Match connections like .port_name(net_name) or .port_name(expression)
        conn_pattern = r'\.(\w+)\s*\(\s*([^)]+?)\s*\)'
        matches = re.findall(conn_pattern, connections_str)
        
        for port_name, net_expr in matches:
            # Clean up the net expression
            net_name = net_expr.strip()
            # Remove any array indices or expressions, keep just the net name
            net_name = re.split(r'[\[<]', net_name)[0].strip()
            
            connections.append((port_name, net_name))
        
        return connections
    
    def _parse_ports(self, content: str):
        """Parse module port declarations"""
        self.logger.debug("Parsing module ports.")
        # Match input/output/inout declarations, capturing direction, optional width, and port names
        # Example: input [31:0] data_in,
        #          output ready,
        #          inout [7:0] data_bus
        port_pattern = r'(input|output|inout)\s*(?:\[\s*(\d+)\s*:\s*(\d+)\s*\])?\s+([\w,\s]+)\s*;'
        matches = re.finditer(port_pattern, content, re.IGNORECASE)
        
        for match in matches:
            direction = match.group(1).lower()
            msb_str = match.group(2)
            lsb_str = match.group(3)
            port_list_str = match.group(4)
            
            width = 1 # Default width for single-bit ports
            if msb_str is not None and lsb_str is not None:
                msb = int(msb_str)
                lsb = int(lsb_str)
                width = abs(msb - lsb) + 1
            
            # Split port list by commas and clean up
            ports = [p.strip() for p in port_list_str.split(',') if p.strip()]
            for port in ports:
                port_info = {
                    'name': port,
                    'direction': direction,
                    'width': width
                }
                self.ports.append(port_info)
                self.logger.debug(f"  Parsed port: {port} (direction: {direction}, width: {width})")
    
    def _parse_nets(self, content: str):
        """Parse net declarations"""
        self.logger.debug("Parsing nets.")
        # Match wire/reg declarations, capturing type, optional width, and net names
        net_pattern = r'(wire|reg)\s*(?:\[\s*(\d+)\s*:\s*(\d+)\s*\])?\s+([\w,\s]+)\s*;'
        matches = re.finditer(net_pattern, content, re.IGNORECASE)
        
        for match in matches:
            net_type = match.group(1).lower()
            msb_str = match.group(2)
            lsb_str = match.group(3)
            net_list_str = match.group(4)
            
            width = 1 # Default width for single-bit nets
            if msb_str is not None and lsb_str is not None:
                msb = int(msb_str)
                lsb = int(lsb_str)
                # Handle both [MSB:LSB] and [LSB:MSB]
                width = abs(msb - lsb) + 1
            
            # Split net list by commas and clean up
            nets = [n.strip() for n in net_list_str.split(',') if n.strip()]
            for net in nets:
                net_info = {
                    'name': net,
                    'type': net_type,
                    'width': width
                }
                self.nets.append(net_info)
                self.logger.debug(f"  Parsed net: {net} (type: {net_type}, width: {width})")
        """Parse net declarations"""
        # Match wire/reg declarations
        net_pattern = r'(wire|reg)(?:\s*\[\s*(\d+)\s*:\s*(\d+)\s*\])?\s+([\w,\s]+)'
        matches = re.findall(net_pattern, content, re.IGNORECASE)
        
        for net_type, msb, lsb, net_list in matches:
            # Calculate width
            if msb and lsb:
                width = abs(int(msb) - int(lsb)) + 1
            else:
                width = 1
            
            # Split net list by commas and clean up
            nets = [n.strip() for n in net_list.split(',')]
            for net in nets:
                if net:  # Skip empty strings
                    net_info = {
                        'name': net,
                        'type': net_type.lower(),
                        'width': width
                    }
                    self.nets.append(net_info)
    
    def _parse_assignments(self, content: str):
        """Parse continuous assignments"""
        # Match assign statements
        assign_pattern = r'assign\s+(\w+(?:\[[^]]*\])?)\s*=\s*([^;]+);'
        matches = re.findall(assign_pattern, content, re.IGNORECASE)
        
        for lhs, rhs in matches:
            # Clean up the left-hand side
            lhs_clean = re.split(r'[\[<]', lhs)[0].strip()
            
            self.assignments.append({
                'lhs': lhs_clean,
                'rhs': rhs.strip()
            })
    
    def _parse_parameters(self, content: str):
        """Parse parameter declarations"""
        # Match parameter declarations
        param_pattern = r'parameter\s+(?:\w+\s+)?(\w+)\s*=\s*([^;,]+)'
        matches = re.findall(param_pattern, content, re.IGNORECASE)
        
        for param_name, param_value in matches:
            self.parameters[param_name] = param_value.strip()
    
    def _identify_top_module(self) -> str:
        """Try to identify the top-level module"""
        # Get all instantiated module types
        instantiated_types = set(inst['type'] for inst in self.instances)
        
        # Find modules that are NOT instantiated (potential top modules)
        potential_tops = []
        for mod_name in self.modules:
            if mod_name not in instantiated_types:
                potential_tops.append(mod_name)
        
        # If we have exactly one potential top, return it
        if len(potential_tops) == 1:
            return potential_tops[0]
        
        # If we have multiple potential tops, return the one with the most instances
        if potential_tops:
            max_instances = -1
            top_module = potential_tops[0]
            for mod_name in potential_tops:
                inst_count = len([i for i in self.instances if i['type'] == mod_name])
                if inst_count > max_instances:
                    max_instances = inst_count
                    top_module = mod_name
            return top_module
        
        # Otherwise, return the first module
        return list(self.modules.keys())[0] if self.modules else ""
