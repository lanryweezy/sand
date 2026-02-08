"""
OpenROAD Interface - Integration with OpenROAD as baseline P&R engine

This module provides an interface to OpenROAD tools for physical implementation,
allowing the AI system to leverage proven P&R capabilities while adding
intelligent optimization and decision-making.
"""

import os
import subprocess
import tempfile
from typing import Dict, List, Any, Optional
from pathlib import Path
import shutil

import re # Added import for regular expressions

from silicon_intelligence.utils.logger import get_logger


class OpenROADInterface:
    """
    Interface to OpenROAD tools for physical implementation
    """
    
    def __init__(self, openroad_executable: str = "openroad"):
        self.logger = get_logger(__name__)
        self.openroad_executable = openroad_executable
        self.check_availability()
    
    def check_availability(self):
        """Check if OpenROAD is available in the system"""
        try:
            result = subprocess.run([self.openroad_executable, "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.available = True
                self.version = result.stdout.strip()
                self.logger.info(f"OpenROAD available: {self.version}")
            else:
                self.available = False
                self.logger.warning("OpenROAD not available")
        except FileNotFoundError:
            self.available = False
            self.logger.warning("OpenROAD executable not found")
        except Exception as e:
            self.available = False
            self.logger.warning(f"Error checking OpenROAD availability: {str(e)}")
    
    def run_flow_script(self, script_content: str, design_dir: str = None, timeout_seconds: int = 300) -> Dict[str, Any]:
        """
        Run an OpenROAD Tcl script
        
        Args:
            script_content: Tcl script content to execute
            design_dir: Directory for the design (uses temp if None)
            timeout_seconds: Timeout for the OpenROAD execution in seconds.
            
        Returns:
            Dictionary with execution results
        """
        if not self.available:
            raise RuntimeError("OpenROAD is not available")
        
        # Create temporary directory if not provided
        if design_dir is None:
            design_dir = tempfile.mkdtemp(prefix="si_openroad_")
        
        # Write script to temporary file
        script_path = os.path.join(design_dir, "flow_script.tcl")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        self.logger.info(f"Running OpenROAD script: {script_path} with timeout {timeout_seconds}s")
        
        try:
            # Execute the script
            result = subprocess.run(
                [self.openroad_executable, script_path],
                cwd=design_dir,
                capture_output=True,
                text=True,
                timeout=timeout_seconds # Use the configurable timeout
            )
            
            # Parse output for metrics
            parsed_metrics = self._parse_openroad_output(result.stdout, result.stderr)

            execution_result = {
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'script_path': script_path,
                'design_dir': design_dir,
                'success': result.returncode == 0,
                'metrics': parsed_metrics # Include parsed metrics
            }
            
            if result.returncode != 0:
                self.logger.error(f"OpenROAD script failed: {result.stderr}")
            else:
                self.logger.info("OpenROAD script executed successfully")
            
            return execution_result
            
        except subprocess.TimeoutExpired:
            self.logger.error("OpenROAD script timed out")
            return {
                'return_code': -1,
                'stdout': '',
                'stderr': 'Script execution timed out',
                'script_path': script_path,
                'design_dir': design_dir,
                'success': False,
                'metrics': {}
            }
        except Exception as e:
            self.logger.error(f"Error running OpenROAD script: {str(e)}")
            return {
                'return_code': -1,
                'stdout': '',
                'stderr': str(e),
                'script_path': script_path,
                'design_dir': design_dir,
                'success': False,
                'metrics': {}
            }
                'design_dir': design_dir,
                'success': result.returncode == 0
            }
            
            if result.returncode != 0:
                self.logger.error(f"OpenROAD script failed: {result.stderr}")
            else:
                self.logger.info("OpenROAD script executed successfully")
            
            return execution_result
            
        except subprocess.TimeoutExpired:
            self.logger.error("OpenROAD script timed out")
            return {
                'return_code': -1,
                'stdout': '',
                'stderr': 'Script execution timed out',
                'script_path': script_path,
                'design_dir': design_dir,
                'success': False
            }
        except Exception as e:
            self.logger.error(f"Error running OpenROAD script: {str(e)}")
            return {
                'return_code': -1,
                'stdout': '',
                'stderr': str(e),
                'script_path': script_path,
                'design_dir': design_dir,
                'success': False
            }
    
    def generate_floorplan_script(self, design_data: Dict[str, Any]) -> str:
        """
        Generate OpenROAD Tcl script for floorplanning
        
        Args:
            design_data: Design information including macros, dimensions, etc.
            
        Returns:
            Tcl script content for floorplanning
        """
        self.logger.debug("Generating OpenROAD floorplan script.")
        # Extract design parameters
        design_file = design_data.get('design_file', 'temp_design.v')
        top_module = design_data.get('top_module', 'top')
        die_area = design_data.get('die_area', [0, 0, 1000, 1000])  # [lx, ly, ux, uy]
        core_area = design_data.get('core_area', [50, 50, 950, 950])  # [lx, ly, ux, uy]
        macros = design_data.get('macros', [])
        placement_grid = design_data.get('placement_grid', None)
        
        script_lines = [
            "# Generated Floorplan Script by Silicon Intelligence System",
            "# Purpose: To create a physical floorplan for the design",
            f"# Design File: {design_file}",
            f"# Top Module: {top_module}",
            "",
            "# Read design",
            f"read_lef {{./tech_lef/tech.lef}}", # Assuming standard cell LEF is available
            f"read_def -order_lef {{./design_def/{design_file.replace('.v', '.def')}}}", # Assuming a DEF for hierarchy
            f"read_verilog {{./design_verilog/{design_file}}}",
            f"link_design {top_module}",
            "",
            "# Define die and core areas",
            f"set die_lx {die_area[0]}",
            f"set die_ly {die_area[1]}",
            f"set die_ux {die_area[2]}",
            f"set die_uy {die_area[3]}",
            "",
            f"set core_lx {core_area[0]}",
            f"set core_ly {core_area[1]}",
            f"set core_ux {core_area[2]}",
            f"set core_uy {core_area[3]}",
            "",
            "# Set floorplan",
            "floorplan -die_area $die_lx $die_ly $die_ux $die_uy \\",
            "          -core_area $core_lx $core_ly $core_ux $core_uy",
            ""
        ]
        
        # Set placement grid if specified
        if placement_grid:
            script_lines.append(f"set_placement_grid -site {placement_grid.get('site_name', 'default')}")
            script_lines.append("")

        # Add macro placements and dont_touch attributes
        for i, macro in enumerate(macros):
            name = macro.get('name', f'macro_{i}')
            origin = macro.get('origin', [100, 100])
            orientation = macro.get('orientation', 'N')
            is_dont_touch = macro.get('dont_touch', False)
            
            script_lines.extend([
                f"# Place macro {name}",
                f"place_macro {name} {origin[0]} {origin[1]} -orientation {orientation}"
            ])
            if is_dont_touch:
                script_lines.append(f"set_dont_touch [get_inst {name}]")
            script_lines.append("")
        
        script_lines.extend([
            "# Save intermediate results",
            "write_def {output.def}", # Use generic names for intermediate files
            "write_db {output.db}",
            "write_floorplan_tcl {output_floorplan.tcl}",
            ""
        ])
        
        return "\n".join(script_lines)
    
    def generate_placement_script(self, design_data: Dict[str, Any]) -> str:
        """
        Generate OpenROAD Tcl script for placement
        
        Args:
            design_data: Design information for placement
            
        Returns:
            Tcl script content for placement
        """
        self.logger.debug("Generating OpenROAD placement script.")
        # Extract design parameters
        input_def = design_data.get('input_def', 'floorplan.def')
        input_db = design_data.get('input_db', 'floorplan.db')
        output_def = design_data.get('output_def', 'placement.def')
        output_db = design_data.get('output_db', 'placement.db')
        
        placement_strategy = design_data.get('placement_strategy', 'default')
        placement_effort = design_data.get('placement_effort', 'medium') # low, medium, high
        timing_driven_weight = design_data.get('timing_driven_weight', 1.0)
        congestion_driven_weight = design_data.get('congestion_driven_weight', 1.0)
        
        script_lines = [
            "# Generated Placement Script by Silicon Intelligence System",
            "# Purpose: To perform global and detailed placement",
            "",
            "# Read design and floorplan",
            f"read_def {{./design_def/{input_def}}}",
            f"read_db {{./design_db/{input_db}}}",
            "",
            "# Global placement",
            "estimate_parasitics -placement",
            "global_placement",
            f"set_global_placement_params -timing_driven_weight {timing_driven_weight} \\",
            f"                          -congestion_driven_weight {congestion_driven_weight} \\",
            f"                          -effort {placement_effort}",
            "global_placement", # Rerun with params
            "",
            "# Detailed placement",
            "detailed_placement",
            "",
            "# Cell padding (if specified, e.g. for density control)",
            f"set_pad_physical_constraints -global_cpp {design_data.get('pad_cpp', 0)}", # Placeholder
            "",
            "# Save results",
            f"write_def {{./design_def/{output_def}}}",
            f"write_db {{./design_db/{output_db}}}",
            f"write_placement_tcl {{output_placement.tcl}}",
            ""
        ]
        
        return "\n".join(script_lines)
    
    def generate_cts_script(self, design_data: Dict[str, Any]) -> str:
        """
        Generate OpenROAD Tcl script for clock tree synthesis
        
        Args:
            design_data: Design information for CTS
            
        Returns:
            Tcl script content for CTS
        """
        self.logger.debug("Generating OpenROAD CTS script.")
        # Extract design parameters
        input_def = design_data.get('input_def', 'placement.def')
        input_db = design_data.get('input_db', 'placement.db')
        output_def = design_data.get('output_def', 'cts.def')
        output_db = design_data.get('output_db', 'cts.db')
        
        clock_tree_name = design_data.get('clock_tree_name', 'clk_tree')
        target_skew = design_data.get('target_skew', 0.1) # ns
        max_transition = design_data.get('max_transition', 0.5) # ns
        buffer_list = design_data.get('buffer_list', ['CLKBUF_X1', 'CLKBUF_X2'])
        
        clocks = design_data.get('clocks', []) # List of {'name': 'clk', 'period': 1000, 'source': 'clk_pin'}

        script_lines = [
            "# Generated CTS Script by Silicon Intelligence System",
            "# Purpose: To perform clock tree synthesis",
            "",
            "# Read design and placement",
            f"read_def {{./design_def/{input_def}}}",
            f"read_db {{./design_db/{input_db}}}",
            "",
            "# Set design units",
            "set_cmd_units -time ns",
            "set_cmd_units -capacitance pF",
            "set_cmd_units -resistance kOhms",
            "set_cmd_units -voltage V",
            "set_cmd_units -current mA",
            "",
            "# Define clocks",
            "# For simplicity, assume one clock for now unless specified",
            ]
        
        if clocks:
            for clk in clocks:
                script_lines.append(f"create_clock -name {clk['name']} -period {clk['period']} -waveform {{0 {clk['period']/2}}} [get_ports {clk['source']}]")
        else:
            # Default clock definition
            script_lines.append(f"create_clock -name default_clk -period 1000 -waveform {{0 500}} [get_ports clk_pin]")
        script_lines.append("")

        script_lines.extend([
            "# Configure CTS parameters",
            f"set_ccopt_property max_transition {max_transition} -clock_tree {clock_tree_name}",
            f"set_ccopt_property target_skew {target_skew} -clock_tree {clock_tree_name}",
            f"set_ccopt_property buffer_list {{ {' '.join(buffer_list)} }} -clock_tree {clock_tree_name}",
            "",
            "# Perform CTS",
            "create_clock_tree_spec",
            "ccopt_design",
            "",
            "# Report CTS results",
            "report_ccopt_skew",
            "report_ccopt_summary",
            "",
            "# Save results",
            f"write_def {{./design_def/{output_def}}}",
            f"write_db {{./design_db/{output_db}}}",
            f"write_cts_tcl {{output_cts.tcl}}",
            ""
        ])
        
        return "\n".join(script_lines)
    
    def generate_routing_script(self, design_data: Dict[str, Any]) -> str:
        """
        Generate OpenROAD Tcl script for routing
        
        Args:
            design_data: Design information for routing
            
        Returns:
            Tcl script content for routing
        """
        self.logger.debug("Generating OpenROAD routing script.")
        # Extract design parameters
        input_def = design_data.get('input_def', 'cts.def')
        input_db = design_data.get('input_db', 'cts.db')
        output_def = design_data.get('output_def', 'route.def')
        output_gds = design_data.get('output_gds', 'route.gds')
        output_db = design_data.get('output_db', 'route.db')
        
        num_threads = design_data.get('num_threads', 8)
        min_routing_layer = design_data.get('min_routing_layer', 'metal1')
        max_routing_layer = design_data.get('max_routing_layer', 'metal7')
        clock_routing_layers = design_data.get('clock_routing_layers', ['metal5', 'metal6', 'metal7'])
        routing_effort = design_data.get('routing_effort', 'medium') # low, medium, high
        
        script_lines = [
            "# Generated Routing Script by Silicon Intelligence System",
            "# Purpose: To perform global and detailed routing",
            "",
            "# Read design and CTS",
            f"read_def {{./design_def/{input_def}}}",
            f"read_db {{./design_db/{input_db}}}",
            "",
            "# Global routing",
            f"set_thread_count {num_threads}",
            "estimate_parasitics -placement",
            "",
            "# Configure global router",
            "global_route \\",
            f"  -min_routing_layer {min_routing_layer} \\",
            f"  -max_routing_layer {max_routing_layer} \\",
            f"  -clock_routing_layers {{ {' '.join(clock_routing_layers)} }} \\",
            f"  -effort {routing_effort}",
            "",
            "# Detailed routing",
            "detailed_route \\",
            f"  -min_routing_layer {min_routing_layer} \\",
            f"  -max_routing_layer {max_routing_layer} \\",
            f"  -clock_min_routing_layer {min_routing_layer} \\",
            f"  -clock_max_routing_layer {max_routing_layer} \\", # Assuming clock routing can use all layers
            f"  -effort {routing_effort}",
            "",
            "# Via optimization",
            "# set_dont_use {dont_use_cells}", # Example, would come from design_data
            "optimize_metal_hop",
            "",
            "# Save results",
            f"write_def {{./design_def/{output_def}}}",
            f"write_gdsii {{./design_gds/{output_gds}}}",
            f"write_db {{./design_db/{output_db}}}",
            f"write_route_tcl {{output_route.tcl}}",
            ""
        ]
        
        return "\n".join(script_lines)
    
    def integrate_with_agents(self, design_data: Dict[str, Any], 
                           floorplan_proposal: Optional[Dict] = None,
                           placement_proposal: Optional[Dict] = None,
                           cts_proposal: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Integrate OpenROAD execution with agent proposals
        
        Args:
            design_data: Base design information
            floorplan_proposal: Proposal from floorplan agent
            placement_proposal: Proposal from placement agent
            cts_proposal: Proposal from clock agent
            
        Returns:
            Results from integrated flow
        """
        self.logger.info("Integrating OpenROAD with agent proposals")
        
        # Create temporary directory for the flow
        flow_dir = tempfile.mkdtemp(prefix="si_integration_")
        
        try:
            # Modify design data based on agent proposals
            modified_design = self._apply_agent_proposals(
                design_data, floorplan_proposal, placement_proposal, cts_proposal
            )
            
            # Generate and run floorplan step
            floorplan_script = self.generate_floorplan_script(modified_design)
            floorplan_result = self.run_flow_script(floorplan_script, flow_dir)
            
            if not floorplan_result['success']:
                return {'success': False, 'error': 'Floorplan step failed'}
            
            # Generate and run placement step
            placement_script = self.generate_placement_script(modified_design)
            placement_result = self.run_flow_script(placement_script, flow_dir)
            
            if not placement_result['success']:
                return {'success': False, 'error': 'Placement step failed'}
            
            # Generate and run CTS step
            cts_script = self.generate_cts_script(modified_design)
            cts_result = self.run_flow_script(cts_script, flow_dir)
            
            if not cts_result['success']:
                return {'success': False, 'error': 'CTS step failed'}
            
            # Generate and run routing step
            routing_script = self.generate_routing_script(modified_design)
            routing_result = self.run_flow_script(routing_script, flow_dir)
            
            if not routing_result['success']:
                return {'success': False, 'error': 'Routing step failed'}
            
            # Compile results
            integration_results = {
                'success': True,
                'flow_directory': flow_dir,
                'steps_completed': ['floorplan', 'placement', 'cts', 'routing'],
                'results': {
                    'floorplan': floorplan_result,
                    'placement': placement_result,
                    'cts': cts_result,
                    'routing': routing_result
                }
            }
            
            self.logger.info("Integration flow completed successfully")
            return integration_results
            
        except Exception as e:
            self.logger.error(f"Integration flow failed: {str(e)}")
            return {'success': False, 'error': str(e)}
        finally:
            # Optionally clean up temp directory based on success
            # For now, we'll leave it for debugging
            pass
    
    def _apply_agent_proposals(self, design_data: Dict[str, Any],
                             floorplan_proposal: Optional[Dict],
                             placement_proposal: Optional[Dict],
                             cts_proposal: Optional[Dict]) -> Dict[str, Any]:
        """
        Apply agent proposals to modify design data
        """
        modified_design = design_data.copy()
        
        # Apply floorplan proposal
        if floorplan_proposal:
            params = floorplan_proposal.get('parameters', {})
            strategy = params.get('strategy')
            
            if strategy == 'compact':
                # Adjust areas for compact placement
                modified_design['core_area'] = [100, 100, 900, 900]
            elif strategy == 'hierarchical':
                # Prepare for hierarchical placement
                modified_design['hierarchical'] = True
            elif strategy == 'ring':
                # Prepare for ring placement
                modified_design['ring_placement'] = True
            
            # Apply macro placements
            macro_placements = params.get('macro_placements', {})
            if 'macros' in modified_design:
                for macro_name, placement_info in macro_placements.items():
                    for i, macro in enumerate(modified_design['macros']):
                        if macro['name'] == macro_name:
                            modified_design['macros'][i].update(placement_info)
        
        # Apply placement proposal
        if placement_proposal:
            params = placement_proposal.get('parameters', {})
            strategy = params.get('strategy')
            
            if strategy:
                modified_design['placement_strategy'] = strategy
            
            # Apply cell-specific constraints
            cell_placements = params.get('cell_placements', {})
            modified_design['cell_constraints'] = cell_placements
        
        # Apply CTS proposal
        if cts_proposal:
            params = cts_proposal.get('parameters', {})
            strategy = params.get('strategy')
            
            if strategy:
                modified_design['cts_strategy'] = strategy
            
            # Apply clock-specific constraints
            skew_reqs = params.get('skew_requirements', {})
            modified_design['skew_requirements'] = skew_reqs
        
                
        
                return modified_design
        
        
        
            def _parse_openroad_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        
                """
        
                Parses OpenROAD stdout/stderr to extract key metrics.
        
                This is a simplified parser and needs to be expanded for production use.
        
                """
        
                self.logger.debug("Parsing OpenROAD output for metrics.")
        
                metrics = {}
        
                
        
                # Example: Extract Total Area
        
                area_match = re.search(r"Total Area:\s+([\d\.]+)\s+um\^2", stdout)
        
                if area_match:
        
                    metrics['total_area_um2'] = float(area_match.group(1))
        
                
        
                # Example: Extract Wirelength (HPWL)
        
                wirelength_match = re.search(r"HPWL:\s+([\d\.]+)\s+um", stdout)
        
                if wirelength_match:
        
                    metrics['hpwl_um'] = float(wirelength_match.group(1))
        
                    
        
                # Example: Extract WNS (Worst Negative Slack)
        
                wns_match = re.search(r"Worst Negative Slack \(WNS\):\s+([-\d\.]+)\s+ps", stdout)
        
                if wns_match:
        
                    metrics['wns_ps'] = float(wns_match.group(1))
        
                    
        
                # Example: Extract TNS (Total Negative Slack)
        
                tns_match = re.search(r"Total Negative Slack \(TNS\):\s+([-\d\.]+)\s+ps", stdout)
        
                if tns_match:
        
                    metrics['tns_ps'] = float(tns_match.group(1))
        
        
        
                # Example: Extract Number of DRC Violations
        
                drc_match = re.search(r"Total DRC Violations:\s+(\d+)", stdout)
        
                if drc_match:
        
                    metrics['total_drc_violations'] = int(drc_match.group(1))
        
        
        
                # Example: Extract estimated congestion (can be complex, simplified here)
        
                congestion_match = re.search(r"Estimated Congestion:\s+([\d\.]+)", stdout)
        
                if congestion_match:
        
                    metrics['estimated_congestion'] = float(congestion_match.group(1))
        
                
        
                self.logger.debug(f"Extracted metrics: {metrics}")
        
                return metrics
        
        
        
        # Example usage function
def example_integration():
    """
    Example of how to integrate OpenROAD with the Silicon Intelligence System
    """
    logger = get_logger(__name__)
    
    # Initialize the interface
    try:
        interface = OpenROADInterface()
        logger.info("OpenROAD interface initialized")
    except Exception as e:
        logger.error(f"Failed to initialize OpenROAD interface: {str(e)}")
        return
    
    if not interface.available:
        logger.warning("OpenROAD not available, skipping integration example")
        return
    
    # Example design data
    design_data = {
        'design_file': 'my_design.v',
        'top_module': 'top',
        'die_area': [0, 0, 1000, 1000],
        'core_area': [50, 50, 950, 950],
        'macros': [
            {'name': 'memory_1', 'origin': [100, 100], 'orientation': 'N'},
            {'name': 'ip_block_1', 'origin': [200, 200], 'orientation': 'FN'}
        ]
    }
    
    # Example agent proposals (these would come from the agents)
    floorplan_proposal = {
        'parameters': {
            'strategy': 'hierarchical',
            'macro_placements': {
                'memory_1': {'region': 'data_path', 'preferred_orientation': 'N'},
                'ip_block_1': {'region': 'control_path', 'preferred_orientation': 'FN'}
            }
        }
    }
    
    placement_proposal = {
        'parameters': {
            'strategy': 'analytical',
            'cell_placements': {
                'critical_ff_1': {'priority': 2.0, 'region_constraint': 'timing_critical'}
            }
        }
    }
    
    cts_proposal = {
        'parameters': {
            'strategy': 'balanced_tree',
            'skew_requirements': {'target_ppm': 50, 'max_local_skew_ps': 10}
        }
    }
    
    # Run integrated flow
    results = interface.integrate_with_agents(
        design_data, floorplan_proposal, placement_proposal, cts_proposal
    )
    
    if results['success']:
        logger.info(f"Integration flow completed successfully in: {results['flow_directory']}")
        logger.info(f"Steps completed: {results['steps_completed']}")
    else:
        logger.error(f"Integration flow failed: {results['error']}")


if __name__ == "__main__":
    example_integration()