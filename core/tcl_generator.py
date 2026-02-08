"""
TCL Generation Service - Produces industry-standard scripts for EDA tools.
Supports Cadence Innovus and Synopsys Fusion Compiler flows.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import os
from core.pdk_manager import PDKManager

class BaseTCLGenerator(ABC):
    """Abstract base class for TCL generation"""
    
    @abstractmethod
    def generate_full_flow(self, design_config: Dict[str, Any]) -> str:
        """Generate a complete physical design flow script"""
        pass

class InnovusTCLGenerator(BaseTCLGenerator):
    """Generates TCL scripts for Cadence Innovus"""
    
    def generate_full_flow(self, config: Dict[str, Any]) -> str:
        design_name = config.get('design_name', 'top')
        verilog_file = config.get('verilog', '')
        sdc_file = config.get('sdc', '')
        
        # PDK Integration
        pdk_variant = config.get('pdk_variant', 'hd')
        pdk_root = config.get('pdk_root')
        pdk_mgr = PDKManager(pdk_root=pdk_root)
        
        lefs = config.get('lefs', [])
        if not lefs:
            lefs = pdk_mgr.get_lef_paths(variant=pdk_variant)
        
        libs = config.get('libs', [])
        if not libs:
            libs = pdk_mgr.get_lib_paths(variant=pdk_variant)
            
        lef_files = " ".join(lefs)
        lib_files = " ".join(libs)
        
        tcl = [
            "# --- Innovus Physical Design Flow (SkyWater Aware) ---",
            "set_multi_cpu_usage -local_cpu 4",
            f"set_db design_process_node {config.get('node', 130)}",
            "",
            "# 1. Import Design",
            f"set_db init_lef_file \"{lef_files}\"",
            f"set_db init_verilog \"{verilog_file}\"",
            f"set_db init_top_cell \"{design_name}\"",
            "init_design",
            f"read_sdc \"{sdc_file}\"",
            "",
            "# 2. Common Setup",
            "set_db connect_global_net VDD -type pg_pin -pin_base_name VDD",
            "set_db connect_global_net VSS -type pg_pin -pin_base_name VSS",
            "",
            "# 3. Floorplan",
            "create_floorplan -core_density 0.7 -aspect_ratio 1.0",
            "",
            "# 4. Placement",
            "set_db place_design_style indigenous",
            "place_design",
            "check_place",
            "",
            "# 5. Clock Tree Synthesis (CTS)",
            "create_ccopt_clock_tree_spec",
            "ccopt_design",
            "",
            "# 6. Routing",
            "set_db route_design_detail_post_route_spread_wire false",
            "route_design -global -detail",
            "",
            "# 7. Reporting",
            "report_timing > timing_report.txt",
            "report_power > power_report.txt",
            "report_area > area_report.txt",
            "exit"
        ]
        return "\n".join(tcl)

class FusionCompilerTCLGenerator(BaseTCLGenerator):
    """Generates TCL scripts for Synopsys Fusion Compiler"""
    
    def generate_full_flow(self, config: Dict[str, Any]) -> str:
        design_name = config.get('design_name', 'top')
        
        tcl = [
            "# --- Fusion Compiler Physical Design Flow ---",
            "set_app_var additive_timing_derating true",
            f"set design_name {design_name}",
            "",
            "# 1. Setup & Read Design",
            "create_lib -technology $tech_file $lib_name",
            f"read_verilog {config.get('verilog', '')}",
            f"link_block",
            f"read_sdc {config.get('sdc', '')}",
            "",
            "# 2. Place & Optimize",
            "place_opt",
            "check_legality",
            "",
            "# 3. Clock Tree Synthesis",
            "clock_opt",
            "",
            "# 4. Route & Post-Route Opt",
            "route_opt",
            "",
            "# 5. Reporting",
            "report_qor > qor_summary.txt",
            "report_timing -significant_digits 3 > timing.rpt",
            "exit"
        ]
        return "\n".join(tcl)

class TCLGeneratorFactory:
    """Factory to produce the appropriate generator for a platform"""
    
    _generators = {
        'innovus': InnovusTCLGenerator,
        'fusion_compiler': FusionCompilerTCLGenerator
    }
    
    @classmethod
    def get_generator(cls, platform: str) -> BaseTCLGenerator:
        generator_class = cls._generators.get(platform.lower())
        if not generator_class:
            raise ValueError(f"Unsupported platform: {platform}")
        return generator_class()
