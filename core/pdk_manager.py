"""
PDK Manager Service - Manages technology libraries and process-specific rules.
Focuses on the SkyWater 130nm (sky130) Process Design Kit.
"""

from typing import Dict, List, Any, Optional
import os

class PDKManager:
    """
    Manages Paths to LEFs, Libs, and technology rules for specific process nodes.
    Essential for professional-grade physical design scripts.
    """
    
    def __init__(self, pdk_root: Optional[str] = None):
        self.pdk_root = pdk_root or os.environ.get('PDK_ROOT', '/usr/local/pdk')
        self.current_node = "sky130"
        
        # Professional standard cell library configuration for SkyWater 130nm
        self.tech_configs = {
            'sky130': {
                'name': 'SkyWater 130nm',
                'process_node_nm': 130,
                'layers': ['li1', 'met1', 'met2', 'met3', 'met4', 'met5'],
                'sc_libs': {
                    'hd': {
                        'lef': 'sky130_fd_sc_hd/lef/sky130_fd_sc_hd.merged.lef',
                        'lib': {
                            'tt': 'sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib',
                            'ss': 'sky130_fd_sc_hd/lib/sky130_fd_sc_hd__ss_100C_1v60.lib',
                            'ff': 'sky130_fd_sc_hd/lib/sky130_fd_sc_hd__ff_n40C_1v95.lib'
                        }
                    }
                },
                'power_nets': {'vdd': 'VDD', 'vss': 'VSS'},
                'site_name': 'unithd'
            }
        }

    def get_lef_paths(self, node: str = 'sky130', variant: str = 'hd') -> List[str]:
        """Returns the full paths to relevant LEF files"""
        config = self.tech_configs.get(node)
        if not config:
            return []
        
        rel_path = config['sc_libs'][variant]['lef']
        return [os.path.join(self.pdk_root, 'sky130A/libs.ref', rel_path)]

    def get_lib_paths(self, node: str = 'sky130', variant: str = 'hd', corner: str = 'tt') -> List[str]:
        """Returns the full paths to relevant .lib files for a specific corner"""
        config = self.tech_configs.get(node)
        if not config:
            return []
        
        rel_path = config['sc_libs'][variant]['lib'][corner]
        return [os.path.join(self.pdk_root, 'sky130A/libs.ref', rel_path)]

    def get_tech_summary(self, node: str = 'sky130') -> Dict[str, Any]:
        """Returns a summary of the technology for EDA scripts"""
        return self.tech_configs.get(node, {})

def example_pdk_usage():
    manager = PDKManager(pdk_root="/home/designs/pdk")
    print(f"Technology: {manager.get_tech_summary()['name']}")
    print(f"LEF Path: {manager.get_lef_paths()[0]}")
    print(f"LIB (TT) Path: {manager.get_lib_paths(corner='tt')[0]}")

if __name__ == "__main__":
    example_pdk_usage()
