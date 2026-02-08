# silicon_intelligence/integration/eda_integration.py

from typing import Dict, Any
from core.canonical_silicon_graph import CanonicalSiliconGraph
from core.openroad_interface import OpenROADInterface, OpenROADConfig


class EDAIntegrationLayer:
    """Integration layer for connecting to various EDA tools"""
    
    def __init__(self, pdk_path: str = "", scl_path: str = "", lib_path: str = ""):
        self.pdk_path = pdk_path
        self.scl_path = scl_path
        self.lib_path = lib_path
    
    def run_openroad_flow(self, graph: CanonicalSiliconGraph, design_name: str) -> Dict[str, Any]:
        """Run complete OpenROAD flow (placement + routing)"""
        config = OpenROADConfig(
            pdk_path=self.pdk_path,
            scl_path=self.scl_path,
            design_name=design_name,
            top_module=design_name
        )
        
        interface = OpenROADInterface(config)
        
        results = {}
        
        try:
            # Run placement
            placement_results = interface.run_placement(graph)
            results['placement'] = placement_results
            
            # Run routing
            routing_results = interface.run_routing(graph)
            results['routing'] = routing_results
            
            # Combine results
            results['overall_ppa'] = self._combine_ppa_results(placement_results, routing_results)
            
            return results
            
        except Exception as e:
            return {
                'error': str(e),
                'placement': None,
                'routing': None,
                'overall_ppa': None
            }
    
    def _combine_ppa_results(self, placement_results: Dict, routing_results: Dict) -> Dict[str, Any]:
        """Combine placement and routing results into overall PPA metrics"""
        p_metrics = placement_results.get('metrics', {})
        r_metrics = routing_results.get('metrics', {})
        
        return {
            'area_um2': p_metrics.get('utilization', 0) * 1000000,  # Rough estimate
            'power_mw': p_metrics.get('cell_count', 1000) * 0.001,  # Rough estimate
            'timing_ns': r_metrics.get('timing_slack', 0),
            'drc_violations': r_metrics.get('drc_violations', 0),
            'wire_length_um': r_metrics.get('wire_length', 0),
            'congestion': r_metrics.get('routing_congestion', 0)
        }
    
    def run_tool_comparison(self, graph: CanonicalSiliconGraph, design_name: str) -> Dict[str, Any]:
        """Run design through multiple tools for comparison (mock implementation)"""
        results = {}
        
        # For now, only run OpenROAD as it's the only implemented interface
        results['openroad'] = self.run_openroad_flow(graph, design_name)
        
        # In a real implementation, we would also run Innovus and Fusion Compiler
        results['innovus'] = {
            'error': 'Innovus interface not yet implemented',
            'available': False
        }
        
        results['fusion_compiler'] = {
            'error': 'Fusion Compiler interface not yet implemented',
            'available': False
        }
        
        return results