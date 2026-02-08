# silicon_intelligence/validation/design_validator.py

from typing import Dict, Any
from silicon_intelligence.core.canonical_silicon_graph import CanonicalSiliconGraph
from silicon_intelligence.integration.eda_integration import EDAIntegrationLayer


class DesignValidator:
    """Validates designs and compares predictions against actual results"""
    
    def __init__(self):
        self.eda_layer = EDAIntegrationLayer()
    
    def validate_predictions(self, predictions: Dict, actual_results: Dict) -> Dict[str, float]:
        """Validate predictions against actual results"""
        validation_results = {}
        
        # Compare each metric
        for metric in ['area', 'power', 'timing', 'drc_violations']:
            pred_key = metric
            actual_key = f'actual_{metric}'
            
            if pred_key in predictions and actual_key in actual_results:
                predicted = predictions[pred_key]
                actual = actual_results[actual_key]
                
                # Calculate error metrics
                abs_error = abs(predicted - actual)
                rel_error = abs_error / max(abs(actual), 1e-9)
                
                # Calculate accuracy (higher is better)
                accuracy = 1.0 / (1.0 + rel_error)
                
                validation_results[f'{metric}_abs_error'] = abs_error
                validation_results[f'{metric}_rel_error'] = rel_error
                validation_results[f'{metric}_accuracy'] = accuracy
            else:
                validation_results[f'{metric}_accuracy'] = 0.0
        
        return validation_results
    
    def run_validation_flow(self, graph: CanonicalSiliconGraph, design_name: str) -> Dict[str, Any]:
        """Run complete validation flow"""
        # Get predictions from the system (these would come from ML models)
        predictions = self._get_system_predictions(graph)
        
        # Run through EDA tools to get actual results
        tool_results = self.eda_layer.run_tool_comparison(graph, design_name)
        
        # Extract actual results from tool output
        actual_results = self._extract_actual_results(tool_results)
        
        # Validate predictions against actual results
        validation = self.validate_predictions(predictions, actual_results)
        
        return {
            'predictions': predictions,
            'actual_results': actual_results,
            'validation': validation,
            'tool_results': tool_results
        }
    
    def _get_system_predictions(self, graph: CanonicalSiliconGraph) -> Dict[str, float]:
        """Get predictions from the system (mock implementation)"""
        # In a real system, this would call ML models
        node_count = graph.graph.number_of_nodes()
        
        return {
            'area': node_count * 10,  # 10 um^2 per node estimate
            'power': node_count * 0.001,  # 0.001 mW per node estimate
            'timing': 2.0 + (node_count / 1000),  # Base 2ns + complexity
            'drc_violations': max(0, node_count // 200)  # Estimate DRC violations
        }
    
    def _extract_actual_results(self, tool_results: Dict) -> Dict[str, float]:
        """Extract actual results from tool output"""
        # Extract from OpenROAD results
        openroad_results = tool_results.get('openroad', {})
        overall_ppa = openroad_results.get('overall_ppa', {})
        
        return {
            'actual_area': overall_ppa.get('area_um2', 0),
            'actual_power': overall_ppa.get('power_mw', 0),
            'actual_timing': overall_ppa.get('timing_ns', 0),
            'actual_drc_violations': overall_ppa.get('drc_violations', 0)
        }