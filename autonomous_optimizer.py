#!/usr/bin/env python3
"""
Autonomous Design Optimizer
Uses predictions and insights to automatically optimize designs
"""

import copy
from typing import Dict, List, Any, Tuple
from enum import Enum

from physical_design_intelligence import PhysicalDesignIntelligence
from synthetic_design_generator import DesignSpec
from core.rtl_transformer import RTLTransformer
import tempfile
import os


class OptimizationStrategy(Enum):
    """Types of optimization strategies"""
    PIPELINE_CRITICAL_PATHS = "pipeline_critical_paths"
    CLUSTER_CONGESTED_AREAS = "cluster_congested_areas"
    REDUCE_REGISTER_COUNT = "reduce_register_count"
    OPTIMIZE_FANOUT = "optimize_fanout"
    BALANCE_AREA_POWER = "balance_area_power"
    CLOCK_GATING = "clock_gating"


class AutonomousOptimizer:
    """
    Autonomous optimization system that uses predictions and insights
    to automatically improve designs
    """
    
    def __init__(self):
        self.design_intelligence = PhysicalDesignIntelligence()
        self.rtl_transformer = RTLTransformer()
        self.optimization_strategies = {
            OptimizationStrategy.PIPELINE_CRITICAL_PATHS: self._apply_pipelining,
            OptimizationStrategy.CLUSTER_CONGESTED_AREAS: self._apply_clustering,
            OptimizationStrategy.REDUCE_REGISTER_COUNT: self._reduce_registers,
            OptimizationStrategy.OPTIMIZE_FANOUT: self._optimize_fanout,
            OptimizationStrategy.BALANCE_AREA_POWER: self._balance_area_power,
            OptimizationStrategy.CLOCK_GATING: self._apply_clock_gating
        }
    
    def analyze_and_optimize(self, rtl_content: str, design_name: str = "unnamed") -> Dict[str, Any]:
        """Analyze design and suggest/implement optimizations"""
        
        # Initial analysis
        initial_analysis = self.design_intelligence.analyze_design(rtl_content, design_name)
        
        # Identify optimization opportunities
        optimization_opportunities = self._identify_optimizations(initial_analysis)
        
        results = {
            'initial_analysis': initial_analysis,
            'optimization_opportunities': optimization_opportunities,
            'optimizations_applied': [],
            'final_analysis': None
        }
        
        # Apply optimizations
        optimized_rtl = rtl_content
        for strategy, params in optimization_opportunities:
            optimized_rtl = self._apply_strategy(optimized_rtl, strategy, params)
            results['optimizations_applied'].append({
                'strategy': strategy.value,
                'params': params,
                'applied': True
            })
        
        # Re-analyze optimized design
        if optimized_rtl != rtl_content:
            final_analysis = self.design_intelligence.analyze_design(optimized_rtl, f"optimized_{design_name}")
            results['final_analysis'] = final_analysis
            results['improvement'] = self._calculate_improvement(
                initial_analysis, final_analysis
            )
        else:
            results['final_analysis'] = initial_analysis
            results['improvement'] = {}
        
        return results
    
    def _identify_optimizations(self, analysis: Dict[str, Any]) -> List[Tuple[OptimizationStrategy, Dict]]:
        """Identify potential optimizations based on analysis"""
        
        opportunities = []
        
        # Check for timing violations
        timing_slack = analysis['openroad_results']['placement']['timing_slack_ps']
        if timing_slack < 0:
            # Critical path needs pipelining
            opportunities.append((
                OptimizationStrategy.PIPELINE_CRITICAL_PATHS,
                {'slack_required': abs(timing_slack)}
            ))
        
        # Check for congestion
        congestion_map = analysis['openroad_results']['placement']['congestion_map']
        if congestion_map and any(c['congestion_level'] > 0.8 for c in congestion_map):
            opportunities.append((
                OptimizationStrategy.CLUSTER_CONGESTED_AREAS,
                {'high_congestion_threshold': 0.8}
            ))
        
        # Check for high register count relative to combinational logic
        phys_stats = analysis['physical_ir_stats']
        reg_count = phys_stats.get('node_types', {}).get('register', 0)
        comb_count = phys_stats.get('node_types', {}).get('combinational', 0)
        
        if reg_count > 0 and comb_count > 0:
            reg_to_comb_ratio = reg_count / comb_count
            if reg_to_comb_ratio > 2.0:  # Too many registers
                opportunities.append((
                    OptimizationStrategy.REDUCE_REGISTER_COUNT,
                    {'ratio_threshold': 2.0, 'current_ratio': reg_to_comb_ratio}
                ))
        
        # Check for high average fanout
        if phys_stats.get('avg_fanout', 0) > 5.0:
            opportunities.append((
                OptimizationStrategy.OPTIMIZE_FANOUT,
                {'fanout_threshold': 5.0, 'current_avg': phys_stats['avg_fanout']}
            ))
        
        # Check for area vs power balance
        area = analysis['openroad_results']['overall_ppa']['area_um2']
        power = analysis['openroad_results']['overall_ppa']['power_mw']
        
        if area > 1000 and power > 1.0:  # Large and power-hungry
            opportunities.append((
                OptimizationStrategy.BALANCE_AREA_POWER,
                {'area_threshold': 1000, 'power_threshold': 1.0}
            ))
        
        return opportunities
    
    def _apply_strategy(self, rtl_content: str, strategy: OptimizationStrategy, params: Dict) -> str:
        """Apply a specific optimization strategy to RTL"""
        
        strategy_func = self.optimization_strategies.get(strategy)
        if strategy_func:
            return strategy_func(rtl_content, params)
        else:
            print(f"Unknown strategy: {strategy}")
            return rtl_content
    
    def _apply_pipelining(self, rtl_content: str, params: Dict) -> str:
        """Apply pipelining to critical paths using AST transformation"""
        print(f"Applying AST-based pipelining for parameters: {params}")
        
        # 1. Write RTL to temp file for the transformer
        with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as tmp:
            tmp.write(rtl_content)
            tmp_path = tmp.name
        
        try:
            # 2. Parse and Transform
            ast = self.rtl_transformer.parse_rtl(tmp_path)
            
            # For demonstration, we'll pipeline 'processed_data' if it exists, 
            # or try to find a suitable candidate from params
            target_signal = params.get('target_signal', 'processed_data')
            module_name = params.get('module_name', 'test_engine')
            
            try:
                ast, reg_name = self.rtl_transformer.add_pipeline_stage(ast, module_name, target_signal)
                
                # Robustly update all sinks of the original signal to use the new pipe reg
                ast = self.rtl_transformer.update_signal_sinks(ast, module_name, target_signal, reg_name)
                
                # 3. Generate new RTL
                optimized_rtl = self.rtl_transformer.generate_verilog(ast)
                return optimized_rtl
            except Exception as transform_err:
                print(f"AST Transformation failed: {transform_err}. Returning original.")
                return rtl_content
                
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _apply_clock_gating(self, rtl_content: str, params: Dict) -> str:
        """Apply clock gating to reduce dynamic power using AST transformation"""
        print(f"Applying AST-based clock gating for parameters: {params}")
        
        # 1. Write RTL to temp file for the transformer
        with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as tmp:
            tmp.write(rtl_content)
            tmp_path = tmp.name
        
        try:
            # 2. Parse and Transform
            ast = self.rtl_transformer.parse_rtl(tmp_path)
            
            target_signal = params.get('target_signal', 'reg_data')
            module_name = params.get('module_name', 'top_module')
            enable_signal = params.get('enable_signal', 'en')
            
            try:
                ast, gated_clk = self.rtl_transformer.insert_clock_gate(
                    ast, module_name, target_signal, enable_signal
                )
                
                # 3. Generate new RTL
                optimized_rtl = self.rtl_transformer.generate_verilog(ast)
                return optimized_rtl
            except Exception as transform_err:
                print(f"AST Power Transformation failed: {transform_err}. Returning original.")
                return rtl_content
                
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def _apply_clustering(self, rtl_content: str, params: Dict) -> str:
        """Apply clustering to reduce congestion"""
        # Placeholder - would involve identifying related modules/signals
        # and restructuring RTL to group them together
        return rtl_content
    
    def _reduce_registers(self, rtl_content: str, params: Dict) -> str:
        """Attempt to reduce register count"""
        # Placeholder - would involve retiming analysis and transformations
        return rtl_content
    
    def _optimize_fanout(self, rtl_content: str, params: Dict) -> str:
        """Optimize high fanout nets"""
        # Placeholder - would involve buffering and replication strategies
        return rtl_content
    
    def _balance_area_power(self, rtl_content: str, params: Dict) -> str:
        """Balance area and power trade-offs"""
        # Placeholder - would involve various area/power optimization techniques
        return rtl_content
    
    def _calculate_improvement(self, initial_analysis: Dict, final_analysis: Dict) -> Dict:
        """Calculate improvement metrics"""
        
        initial_ppa = initial_analysis['openroad_results']['overall_ppa']
        final_ppa = final_analysis['openroad_results']['overall_ppa']
        
        improvement = {
            'area_improvement_pct': ((initial_ppa['area_um2'] - final_ppa['area_um2']) / initial_ppa['area_um2']) * 100 if initial_ppa['area_um2'] > 0 else 0,
            'power_improvement_pct': ((initial_ppa['power_mw'] - final_ppa['power_mw']) / initial_ppa['power_mw']) * 100 if initial_ppa['power_mw'] > 0 else 0,
            'timing_improvement_pct': ((initial_ppa['timing_ns'] - final_ppa['timing_ns']) / initial_ppa['timing_ns']) * 100 if initial_ppa['timing_ns'] > 0 else 0,
            'drc_improvement': initial_analysis['openroad_results']['routing']['drc_violations'] - final_analysis['openroad_results']['routing']['drc_violations']
        }
        
        return improvement


class AdvancedAutonomousOptimizer(AutonomousOptimizer):
    """
    Advanced version that uses ML predictions to guide optimizations
    """
    
    def __init__(self):
        super().__init__()
        # In a real implementation, this would include ML models
        # for predicting the impact of each optimization
        self.prediction_models = None  # Would be loaded ML models
    
    def predict_optimization_impact(self, rtl_content: str, strategy: OptimizationStrategy, params: Dict) -> Dict[str, float]:
        """Predict the impact of applying an optimization"""
        # Placeholder - would use ML models to predict outcome
        return {
            'predicted_area_change_pct': -5.0,  # Example prediction
            'predicted_power_change_pct': -3.0,
            'predicted_timing_improvement': 0.1,
            'confidence': 0.7
        }
    
    def optimize_with_prediction_guidance(self, rtl_content: str, design_name: str = "unnamed") -> Dict[str, Any]:
        """Optimize using ML predictions to guide optimization selection"""
        
        # Analyze initial design
        initial_analysis = self.design_intelligence.analyze_design(rtl_content, design_name)
        
        # Identify possible optimizations
        opportunities = self._identify_optimizations(initial_analysis)
        
        # Predict impact of each optimization
        scored_opportunities = []
        for strategy, params in opportunities:
            predicted_impact = self.predict_optimization_impact(rtl_content, strategy, params)
            score = self._score_optimization(predicted_impact)
            scored_opportunities.append((strategy, params, predicted_impact, score))
        
        # Sort by score (higher is better)
        scored_opportunities.sort(key=lambda x: x[3], reverse=True)
        
        # Apply optimizations in order of predicted benefit
        optimized_rtl = rtl_content
        applied_optimizations = []
        
        for strategy, params, predicted_impact, score in scored_opportunities:
            if score > 0.1:  # Only apply if predicted positive impact
                optimized_rtl = self._apply_strategy(optimized_rtl, strategy, params)
                applied_optimizations.append({
                    'strategy': strategy.value,
                    'params': params,
                    'predicted_impact': predicted_impact,
                    'score': score
                })
        
        # Final analysis
        final_analysis = self.design_intelligence.analyze_design(optimized_rtl, f"optimized_{design_name}")
        
        results = {
            'initial_analysis': initial_analysis,
            'predicted_optimizations': scored_opportunities,
            'applied_optimizations': applied_optimizations,
            'final_analysis': final_analysis,
            'improvement': self._calculate_improvement(initial_analysis, final_analysis)
        }
        
        return results
    
    def _score_optimization(self, predicted_impact: Dict[str, float]) -> float:
        """Score an optimization based on predicted impact"""
        # Weight different metrics (this would be learned in practice)
        area_weight = 0.4
        power_weight = 0.3
        timing_weight = 0.3
        
        score = (
            area_weight * abs(predicted_impact.get('predicted_area_change_pct', 0)) +
            power_weight * abs(predicted_impact.get('predicted_power_change_pct', 0)) +
            timing_weight * abs(predicted_impact.get('predicted_timing_improvement', 0))
        )
        
        # Adjust for confidence
        confidence = predicted_impact.get('confidence', 0.5)
        score *= confidence
        
        return score


def test_autonomous_optimizer():
    """Test the autonomous optimizer"""
    print("Testing Autonomous Optimizer...")
    
    optimizer = AdvancedAutonomousOptimizer()
    
    # Test with a simple design
    test_rtl = '''
module test_adder (
    input [7:0] a,
    input [7:0] b,
    output [8:0] sum
);
    assign sum = a + b;
endmodule
    '''
    
    print("Original RTL:")
    print(test_rtl)
    
    # Apply optimization
    results = optimizer.optimize_with_prediction_guidance(test_rtl, "test_adder")
    
    print(f"\nInitial Area: {results['initial_analysis']['openroad_results']['overall_ppa']['area_um2']:.2f}")
    print(f"Final Area: {results['final_analysis']['openroad_results']['overall_ppa']['area_um2']:.2f}")
    print(f"Improvement: {results['improvement']['area_improvement_pct']:.2f}%")
    
    print(f"\nApplied {len(results['applied_optimizations'])} optimizations:")
    for opt in results['applied_optimizations']:
        print(f"  - {opt['strategy']}: Score={opt['score']:.3f}")
    
    return optimizer


if __name__ == "__main__":
    optimizer = test_autonomous_optimizer()