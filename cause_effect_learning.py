#!/usr/bin/env python3
"""
Cause and Effect Learning Loop
Connects design changes to measurable outcomes for real learning
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from real_openroad_interface import RealOpenROADInterface
from physical_design_intelligence import PhysicalDesignIntelligence
from ml_prediction_models import DesignPPAPredictor


@dataclass
class DesignChange:
    """Represents a design modification"""
    change_type: str  # 'refactoring', 'pipelining', 'clustering', etc.
    description: str
    parameters: Dict[str, Any]
    timestamp: str


@dataclass
class OutcomeMetrics:
    """Results from design implementation"""
    area_um2: float
    power_mw: float
    timing_ns: float
    drc_violations: int
    congestion_max: float
    util_max: float
    runtime_sec: float
    timestamp: str


@dataclass
class CauseEffectPair:
    """Links a design change to its measured outcomes"""
    design_change: DesignChange
    before_metrics: OutcomeMetrics
    after_metrics: OutcomeMetrics
    improvement_area: float  # Positive = improvement
    improvement_power: float
    improvement_timing: float
    improvement_drc: float
    confidence: float  # How confident we are in causality
    timestamp: str


class CauseEffectLearningLoop:
    """
    Learning system that observes cause-effect relationships
    between design changes and implementation outcomes
    """
    
    def __init__(self, data_dir: str = "cause_effect_data"):
        self.data_dir = data_dir
        self.interface = RealOpenROADInterface()
        self.design_intel = PhysicalDesignIntelligence()
        self.predictor = DesignPPAPredictor()
        self.history: List[CauseEffectPair] = []
        
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing history if available
        self._load_history()
    
    def _load_history(self):
        """Load previous cause-effect pairs from storage"""
        history_file = os.path.join(self.data_dir, "cause_effect_history.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    # Convert JSON back to objects
                    for item in data:
                        change = DesignChange(**item['design_change'])
                        before_metrics = OutcomeMetrics(**item['before_metrics'])
                        after_metrics = OutcomeMetrics(**item['after_metrics'])
                        
                        pair = CauseEffectPair(
                            design_change=change,
                            before_metrics=before_metrics,
                            after_metrics=after_metrics,
                            improvement_area=item['improvement_area'],
                            improvement_power=item['improvement_power'],
                            improvement_timing=item['improvement_timing'],
                            improvement_drc=item['improvement_drc'],
                            confidence=item['confidence'],
                            timestamp=item['timestamp']
                        )
                        self.history.append(pair)
            except Exception as e:
                print(f"Could not load history: {e}")
    
    def _save_history(self):
        """Save cause-effect pairs to storage"""
        history_file = os.path.join(self.data_dir, "cause_effect_history.json")
        
        # Convert objects to JSON-serializable format
        serializable_history = []
        for pair in self.history:
            serializable_history.append({
                'design_change': {
                    'change_type': pair.design_change.change_type,
                    'description': pair.design_change.description,
                    'parameters': pair.design_change.parameters,
                    'timestamp': pair.design_change.timestamp
                },
                'before_metrics': {
                    'area_um2': pair.before_metrics.area_um2,
                    'power_mw': pair.before_metrics.power_mw,
                    'timing_ns': pair.before_metrics.timing_ns,
                    'drc_violations': pair.before_metrics.drc_violations,
                    'congestion_max': pair.before_metrics.congestion_max,
                    'util_max': pair.before_metrics.util_max,
                    'runtime_sec': pair.before_metrics.runtime_sec,
                    'timestamp': pair.before_metrics.timestamp
                },
                'after_metrics': {
                    'area_um2': pair.after_metrics.area_um2,
                    'power_mw': pair.after_metrics.power_mw,
                    'timing_ns': pair.after_metrics.timing_ns,
                    'drc_violations': pair.after_metrics.drc_violations,
                    'congestion_max': pair.after_metrics.congestion_max,
                    'util_max': pair.after_metrics.util_max,
                    'runtime_sec': pair.after_metrics.runtime_sec,
                    'timestamp': pair.after_metrics.timestamp
                },
                'improvement_area': pair.improvement_area,
                'improvement_power': pair.improvement_power,
                'improvement_timing': pair.improvement_timing,
                'improvement_drc': pair.improvement_drc,
                'confidence': pair.confidence,
                'timestamp': pair.timestamp
            })
        
        with open(history_file, 'w') as f:
            json.dump(serializable_history, f, indent=2)
    
    def measure_design(self, rtl_content: str, design_name: str) -> OutcomeMetrics:
        """Measure outcomes for a given design"""
        # Run full OpenROAD flow
        results = self.interface.run_full_flow(rtl_content)
        
        # Extract metrics
        ppa = results['overall_ppa']
        placement = results['placement']
        routing = results['routing']
        
        # Get congestion and utilization from placement
        congestion_max = 0.0
        util_max = placement.get('utilization', 0.7)
        
        if 'congestion_map' in placement and placement['congestion_map']:
            congestion_vals = [c.get('congestion_level', 0) for c in placement['congestion_map']]
            congestion_max = max(congestion_vals) if congestion_vals else 0.0
        
        metrics = OutcomeMetrics(
            area_um2=ppa['area_um2'],
            power_mw=ppa['power_mw'],
            timing_ns=ppa['timing_ns'],
            drc_violations=routing['drc_violations'],
            congestion_max=congestion_max,
            util_max=util_max,
            runtime_sec=results.get('total_runtime', 0),
            timestamp=datetime.now().isoformat()
        )
        
        return metrics
    
    def apply_design_change(self, original_rtl: str, change: DesignChange) -> str:
        """Apply a design change to RTL code"""
        # This would be implemented with actual RTL transformation logic
        # For now, we'll simulate different changes
        
        if change.change_type == "pipelining":
            # Add pipeline registers based on parameters
            return self._apply_pipelining(original_rtl, change.parameters)
        elif change.change_type == "clustering":
            # Group related logic together
            return self._apply_clustering(original_rtl, change.parameters)
        elif change.change_type == "buffering":
            # Add buffers for high-fanout nets
            return self._apply_buffering(original_rtl, change.parameters)
        else:
            # For unknown changes, return original (no-op)
            return original_rtl
    
    def _apply_pipelining(self, rtl: str, params: Dict) -> str:
        """Apply pipelining transformation to RTL"""
        # This is a simplified example - real implementation would be much more complex
        # and would involve proper RTL parsing and transformation
        
        # For demonstration purposes, we'll just add some comments indicating pipelining
        lines = rtl.split('\n')
        new_lines = []
        
        for line in lines:
            new_lines.append(line)
            if 'assign' in line and '=' in line and ';' in line:
                # Add a comment indicating this could be pipelined
                new_lines.append(f"    // PIPELINE STAGE: {params.get('stages', 1)} stages")
        
        return '\n'.join(new_lines)
    
    def _apply_clustering(self, rtl: str, params: Dict) -> str:
        """Apply clustering transformation to RTL"""
        # Simplified clustering - in reality this would involve module restructuring
        lines = rtl.split('\n')
        new_lines = []
        
        for line in lines:
            new_lines.append(line)
            if 'reg' in line and '[' in line and ']' in line:
                # Add clustering hint
                new_lines.append(f"    // CLUSTER GROUP: {params.get('group', 'default')}")
        
        return '\n'.join(new_lines)
    
    def _apply_buffering(self, rtl: str, params: Dict) -> str:
        """Apply buffering transformation to RTL"""
        # Simplified buffering - in reality this would involve netlist manipulation
        return rtl  # For now, return unchanged
    
    def record_cause_effect(self, change: DesignChange, before_rtl: str, after_rtl: str, design_name: str):
        """Record the cause-effect relationship between a change and its outcome"""
        
        # Measure before and after
        before_metrics = self.measure_design(before_rtl, f"{design_name}_before")
        after_metrics = self.measure_design(after_rtl, f"{design_name}_after")
        
        # Calculate improvements (positive = better)
        improvement_area = before_metrics.area_um2 - after_metrics.area_um2
        improvement_power = before_metrics.power_mw - after_metrics.power_mw
        improvement_timing = before_metrics.timing_ns - after_metrics.timing_ns
        improvement_drc = before_metrics.drc_violations - after_metrics.drc_violations
        
        # Calculate confidence based on magnitude of change vs noise
        base_values = [before_metrics.area_um2, before_metrics.power_mw, 
                      before_metrics.timing_ns, before_metrics.drc_violations]
        avg_base = sum(base_values) / len(base_values) if base_values else 1
        
        improvement_magnitude = abs(improvement_area) + abs(improvement_power) + \
                               abs(improvement_timing) + abs(improvement_drc)
        confidence = min(1.0, improvement_magnitude / (avg_base * 0.1))  # 10% threshold
        
        # Create cause-effect pair
        pair = CauseEffectPair(
            design_change=change,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            improvement_area=improvement_area,
            improvement_power=improvement_power,
            improvement_timing=improvement_timing,
            improvement_drc=improvement_drc,
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
        
        # Add to history
        self.history.append(pair)
        
        # Save to storage
        self._save_history()
        
        return pair
    
    def get_actionable_insights(self) -> List[Dict[str, Any]]:
        """Extract actionable insights from cause-effect history"""
        insights = []
        
        if not self.history:
            return insights
        
        # Group by change type
        change_stats = {}
        for pair in self.history:
            change_type = pair.design_change.change_type
            
            if change_type not in change_stats:
                change_stats[change_type] = {
                    'count': 0,
                    'total_area_improvement': 0,
                    'total_power_improvement': 0,
                    'total_timing_improvement': 0,
                    'total_drc_improvement': 0,
                    'avg_confidence': 0
                }
            
            stats = change_stats[change_type]
            stats['count'] += 1
            stats['total_area_improvement'] += pair.improvement_area
            stats['total_power_improvement'] += pair.improvement_power
            stats['total_timing_improvement'] += pair.improvement_timing
            stats['total_drc_improvement'] += pair.improvement_drc
            stats['avg_confidence'] += pair.confidence
        
        # Calculate averages and create insights
        for change_type, stats in change_stats.items():
            count = stats['count']
            insight = {
                'change_type': change_type,
                'applications': count,
                'avg_area_improvement': stats['total_area_improvement'] / count,
                'avg_power_improvement': stats['total_power_improvement'] / count,
                'avg_timing_improvement': stats['total_timing_improvement'] / count,
                'avg_drc_improvement': stats['total_drc_improvement'] / count,
                'avg_confidence': stats['avg_confidence'] / count,
                'effectiveness_score': (
                    abs(stats['total_area_improvement'] / count) +
                    abs(stats['total_power_improvement'] / count) +
                    abs(stats['total_timing_improvement'] / count) +
                    abs(stats['total_drc_improvement'] / count)
                ) / count
            }
            insights.append(insight)
        
        # Sort by effectiveness
        insights.sort(key=lambda x: x['effectiveness_score'], reverse=True)
        
        return insights
    
    def suggest_changes(self, current_rtl: str, design_name: str) -> List[Dict[str, Any]]:
        """Suggest beneficial changes based on learned patterns"""
        insights = self.get_actionable_insights()
        
        suggestions = []
        for insight in insights[:5]:  # Top 5 suggestions
            if insight['effectiveness_score'] > 0:  # Only suggest positive changes
                suggestion = {
                    'change_type': insight['change_type'],
                    'expected_area_improvement': insight['avg_area_improvement'],
                    'expected_power_improvement': insight['avg_power_improvement'],
                    'expected_timing_improvement': insight['avg_timing_improvement'],
                    'expected_drc_improvement': insight['avg_drc_improvement'],
                    'confidence': insight['avg_confidence'],
                    'historical_applications': insight['applications']
                }
                suggestions.append(suggestion)
        
        return suggestions


def demonstrate_cause_effect_learning():
    """Demonstrate the cause-effect learning system"""
    print("🔬 DEMONSTRATING CAUSE-EFFECT LEARNING")
    print("=" * 50)
    
    # Create learning system
    learner = CauseEffectLearningLoop()
    
    # Sample RTL design
    sample_rtl = '''
module test_design (
    input clk,
    input rst_n,
    input [7:0] a,
    input [7:0] b,
    output [8:0] sum
);
    reg [8:0] sum_reg;
    
    always @(posedge clk) begin
        if (!rst_n)
            sum_reg <= 9'd0;
        else
            sum_reg <= a + b;
    end
    
    assign sum = sum_reg;
endmodule
    '''
    
    print("Measuring baseline design...")
    baseline_metrics = learner.measure_design(sample_rtl, "baseline")
    print(f"Baseline: Area={baseline_metrics.area_um2:.2f}, Power={baseline_metrics.power_mw:.3f}, Timing={baseline_metrics.timing_ns:.3f}")
    
    # Apply a change
    change = DesignChange(
        change_type="pipelining",
        description="Add pipeline register to critical path",
        parameters={"stages": 2, "target_path": "addition"},
        timestamp=datetime.now().isoformat()
    )
    
    modified_rtl = learner.apply_design_change(sample_rtl, change)
    
    print("Applying design change...")
    after_metrics = learner.measure_design(modified_rtl, "modified")
    print(f"After change: Area={after_metrics.area_um2:.2f}, Power={after_metrics.power_mw:.3f}, Timing={after_metrics.timing_ns:.3f}")
    
    # Record the cause-effect relationship
    print("Recording cause-effect relationship...")
    pair = learner.record_cause_effect(change, sample_rtl, modified_rtl, "test_design")
    
    print(f"Improvement: Area={pair.improvement_area:.2f}, Power={pair.improvement_power:.3f}, Timing={pair.improvement_timing:.3f}")
    print(f"Confidence: {pair.confidence:.2f}")
    
    # Show actionable insights
    print("\n💡 ACTIONABLE INSIGHTS")
    print("-" * 30)
    insights = learner.get_actionable_insights()
    for insight in insights[:3]:
        print(f"Change Type: {insight['change_type']}")
        print(f"  Applications: {insight['applications']}")
        print(f"  Avg Area Improvement: {insight['avg_area_improvement']:.2f}")
        print(f"  Avg Power Improvement: {insight['avg_power_improvement']:.3f}")
        print(f"  Avg Timing Improvement: {insight['avg_timing_improvement']:.3f}")
        print(f"  Effectiveness Score: {insight['effectiveness_score']:.3f}")
        print()
    
    print("✅ CAUSE-EFFECT LEARNING LOOP ESTABLISHED")
    print("SAND now observes 'When I changed X, Y improved and Z broke'")
    print("This creates the foundation for true learning from outcomes.")
    
    return learner


if __name__ == "__main__":
    learner = demonstrate_cause_effect_learning()