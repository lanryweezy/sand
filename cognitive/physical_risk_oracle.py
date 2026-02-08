"""
Physical Risk Oracle - Predicts physical implementation challenges before layout

This module implements the core capability to predict where and why physical 
implementation will fail, before placement occurs.
"""

import os
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
import networkx as nx
from data.rtl_parser import RTLParser
from core.canonical_silicon_graph import CanonicalSiliconGraph
from models.congestion_predictor import CongestionPredictor
from models.timing_analyzer import TimingAnalyzer
from utils.logger import get_logger


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PhysicalRiskAssessment:
    """Data class for physical risk assessment results"""
    congestion_heatmap: Dict[str, float]
    timing_risk_zones: List[Dict[str, Any]]
    clock_skew_sensitivity: Dict[str, float]
    power_density_hotspots: List[Dict[str, float]]
    drc_risk_classes: List[Dict[str, str]]
    overall_confidence: float
    recommendations: List[str]


class PhysicalRiskOracle:
    """
    The Physical Risk Oracle - Predicts physical implementation challenges
    
    Given RTL + constraints, predicts where and why physical implementation 
    will fail, before placement.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.rtl_parser = RTLParser()
        self.graph_builder = CanonicalSiliconGraph()
        self.congestion_predictor = CongestionPredictor()
        self.timing_analyzer = TimingAnalyzer()
        
        # Initialize with default models for 7nm
        self._load_models_for_node("7nm")
    
    def _load_models_for_node(self, node: str):
        """Load appropriate models for the target process node"""
        self.logger.info(f"Loading models for {node} process node")
        # In a real implementation, this would load node-specific models
        # For now, we'll use generic models
        pass
    
    def predict_physical_risks(
        self, 
        rtl_file: str, 
        constraints_file: str, 
        node: str = "7nm",
        floorplan_hints: Dict[str, Any] = None
    ) -> PhysicalRiskAssessment:
        """
        Predict physical risks given RTL and constraints
        
        Args:
            rtl_file: Path to RTL file
            constraints_file: Path to constraints file (SDC)
            node: Target process node (e.g., "7nm", "5nm", "3nm")
            floorplan_hints: Optional floorplan hints
            
        Returns:
            PhysicalRiskAssessment with predictions
        """
        self.logger.info(f"Analyzing physical risks for RTL: {rtl_file}")
        
        # Parse RTL and build graph representation
        rtl_data = self.rtl_parser.parse(rtl_file)
        constraint_data = self._parse_constraints(constraints_file)
        
        # Build canonical silicon graph
        silicon_graph = self.graph_builder.build_from_rtl(
            rtl_data, constraint_data, floorplan_hints
        )
        
        # Generate predictions
        congestion_map = self._predict_congestion(silicon_graph)
        timing_risks = self._analyze_timing_risks(silicon_graph, constraint_data)
        clock_sensitivity = self._analyze_clock_sensitivity(silicon_graph)
        power_hotspots = self._analyze_power_density(silicon_graph)
        drc_risks = self._predict_drc_risks(silicon_graph, node)
        
        # Calculate overall confidence
        confidence = self._calculate_confidence([
            congestion_map, timing_risks, clock_sensitivity, 
            power_hotspots, drc_risks
        ])
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            congestion_map, timing_risks, clock_sensitivity, 
            power_hotspots, drc_risks
        )
        
        return PhysicalRiskAssessment(
            congestion_heatmap=congestion_map,
            timing_risk_zones=timing_risks,
            clock_skew_sensitivity=clock_sensitivity,
            power_density_hotspots=power_hotspots,
            drc_risk_classes=drc_risks,
            overall_confidence=confidence,
            recommendations=recommendations
        )
    
    def _parse_constraints(self, constraints_file: str) -> Dict[str, Any]:
        """Parse constraints file (SDC format)"""
        # Simplified constraint parsing - in reality would use proper SDC parser
        constraints = {}
        with open(constraints_file, 'r') as f:
            content = f.read()
            # Extract basic timing constraints
            constraints['clocks'] = []
            constraints['timing_paths'] = []
            constraints['power_domains'] = []
            
            # This is a simplified version - real implementation would parse SDC properly
            if 'create_clock' in content:
                # Parse clock definitions
                pass
            if 'set_input_delay' in content or 'set_output_delay' in content:
                # Parse I/O constraints
                pass
                
        return constraints
    
    def _predict_congestion(self, silicon_graph: CanonicalSiliconGraph) -> Dict[str, float]:
        """Predict routing congestion based on graph structure"""
        return self.congestion_predictor.predict(silicon_graph)
    
    def _analyze_timing_risks(self, silicon_graph: CanonicalSiliconGraph, constraints: Dict) -> List[Dict[str, Any]]:
        """Analyze timing risk zones"""
        return self.timing_analyzer.analyze(silicon_graph, constraints)
    
    def _analyze_clock_sensitivity(self, silicon_graph: CanonicalSiliconGraph) -> Dict[str, float]:
        """Analyze clock skew sensitivity"""
        # Find clock domains and analyze sensitivity
        clock_nodes = [n for n, attr in silicon_graph.nodes(data=True) 
                      if attr.get('type') == 'clock']
        
        sensitivity_map = {}
        for clk_node in clock_nodes:
            # Calculate sensitivity based on fanout and path characteristics
            fanout = len(list(silicon_graph.successors(clk_node)))
            # More sophisticated analysis would go here
            sensitivity = min(fanout / 100.0, 1.0)  # Normalize to 0-1 range
            sensitivity_map[clk_node] = sensitivity
            
        return sensitivity_map
    
    def _analyze_power_density(self, silicon_graph: CanonicalSiliconGraph) -> List[Dict[str, float]]:
        """Analyze power density hotspots"""
        hotspots = []
        
        # Group nodes by physical regions (simplified approach)
        # In reality, this would use placement information or clustering
        node_attrs = dict(silicon_graph.nodes(data=True))
        
        # Find high-power components (macros, memory, etc.)
        for node_id, attrs in node_attrs.items():
            if attrs.get('power_class') in ['high', 'macro', 'memory']:
                hotspots.append({
                    'node': node_id,
                    'estimated_power': attrs.get('estimated_power', 1.0),
                    'region': attrs.get('region', 'unknown'),
                    'risk_level': 'high'
                })
                
        return hotspots
    
    def _predict_drc_risks(self, silicon_graph: CanonicalSiliconGraph, node: str) -> List[Dict[str, str]]:
        """Predict DRC risk classes based on node and design characteristics"""
        drc_risks = []
        
        # Different process nodes have different DRC challenges
        if node in ['3nm', '5nm', '7nm']:
            # Advanced nodes have more complex DRC rules
            drc_risks.append({
                'rule_class': 'density',
                'severity': 'high',
                'description': f'{node} nodes have strict density rules'
            })
            drc_risks.append({
                'rule_class': 'spacing',
                'severity': 'high', 
                'description': f'{node} nodes have complex spacing requirements'
            })
        else:
            # Older nodes have different challenges
            drc_risks.append({
                'rule_class': 'metal_stack',
                'severity': 'medium',
                'description': f'{node} nodes have specific metal stack rules'
            })
            
        return drc_risks
    
    def _calculate_confidence(self, predictions: List[Any]) -> float:
        """Calculate overall confidence in predictions"""
        # Simple confidence calculation - in reality would be more sophisticated
        # based on model certainty, data quality, etc.
        return 0.85  # High confidence for initial implementation
    
    def _generate_recommendations(
        self, 
        congestion_map: Dict[str, float], 
        timing_risks: List[Dict[str, Any]],
        clock_sensitivity: Dict[str, float],
        power_hotspots: List[Dict[str, float]],
        drc_risks: List[Dict[str, str]]
    ) -> List[str]:
        """Generate actionable recommendations based on all analyses"""
        recommendations = []
        
        # Congestion recommendations
        high_congestion = {k: v for k, v in congestion_map.items() if v > 0.7}
        if high_congestion:
            recommendations.append(
                f"Avoid dense packing in regions: {list(high_congestion.keys())[:3]}"
            )
        
        # Timing recommendations
        if timing_risks:
            critical_paths = [tr for tr in timing_risks if tr.get('risk_level') == 'critical']
            if critical_paths:
                recommendations.append(
                    f"Consider hierarchical design for critical paths: {[cp['path'] for cp in critical_paths[:2]]}"
                )
        
        # Clock recommendations
        high_sensitivity = {k: v for k, v in clock_sensitivity.items() if v > 0.5}
        if high_sensitivity:
            recommendations.append(
                f"Use low-skew clock routing for clocks: {list(high_sensitivity.keys())}"
            )
        
        # Power recommendations
        if power_hotspots:
            recommendations.append(
                f"Plan dedicated power grid for hotspots: {[ph['node'] for ph in power_hotspots[:3]]}"
            )
        
        # DRC recommendations
        if drc_risks:
            recommendations.append(
                f"Apply {len(drc_risks)} DRC mitigation strategies for {drc_risks[0]['rule_class']} rules"
            )
        
        return recommendations
    
    def visualize_assessment(self, assessment: PhysicalRiskAssessment, output_file: str = None):
        """Visualize the risk assessment results"""
        # This would create visualizations of the risk map
        # For now, we'll just print a summary
        print(f"Physical Risk Assessment Summary:")
        print(f"- Congestion risk areas: {len(assessment.congestion_heatmap)}")
        print(f"- Timing risk zones: {len(assessment.timing_risk_zones)}")
        print(f"- Clock sensitivity issues: {len(assessment.clock_skew_sensitivity)}")
        print(f"- Power hotspots: {len(assessment.power_density_hotspots)}")
        print(f"- DRC risk classes: {len(assessment.drc_risk_classes)}")
        print(f"- Overall confidence: {assessment.overall_confidence:.2f}")
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump({
                    'congestion_heatmap': assessment.congestion_heatmap,
                    'timing_risk_zones': assessment.timing_risk_zones,
                    'clock_skew_sensitivity': assessment.clock_skew_sensitivity,
                    'power_density_hotspots': assessment.power_density_hotspots,
                    'drc_risk_classes': assessment.drc_risk_classes,
                    'overall_confidence': assessment.overall_confidence,
                    'recommendations': assessment.recommendations
                }, f, indent=2)