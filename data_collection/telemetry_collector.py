#!/usr/bin/env python3
"""
Telemetry Collector for Silicon Intelligence System

Collects post-layout telemetry and intermediate data to teach the system
how pain unfolds over time, not just final states.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import pickle
import hashlib

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from core.canonical_silicon_graph import CanonicalSiliconGraph


@dataclass
class TelemetrySnapshot:
    """Represents a snapshot of design state during implementation flow"""
    timestamp: datetime
    stage: str  # e.g., 'initial', 'post-floorplan', 'post-placement', 'post-routing', 'signoff'
    metrics: Dict[str, Any]  # congestion, timing, power, etc.
    design_signature: str  # hash of design characteristics
    iteration: int
    flow_state: str  # serialized flow state
    risk_predictions: Dict[str, Any]  # Oracle predictions at this stage
    decisions_made: List[Dict[str, Any]]  # Decisions made at this stage
    delta_from_prev: Dict[str, Any]  # Changes from previous snapshot


@dataclass
class FailureMemory:
    """Records what broke, where, when, and why"""
    design_name: str
    failure_stage: str
    failure_type: str  # congestion, timing, drc, power
    failure_location: str  # specific region/cell/path
    failure_severity: str  # low, medium, high, critical
    root_cause: str
    contributing_factors: List[str]
    timestamp: datetime
    recovery_attempts: List[Dict[str, Any]]
    outcome: str  # fixed, abandoned, workaround
    prevention_opportunities: List[str]


@dataclass
class ContextualIntent:
    """Captures constraints, goals, tradeoffs humans made"""
    design_name: str
    constraints_applied: Dict[str, Any]
    optimization_goals: Dict[str, Any]  # timing, power, area priorities
    tradeoff_decisions: List[Dict[str, Any]]  # "chose X over Y because..."
    human_override_reasons: List[Dict[str, Any]]
    priority_shifts: List[Dict[str, Any]]  # When priorities changed during flow
    timestamp: datetime


class TelemetryCollector:
    """
    Collects and stores various types of implementation data to build the learning system
    """
    
    def __init__(self, storage_path: str = "telemetry_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Create subdirectories for different data types
        (self.storage_path / "snapshots").mkdir(exist_ok=True)
        (self.storage_path / "failures").mkdir(exist_ok=True)
        (self.storage_path / "intent").mkdir(exist_ok=True)
        (self.storage_path / "models").mkdir(exist_ok=True)
        
        self.snapshots: List[TelemetrySnapshot] = []
        self.failures: List[FailureMemory] = []
        self.intents: List[ContextualIntent] = []
        
    def collect_snapshot(self, 
                       stage: str, 
                       graph: CanonicalSiliconGraph, 
                       risk_predictions: Dict[str, Any],
                       decisions_made: List[Dict[str, Any]],
                       iteration: int = 0) -> TelemetrySnapshot:
        """Collect a snapshot of the design state during implementation"""
        
        # Calculate metrics from the graph
        metrics = self._extract_graph_metrics(graph)
        
        # Create design signature
        design_signature = self._create_design_signature(metrics, stage)
        
        # Calculate deltas from previous snapshot if available
        delta_from_prev = {}
        if self.snapshots:
            prev_snapshot = self.snapshots[-1]
            delta_from_prev = self._calculate_deltas(prev_snapshot.metrics, metrics)
        
        snapshot = TelemetrySnapshot(
            timestamp=datetime.now(),
            stage=stage,
            metrics=metrics,
            design_signature=design_signature,
            iteration=iteration,
            flow_state=self._serialize_flow_state(graph),  # Simplified
            risk_predictions=risk_predictions,
            decisions_made=decisions_made,
            delta_from_prev=delta_from_prev
        )
        
        self.snapshots.append(snapshot)
        
        # Save to persistent storage
        self._save_snapshot(snapshot)
        
        return snapshot
    
    def _extract_graph_metrics(self, graph: CanonicalSiliconGraph) -> Dict[str, Any]:
        """Extract metrics from the canonical silicon graph"""
        metrics = {
            'node_count': len(graph.graph.nodes()),
            'edge_count': len(graph.graph.edges()),
            'macro_count': len(graph.get_macros()),
            'clock_roots': len(graph.get_clock_roots()),
            'timing_critical_nodes': len(graph.get_timing_critical_nodes()),
            'estimated_congestion': self._calculate_congestion_estimate(graph),
            'power_estimates': self._calculate_power_estimates(graph),
            'timing_slack_stats': self._calculate_timing_stats(graph)
        }
        return metrics
    
    def _calculate_congestion_estimate(self, graph: CanonicalSiliconGraph) -> Dict[str, float]:
        """Calculate congestion estimates from graph attributes"""
        congestion_map = {}
        for node, attrs in graph.graph.nodes(data=True):
            congestion = attrs.get('estimated_congestion', 0.0)
            if congestion > 0.1:  # Only include non-trivial congestion
                congestion_map[node] = congestion
        return congestion_map
    
    def _calculate_power_estimates(self, graph: CanonicalSiliconGraph) -> Dict[str, float]:
        """Calculate power estimates from graph attributes"""
        power_map = {}
        for node, attrs in graph.graph.nodes(data=True):
            power = attrs.get('power', 0.01)  # Default low power
            if power > 0.05:  # Only include non-trivial power
                power_map[node] = power
        return power_map
    
    def _calculate_timing_stats(self, graph: CanonicalSiliconGraph) -> Dict[str, Any]:
        """Calculate timing statistics from graph"""
        critical_nodes = graph.get_timing_critical_nodes()
        return {
            'critical_node_count': len(critical_nodes),
            'critical_ratio': len(critical_nodes) / max(len(graph.graph.nodes()), 1),
            'max_criticality': max([graph.graph.nodes[n].get('timing_criticality', 0.0) for n in critical_nodes], default=0.0)
        }
    
    def _create_design_signature(self, metrics: Dict[str, Any], stage: str) -> str:
        """Create a signature that identifies this design's characteristics"""
        signature_data = {
            'stage': stage,
            'node_count': metrics['node_count'],
            'macro_count': metrics['macro_count'],
            'clock_roots': metrics['clock_roots'],
            'estimated_congestion_avg': sum(metrics['estimated_congestion'].values()) / max(len(metrics['estimated_congestion']), 1) if metrics['estimated_congestion'] else 0.0
        }
        signature_str = json.dumps(signature_data, sort_keys=True)
        return hashlib.sha256(signature_str.encode()).hexdigest()[:16]
    
    def _calculate_deltas(self, prev_metrics: Dict[str, Any], curr_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate deltas between two metric sets"""
        deltas = {}
        
        for key in prev_metrics:
            if key in curr_metrics and isinstance(prev_metrics[key], (int, float)):
                deltas[f"{key}_delta"] = curr_metrics[key] - prev_metrics[key]
            elif key in curr_metrics and isinstance(prev_metrics[key], dict):
                # For dict metrics like congestion maps, calculate summary stats
                prev_sum = sum(prev_metrics[key].values()) if prev_metrics[key] else 0
                curr_sum = sum(curr_metrics[key].values()) if curr_metrics[key] else 0
                deltas[f"{key}_sum_delta"] = curr_sum - prev_sum
        
        return deltas
    
    def _serialize_flow_state(self, graph: CanonicalSiliconGraph) -> str:
        """Create a simplified serialization of flow state"""
        # For now, just return a hash of the graph structure
        # In practice, this would serialize the complete flow state
        return f"graph_nodes_{len(graph.graph.nodes())}_edges_{len(graph.graph.edges())}"
    
    def _save_snapshot(self, snapshot: TelemetrySnapshot):
        """Save snapshot to persistent storage"""
        filename = f"snapshot_{snapshot.timestamp.strftime('%Y%m%d_%H%M%S')}_{snapshot.stage}_{snapshot.iteration}.json"
        filepath = self.storage_path / "snapshots" / filename
        
        snapshot_dict = asdict(snapshot)
        snapshot_dict['timestamp'] = snapshot.timestamp.isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(snapshot_dict, f, indent=2, default=str)
    
    def record_failure(self, 
                     design_name: str,
                     failure_stage: str, 
                     failure_type: str,
                     failure_location: str,
                     failure_severity: str,
                     root_cause: str,
                     contributing_factors: List[str],
                     recovery_attempts: List[Dict[str, Any]],
                     outcome: str) -> FailureMemory:
        """Record a failure event with full context"""
        
        failure = FailureMemory(
            design_name=design_name,
            failure_stage=failure_stage,
            failure_type=failure_type,
            failure_location=failure_location,
            failure_severity=failure_severity,
            root_cause=root_cause,
            contributing_factors=contributing_factors,
            timestamp=datetime.now(),
            recovery_attempts=recovery_attempts,
            outcome=outcome,
            prevention_opportunities=[]  # Will be filled by analysis
        )
        
        self.failures.append(failure)
        
        # Save to persistent storage
        self._save_failure(failure)
        
        return failure
    
    def _save_failure(self, failure: FailureMemory):
        """Save failure record to persistent storage"""
        filename = f"failure_{failure.timestamp.strftime('%Y%m%d_%H%M%S')}_{failure.design_name}_{failure.failure_type}.json"
        filepath = self.storage_path / "failures" / filename
        
        failure_dict = asdict(failure)
        failure_dict['timestamp'] = failure.timestamp.isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(failure_dict, f, indent=2, default=str)
    
    def record_intent(self, 
                     design_name: str,
                     constraints_applied: Dict[str, Any],
                     optimization_goals: Dict[str, Any],
                     tradeoff_decisions: List[Dict[str, Any]],
                     human_override_reasons: List[Dict[str, Any]],
                     priority_shifts: List[Dict[str, Any]]) -> ContextualIntent:
        """Record the contextual intent and human decisions"""
        
        intent = ContextualIntent(
            design_name=design_name,
            constraints_applied=constraints_applied,
            optimization_goals=optimization_goals,
            tradeoff_decisions=tradeoff_decisions,
            human_override_reasons=human_override_reasons,
            priority_shifts=priority_shifts,
            timestamp=datetime.now()
        )
        
        self.intents.append(intent)
        
        # Save to persistent storage
        self._save_intent(intent)
        
        return intent
    
    def _save_intent(self, intent: ContextualIntent):
        """Save intent record to persistent storage"""
        filename = f"intent_{intent.timestamp.strftime('%Y%m%d_%H%M%S')}_{intent.design_name}.json"
        filepath = self.storage_path / "intent" / filename
        
        intent_dict = asdict(intent)
        intent_dict['timestamp'] = intent.timestamp.isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(intent_dict, f, indent=2, default=str)
    
    def generate_training_data(self) -> Dict[str, Any]:
        """Generate training data from collected telemetry"""
        
        # Combine all data sources into training format
        training_data = {
            'snapshots': [asdict(s) for s in self.snapshots],
            'failures': [asdict(f) for f in self.failures],
            'intents': [asdict(i) for i in self.intents],
            'derived_patterns': self._extract_derived_patterns(),
            'signature_to_failures': self._build_signature_failure_map()
        }
        
        return training_data
    
    def _extract_derived_patterns(self) -> List[Dict[str, Any]]:
        """Extract patterns from the collected data"""
        patterns = []
        
        # Pattern: Common failure precursors
        for failure in self.failures:
            # Look for snapshots that preceded this failure
            preceding_snapshots = [
                s for s in self.snapshots 
                if s.timestamp < failure.timestamp 
                and s.design_signature[:8] == failure.design_name[:8]  # Approximate match
            ]
            
            if preceding_snapshots:
                latest_before_failure = max(preceding_snapshots, key=lambda x: x.timestamp)
                
                patterns.append({
                    'pattern_type': 'failure_precursor',
                    'failure_type': failure.failure_type,
                    'precursor_metrics': latest_before_failure.metrics,
                    'time_to_failure': (failure.timestamp - latest_before_failure.timestamp).total_seconds(),
                    'stage_at_warning': latest_before_failure.stage
                })
        
        return patterns
    
    def _build_signature_failure_map(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build mapping from design signatures to associated failures"""
        sig_map = {}
        
        for failure in self.failures:
            # This is a simplified signature matching
            # In practice, you'd use more sophisticated matching
            key = failure.design_name[:8]  # First 8 chars as proxy for signature
            if key not in sig_map:
                sig_map[key] = []
            sig_map[key].append(asdict(failure))
        
        return sig_map
    
    def save_training_data(self, filename: str = "training_data.json"):
        """Save generated training data to file"""
        training_data = self.generate_training_data()
        
        filepath = self.storage_path / filename
        with open(filepath, 'w') as f:
            json.dump(training_data, f, indent=2, default=str)
        
        print(f"Training data saved to {filepath}")
        print(f"Records: {len(training_data['snapshots'])} snapshots, "
              f"{len(training_data['failures'])} failures, "
              f"{len(training_data['intents'])} intents")


def main():
    """Example usage of the telemetry collector"""
    print("Silicon Intelligence - Telemetry Collection System")
    print("=" * 60)
    
    collector = TelemetryCollector()
    
    print("Telemetry collection system initialized.")
    print(f"Storage path: {collector.storage_path}")
    print(f"Ready to collect: snapshots, failures, and intent data")
    
    # Example of how this would be used in the flow
    print("\nIn a real implementation, this would be called at each stage:")
    print("- collect_snapshot() after each major flow step")
    print("- record_failure() when issues occur")
    print("- record_intent() when constraints/human decisions are made")
    print("- save_training_data() periodically to build ML datasets")


if __name__ == "__main__":
    main()