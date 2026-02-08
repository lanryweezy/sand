#!/usr/bin/env python3
"""
Override Tracker for Silicon Intelligence System

Tracks human overrides of AI decisions to learn which engineers to trust less
and improve system authority through empirical feedback.
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class OverrideRecord:
    """Record of a human override"""
    timestamp: datetime
    engineer_id: str
    design_name: str
    ai_recommendation: Dict[str, Any]
    human_action: Dict[str, Any]
    actual_outcome: Optional[Dict[str, Any]]
    override_reason: str
    ai_was_correct: bool
    severity: str  # low, medium, high
    learning_impact: float  # 0.0 to 1.0


@dataclass
class EngineerTrustScore:
    """Trust score for individual engineers"""
    engineer_id: str
    trust_score: float  # 0.0 (don't trust) to 1.0 (fully trust)
    override_count: int
    correct_override_rate: float
    ai_accuracy_when_followed: float
    last_updated: datetime


class OverrideTracker:
    """
    Tracks and learns from human overrides to improve system authority
    
    Key principle: "Learn which engineers to trust less"
    """
    
    def __init__(self, trust_decay_factor: float = 0.95):
        self.trust_decay_factor = trust_decay_factor
        self.override_records: List[OverrideRecord] = []
        self.engineer_scores: Dict[str, EngineerTrustScore] = {}
        self.ai_accuracy_history: List[float] = []
        
    def record_override(self, 
                       engineer_id: str,
                       design_name: str,
                       ai_recommendation: Dict[str, Any],
                       human_action: Dict[str, Any],
                       override_reason: str,
                       severity: str = "medium") -> OverrideRecord:
        """
        Record a human override of an AI recommendation
        """
        # Determine if AI was correct (this would be populated later with actual outcomes)
        ai_was_correct = self._evaluate_ai_correctness(ai_recommendation, human_action)
        
        record = OverrideRecord(
            timestamp=datetime.now(),
            engineer_id=engineer_id,
            design_name=design_name,
            ai_recommendation=ai_recommendation,
            human_action=human_action,
            actual_outcome=None,  # Will be populated later
            override_reason=override_reason,
            ai_was_correct=ai_was_correct,
            severity=severity,
            learning_impact=self._calculate_learning_impact(severity, ai_was_correct)
        )
        
        self.override_records.append(record)
        
        # Update engineer trust score
        self._update_engineer_trust(engineer_id, ai_was_correct)
        
        # Update AI accuracy metrics
        self._update_ai_accuracy(ai_was_correct)
        
        return record
    
    def _evaluate_ai_correctness(self, ai_recommendation: Dict, human_action: Dict) -> bool:
        """
        Evaluate whether the AI recommendation was correct compared to human action
        """
        # This is a simplified evaluation - in practice this would compare against
        # actual implementation results or known good outcomes
        
        ai_congestion_pred = len(ai_recommendation.get('congestion_heatmap', {}))
        human_congestion_actual = len(human_action.get('congestion_issues_found', []))
        
        ai_timing_pred = len(ai_recommendation.get('timing_risk_zones', []))
        human_timing_actual = len(human_action.get('timing_violations_found', []))
        
        # Simple accuracy check - if AI predicted issues and human found similar issues, AI was right
        congestion_match = abs(ai_congestion_pred - human_congestion_actual) <= 2
        timing_match = abs(ai_timing_pred - human_timing_actual) <= 1
        
        return congestion_match and timing_match
    
    def _calculate_learning_impact(self, severity: str, ai_was_correct: bool) -> float:
        """Calculate learning impact of this override"""
        severity_weights = {'low': 0.3, 'medium': 0.6, 'high': 1.0}
        weight = severity_weights.get(severity, 0.5)
        
        # Higher impact when AI was correct (we lost a good decision)
        impact = weight * (1.0 if ai_was_correct else 0.5)
        return impact
    
    def _update_engineer_trust(self, engineer_id: str, ai_was_correct: bool):
        """Update trust score for an engineer based on override"""
        if engineer_id not in self.engineer_scores:
            self.engineer_scores[engineer_id] = EngineerTrustScore(
                engineer_id=engineer_id,
                trust_score=0.8,  # Start with moderate trust
                override_count=0,
                correct_override_rate=0.0,
                ai_accuracy_when_followed=1.0,
                last_updated=datetime.now()
            )
        
        engineer = self.engineer_scores[engineer_id]
        engineer.override_count += 1
        
        # Decay trust when AI was correct but overridden
        if ai_was_correct:
            engineer.trust_score *= self.trust_decay_factor
        else:
            # Slight trust increase when override was justified
            engineer.trust_score = min(engineer.trust_score + 0.02, 1.0)
        
        engineer.last_updated = datetime.now()
    
    def _update_ai_accuracy(self, ai_was_correct: bool):
        """Update overall AI accuracy metrics"""
        self.ai_accuracy_history.append(1.0 if ai_was_correct else 0.0)
        
        # Keep rolling average of last 50 decisions
        if len(self.ai_accuracy_history) > 50:
            self.ai_accuracy_history.pop(0)
    
    def populate_actual_outcomes(self, design_name: str, actual_results: Dict[str, Any]):
        """
        Populate actual outcomes for records (called after implementation)
        """
        for record in self.override_records:
            if record.design_name == design_name:
                record.actual_outcome = actual_results
                
                # Re-evaluate AI correctness with actual data
                record.ai_was_correct = self._evaluate_against_actual(
                    record.ai_recommendation, actual_results
                )
                
                # Update engineer trust based on new information
                self._update_engineer_trust(record.engineer_id, record.ai_was_correct)
    
    def _evaluate_against_actual(self, ai_recommendation: Dict, actual_results: Dict) -> bool:
        """Evaluate AI prediction against actual implementation results"""
        # Compare predicted vs actual congestion
        predicted_congestion = len(ai_recommendation.get('congestion_heatmap', {}))
        actual_congestion = len(actual_results.get('actual_congestion_areas', []))
        
        # Compare predicted vs actual timing violations
        predicted_timing = len(ai_recommendation.get('timing_risk_zones', []))
        actual_timing = len(actual_results.get('actual_timing_violations', []))
        
        # AI is considered correct if predictions are within reasonable bounds
        congestion_accurate = abs(predicted_congestion - actual_congestion) <= 3
        timing_accurate = abs(predicted_timing - actual_timing) <= 2
        
        return congestion_accurate and timing_accurate
    
    def get_engineer_trust_report(self) -> Dict[str, Any]:
        """Generate trust report for all engineers"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_overrides': len(self.override_records),
            'overall_ai_accuracy': np.mean(self.ai_accuracy_history) if self.ai_accuracy_history else 0.0,
            'engineers': []
        }
        
        for engineer_id, score in self.engineer_scores.items():
            engineer_data = {
                'engineer_id': engineer_id,
                'trust_score': score.trust_score,
                'override_count': score.override_count,
                'trust_level': self._classify_trust_level(score.trust_score),
                'recommendations': self._generate_engineer_recommendations(score)
            }
            report['engineers'].append(engineer_data)
        
        # Sort by trust score (lowest first - these need attention)
        report['engineers'].sort(key=lambda x: x['trust_score'])
        
        return report
    
    def _classify_trust_level(self, trust_score: float) -> str:
        """Classify trust level"""
        if trust_score >= 0.8:
            return "High Trust"
        elif trust_score >= 0.6:
            return "Medium Trust"
        elif trust_score >= 0.4:
            return "Low Trust"
        else:
            return "Very Low Trust"
    
    def _generate_engineer_recommendations(self, score: EngineerTrustScore) -> List[str]:
        """Generate recommendations for handling this engineer"""
        recommendations = []
        
        if score.trust_score < 0.4:
            recommendations.append("Require peer review for all overrides")
            recommendations.append("Mandatory justification for AI disagreements")
            recommendations.append("Consider additional training on AI system")
        elif score.trust_score < 0.6:
            recommendations.append("Flag high-severity overrides for review")
            recommendations.append("Provide feedback on override decisions")
        else:
            recommendations.append("Standard override process acceptable")
            if score.trust_score > 0.8:
                recommendations.append("Consider this engineer as AI system advocate")
        
        return recommendations
    
    def get_authority_metrics(self) -> Dict[str, Any]:
        """Get metrics that demonstrate system authority"""
        if not self.override_records:
            return {'authority_score': 0.0, 'overrides_processed': 0}
        
        # Authority increases when AI is correct and decreases when wrong
        correct_overrides = sum(1 for r in self.override_records if r.ai_was_correct)
        total_overrides = len(self.override_records)
        
        # Authority score: percentage of times AI was right when challenged
        authority_score = correct_overrides / total_overrides if total_overrides > 0 else 0.0
        
        # Weight by severity and learning impact
        weighted_authority = np.average(
            [r.learning_impact for r in self.override_records],
            weights=[1.0 if r.ai_was_correct else 0.5 for r in self.override_records]
        ) if self.override_records else 0.0
        
        return {
            'authority_score': authority_score,
            'weighted_authority': weighted_authority,
            'overrides_processed': total_overrides,
            'ai_correct_when_challenged': correct_overrides,
            'challenge_success_rate': f"{authority_score:.1%}",
            'engineers_requiring_attention': len([e for e in self.engineer_scores.values() if e.trust_score < 0.6])
        }
    
    def save_override_data(self, filepath: str):
        """Save override tracking data to file"""
        data = {
            'override_records': [asdict(record) for record in self.override_records],
            'engineer_scores': {k: asdict(v) for k, v in self.engineer_scores.items()},
            'ai_accuracy_history': self.ai_accuracy_history,
            'trust_decay_factor': self.trust_decay_factor
        }
        
        # Convert datetime objects to strings
        for record in data['override_records']:
            record['timestamp'] = record['timestamp'].isoformat()
        
        for score in data['engineer_scores'].values():
            score['last_updated'] = score['last_updated'].isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_override_data(self, filepath: str):
        """Load override tracking data from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert string timestamps back to datetime objects
        self.override_records = []
        for record_data in data['override_records']:
            record_data['timestamp'] = datetime.fromisoformat(record_data['timestamp'])
            self.override_records.append(OverrideRecord(**record_data))
        
        self.engineer_scores = {}
        for engineer_id, score_data in data['engineer_scores'].items():
            score_data['last_updated'] = datetime.fromisoformat(score_data['last_updated'])
            self.engineer_scores[engineer_id] = EngineerTrustScore(**score_data)
        
        self.ai_accuracy_history = data['ai_accuracy_history']
        self.trust_decay_factor = data['trust_decay_factor']


# Example usage and integration with the main system
class AutonomousFlowController:
    """
    Controller that integrates override tracking with autonomous decision making
    """
    
    def __init__(self):
        self.override_tracker = OverrideTracker()
        self.autonomy_level = 0.8  # Start with high autonomy
    
    def make_autonomous_decision(self, 
                               engineer_id: str,
                               design_name: str,
                               ai_recommendation: Dict[str, Any],
                               human_proposed_action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make autonomous decision about whether to allow human override
        """
        engineer_trust = self.override_tracker.engineer_scores.get(engineer_id, 
                                                                 EngineerTrustScore(engineer_id, 0.8, 0, 0.0, 1.0, datetime.now()))
        
        # Calculate override probability based on engineer trust and AI confidence
        ai_confidence = ai_recommendation.get('overall_confidence', 0.5)
        override_probability = (1 - engineer_trust.trust_score) * (1 - ai_confidence)
        
        # Decision thresholds
        if override_probability < 0.2:
            # Allow override without question
            decision = "ALLOW_OVERRIDE"
            reason = "Low conflict probability"
        elif override_probability < 0.5:
            # Require justification
            decision = "REQUIRE_JUSTIFICATION"
            reason = "Moderate conflict probability"
        else:
            # Challenge override - require strong justification
            decision = "CHALLENGE_OVERRIDE"
            reason = "High conflict probability"
        
        # Record the interaction
        self.override_tracker.record_override(
            engineer_id=engineer_id,
            design_name=design_name,
            ai_recommendation=ai_recommendation,
            human_action=human_proposed_action,
            override_reason=f"Autonomous decision: {decision}",
            severity="high" if override_probability > 0.5 else "medium"
        )
        
        return {
            'decision': decision,
            'reason': reason,
            'override_probability': override_probability,
            'engineer_trust': engineer_trust.trust_score,
            'ai_confidence': ai_confidence,
            'autonomy_level': self.autonomy_level
        }
    
    def adjust_autonomy_level(self):
        """Adjust system autonomy based on override tracking results"""
        authority_metrics = self.override_tracker.get_authority_metrics()
        
        # Increase autonomy when AI authority is high
        if authority_metrics['authority_score'] > 0.7:
            self.autonomy_level = min(self.autonomy_level + 0.05, 1.0)
        # Decrease autonomy when AI authority is low
        elif authority_metrics['authority_score'] < 0.3:
            self.autonomy_level = max(self.autonomy_level - 0.05, 0.5)
        
        return self.autonomy_level


def main():
    """Example demonstrating override tracking in action"""
    print("Silicon Intelligence Override Tracking System")
    print("=" * 50)
    
    tracker = OverrideTracker()
    
    # Simulate some override scenarios
    scenarios = [
        {
            'engineer': 'eng_001',
            'design': 'mac_array_small',
            'ai_rec': {'congestion_heatmap': {'region1': 0.8, 'region2': 0.6}, 'overall_confidence': 0.9},
            'human_action': {'congestion_issues_found': ['region1'], 'timing_violations_found': []},
            'reason': 'Different congestion assessment',
            'severity': 'high'
        },
        {
            'engineer': 'eng_002', 
            'design': 'convolution_core',
            'ai_rec': {'timing_risk_zones': [{'path': 'critical_path_1'}], 'overall_confidence': 0.85},
            'human_action': {'timing_violations_found': ['critical_path_1', 'critical_path_2']},
            'reason': 'Additional timing analysis',
            'severity': 'medium'
        }
    ]
    
    for scenario in scenarios:
        record = tracker.record_override(
            engineer_id=scenario['engineer'],
            design_name=scenario['design'],
            ai_recommendation=scenario['ai_rec'],
            human_action=scenario['human_action'],
            override_reason=scenario['reason'],
            severity=scenario['severity']
        )
        
        print(f"Recorded override by {scenario['engineer']} on {scenario['design']}")
        print(f"  AI was {'correct' if record.ai_was_correct else 'incorrect'}")
        print(f"  Learning impact: {record.learning_impact:.2f}")
    
    # Generate reports
    trust_report = tracker.get_engineer_trust_report()
    authority_metrics = tracker.get_authority_metrics()
    
    print(f"\nAuthority Metrics:")
    print(f"  Challenge success rate: {authority_metrics['challenge_success_rate']}")
    print(f"  Engineers requiring attention: {authority_metrics['engineers_requiring_attention']}")
    
    print(f"\nTrust Report:")
    for engineer in trust_report['engineers']:
        print(f"  {engineer['engineer_id']}: {engineer['trust_level']} ({engineer['trust_score']:.2f})")


if __name__ == "__main__":
    main()