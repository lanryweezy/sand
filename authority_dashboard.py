#!/usr/bin/env python3
"""
Authority Dashboard for Silicon Intelligence System

Provides a real-time view of system authority metrics and performance.
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

try:
    from evaluation_harness import EvaluationHarness, EvaluationResult
    from override_tracker import OverrideTracker
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required modules are available")
    sys.exit(1)


class AuthorityDashboard:
    """Dashboard for monitoring system authority metrics"""
    
    def __init__(self):
        self.override_tracker = OverrideTracker()
        self.evaluation_harness = EvaluationHarness()
        
    def load_recent_data(self) -> Dict[str, Any]:
        """Load recent evaluation and override data"""
        # Try to load evaluation results
        eval_results = None
        try:
            with open('evaluation_results.json', 'r') as f:
                eval_results = json.load(f)
        except FileNotFoundError:
            print("No evaluation results found, running new evaluation...")
            eval_results = self.evaluation_harness.run_comprehensive_evaluation()
        
        # Try to load validation results
        validation_results = None
        try:
            with open('validation_results.json', 'r') as f:
                validation_results = json.load(f)
        except FileNotFoundError:
            print("No validation results found")
        
        # Load override data
        try:
            self.override_tracker.load_override_data('override_data.json')
        except FileNotFoundError:
            print("No override data found")
        
        return {
            'eval_results': eval_results,
            'validation_results': validation_results,
            'override_tracker': self.override_tracker
        }
    
    def generate_authority_report(self) -> Dict[str, Any]:
        """Generate comprehensive authority report"""
        data = self.load_recent_data()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'prediction_metrics': {},
            'authority_metrics': {},
            'engineer_insights': {},
            'system_performance': {}
        }
        
        # Add prediction metrics
        if data['eval_results']:
            report['prediction_metrics'] = {
                'accuracy': data['eval_results']['summary']['prediction_accuracy'],
                'time_saved': data['eval_results']['summary']['time_saved_per_design'],
                'decisions_prevented': data['eval_results']['summary']['bad_decisions_prevented_total'],
                'prediction_speed': data['eval_results']['summary']['prediction_speed']
            }
        
        # Add authority metrics from override tracker
        authority_metrics = self.override_tracker.get_authority_metrics()
        report['authority_metrics'] = authority_metrics
        
        # Add engineer insights
        trust_report = self.override_tracker.get_engineer_trust_report()
        report['engineer_insights'] = {
            'total_engineers': len(trust_report.get('engineers', [])),
            'engineers_requiring_attention': trust_report.get('engineers', []),
            'overall_ai_accuracy': trust_report.get('overall_ai_accuracy', 0.0),
            'total_overrides': trust_report.get('total_overrides', 0)
        }
        
        # Add system performance metrics
        if data['validation_results']:
            report['system_performance'] = {
                'overall_authority': data['validation_results']['overall_authority'],
                'authority_score': data['validation_results']['authority_score'],
                'accuracy_target_met': data['validation_results']['accuracy_target_met']
            }
        
        return report
    
    def display_authority_metrics(self, report: Dict[str, Any]):
        """Display authority metrics in console"""
        print("SILICON INTELLIGENCE SYSTEM - AUTHORITY DASHBOARD")
        print("=" * 70)
        
        print("\n📈 PREDICTION ACCURACY")
        print("-" * 30)
        pred_metrics = report.get('prediction_metrics', {})
        print(f"  Overall Accuracy: {pred_metrics.get('accuracy', 'N/A')}")
        print(f"  Time Saved: {pred_metrics.get('time_saved', 'N/A')}")
        print(f"  Bad Decisions Prevented: {pred_metrics.get('decisions_prevented', 'N/A')}")
        print(f"  Prediction Speed: {pred_metrics.get('prediction_speed', 'N/A')}")
        
        print("\n⚖️  AUTHORITY METRICS")
        print("-" * 30)
        auth_metrics = report.get('authority_metrics', {})
        print(f"  Authority Score: {auth_metrics.get('challenge_success_rate', 'N/A')}")
        print(f"  Total Overrides Processed: {auth_metrics.get('overrides_processed', 'N/A')}")
        print(f"  AI Correct When Challenged: {auth_metrics.get('ai_correct_when_challenged', 'N/A')}")
        print(f"  Engineers Needing Attention: {auth_metrics.get('engineers_requiring_attention', 'N/A')}")
        
        print("\n👥 ENGINEER INSIGHTS")
        print("-" * 30)
        eng_insights = report.get('engineer_insights', {})
        print(f"  Total Engineers Tracked: {eng_insights.get('total_engineers', 'N/A')}")
        print(f"  Overall AI Accuracy: {eng_insights.get('overall_ai_accuracy', 0):.2%}")
        print(f"  Total Overrides: {eng_insights.get('total_overrides', 'N/A')}")
        
        # Show top engineers needing attention
        if eng_insights.get('engineers_requiring_attention'):
            print("  Top Engineers Needing Attention:")
            for i, eng in enumerate(eng_insights['engineers_requiring_attention'][:3]):
                print(f"    {i+1}. {eng['engineer_id']}: {eng['trust_level']} (score: {eng['trust_score']:.2f})")
        
        print("\n📊 SYSTEM PERFORMANCE")
        print("-" * 30)
        perf_metrics = report.get('system_performance', {})
        print(f"  Overall Authority: {'✅ PASS' if perf_metrics.get('overall_authority') else '❌ FAIL'}")
        print(f"  Authority Score: {perf_metrics.get('authority_score', 0):.2%}")
        print(f"  Accuracy Target Met: {'✅ YES' if perf_metrics.get('accuracy_target_met') else '❌ NO'}")
        
        print(f"\n🔄 Last Updated: {report['timestamp']}")
        
        # Authority assessment
        authority_score = auth_metrics.get('authority_score', 0)
        if authority_score >= 0.8:
            status = "🌟 EXCELLENT - System gaining strong authority"
        elif authority_score >= 0.6:
            status = "👍 GOOD - System building authority"
        elif authority_score >= 0.4:
            status = "⚠️  FAIR - System authority developing"
        else:
            status = "🚨 POOR - System authority needs work"
        
        print(f"\n🎯 AUTHORITY ASSESSMENT: {status}")
    
    def plot_authority_trends(self, report: Dict[str, Any]):
        """Plot authority trends and metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Silicon Intelligence System - Authority Dashboard', fontsize=16)
        
        # Plot 1: Prediction Accuracy
        pred_metrics = report.get('prediction_metrics', {})
        if pred_metrics:
            accuracy_val = float(pred_metrics['accuracy'].strip('%')) / 100 if isinstance(pred_metrics.get('accuracy'), str) else 0
            ax1.bar(['Prediction Accuracy'], [accuracy_val], color='skyblue')
            ax1.set_ylim(0, 1)
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Prediction Accuracy')
            ax1.text(0, accuracy_val + 0.05, f'{accuracy_val:.2%}', ha='center')
        
        # Plot 2: Authority Score
        auth_metrics = report.get('authority_metrics', {})
        authority_score = auth_metrics.get('authority_score', 0)
        ax2.bar(['Authority Score'], [authority_score], color='lightgreen')
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Score')
        ax2.set_title('System Authority Score')
        ax2.text(0, authority_score + 0.05, f'{authority_score:.2%}', ha='center')
        
        # Plot 3: Engineer Trust Distribution
        eng_insights = report.get('engineer_insights', {})
        engineers = eng_insights.get('engineers_requiring_attention', [])
        if engineers:
            trust_scores = [e['trust_score'] for e in engineers[:10]]  # Top 10
            engineer_ids = [e['engineer_id'] for e in engineers[:10]]
            ax3.barh(range(len(trust_scores)), trust_scores, color='orange')
            ax3.set_yticks(range(len(engineer_ids)))
            ax3.set_yticklabels(engineer_ids)
            ax3.set_xlabel('Trust Score')
            ax3.set_title('Engineer Trust Scores (Top 10)')
        
        # Plot 4: Performance Summary
        perf_metrics = report.get('system_performance', {})
        labels = ['Accuracy Target', 'Authority Achieved']
        values = [
            1 if perf_metrics.get('accuracy_target_met') else 0,
            perf_metrics.get('authority_score', 0)
        ]
        colors = ['green' if perf_metrics.get('accuracy_target_met') else 'red', 'blue']
        bars = ax4.bar(labels, values, color=colors)
        ax4.set_ylim(0, 1)
        ax4.set_title('Performance Status')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.2f}' if isinstance(value, float) else ('YES' if value == 1 else 'NO'),
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('authority_dashboard.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Dashboard chart saved as: authority_dashboard.png")
        
        plt.show()
    
    def run_dashboard(self, auto_refresh: bool = False, refresh_interval: int = 30):
        """Run the authority dashboard"""
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("Silicon Intelligence Authority Dashboard")
            print("Press Ctrl+C to exit\n")
            
            report = self.generate_authority_report()
            self.display_authority_metrics(report)
            
            try:
                self.plot_authority_trends(report)
            except ImportError:
                print("\n⚠️  Matplotlib not available, skipping charts")
            
            if not auto_refresh:
                break
            
            print(f"\n🔄 Auto-refresh in {refresh_interval} seconds (Ctrl+C to stop)...")
            time.sleep(refresh_interval)


def main():
    """Main dashboard function"""
    print("Initializing Silicon Intelligence Authority Dashboard...")
    
    try:
        dashboard = AuthorityDashboard()
        
        # Generate and display initial report
        report = dashboard.generate_authority_report()
        dashboard.display_authority_metrics(report)
        
        # Plot charts
        try:
            dashboard.plot_authority_trends(report)
        except ImportError:
            print("Matplotlib not available, skipping charts")
        
        # Ask user if they want continuous monitoring
        print(f"\nWould you like to run continuous monitoring? (y/n): ", end="")
        try:
            response = input().lower().strip()
            if response in ['y', 'yes']:
                print("Starting continuous monitoring (press Ctrl+C to stop)...")
                dashboard.run_dashboard(auto_refresh=True)
        except KeyboardInterrupt:
            print("\nContinuous monitoring stopped.")
        
    except KeyboardInterrupt:
        print("\nDashboard interrupted by user.")
    except Exception as e:
        print(f"Error running dashboard: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()