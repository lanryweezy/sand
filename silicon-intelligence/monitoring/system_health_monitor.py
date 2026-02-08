"""
System Health Monitor and Dashboard for Silicon Intelligence System

This module provides real-time monitoring and visualization of system health,
performance metrics, and optimization progress.
"""

import time
import threading
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from utils.logger import get_logger


class SystemHealthMonitor:
    """
    Monitors the health and performance of the Silicon Intelligence System
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.metrics_history = []
        self.current_metrics = {}
        self.monitoring_active = False
        self.monitoring_thread = None
        self.agent_performance = {}
        self.ml_model_performance = {}
        
    def start_monitoring(self):
        """Start system health monitoring"""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("System health monitoring started")
    
    def stop_monitoring(self):
        """Stop system health monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        self.logger.info("System health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop running in background thread"""
        while self.monitoring_active:
            try:
                # Collect current metrics
                current_metrics = self._collect_current_metrics()
                self.current_metrics = current_metrics
                self.metrics_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'metrics': current_metrics
                })
                
                # Keep only last 1000 metrics to prevent memory overflow
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                # Sleep for monitoring interval
                time.sleep(5.0)  # Monitor every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(5.0)  # Continue monitoring even if there's an error
    
    def _collect_current_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        metrics = {
            'system_uptime': self._get_system_uptime(),
            'active_agents': self._get_active_agent_count(),
            'active_ml_models': self._get_active_ml_model_count(),
            'current_flow_stage': self._get_current_flow_stage(),
            'resource_utilization': self._get_resource_utilization(),
            'prediction_accuracy': self._get_prediction_accuracy(),
            'agent_authority_levels': self._get_agent_authority_levels(),
            'model_confidence': self._get_model_confidence_levels(),
            'optimization_progress': self._get_optimization_progress(),
            'error_rates': self._get_error_rates()
        }
        return metrics
    
    def _get_system_uptime(self) -> float:
        """Get system uptime in seconds"""
        # In a real implementation, this would track actual uptime
        return time.time() - getattr(self, '_start_time', time.time())
    
    def _get_active_agent_count(self) -> int:
        """Get count of active agents"""
        # This would interface with the agent manager in a real system
        return 7  # Mock value - would be dynamic in real implementation
    
    def _get_active_ml_model_count(self) -> int:
        """Get count of active ML models"""
        # This would interface with the model manager in a real system
        return 5  # Mock value - would be dynamic in real implementation
    
    def _get_current_flow_stage(self) -> str:
        """Get current flow stage"""
        # This would interface with the flow orchestrator in a real system
        stages = ['risk_assessment', 'graph_construction', 'agent_negotiation', 'parallel_exploration', 'drc_optimization']
        return np.random.choice(stages)  # Mock current stage
    
    def _get_resource_utilization(self) -> Dict[str, float]:
        """Get resource utilization metrics"""
        import psutil
        import os
        
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent,
            'gpu_memory_percent': 0.0,  # Would be actual GPU usage in real implementation
            'threads_active': threading.active_count()
        }
    
    def _get_prediction_accuracy(self) -> Dict[str, float]:
        """Get prediction accuracy metrics"""
        # This would interface with the learning loop in a real system
        return {
            'congestion_prediction_accuracy': 0.85,
            'timing_prediction_accuracy': 0.82,
            'drc_prediction_accuracy': 0.78,
            'overall_accuracy': 0.82
        }
    
    def _get_agent_authority_levels(self) -> Dict[str, float]:
        """Get current authority levels for agents"""
        # This would interface with the agent manager in a real system
        return {
            'floorplan_agent': 0.85,
            'placement_agent': 0.92,
            'clock_agent': 0.88,
            'power_agent': 0.80,
            'yield_agent': 0.75,
            'routing_agent': 0.83,
            'thermal_agent': 0.79
        }
    
    def _get_model_confidence_levels(self) -> Dict[str, float]:
        """Get current confidence levels for ML models"""
        # This would interface with the model manager in a real system
        return {
            'congestion_predictor': 0.87,
            'timing_analyzer': 0.84,
            'drc_predictor': 0.81,
            'intent_interpreter': 0.90,
            'knowledge_model': 0.89
        }
    
    def _get_optimization_progress(self) -> Dict[str, float]:
        """Get optimization progress metrics"""
        # This would interface with the optimization engines in a real system
        return {
            'congestion_reduction': 0.25,
            'timing_improvement': 0.18,
            'power_reduction': 0.12,
            'area_efficiency': 0.78,
            'yield_improvement': 0.15
        }
    
    def _get_error_rates(self) -> Dict[str, float]:
        """Get current error rates"""
        # This would interface with the error tracking system in a real system
        return {
            'proposal_rejection_rate': 0.15,
            'model_prediction_error_rate': 0.18,
            'flow_failure_rate': 0.05,
            'communication_error_rate': 0.02
        }
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        return {
            'current_metrics': self.current_metrics,
            'historical_trend': self._get_historical_trend(),
            'health_score': self._calculate_health_score(),
            'recommendations': self._generate_recommendations(),
            'last_updated': datetime.now().isoformat()
        }
    
    def _get_historical_trend(self) -> Dict[str, List[float]]:
        """Get historical trend data"""
        if not self.metrics_history:
            return {}
        
        # Extract metrics over time
        timestamps = [entry['timestamp'] for entry in self.metrics_history]
        cpu_usage = [entry['metrics']['resource_utilization']['cpu_percent'] for entry in self.metrics_history]
        memory_usage = [entry['metrics']['resource_utilization']['memory_percent'] for entry in self.metrics_history]
        accuracy = [entry['metrics']['prediction_accuracy']['overall_accuracy'] for entry in self.metrics_history]
        
        return {
            'timestamps': timestamps,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'prediction_accuracy': accuracy
        }
    
    def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0-1)"""
        if not self.current_metrics:
            return 0.5  # Default score if no metrics available
        
        metrics = self.current_metrics
        
        # Calculate weighted health score
        cpu_health = max(0, 1 - metrics['resource_utilization']['cpu_percent'] / 100.0)
        memory_health = max(0, 1 - metrics['resource_utilization']['memory_percent'] / 100.0)
        accuracy_health = metrics['prediction_accuracy']['overall_accuracy']
        error_health = max(0, 1 - np.mean(list(metrics['error_rates'].values())))
        
        # Weighted average
        health_score = (
            0.2 * cpu_health +
            0.2 * memory_health +
            0.3 * accuracy_health +
            0.3 * error_health
        )
        
        return min(max(health_score, 0.0), 1.0)  # Clamp to [0, 1]
    
    def _generate_recommendations(self) -> List[str]:
        """Generate system health recommendations"""
        recommendations = []
        
        if not self.current_metrics:
            return ["System metrics not available"]
        
        metrics = self.current_metrics
        
        # Resource recommendations
        if metrics['resource_utilization']['cpu_percent'] > 80:
            recommendations.append("CPU usage high - consider scaling or optimization")
        
        if metrics['resource_utilization']['memory_percent'] > 85:
            recommendations.append("Memory usage high - consider memory optimization")
        
        # Performance recommendations
        if metrics['prediction_accuracy']['overall_accuracy'] < 0.8:
            recommendations.append("Prediction accuracy below threshold - investigate model performance")
        
        if metrics['error_rates']['proposal_rejection_rate'] > 0.2:
            recommendations.append("High proposal rejection rate - review agent strategies")
        
        if metrics['error_rates']['flow_failure_rate'] > 0.1:
            recommendations.append("High flow failure rate - investigate flow stability")
        
        # Default recommendation if no issues found
        if not recommendations:
            recommendations.append("System health is optimal")
        
        return recommendations
    
    def save_metrics_history(self, filepath: str):
        """Save metrics history to file"""
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        self.logger.info(f"Metrics history saved to {filepath}")


class SystemDashboard:
    """
    Interactive dashboard for visualizing system health and performance
    """
    
    def __init__(self, health_monitor: SystemHealthMonitor):
        self.health_monitor = health_monitor
        self.logger = get_logger(__name__)
    
    def create_dashboard(self) -> go.Figure:
        """Create an interactive dashboard with system metrics"""
        # Get current metrics
        health_report = self.health_monitor.get_health_report()
        current_metrics = health_report['current_metrics']
        historical_data = health_report.get('historical_trend', {})
        
        # Create subplot figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Resource Utilization', 'Prediction Accuracy', 
                'Agent Authority Levels', 'Optimization Progress',
                'Error Rates', 'System Health Trend'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ]
        )
        
        # 1. Resource Utilization
        if 'resource_utilization' in current_metrics:
            res_util = current_metrics['resource_utilization']
            fig.add_trace(
                go.Bar(
                    x=list(res_util.keys()),
                    y=list(res_util.values()),
                    name="Resource Utilization",
                    marker_color=['red' if v > 80 else 'orange' if v > 60 else 'green' 
                                for v in res_util.values()]
                ),
                row=1, col=1
            )
        
        # 2. Prediction Accuracy
        if 'prediction_accuracy' in current_metrics:
            pred_acc = current_metrics['prediction_accuracy']
            fig.add_trace(
                go.Bar(
                    x=list(pred_acc.keys()),
                    y=list(pred_acc.values()),
                    name="Prediction Accuracy",
                    marker_color='blue'
                ),
                row=1, col=2
            )
        
        # 3. Agent Authority Levels
        if 'agent_authority_levels' in current_metrics:
            agent_auth = current_metrics['agent_authority_levels']
            fig.add_trace(
                go.Bar(
                    x=list(agent_auth.keys()),
                    y=list(agent_auth.values()),
                    name="Agent Authority",
                    marker_color='purple'
                ),
                row=2, col=1
            )
        
        # 4. Optimization Progress
        if 'optimization_progress' in current_metrics:
            opt_prog = current_metrics['optimization_progress']
            fig.add_trace(
                go.Bar(
                    x=list(opt_prog.keys()),
                    y=list(opt_prog.values()),
                    name="Optimization Progress",
                    marker_color='orange'
                ),
                row=2, col=2
            )
        
        # 5. Error Rates
        if 'error_rates' in current_metrics:
            err_rates = current_metrics['error_rates']
            fig.add_trace(
                go.Bar(
                    x=list(err_rates.keys()),
                    y=list(err_rates.values()),
                    name="Error Rates",
                    marker_color=['red' if v > 0.2 else 'orange' if v > 0.1 else 'green' 
                                for v in err_rates.values()]
                ),
                row=3, col=1
            )
        
        # 6. System Health Trend (if historical data available)
        if historical_data and 'timestamps' in historical_data:
            if 'prediction_accuracy' in historical_data:
                fig.add_trace(
                    go.Scatter(
                        x=historical_data['timestamps'],
                        y=historical_data['prediction_accuracy'],
                        mode='lines+markers',
                        name="Accuracy Trend",
                        line=dict(color='blue', width=2)
                    ),
                    row=3, col=2
                )
        
        # Update layout
        fig.update_layout(
            title_text="Silicon Intelligence System - Real-Time Dashboard",
            showlegend=False,
            height=900,
            width=1200
        )
        
        # Update axes
        fig.update_xaxes(title_text="Metrics", row=1, col=1)
        fig.update_yaxes(title_text="Percentage", row=1, col=1)
        
        fig.update_xaxes(title_text="Accuracy Type", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy Score", row=1, col=2)
        
        fig.update_xaxes(title_text="Agent Type", row=2, col=1)
        fig.update_yaxes(title_text="Authority Level", row=2, col=1)
        
        fig.update_xaxes(title_text="Optimization Aspect", row=2, col=2)
        fig.update_yaxes(title_text="Improvement Score", row=2, col=2)
        
        fig.update_xaxes(title_text="Error Type", row=3, col=1)
        fig.update_yaxes(title_text="Error Rate", row=3, col=1)
        
        fig.update_xaxes(title_text="Time", row=3, col=2)
        fig.update_yaxes(title_text="Accuracy", row=3, col=2)
        
        return fig
    
    def create_agent_performance_dashboard(self) -> go.Figure:
        """Create dashboard specifically for agent performance"""
        # Get current metrics
        health_report = self.health_monitor.get_health_report()
        current_metrics = health_report['current_metrics']
        
        if 'agent_authority_levels' not in current_metrics:
            # Create empty figure if no data
            fig = go.Figure()
            fig.add_annotation(text="No agent data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            fig.update_layout(title="Agent Performance Dashboard - No Data")
            return fig
        
        agent_auth = current_metrics['agent_authority_levels']
        agent_names = list(agent_auth.keys())
        authority_levels = list(agent_auth.values())
        
        # Get optimization impact data if available
        opt_impact = current_metrics.get('optimization_progress', {})
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Agent Authority Levels', 'Recent Performance Trends'),
            specs=[[{"type": "bar"}], [{"type": "scatter"}]]
        )
        
        # Authority levels
        fig.add_trace(
            go.Bar(
                x=agent_names,
                y=authority_levels,
                name="Current Authority",
                marker_color=['red' if v < 0.7 else 'orange' if v < 0.85 else 'green' 
                            for v in authority_levels]
            ),
            row=1, col=1
        )
        
        # Add threshold line
        fig.add_hline(y=0.85, line_dash="dash", line_color="green", 
                     annotation_text="High Authority Threshold", row=1, col=1)
        fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                     annotation_text="Low Authority Threshold", row=1, col=1)
        
        # Performance trends (mock data for demonstration)
        time_points = list(range(len(agent_names)))
        performance_trends = [auth * (1 + 0.1*np.random.randn()) for auth in authority_levels]  # Add some noise
        
        fig.add_trace(
            go.Scatter(
                x=agent_names,
                y=performance_trends,
                mode='lines+markers',
                name="Performance Trend",
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Silicon Intelligence System - Agent Performance Dashboard",
            height=600,
            width=1000
        )
        
        fig.update_yaxes(title_text="Authority Level", row=1, col=1)
        fig.update_yaxes(title_text="Performance Score", row=2, col=1)
        
        return fig
    
    def create_ml_model_dashboard(self) -> go.Figure:
        """Create dashboard specifically for ML model performance"""
        # Get current metrics
        health_report = self.health_monitor.get_health_report()
        current_metrics = health_report['current_metrics']
        
        if 'model_confidence_levels' not in current_metrics:
            # Create empty figure if no data
            fig = go.Figure()
            fig.add_annotation(text="No ML model data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            fig.update_layout(title="ML Model Performance Dashboard - No Data")
            return fig
        
        model_conf = current_metrics['model_confidence_levels']
        model_names = list(model_conf.keys())
        confidence_levels = list(model_conf.values())
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=confidence_levels,
                name="Model Confidence",
                marker_color=['red' if v < 0.75 else 'orange' if v < 0.85 else 'green' 
                            for v in confidence_levels]
            )
        )
        
        # Add threshold lines
        fig.add_hline(y=0.85, line_dash="dash", line_color="green", 
                     annotation_text="High Confidence Threshold")
        fig.add_hline(y=0.75, line_dash="dash", line_color="red", 
                     annotation_text="Low Confidence Threshold")
        
        fig.update_layout(
            title="Silicon Intelligence System - ML Model Confidence Dashboard",
            xaxis_title="Model Type",
            yaxis_title="Confidence Level",
            height=500,
            width=900
        )
        
        return fig
    
    def save_dashboard_html(self, fig: go.Figure, filepath: str):
        """Save dashboard as HTML file"""
        fig.write_html(filepath)
        self.logger.info(f"Dashboard saved to {filepath}")
    
    def start_web_server(self, port: int = 8050):
        """Start a web server to serve the dashboard"""
        import dash
        from dash import dcc, html
        import plotly.express as px
        
        app = dash.Dash(__name__)
        
        # Create initial figures
        main_fig = self.create_dashboard()
        agent_fig = self.create_agent_performance_dashboard()
        ml_fig = self.create_ml_model_dashboard()
        
        app.layout = html.Div([
            html.H1("Silicon Intelligence System - Real-Time Dashboard"),
            
            html.Div([
                html.H2("System Health Overview"),
                dcc.Graph(id='main-dashboard', figure=main_fig)
            ]),
            
            html.Div([
                html.H2("Agent Performance"),
                dcc.Graph(id='agent-dashboard', figure=agent_fig)
            ]),
            
            html.Div([
                html.H2("ML Model Performance"),
                dcc.Graph(id='ml-dashboard', figure=ml_fig)
            ]),
            
            dcc.Interval(
                id='interval-component',
                interval=10*1000,  # Update every 10 seconds
                n_intervals=0
            )
        ])
        
        # Callback to update figures periodically
        @app.callback(
            [dash.Output('main-dashboard', 'figure'),
             dash.Output('agent-dashboard', 'figure'),
             dash.Output('ml-dashboard', 'figure')],
            [dash.Input('interval-component', 'n_intervals')]
        )
        def update_figures(n):
            # Refresh metrics
            main_fig = self.create_dashboard()
            agent_fig = self.create_agent_performance_dashboard()
            ml_fig = self.create_ml_model_dashboard()
            
            return main_fig, agent_fig, ml_fig
        
        self.logger.info(f"Starting dashboard web server on port {port}")
        app.run_server(debug=False, port=port)


def run_system_monitoring_demo():
    """Run a demonstration of the system monitoring capabilities"""
    logger = get_logger(__name__)
    logger.info("Starting Silicon Intelligence System monitoring demo")
    
    # Initialize health monitor
    health_monitor = SystemHealthMonitor()
    health_monitor.start_monitoring()
    
    # Initialize dashboard
    dashboard = SystemDashboard(health_monitor)
    
    print("\n" + "="*60)
    print("SILICON INTELLIGENCE SYSTEM - MONITORING DASHBOARD")
    print("="*60)
    print("\nSystem monitoring is now active. Available dashboards:")
    print("1. Main System Dashboard - Overall health and metrics")
    print("2. Agent Performance Dashboard - Agent authority and performance")
    print("3. ML Model Dashboard - Model confidence and accuracy")
    print("\nThe system continuously monitors:")
    print("• Resource utilization (CPU, memory, disk)")
    print("• Prediction accuracy across all models")
    print("• Agent authority levels and performance")
    print("• Optimization progress and PPA metrics")
    print("• Error rates and system stability")
    print("\nTo view dashboards in browser: dashboard.start_web_server(port=8050)")
    print("To get health report: health_monitor.get_health_report()")
    print("To stop monitoring: health_monitor.stop_monitoring()")
    print("="*60)
    
    # Show initial health report
    time.sleep(2)  # Allow some metrics to accumulate
    health_report = health_monitor.get_health_report()
    
    print(f"\nInitial Health Report:")
    print(f"• Health Score: {health_report['health_score']:.3f}")
    print(f"• Recommendations: {len(health_report['recommendations'])}")
    for rec in health_report['recommendations']:
        print(f"  - {rec}")
    
    return health_monitor, dashboard


if __name__ == "__main__":
    monitor, dash = run_system_monitoring_demo()
    
    # Uncomment the next line to start the web dashboard
    # dash.start_web_server(port=8050)