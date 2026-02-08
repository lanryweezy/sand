"""
Visualization Module for Silicon Intelligence System

This module provides visualization capabilities to understand the system's
decision-making process and results.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from typing import Dict, List, Any, Optional
import pandas as pd
from matplotlib.patches import Rectangle
from silicon_intelligence.core.canonical_silicon_graph import CanonicalSiliconGraph
from silicon_intelligence.cognitive.physical_risk_oracle import PhysicalRiskAssessment
from silicon_intelligence.agents.base_agent import AgentProposal
from silicon_intelligence.core.parallel_reality_engine import ParallelUniverse
from silicon_intelligence.models.drc_predictor import DRCPredictor


class SiliconVisualizer:
    """
    Silicon Visualizer - creates visual representations of the system's analysis
    """
    
    def __init__(self):
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def visualize_risk_assessment(self, assessment: PhysicalRiskAssessment, 
                                title: str = "Physical Risk Assessment"):
        """
        Visualize the physical risk assessment results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)
        
        # 1. Congestion heatmap
        ax1 = axes[0, 0]
        if assessment.congestion_heatmap:
            nodes = list(assessment.congestion_heatmap.keys())[:20]  # Limit for readability
            congestion_values = [assessment.congestion_heatmap[node] for node in nodes]
            ax1.bar(range(len(nodes)), congestion_values)
            ax1.set_title("Congestion Risk Heatmap")
            ax1.set_xlabel("Nodes")
            ax1.set_ylabel("Congestion Risk")
            ax1.set_xticks(range(len(nodes)))
            ax1.set_xticklabels(nodes, rotation=45, ha="right")
        else:
            ax1.text(0.5, 0.5, "No congestion data", horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
            ax1.set_title("Congestion Risk Heatmap")
        
        # 2. Timing risk zones
        ax2 = axes[0, 1]
        if assessment.timing_risk_zones:
            risk_levels = [zone.get('risk_level', 'low') for zone in assessment.timing_risk_zones]
            risk_counts = {level: risk_levels.count(level) for level in set(risk_levels)}
            ax2.bar(risk_counts.keys(), risk_counts.values())
            ax2.set_title("Timing Risk Zone Distribution")
            ax2.set_xlabel("Risk Level")
            ax2.set_ylabel("Count")
        else:
            ax2.text(0.5, 0.5, "No timing risk data", horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
            ax2.set_title("Timing Risk Zone Distribution")
        
        # 3. Clock skew sensitivity
        ax3 = axes[1, 0]
        if assessment.clock_skew_sensitivity:
            nodes = list(assessment.clock_skew_sensitivity.keys())[:15]  # Limit for readability
            sensitivity_values = [assessment.clock_skew_sensitivity[node] for node in nodes]
            ax3.scatter(range(len(nodes)), sensitivity_values, alpha=0.7)
            ax3.set_title("Clock Skew Sensitivity")
            ax3.set_xlabel("Clock Elements")
            ax3.set_ylabel("Sensitivity")
            ax3.set_xticks(range(len(nodes)))
            ax3.set_xticklabels(nodes, rotation=45, ha="right")
        else:
            ax3.text(0.5, 0.5, "No clock sensitivity data", horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
            ax3.set_title("Clock Skew Sensitivity")
        
        # 4. Power density hotspots
        ax4 = axes[1, 1]
        if assessment.power_density_hotspots:
            powers = [spot.get('estimated_power', 0) for spot in assessment.power_density_hotspots]
            regions = [spot.get('region', 'unknown') for spot in assessment.power_density_hotspots]
            unique_regions = list(set(regions))
            region_powers = {region: [] for region in unique_regions}
            
            for i, region in enumerate(regions):
                region_powers[region].append(powers[i])
            
            avg_powers = [np.mean(region_powers[region]) for region in unique_regions]
            ax4.bar(unique_regions, avg_powers)
            ax4.set_title("Average Power by Region")
            ax4.set_xlabel("Regions")
            ax4.set_ylabel("Average Power")
            ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.text(0.5, 0.5, "No power hotspot data", horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
            ax4.set_title("Power Density Hotspots")
        
        plt.tight_layout()
        return fig
    
    def visualize_canonical_graph(self, graph: CanonicalSiliconGraph, 
                                title: str = "Canonical Silicon Graph"):
        """
        Visualize the canonical silicon graph
        """
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle(title, fontsize=16)
        
        # Create a subgraph for visualization (too many nodes can be overwhelming)
        nodes = list(graph.graph.nodes())[:100]  # Limit to first 100 nodes
        subgraph = graph.graph.subgraph(nodes)
        
        # 1. Graph structure visualization
        ax1 = axes[0]
        pos = nx.spring_layout(subgraph, seed=42)  # Consistent layout
        
        # Color nodes by type
        node_colors = []
        for node in subgraph.nodes():
            attrs = subgraph.nodes[node]
            node_type = attrs.get('node_type', 'unknown')
            if node_type == 'cell':
                node_colors.append('lightblue')
            elif node_type == 'macro':
                node_colors.append('orange')
            elif node_type == 'clock':
                node_colors.append('red')
            elif node_type == 'power':
                node_colors.append('green')
            elif node_type == 'port':
                node_colors.append('purple')
            else:
                node_colors.append('gray')
        
        nx.draw(subgraph, pos, ax=ax1, node_color=node_colors, 
                with_labels=True, font_size=8, node_size=300)
        ax1.set_title("Graph Structure (Node Types)")
        
        # 2. Attribute distribution
        ax2 = axes[1]
        
        # Plot power distribution
        powers = [attrs.get('power', 0) for _, attrs in subgraph.nodes(data=True)]
        if any(p > 0 for p in powers):
            ax2.hist(powers, bins=20, alpha=0.7, label='Power', color='blue')
            ax2.set_title("Power Distribution")
            ax2.set_xlabel("Power")
            ax2.set_ylabel("Frequency")
        else:
            ax2.text(0.5, 0.5, "No power data", horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
            ax2.set_title("Power Distribution")
        
        plt.tight_layout()
        return fig
    
    def visualize_agent_proposals(self, proposals: List[AgentProposal], 
                                title: str = "Agent Proposals Analysis"):
        """
        Visualize agent proposals and their characteristics
        """
        if not proposals:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No proposals to visualize", 
                   horizontalalignment='center', verticalalignment='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(title)
            return fig
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)
        
        # Convert proposals to DataFrame for easier analysis
        proposal_data = []
        for prop in proposals:
            proposal_data.append({
                'agent_type': prop.agent_type.value,
                'confidence': prop.confidence_score,
                'action_type': prop.action_type,
                'target_count': len(prop.targets),
                'risk_timing': prop.risk_profile.get('timing_risk', 0),
                'risk_congestion': prop.risk_profile.get('congestion_risk', 0),
                'cost_power': prop.cost_vector.get('power', 0),
                'cost_performance': prop.cost_vector.get('performance', 0)
            })
        
        df = pd.DataFrame(proposal_data)
        
        # 1. Agent type distribution
        ax1 = axes[0, 0]
        agent_counts = df['agent_type'].value_counts()
        ax1.pie(agent_counts.values, labels=agent_counts.index, autopct='%1.1f%%')
        ax1.set_title("Agent Type Distribution")
        
        # 2. Confidence scores by agent type
        ax2 = axes[0, 1]
        sns.boxplot(data=df, x='agent_type', y='confidence', ax=ax2)
        ax2.set_title("Confidence Scores by Agent Type")
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Cost vectors (Power vs Performance)
        ax3 = axes[1, 0]
        scatter = ax3.scatter(df['cost_power'], df['cost_performance'], 
                             c=pd.Categorical(df['agent_type']).codes, 
                             cmap='viridis', alpha=0.7)
        ax3.set_xlabel("Power Cost")
        ax3.set_ylabel("Performance Cost")
        ax3.set_title("Cost Vector Analysis (Power vs Performance)")
        plt.colorbar(scatter, ax=ax3, label="Agent Type")
        
        # 4. Risk profiles
        ax4 = axes[1, 1]
        x = np.arange(len(df))
        width = 0.35
        ax4.bar(x - width/2, df['risk_timing'], width, label='Timing Risk', alpha=0.7)
        ax4.bar(x + width/2, df['risk_congestion'], width, label='Congestion Risk', alpha=0.7)
        ax4.set_xlabel("Proposals")
        ax4.set_ylabel("Risk Level")
        ax4.set_title("Risk Profile Comparison")
        ax4.legend()
        ax4.set_xticks(x)
        ax4.set_xticklabels([f"P{i}" for i in range(len(df))], rotation=45)
        
        plt.tight_layout()
        return fig
    
    def visualize_parallel_universes(self, universes: List[ParallelUniverse], 
                                   title: str = "Parallel Universe Analysis"):
        """
        Visualize the parallel universes and their outcomes
        """
        if not universes:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No universes to visualize", 
                   horizontalalignment='center', verticalalignment='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(title)
            return fig
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)
        
        # Prepare data
        universe_ids = [u.id for u in universes]
        scores = [u.score for u in universes]
        execution_times = [u.execution_time for u in universes]
        active_status = [u.active for u in universes]
        proposal_counts = [len(u.proposals) for u in universes]
        
        # 1. Universe scores
        ax1 = axes[0, 0]
        colors = ['green' if active else 'red' for active in active_status]
        bars = ax1.bar(range(len(universe_ids)), scores, color=colors, alpha=0.7)
        ax1.set_xlabel("Universes")
        ax1.set_ylabel("Score")
        ax1.set_title("Universe Scores (Green=Active, Red=Inactive)")
        ax1.set_xticks(range(len(universe_ids)))
        ax1.set_xticklabels([uid.split('_')[-1] for uid in universe_ids], rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.2f}', ha='center', va='bottom')
        
        # 2. Execution times
        ax2 = axes[0, 1]
        ax2.bar(range(len(universe_ids)), execution_times, alpha=0.7, color='orange')
        ax2.set_xlabel("Universes")
        ax2.set_ylabel("Execution Time (s)")
        ax2.set_title("Execution Time by Universe")
        ax2.set_xticks(range(len(universe_ids)))
        ax2.set_xticklabels([uid.split('_')[-1] for uid in universe_ids], rotation=45)
        
        # 3. Proposal counts
        ax3 = axes[1, 0]
        ax3.bar(range(len(universe_ids)), proposal_counts, alpha=0.7, color='purple')
        ax3.set_xlabel("Universes")
        ax3.set_ylabel("Number of Proposals")
        ax3.set_title("Proposals Applied by Universe")
        ax3.set_xticks(range(len(universe_ids)))
        ax3.set_xticklabels([uid.split('_')[-1] for uid in universe_ids], rotation=45)
        
        # 4. Score vs Execution time scatter
        ax4 = axes[1, 1]
        scatter_colors = ['green' if active else 'red' for active in active_status]
        scatter = ax4.scatter(execution_times, scores, c=scatter_colors, s=100, alpha=0.7)
        ax4.set_xlabel("Execution Time (s)")
        ax4.set_ylabel("Score")
        ax4.set_title("Score vs Execution Time (Green=Active, Red=Inactive)")
        
        # Add universe labels
        for i, txt in enumerate([uid.split('_')[-1] for uid in universe_ids]):
            ax4.annotate(txt, (execution_times[i], scores[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def visualize_drc_predictions(self, drc_predictions: Dict[str, Any], 
                                title: str = "DRC Prediction Analysis"):
        """
        Visualize DRC predictions and risk analysis
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)
        
        # 1. Overall risk score
        ax1 = axes[0, 0]
        ax1.bar(['Overall Risk'], [drc_predictions.get('overall_risk_score', 0)], 
                color='red' if drc_predictions.get('overall_risk_score', 0) > 0.5 else 'green')
        ax1.set_ylim(0, 1)
        ax1.set_ylabel("Risk Score")
        ax1.set_title("Overall DRC Risk Score")
        
        # 2. Individual risk categories
        ax2 = axes[0, 1]
        risk_categories = ['spacing_violations', 'density_violations', 'antenna_violations', 'via_violations']
        risk_scores = []
        category_labels = []
        
        for cat in risk_categories:
            if cat in drc_predictions:
                risk_scores.append(drc_predictions[cat].get('risk_score', 0))
                category_labels.append(cat.replace('_', ' ').title())
        
        if risk_scores:
            ax2.bar(category_labels, risk_scores)
            ax2.set_ylabel("Risk Score")
            ax2.set_title("DRC Risk by Category")
            ax2.tick_params(axis='x', rotation=45)
        else:
            ax2.text(0.5, 0.5, "No risk category data", horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
            ax2.set_title("DRC Risk by Category")
        
        # 3. High risk areas
        ax3 = axes[1, 0]
        high_risk_items = []
        high_risk_values = []
        
        for cat in risk_categories:
            if cat in drc_predictions and 'high_risk_areas' in drc_predictions[cat]:
                areas = drc_predictions[cat]['high_risk_areas']
                for area in areas:
                    prob = area.get('estimated_violation_probability', 0)
                    if prob > 0.3:  # High risk threshold
                        high_risk_items.append(f"{cat[:3]}-{len(high_risk_items)}")
                        high_risk_values.append(prob)
        
        if high_risk_items:
            ax3.barh(high_risk_items, high_risk_values)
            ax3.set_xlabel("Violation Probability")
            ax3.set_title("High-Risk Areas (>30%)")
        else:
            ax3.text(0.5, 0.5, "No high-risk areas", horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
            ax3.set_title("High-Risk Areas (>30%)")
        
        # 4. Process node information
        ax4 = axes[1, 1]
        ax4.text(0.1, 0.8, f"Process Node: {drc_predictions.get('process_node', 'Unknown')}", 
                transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.6, f"Confidence: {drc_predictions.get('confidence', 0):.2f}", 
                transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.4, f"Total Predicted Violations: {sum(len(drc_predictions.get(cat, {}).get('predicted_violations', [])) for cat in risk_categories)}", 
                transform=ax4.transAxes, fontsize=12)
        ax4.axis('off')
        ax4.set_title("DRC Prediction Metadata")
        
        plt.tight_layout()
        return fig
    
    def visualize_flow_progress(self, step_results: List[Dict], 
                              title: str = "Flow Progress Visualization"):
        """
        Visualize the progress of the flow execution
        """
        if not step_results:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No flow results to visualize", 
                   horizontalalignment='center', verticalalignment='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(title)
            return fig
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)
        
        # Extract data
        stages = [result['stage'] for result in step_results]
        durations = [result['duration'] for result in step_results]
        success_status = [result['success'] for result in step_results]
        timestamps = [pd.to_datetime(result['timestamp']) for result in step_results]
        
        # 1. Duration by stage
        ax1 = axes[0, 0]
        colors = ['green' if success else 'red' for success in success_status]
        bars = ax1.bar(stages, durations, color=colors, alpha=0.7)
        ax1.set_ylabel("Duration (seconds)")
        ax1.set_title("Duration by Stage (Green=Success, Red=Failure)")
        ax1.tick_params(axis='x', rotation=45)
        
        # Add duration labels
        for bar, duration in zip(bars, durations):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{duration:.2f}s', ha='center', va='bottom', fontsize=8)
        
        # 2. Success/Failure distribution
        ax2 = axes[0, 1]
        success_counts = [success_status.count(True), success_status.count(False)]
        ax2.pie(success_counts, labels=['Success', 'Failure'], autopct='%1.1f%%', 
                colors=['green', 'red'])
        ax2.set_title("Success/Failure Distribution")
        
        # 3. Cumulative time
        ax3 = axes[1, 0]
        cumulative_time = np.cumsum(durations)
        ax3.plot(range(len(stages)), cumulative_time, marker='o', linewidth=2)
        ax3.set_xlabel("Stage Index")
        ax3.set_ylabel("Cumulative Time (seconds)")
        ax3.set_title("Cumulative Execution Time")
        ax3.grid(True, alpha=0.3)
        
        # 4. Timeline
        ax4 = axes[1, 1]
        start_time = min(timestamps)
        relative_times = [(ts - start_time).total_seconds() for ts in timestamps]
        
        # Create a Gantt-like chart
        for i, (stage, rel_time, duration, success) in enumerate(zip(stages, relative_times, durations, success_status)):
            color = 'green' if success else 'red'
            ax4.barh(i, duration, left=rel_time, color=color, alpha=0.7, edgecolor='black')
            ax4.text(rel_time + duration/2, i, stage.split('.')[-1][:10], 
                    ha='center', va='center', fontsize=8)
        
        ax4.set_xlabel("Time (seconds)")
        ax4.set_yticks(range(len(stages)))
        ax4.set_yticklabels([s.split('.')[-1] for s in stages])
        ax4.set_title("Execution Timeline")
        
        plt.tight_layout()
        return fig
    
    def save_visualization(self, fig, filename: str, dpi: int = 300):
        """
        Save visualization to file
        """
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory


# Example usage function
def example_visualization():
    """
    Example of how to use the visualization module
    """
    import tempfile
    import os
    
    # Create visualizer
    viz = SiliconVisualizer()
    
    # Create some dummy data for visualization examples
    print("Creating example visualizations...")
    
    # Example 1: Risk assessment visualization
    from silicon_intelligence.cognitive.physical_risk_oracle import PhysicalRiskAssessment
    assessment = PhysicalRiskAssessment(
        congestion_heatmap={'node1': 0.7, 'node2': 0.3, 'node3': 0.9},
        timing_risk_zones=[
            {'path': ['a', 'b', 'c'], 'risk_level': 'high', 'slack': -0.1},
            {'path': ['d', 'e'], 'risk_level': 'medium', 'slack': 0.2}
        ],
        clock_skew_sensitivity={'clk1': 0.8, 'clk2': 0.4},
        power_density_hotspots=[
            {'node': 'macro1', 'estimated_power': 1.2, 'region': 'core'},
            {'node': 'macro2', 'estimated_power': 0.8, 'region': 'peripheral'}
        ],
        drc_risk_classes=[
            {'rule_class': 'spacing', 'severity': 'high', 'description': 'Min spacing violation'},
            {'rule_class': 'density', 'severity': 'medium', 'description': 'Density rule violation'}
        ],
        overall_confidence=0.85,
        recommendations=[
            "Increase spacing in high-congestion areas",
            "Review clock tree for skew issues"
        ]
    )
    
    fig1 = viz.visualize_risk_assessment(assessment, "Example Risk Assessment")
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        viz.save_visualization(fig1, tmp.name)
        print(f"Saved risk assessment visualization: {tmp.name}")
    
    # Example 2: Canonical graph visualization
    graph = CanonicalSiliconGraph()
    # Add some dummy nodes for visualization
    graph.graph.add_node('cell1', node_type='cell', power=0.1, area=2.0)
    graph.graph.add_node('macro1', node_type='macro', power=1.0, area=100.0)
    graph.graph.add_node('clk1', node_type='clock', power=0.05, area=1.5)
    graph.graph.add_edge('cell1', 'macro1')
    graph.graph.add_edge('clk1', 'cell1')
    
    fig2 = viz.visualize_canonical_graph(graph, "Example Canonical Graph")
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        viz.save_visualization(fig2, tmp.name)
        print(f"Saved graph visualization: {tmp.name}")
    
    print("Example visualizations created successfully!")


if __name__ == "__main__":
    example_visualization()