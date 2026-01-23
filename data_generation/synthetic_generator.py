#!/usr/bin/env python3
"""
Synthetic Data Generator for Silicon Intelligence System

Generates data by breaking designs on purpose to create labeled causality,
which real-world data rarely has. One intelligently generated failure is worth ten clean designs.
"""

import os
import sys
import json
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from core.canonical_silicon_graph import CanonicalSiliconGraph
from data.comprehensive_rtl_parser import DesignHierarchyBuilder


@dataclass
class SyntheticExperiment:
    """Represents a synthetic experiment designed to break a design in a specific way"""
    experiment_id: str
    original_design: str
    manipulation_type: str  # 'constraint_stress', 'floorplan_break', 'clock_stress', etc.
    parameters: Dict[str, Any]  # Specific parameters for the manipulation
    intended_failure_mode: str  # What we expect to break
    timestamp: datetime
    result: Dict[str, Any]  # Actual results of the experiment


class SyntheticDataGenerator:
    """
    Generates synthetic data by intentionally breaking designs to learn causality
    """
    
    def __init__(self, output_path: str = "synthetic_data"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        self.experiments: List[SyntheticExperiment] = []
        
        # Create subdirectories
        (self.output_path / "rtl").mkdir(exist_ok=True)
        (self.output_path / "experiments").mkdir(exist_ok=True)
        (self.output_path / "results").mkdir(exist_ok=True)
        
    def generate_constraint_stress_tests(self, base_rtl_path: str, num_experiments: int = 10) -> List[SyntheticExperiment]:
        """Generate experiments that stress constraints to cause failures"""
        experiments = []
        
        for i in range(num_experiments):
            # Create variations of timing constraints
            experiment_id = f"constraint_stress_{i:03d}"
            
            # Generate stressed constraints
            stress_params = {
                'clock_period_multiplier': random.uniform(0.5, 1.5),  # Make clocks faster/slower
                'input_delay_reduction': random.uniform(0.1, 0.8),   # Reduce input delays
                'output_delay_reduction': random.uniform(0.1, 0.8),  # Reduce output delays
                'fanout_limit_scaling': random.uniform(0.5, 2.0),    # Change fanout limits
                'area_constraint_scaling': random.uniform(0.3, 1.5)  # Change area constraints
            }
            
            # Manipulate the RTL/constraints file to apply stress
            modified_rtl_path = self._apply_constraint_stress(base_rtl_path, stress_params, experiment_id)
            
            experiment = SyntheticExperiment(
                experiment_id=experiment_id,
                original_design=base_rtl_path,
                manipulation_type='constraint_stress',
                parameters=stress_params,
                intended_failure_mode='timing_violations',
                timestamp=datetime.now(),
                result={}
            )
            
            experiments.append(experiment)
        
        self.experiments.extend(experiments)
        return experiments
    
    def _apply_constraint_stress(self, base_rtl_path: str, params: Dict[str, Any], experiment_id: str) -> str:
        """Apply constraint stress to create a modified RTL file"""
        # Read the original RTL file
        with open(base_rtl_path, 'r') as f:
            rtl_content = f.read()
        
        # Create modified RTL with stressed constraints
        modified_content = self._modify_constraints(rtl_content, params)
        
        # Save to new file
        output_file = self.output_path / "rtl" / f"{experiment_id}_modified.v"
        with open(output_file, 'w') as f:
            f.write(modified_content)
        
        return str(output_file)
    
    def _modify_constraints(self, rtl_content: str, params: Dict[str, Any]) -> str:
        """Modify constraints in the RTL content based on parameters"""
        # This is a simplified implementation
        # In practice, this would parse and modify SDC files properly
        modified_content = rtl_content
        
        # Example: Add aggressive timing constraints
        if 'clock_period_multiplier' in params and params['clock_period_multiplier'] < 1.0:
            # Add more aggressive clock constraints
            aggressive_clock = f"\n// Aggressive clock constraint (multiplier: {params['clock_period_multiplier']})"
            aggressive_clock += f"\ncreate_clock -name test_clk -period {2.0 * params['clock_period_multiplier']}"
            modified_content += aggressive_clock
        
        return modified_content
    
    def generate_floorplan_break_tests(self, base_rtl_path: str, num_experiments: int = 10) -> List[SyntheticExperiment]:
        """Generate experiments that break floorplans to cause congestion"""
        experiments = []
        
        for i in range(num_experiments):
            experiment_id = f"floorplan_break_{i:03d}"
            
            # Generate floorplan stress parameters
            stress_params = {
                'utilization_target': random.uniform(0.8, 1.2),  # Force high utilization
                'aspect_ratio_distortion': random.uniform(0.5, 2.0),  # Distort shape
                'macro_placement_randomization': random.uniform(0.1, 0.9),  # Randomize macro placement
                'bandwidth_constraint_scaling': random.uniform(0.3, 2.0),  # Stress bandwidth
                'power_density_focusing': random.uniform(0.7, 1.5)  # Concentrate power
            }
            
            # Apply floorplan stress
            modified_rtl_path = self._apply_floorplan_stress(base_rtl_path, stress_params, experiment_id)
            
            experiment = SyntheticExperiment(
                experiment_id=experiment_id,
                original_design=base_rtl_path,
                manipulation_type='floorplan_break',
                parameters=stress_params,
                intended_failure_mode='congestion',
                timestamp=datetime.now(),
                result={}
            )
            
            experiments.append(experiment)
        
        self.experiments.extend(experiments)
        return experiments
    
    def _apply_floorplan_stress(self, base_rtl_path: str, params: Dict[str, Any], experiment_id: str) -> str:
        """Apply floorplan stress to create a modified RTL file"""
        with open(base_rtl_path, 'r') as f:
            rtl_content = f.read()
        
        # Apply floorplan modifications
        modified_content = self._modify_for_congestion(rtl_content, params)
        
        # Save to new file
        output_file = self.output_path / "rtl" / f"{experiment_id}_congested.v"
        with open(output_file, 'w') as f:
            f.write(modified_content)
        
        return str(output_file)
    
    def _modify_for_congestion(self, rtl_content: str, params: Dict[str, Any]) -> str:
        """Modify RTL to create congestion-prone structures"""
        modified_content = rtl_content
        
        # Add structures that typically cause congestion
        if params.get('utilization_target', 1.0) > 1.0:
            # Add more interconnects to increase congestion
            extra_connections = "\n// Extra connections to increase congestion"
            for i in range(int(params['utilization_target'] * 10)):
                extra_connections += f"\nwire extra_conn_{i};\n"
            modified_content += extra_connections
        
        return modified_content
    
    def generate_clock_stress_tests(self, base_rtl_path: str, num_experiments: int = 10) -> List[SyntheticExperiment]:
        """Generate experiments that stress clock domains to cause skew/timing issues"""
        experiments = []
        
        for i in range(num_experiments):
            experiment_id = f"clock_stress_{i:03d}"
            
            # Generate clock stress parameters
            stress_params = {
                'clock_fanout_multiplier': random.randint(2, 10),  # Increase fanout
                'clock_tree_depth_max': random.randint(3, 8),      # Limit tree depth
                'skew_tolerance_reduction': random.uniform(0.1, 0.8),  # Reduce skew tolerance
                'multiple_clock_domains': random.choice([True, False]),  # Add multiple domains
                'clock_gating_removal': random.uniform(0.0, 0.5)   # Remove clock gating
            }
            
            # Apply clock stress
            modified_rtl_path = self._apply_clock_stress(base_rtl_path, stress_params, experiment_id)
            
            experiment = SyntheticExperiment(
                experiment_id=experiment_id,
                original_design=base_rtl_path,
                manipulation_type='clock_stress',
                parameters=stress_params,
                intended_failure_mode='clock_skew',
                timestamp=datetime.now(),
                result={}
            )
            
            experiments.append(experiment)
        
        self.experiments.extend(experiments)
        return experiments
    
    def _apply_clock_stress(self, base_rtl_path: str, params: Dict[str, Any], experiment_id: str) -> str:
        """Apply clock stress to create a modified RTL file"""
        with open(base_rtl_path, 'r') as f:
            rtl_content = f.read()
        
        # Apply clock modifications
        modified_content = self._modify_for_clock_stress(rtl_content, params)
        
        # Save to new file
        output_file = self.output_path / "rtl" / f"{experiment_id}_clock_stressed.v"
        with open(output_file, 'w') as f:
            f.write(modified_content)
        
        return str(output_file)
    
    def _modify_for_clock_stress(self, rtl_content: str, params: Dict[str, Any]) -> str:
        """Modify RTL to create clock-stress conditions"""
        modified_content = rtl_content
        
        # Add clock stress indicators
        if params.get('multiple_clock_domains', False):
            # Add extra clock domains to increase complexity
            extra_clocks = "\n// Additional clock domains for stress testing"
            for i in range(random.randint(1, 3)):
                extra_clocks += f"\nwire clk_extra_{i};\n"
            modified_content += extra_clocks
        
        return modified_content
    
    def generate_power_stress_tests(self, base_rtl_path: str, num_experiments: int = 10) -> List[SyntheticExperiment]:
        """Generate experiments that stress power domains to cause hotspots"""
        experiments = []
        
        for i in range(num_experiments):
            experiment_id = f"power_stress_{i:03d}"
            
            # Generate power stress parameters
            stress_params = {
                'activity_factor_increase': random.uniform(1.5, 5.0),  # Increase switching activity
                'power_domain_reduction': random.uniform(0.5, 0.9),    # Reduce power domains
                'voltage_scaling_reduction': random.uniform(0.8, 1.0), # Reduce voltage scaling
                'leakage_injection': random.uniform(0.1, 0.8),        # Increase leakage
                'power_grid_thinning': random.uniform(0.2, 0.8)       # Thin power grid
            }
            
            # Apply power stress
            modified_rtl_path = self._apply_power_stress(base_rtl_path, stress_params, experiment_id)
            
            experiment = SyntheticExperiment(
                experiment_id=experiment_id,
                original_design=base_rtl_path,
                manipulation_type='power_stress',
                parameters=stress_params,
                intended_failure_mode='power_hotspots',
                timestamp=datetime.now(),
                result={}
            )
            
            experiments.append(experiment)
        
        self.experiments.extend(experiments)
        return experiments
    
    def _apply_power_stress(self, base_rtl_path: str, params: Dict[str, Any], experiment_id: str) -> str:
        """Apply power stress to create a modified RTL file"""
        with open(base_rtl_path, 'r') as f:
            rtl_content = f.read()
        
        # Apply power modifications
        modified_content = self._modify_for_power_stress(rtl_content, params)
        
        # Save to new file
        output_file = self.output_path / "rtl" / f"{experiment_id}_power_stressed.v"
        with open(output_file, 'w') as f:
            f.write(modified_content)
        
        return str(output_file)
    
    def _modify_for_power_stress(self, rtl_content: str, params: Dict[str, Any]) -> str:
        """Modify RTL to create power-stress conditions"""
        modified_content = rtl_content
        
        # Add power stress indicators
        if params.get('activity_factor_increase', 1.0) > 1.0:
            # Add toggle-prone signals to increase power
            extra_toggles = "\n// Extra toggling signals for power stress"
            for i in range(int(params['activity_factor_increase'] * 5)):
                extra_toggles += f"\nreg toggle_signal_{i};\n"
                extra_toggles += f"always @(posedge clk) toggle_signal_{i} <= ~toggle_signal_{i};\n"
            modified_content += extra_toggles
        
        return modified_content
    
    def run_all_synthetic_experiments(self, base_rtl_path: str) -> List[SyntheticExperiment]:
        """Run all types of synthetic experiments on a base design"""
        print(f"Generating synthetic experiments for: {base_rtl_path}")
        
        all_experiments = []
        
        # Run all experiment types
        constraint_experiments = self.generate_constraint_stress_tests(base_rtl_path, num_experiments=5)
        floorplan_experiments = self.generate_floorplan_break_tests(base_rtl_path, num_experiments=5)
        clock_experiments = self.generate_clock_stress_tests(base_rtl_path, num_experiments=5)
        power_experiments = self.generate_power_stress_tests(base_rtl_path, num_experiments=5)
        
        all_experiments.extend(constraint_experiments)
        all_experiments.extend(floorplan_experiments)
        all_experiments.extend(clock_experiments)
        all_experiments.extend(power_experiments)
        
        print(f"Generated {len(all_experiments)} synthetic experiments")
        
        # Save experiment metadata
        self._save_experiment_metadata(all_experiments)
        
        return all_experiments
    
    def _save_experiment_metadata(self, experiments: List[SyntheticExperiment]):
        """Save experiment metadata to file"""
        for exp in experiments:
            exp_dict = asdict(exp)
            exp_dict['timestamp'] = exp.timestamp.isoformat()
            
            filename = f"experiment_{exp.experiment_id}.json"
            filepath = self.output_path / "experiments" / filename
            
            with open(filepath, 'w') as f:
                json.dump(exp_dict, f, indent=2, default=str)
    
    def generate_curriculum_data(self) -> List[SyntheticExperiment]:
        """Generate curriculum-style data from simple to complex"""
        print("Generating curriculum-style synthetic data...")
        
        # For this example, we'll create a simple curriculum
        # In practice, you'd have actual simple -> complex design progression
        curriculum_experiments = []
        
        # Start with simple designs
        simple_params = {
            'complexity_level': 'simple',
            'num_modules': random.randint(1, 3),
            'num_clocks': 1,
            'utilization': 0.3
        }
        
        # Progress to complex designs
        complex_params = {
            'complexity_level': 'complex',
            'num_modules': random.randint(10, 20),
            'num_clocks': random.randint(3, 8),
            'utilization': 0.9
        }
        
        # Generate progressive experiments
        for level, params in [('simple', simple_params), ('complex', complex_params)]:
            for i in range(5):
                experiment_id = f"curriculum_{level}_{i:03d}"
                
                experiment = SyntheticExperiment(
                    experiment_id=experiment_id,
                    original_design=f"curriculum_{level}_base",
                    manipulation_type='curriculum_progression',
                    parameters=params,
                    intended_failure_mode='scalability_limits',
                    timestamp=datetime.now(),
                    result={}
                )
                
                curriculum_experiments.append(experiment)
        
        self.experiments.extend(curriculum_experiments)
        self._save_experiment_metadata(curriculum_experiments)
        
        print(f"Generated {len(curriculum_experiments)} curriculum experiments")
        return curriculum_experiments


def main():
    """Example usage of the synthetic data generator"""
    print("Silicon Intelligence - Synthetic Data Generation System")
    print("=" * 60)
    
    generator = SyntheticDataGenerator()
    
    print("Synthetic data generation system initialized.")
    print(f"Output path: {generator.output_path}")
    print("Available experiment types:")
    print("- Constraint stress tests (cause timing violations)")
    print("- Floorplan break tests (cause congestion)")
    print("- Clock stress tests (cause clock skew)")
    print("- Power stress tests (cause power hotspots)")
    print("- Curriculum progression (simple to complex)")
    
    print("\nThis system generates intentional failures to learn causality.")
    print("One intelligently generated failure is worth ten clean designs.")


if __name__ == "__main__":
    main()