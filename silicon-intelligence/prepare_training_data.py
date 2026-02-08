#!/usr/bin/env python3
"""
Prepare training data from open-source designs for the Silicon Intelligence System
"""

import sys
import os
import json
import glob
from pathlib import Path

sys.path.append(os.path.dirname(__file__))

from data.rtl_parser import RTLParser
from core.canonical_silicon_graph import CanonicalSiliconGraph
from mock_openroad import MockOpenROADInterface


def extract_features_from_graph(graph):
    """Extract features from canonical silicon graph for ML training"""
    stats = graph.get_graph_statistics()
    
    features = {
        'num_nodes': stats['num_nodes'],
        'num_edges': stats['num_edges'],
        'total_area': stats.get('total_area', 0),
        'total_power': stats.get('total_power', 0),
        'avg_timing_criticality': stats.get('avg_timing_criticality', 0),
        'avg_congestion': stats.get('avg_congestion', 0),
        'node_type_counts': stats.get('node_types', {}),
        'edge_type_counts': stats.get('edge_types', {}),
        'density': stats.get('density', 0),
        'avg_area': stats.get('avg_area', 0),
        'avg_power': stats.get('avg_power', 0)
    }
    
    return features


def process_design_for_training(design_path, design_name, mock_or):
    """Process a single design for training data"""
    print(f"Processing {design_name} for training...")
    
    try:
        # Read RTL content
        with open(design_path, 'r', errors='ignore') as f:
            rtl_content = f.read()

        # Parse the design
        parser = RTLParser()
        rtl_data = parser.parse_verilog(design_path)
        
        # Build canonical graph
        graph = CanonicalSiliconGraph()
        graph.build_from_rtl(rtl_data)
        
        # Extract features
        features = extract_features_from_graph(graph)
        
        # Run mock PPA generation
        print(f"  Running mock PPA generation for {design_name}...")
        ppa_results = mock_or.run_full_flow(rtl_content, top_module=design_name)
        if not ppa_results or not ppa_results.get('success'):
            print(f"  Warning: Mock PPA generation failed for {design_name}")
            ppa_labels = {}
        else:
            ppa_labels = ppa_results['overall_ppa']
            print(f"  PPA Results: Area={ppa_labels.get('area_um2', 0):.2f}, Power={ppa_labels.get('power_mw', 0):.3f}, Timing={ppa_labels.get('timing_ns', 0):.3f}")

        # Create training sample
        training_sample = {
            'design_name': design_name,
            'design_path': str(design_path),
            'features': features,
            'labels': { # Store PPA as labels for supervised learning
                'ppa': ppa_labels
            },
            'graph_stats': graph.get_graph_statistics(),
            'timestamp': os.path.getmtime(design_path)
        }
        
        print(f"  Processed: {features['num_nodes']} nodes, {features['num_edges']} edges")
        return training_sample

    except Exception as e:
        print(f"  Error processing {design_name}: {str(e)}")
        return None


def find_verilog_files(base_dir):
    """Find all Verilog and SystemVerilog files in the directory"""
    verilog_extensions = ['*.v', '*.sv', '*.vh', '*.svh']
    verilog_files = []
    
    for ext in verilog_extensions:
        pattern = os.path.join(base_dir, '**', ext)
        files = glob.glob(pattern, recursive=True)
        verilog_files.extend(files)
    
    return verilog_files


def main():
    print("Preparing Training Data from Open-Source Designs")
    print("=" * 60)

    # --- Tool Initialization ---
    mock_or = MockOpenROADInterface()
    
    # Base directories with designs
    design_dirs = [
        './open_source_designs/picorv32/picorv32-main/',
        './open_source_designs_extended/ibex/ibex-master/rtl/',
        './open_source_designs_extended/serv/serv-master/',
        './open_source_designs_extended/vexriscv/VexRiscv-master/'
    ]
    
    # Collect all Verilog files
    all_verilog_files = []
    for design_dir in design_dirs:
        if os.path.exists(design_dir):
            verilog_files = find_verilog_files(design_dir)
            all_verilog_files.extend(verilog_files)
            print(f"Found {len(verilog_files)} Verilog files in {design_dir}")
    
    print(f"\nTotal Verilog files found: {len(all_verilog_files)}")
    
    # Process each file for training
    training_samples = []
    processed_count = 0
    
    for i, verilog_file in enumerate(all_verilog_files[:20]):  # Limit to first 20 for demo
        # Extract design name from file path
        design_name = os.path.splitext(os.path.basename(verilog_file))[0]
        design_name = f"{design_name}_{i}"
        
        sample = process_design_for_training(verilog_file, design_name, mock_or)
        if sample:
            training_samples.append(sample)
            processed_count += 1
    
    print(f"\nProcessed {processed_count} designs for training")
    
    # Create training dataset
    training_dataset = {
        'dataset_info': {
            'created_at': os.path.getctime('.'),  # Current time as placeholder
            'total_samples': len(training_samples),
            'source_directories': design_dirs,
            'description': 'Training dataset from open-source silicon designs with MOCK PPA labels'
        },
        'samples': training_samples
    }
    
    # Save training dataset
    output_file = './training_dataset.json'
    with open(output_file, 'w') as f:
        json.dump(training_dataset, f, indent=2)
    
    print(f"\nTraining dataset saved to: {output_file}")
    print(f"Dataset contains {len(training_samples)} samples")
    
    # Show sample statistics
    if training_samples:
        print(f"\nSample Statistics:")
        avg_nodes = sum(s['features']['num_nodes'] for s in training_samples) / len(training_samples)
        avg_edges = sum(s['features']['num_edges'] for s in training_samples) / len(training_samples)
        print(f"  Average nodes per design: {avg_nodes:.1f}")
        print(f"  Average edges per design: {avg_edges:.1f}")
        
        # Show first sample as example
        first_sample = training_samples[0]
        print(f"\nExample sample from {first_sample['design_name']}:")
        print(f"  Nodes: {first_sample['features']['num_nodes']}")
        print(f"  Edges: {first_sample['features']['num_edges']}")
        if 'ppa' in first_sample.get('labels', {}):
            ppa = first_sample['labels']['ppa']
            print(f"  Mock Area: {ppa.get('area_um2', 0):.2f} µm²")
            print(f"  Mock Power: {ppa.get('power_mw', 0):.3f} mW")
            print(f"  Mock Timing: {ppa.get('timing_ns', 0):.3f} ns")
    
    print(f"\n{'='*60}")
    print("TRAINING DATA PREPARATION COMPLETE")
    print(f"{'='*60}")
    print("Next steps:")
    print("1. Use training_dataset.json to train ML models")
    print("2. Implement model training pipeline") 
    print("3. Validate models with held-out test set")
    print("4. Deploy trained models for predictions")


if __name__ == "__main__":
    main()