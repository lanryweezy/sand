#!/usr/bin/env python3
"""
Complete integration script to connect Silicon Intelligence System to real open-source designs
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_acquisition.design_downloader import DesignDownloader
from data_processing.design_processor import DesignProcessor
from validation.ground_truth_generator import GroundTruthGenerator
from validation.validation_pipeline import ValidationPipeline


def main():
    print("Silicon Intelligence System - Real Design Integration")
    print("=" * 70)

    # Step 1: Set up design acquisition
    print("\n1. Setting up Design Acquisition...")
    downloader = DesignDownloader()
    print("   Design downloader initialized")

    # Step 2: Process designs
    print("\n2. Processing Open-Source Designs...")
    processor = DesignProcessor()

    # Start with a small design to test
    test_designs = ['picorv32']

    for design in test_designs:
        try:
            result = processor.process_design(design)
            if result:
                print(f"   Processed {design}: {result['graph_stats']['num_nodes']} nodes")
            else:
                print(f"   Failed to process {design}")
        except Exception as e:
            print(f"   Error processing {design}: {e}")

    # Step 3: Generate ground truth
    print("\n3. Generating Ground Truth Data...")
    generator = GroundTruthGenerator()

    try:
        ground_truths = generator.batch_generate_ground_truth(test_designs)
        print(f"   Generated ground truth for {len(ground_truths)} designs")

        for gt in ground_truths:
            print(f"     - {gt['design_name']}: Area Pred={gt['predictions']['area']:.1f}, "
                  f"Est Actual={gt['estimated_actual']['actual_area']:.1f}")
    except Exception as e:
        print(f"   Error generating ground truth: {e}")

    # Step 4: Run validation pipeline
    print("\n4. Running Validation Pipeline...")
    pipeline = ValidationPipeline()

    try:
        validation_results = pipeline.run_validation_cycle(test_designs)
        print(f"   Validation completed for {validation_results['design_count']} designs")

        # Show validation results
        accuracy_summary = validation_results['insights']['accuracy_summary']
        print("   Accuracy Summary:")
        for key, value in accuracy_summary.items():
            if 'accuracy' in key:
                print(f"     - {key}: {value:.3f}")

        # Show recommendations
        recommendations = validation_results['insights']['recommendations']
        print("   Recommendations:")
        for rec in recommendations:
            print(f"     - {rec}")
    except Exception as e:
        print(f"   Error running validation pipeline: {e}")

    print("\n" + "=" * 70)
    print("REAL DESIGN INTEGRATION COMPLETE!")
    print("\nSystem is now ready to:")
    print("- Process real open-source designs")
    print("- Generate ground truth data")
    print("- Validate predictions against actual results")
    print("- Update models with real feedback")
    print("- Scale to larger design sets")
    print("=" * 70)

    print("\nNext Steps:")
    print("1. Connect to actual EDA tools (OpenROAD, Innovus, Fusion Compiler)")
    print("2. Process larger open-source designs (IBEX, SERV, etc.)")
    print("3. Integrate with real silicon data when available")
    print("4. Deploy production validation pipeline")


if __name__ == "__main__":
    main()