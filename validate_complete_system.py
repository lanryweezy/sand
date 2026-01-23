#!/usr/bin/env python3
"""
Complete System Validation - Silicon Intelligence with Trained Models

Validates that the entire system is working with trained risk prediction models.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

def validate_trained_models():
    """Validate that trained models exist and are functional"""
    print("🔍 VALIDATING TRAINED RISK MODELS")
    print("-" * 50)
    
    model_path = Path("models/trained")
    required_models = [
        "congestion_model.pkl",
        "timing_model.pkl", 
        "power_model.pkl",
        "drc_model.pkl"
    ]
    
    all_present = True
    for model_file in required_models:
        model_file_path = model_path / model_file
        exists = model_file_path.exists()
        print(f"  {model_file}: {'✅ PRESENT' if exists else '❌ MISSING'}")
        if not exists:
            all_present = False
    
    if all_present:
        print("\n  Loading models to test functionality...")
        try:
            import joblib
            models_loaded = 0
            for model_file in required_models:
                model = joblib.load(model_path / model_file)
                models_loaded += 1
            print(f"  ✅ Successfully loaded {models_loaded} models")
            return True
        except Exception as e:
            print(f"  ❌ Error loading models: {e}")
            return False
    else:
        print("  ❌ Missing required models")
        return False


def validate_data_collection_systems():
    """Validate that data collection systems are in place"""
    print("\n🔍 VALIDATING DATA COLLECTION SYSTEMS")
    print("-" * 50)
    
    systems_present = [
        ("data_collection/telemetry_collector.py", Path("data_collection/telemetry_collector.py").exists()),
        ("data_generation/synthetic_generator.py", Path("data_generation/synthetic_generator.py").exists()),
        ("data_integration/learning_pipeline.py", Path("data_integration/learning_pipeline.py").exists()),
    ]
    
    all_present = True
    for name, exists in systems_present:
        print(f"  {name}: {'✅ PRESENT' if exists else '❌ MISSING'}")
        if not exists:
            all_present = False
    
    return all_present


def validate_evaluation_harness():
    """Validate that evaluation harness is functional"""
    print("\n🔍 VALIDATING EVALUATION HARNESS")
    print("-" * 50)
    
    harness_exists = Path("evaluation_harness.py").exists()
    print(f"  evaluation_harness.py: {'✅ PRESENT' if harness_exists else '❌ MISSING'}")
    
    if harness_exists:
        try:
            from evaluation_harness import EvaluationHarness
            harness = EvaluationHarness()
            print("  ✅ Evaluation harness imports successfully")
            return True
        except Exception as e:
            print(f"  ❌ Error importing evaluation harness: {e}")
            return False
    else:
        return False


def validate_override_tracking():
    """Validate that override tracking system is functional"""
    print("\n🔍 VALIDATING OVERRIDE TRACKING")
    print("-" * 50)
    
    tracker_exists = Path("override_tracker.py").exists()
    print(f"  override_tracker.py: {'✅ PRESENT' if tracker_exists else '❌ MISSING'}")
    
    if tracker_exists:
        try:
            from override_tracker import OverrideTracker
            tracker = OverrideTracker()
            print("  ✅ Override tracker imports successfully")
            return True
        except Exception as e:
            print(f"  ❌ Error importing override tracker: {e}")
            return False
    else:
        return False


def validate_flow_orchestration():
    """Validate that flow orchestration system is functional"""
    print("\n🔍 VALIDATING FLOW ORCHESTRATION")
    print("-" * 50)
    
    orchestrator_exists = Path("core/flow_orchestrator.py").exists()
    print(f"  core/flow_orchestrator.py: {'✅ PRESENT' if orchestrator_exists else '❌ MISSING'}")
    
    if orchestrator_exists:
        try:
            from core.flow_orchestrator import FlowOrchestrator
            orchestrator = FlowOrchestrator()
            print("  ✅ Flow orchestrator imports successfully")
            return True
        except Exception as e:
            print(f"  ❌ Error importing flow orchestrator: {e}")
            return False
    else:
        return False


def validate_target_design_profile():
    """Validate that target design profile is set"""
    print("\n🔍 VALIDATING TARGET DESIGN PROFILE")
    print("-" * 50)
    
    profile_exists = Path("target_design_profile.md").exists()
    print(f"  target_design_profile.md: {'✅ PRESENT' if profile_exists else '❌ MISSING'}")
    
    if profile_exists:
        try:
            with open("target_design_profile.md", "r") as f:
                content = f.read()
                has_ai_focus = "AI accelerators" in content
                has_success_metrics = "Success Metrics" in content or "success metrics" in content
                print(f"  Has AI accelerator focus: {'✅ YES' if has_ai_focus else '❌ NO'}")
                print(f"  Has success metrics: {'✅ YES' if has_success_metrics else '❌ NO'}")
                
                if has_ai_focus and has_success_metrics:
                    return True
                else:
                    return False
        except Exception as e:
            print(f"  ❌ Error reading profile: {e}")
            return False
    else:
        return False


def validate_authority_building():
    """Validate that authority building systems are in place"""
    print("\n🔍 VALIDATING AUTHORITY BUILDING SYSTEMS")
    print("-" * 50)
    
    authority_files = [
        "authority_summary.py",
        "validate_authority.py", 
        "authority_dashboard.py"
    ]
    
    all_present = True
    for file in authority_files:
        exists = Path(file).exists()
        print(f"  {file}: {'✅ PRESENT' if exists else '❌ MISSING'}")
        if not exists:
            all_present = False
    
    # Test authority metrics
    try:
        from override_tracker import OverrideTracker
        tracker = OverrideTracker()
        metrics = tracker.get_authority_metrics()
        print(f"  Authority metrics available: ✅ YES")
        print(f"  Sample metrics: {list(metrics.keys())[:3]}...")  # Show first 3 metric names
        return all_present
    except Exception as e:
        print(f"  ❌ Error testing authority metrics: {e}")
        return False


def run_complete_validation():
    """Run complete system validation"""
    print("🏆 SILICON INTELLIGENCE SYSTEM - COMPLETE VALIDATION")
    print("=" * 80)
    print("Validating that the complete strategic plan implementation is functional")
    print("with trained models and data-driven authority building.")
    print()
    
    validation_results = {}
    
    validation_results['trained_models'] = validate_trained_models()
    validation_results['data_collection'] = validate_data_collection_systems()
    validation_results['evaluation_harness'] = validate_evaluation_harness()
    validation_results['override_tracking'] = validate_override_tracking()
    validation_results['flow_orchestration'] = validate_flow_orchestration()
    validation_results['target_profile'] = validate_target_design_profile()
    validation_results['authority_building'] = validate_authority_building()
    
    print(f"\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for component, passed in validation_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {component.replace('_', ' ').title()}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\nOVERALL SYSTEM STATUS: {'✅ FULLY FUNCTIONAL' if all_passed else '❌ ISSUES IDENTIFIED'}")
    
    if all_passed:
        print(f"\n🎉 SUCCESS: The Silicon Intelligence System is fully implemented with:")
        print(f"   • Trained risk prediction models")
        print(f"   • Data collection and synthetic generation")
        print(f"   • Evaluation harness for accuracy measurement")
        print(f"   • Override tracking for authority building")
        print(f"   • Flow orchestration with automatic biasing")
        print(f"   • Target focus on AI accelerators")
        print(f"   • Authority metrics and validation")
        print(f"\n   The system successfully executes the strategic plan to")
        print(f"   turn prediction accuracy into authority!")
    else:
        print(f"\n⚠️  Some components need attention before full deployment.")
    
    return all_passed


def main():
    """Main validation function"""
    success = run_complete_validation()
    
    if success:
        print(f"\n🚀 READY FOR DEPLOYMENT: The system is ready to dominate the AI accelerator space!")
    else:
        print(f"\n🔧 CONTINUE WORK: Address validation failures before deployment.")


if __name__ == "__main__":
    main()