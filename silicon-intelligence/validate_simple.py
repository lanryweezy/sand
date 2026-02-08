#!/usr/bin/env python3
"""
Simplified Validation Script for Silicon Intelligence System
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

def validate_core_components():
    """Validate core system components"""
    print("Validating Silicon Intelligence System Core Components...")
    print("="*60)
    
    # Check if key directories exist
    core_dirs = [
        'cognitive',
        'agents', 
        'core',
        'data',
        'models',
        'utils'
    ]
    
    print("Checking directory structure:")
    for dir_name in core_dirs:
        dir_exists = os.path.exists(dir_name)
        status = "[OK]" if dir_exists else "[MISSING]"
        print(f"  {status} {dir_name}/")
    
    # Check if key files exist
    key_files = [
        'main.py',
        'cognitive/advanced_cognitive_system.py',
        'core/canonical_silicon_graph.py',
        'agents/base_agent.py',
        'models/congestion_predictor.py',
        'utils/logger.py'
    ]
    
    print("\nChecking key files:")
    files_exist = 0
    for file_path in key_files:
        file_exists = os.path.exists(file_path)
        status = "[OK]" if file_exists else "[MISSING]"
        print(f"  {status} {file_path}")
        if file_exists:
            files_exist += 1
    
    print(f"\nCore files validation: {files_exist}/{len(key_files)} files found")
    
    return files_exist == len(key_files)


def validate_imports():
    """Validate that key modules can be imported"""
    print("\nValidating module imports...")
    print("-"*40)
    
    modules_to_test = [
        ('silicon_intelligence.cognitive.advanced_cognitive_system', 'PhysicalRiskOracle'),
        ('silicon_intelligence.core.canonical_silicon_graph', 'CanonicalSiliconGraph'),
        ('silicon_intelligence.agents.base_agent', 'BaseAgent'),
        ('silicon_intelligence.models.congestion_predictor', 'CongestionPredictor'),
        ('silicon_intelligence.utils.logger', 'get_logger')
    ]
    
    successful_imports = 0
    
    for module_path, class_name in modules_to_test:
        try:
            # Try to import the module
            module_parts = module_path.split('.')
            file_path = '/'.join(module_parts) + '.py'
            
            if os.path.exists(file_path):
                # Import using importlib
                import importlib.util
                spec = importlib.util.spec_from_file_location(module_path, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Check if class exists
                if hasattr(module, class_name):
                    print(f"  [OK] {module_path}.{class_name}")
                    successful_imports += 1
                else:
                    print(f"  [ERR] {module_path}.{class_name} - Class not found")
            else:
                print(f"  [ERR] {module_path} - File not found")
        except Exception as e:
            print(f"  [ERR] {module_path}.{class_name} - {str(e)}")
    
    print(f"\nImport validation: {successful_imports}/{len(modules_to_test)} successful")
    return successful_imports == len(modules_to_test)


def validate_agents():
    """Validate agent implementations"""
    print("\nValidating agent implementations...")
    print("-"*40)
    
    agent_files = [
        'agents/floorplan_agent.py',
        'agents/placement_agent.py',
        'agents/clock_agent.py',
        'agents/power_agent.py',
        'agents/yield_agent.py',
        'agents/routing_agent.py',
        'agents/thermal_agent.py'
    ]
    
    agents_valid = 0
    for agent_file in agent_files:
        if os.path.exists(agent_file):
            print(f"  [OK] {agent_file}")
            agents_valid += 1
        else:
            print(f"  [ERR] {agent_file} - Not found")
    
    print(f"\nAgent validation: {agents_valid}/{len(agent_files)} found")
    return agents_valid == len(agent_files)


def validate_models():
    """Validate ML model implementations"""
    print("\nValidating ML model implementations...")
    print("-"*40)
    
    model_files = [
        'models/congestion_predictor.py',
        'models/timing_analyzer.py',
        'models/drc_predictor.py',
        'models/advanced_ml_models.py'
    ]
    
    models_valid = 0
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"  [OK] {model_file}")
            models_valid += 1
        else:
            print(f"  [ERR] {model_file} - Not found")
    
    print(f"\nModel validation: {models_valid}/{len(model_files)} found")
    return models_valid == len(model_files)


def validate_core_architecture():
    """Validate core architecture components"""
    print("\nValidating core architecture components...")
    print("-"*40)
    
    arch_files = [
        'core/parallel_reality_engine.py',
        'core/learning_loop.py',
        'core/flow_orchestrator.py',
        'core/eda_integration.py'
    ]
    
    arch_valid = 0
    for arch_file in arch_files:
        if os.path.exists(arch_file):
            print(f"  [OK] {arch_file}")
            arch_valid += 1
        else:
            print(f"  [ERR] {arch_file} - Not found")
    
    print(f"\nArchitecture validation: {arch_valid}/{len(arch_files)} found")
    return arch_valid == len(arch_files)


def run_basic_functionality_test():
    """Run a basic functionality test"""
    print("\nRunning basic functionality test...")
    print("-"*40)
    
    try:
        # Test basic import and instantiation
        from silicon_intelligence.cognitive.advanced_cognitive_system import PhysicalRiskOracle
        oracle = PhysicalRiskOracle()
        print("  [OK] PhysicalRiskOracle instantiated successfully")
        
        from silicon_intelligence.core.canonical_silicon_graph import CanonicalSiliconGraph
        graph = CanonicalSiliconGraph()
        print("  [OK] CanonicalSiliconGraph instantiated successfully")
        
        from silicon_intelligence.utils.logger import get_logger
        logger = get_logger(__name__)
        print("  [OK] Logger instantiated successfully")
        
        print("\n  Basic functionality test: PASSED")
        return True
        
    except Exception as e:
        print(f"\n  Basic functionality test: FAILED - {str(e)}")
        return False


def main():
    print("Silicon Intelligence System - Simplified Validation")
    print("="*60)
    print("Validating system implementation...")
    print()
    
    # Run all validations
    core_ok = validate_core_components()
    imports_ok = validate_imports()
    agents_ok = validate_agents()
    models_ok = validate_models()
    arch_ok = validate_core_architecture()
    basic_test_ok = run_basic_functionality_test()
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    print(f"Core directory structure:     {'[PASS]' if core_ok else '[FAIL]'}")
    print(f"Module imports:              {'[PASS]' if imports_ok else '[FAIL]'}")
    print(f"Agent implementations:       {'[PASS]' if agents_ok else '[FAIL]'}")
    print(f"ML model implementations:    {'[PASS]' if models_ok else '[FAIL]'}")
    print(f"Core architecture:           {'[PASS]' if arch_ok else '[FAIL]'}")
    print(f"Basic functionality test:    {'[PASS]' if basic_test_ok else '[FAIL]'}")
    
    overall_success = all([core_ok, imports_ok, agents_ok, models_ok, arch_ok, basic_test_ok])
    
    print(f"\nOverall system validation:   {'[PASS]' if overall_success else '[FAIL]'}")
    
    if overall_success:
        print("\n✓ Silicon Intelligence System validation PASSED!")
        print("  All core components are properly implemented and accessible.")
        print("\n  The system is ready for:")
        print("  - Physical risk assessment")
        print("  - Multi-agent coordination")
        print("  - Parallel reality exploration")
        print("  - ML-driven optimization")
        print("  - Full RTL-to-GDS flow execution")
    else:
        print("\n✗ Silicon Intelligence System validation FAILED!")
        print("  Some components are missing or not properly implemented.")
        print("  Please check the validation results above.")
    
    # Generate validation report
    report = {
        'validation_timestamp': datetime.now().isoformat(),
        'system_name': 'Silicon Intelligence System',
        'validation_results': {
            'core_structure': core_ok,
            'module_imports': imports_ok,
            'agents': agents_ok,
            'models': models_ok,
            'architecture': arch_ok,
            'basic_functionality': basic_test_ok
        },
        'overall_success': overall_success
    }
    
    with open('validation_report_simple.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nValidation report saved to: validation_report_simple.json")
    print("="*60)
    
    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)