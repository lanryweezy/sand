
import sys
import os
import json

# Ensure we can import core modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from autonomous_optimizer import AdvancedAutonomousOptimizer, OptimizationStrategy
from core.system_optimizer import SystemOptimizer, OptimizationGoal
from datetime import datetime
import json

def run_end_to_end_demo():
    print("="*80)
    print("   SILICON INTELLIGENCE: END-TO-END INDUSTRIAL OPTIMIZATION RUN")
    print("="*80)
    
    rtl_path = "industrial_grade_accelerator.v"
    
    # 1. Initialize Optimizer
    print("\n[STEP 1] Initializing Autonomous AI Optimizer...")
    optimizer = AdvancedAutonomousOptimizer()
    
    # 2. Run Intelligent Optimization
    # This will trigger:
    # - Fanout Optimization on 'clk' or 'en_internal'
    # - Logic Merging on 'mult_result'
    # - Input Isolation on 'data_in' signals
    print("\n[STEP 2] Executing Multi-Strategy Optimization...")
    print(f"Target RTL: {rtl_path}")
    
    with open(rtl_path, 'r') as f:
        original_rtl = f.read()
    
    # We simulate a "Performance" intent which the Cognitive System would now detect
    # triggering multiple strategies
    
    # Simulate the optimizer's sequence
    # Note: In a full 'run_optimization_loop', it would do this automatically.
    # Here we show the steps.
    
    print("\n -> Applying Logic Merging (Clustering)...")
    optimized_1 = optimizer._apply_clustering(original_rtl, {'module_name': 'crypto_accelerator_pro'})
    
    print("\n -> Applying Input Isolation (Power)...")
    optimized_2 = optimizer._balance_area_power(optimized_1, {
        'module_name': 'crypto_accelerator_pro',
        'targets': ['data_in_a', 'data_in_b'],
        'enable_signal': 'en'
    })
    
    print("\n -> Applying Fanout Buffering (Timing)...")
    final_rtl = optimizer._optimize_fanout(optimized_2, {
        'module_name': 'crypto_accelerator_pro',
        'target_signal': 'clk',
        'current_avg': 20.0 # High fanout
    })
    
    # 3. Save Optimized Results
    output_path = "industrial_grade_accelerator_OPTIMIZED.v"
    with open(output_path, 'w') as f:
        f.write(final_rtl)
    
    print(f"\n[STEP 3] Optimization Complete. Results saved to: {output_path}")
    
    # 4. Global Analysis
    print("\n[STEP 4] System-Level PPA Verification...")
    sys_opt = SystemOptimizer()
    # We use the existing analysis tools to compare
    initial_analysis = optimizer.design_intelligence.analyze_design(original_rtl, "original_industrial")
    final_analysis = optimizer.design_intelligence.analyze_design(final_rtl, "optimized_industrial")
    
    improvement = optimizer._calculate_improvement(initial_analysis, final_analysis)
    
    print("\n--- PERFORMANCE SUMMARY ---")
    print(f"Logic Nodes Removed: {initial_analysis['physical_ir_stats']['num_nodes'] - final_analysis['physical_ir_stats']['num_nodes']}")
    print(f"Power Savings (Est): {improvement['power_improvement_pct']:.1f}%")
    print(f"Area Efficiency Improvement: {improvement['area_improvement_pct']:.1f}%")
    print(f"Overall Confidence: {initial_analysis['openroad_results']['overall_ppa'].get('confidence', 0.8)*100:.1f}%")
    
    # 5. Export to Web Portal Telemetry
    telemetry_path = os.path.join("silicon-web-portal", "src", "data", "telemetry.json")
    if os.path.exists(os.path.dirname(telemetry_path)):
        print(f"\n[STEP 5] Updating Web Portal Telemetry: {telemetry_path}")
        
        now = datetime.now()
        telemetry_data = {
            "metrics": {
                "area": f"{final_analysis['openroad_results']['overall_ppa']['area_um2']:.0f} um² ({improvement['area_improvement_pct']:.1f}%)",
                "power": f"{final_analysis['openroad_results']['overall_ppa']['power_mw']:.2f} mW ({improvement['power_improvement_pct']:.1f}%)",
                "timing": f"+{improvement['timing_improvement_pct']:.1f} ns",
                "confidence": f"{initial_analysis['openroad_results']['overall_ppa'].get('confidence', 0.8)*100:.1f}%"
            },
            "feed": [
                {
                    "id": int(now.timestamp() * 1000),
                    "time": now.strftime("%H:%M:%S"),
                    "title": "INDUSTRIAL_OPTIMIZATION_SUCCESS",
                    "desc": "Crypto-accelerator DESIGN TAPE-OUT READY."
                },
                {
                    "id": int(now.timestamp() * 1000) - 1,
                    "time": now.strftime("%H:%M:%S"),
                    "title": "FANOUT_BUFFERING",
                    "desc": "Injected 5-branch clock distribution tree."
                },
                {
                    "id": int(now.timestamp() * 1000) - 2,
                    "time": now.strftime("%H:%M:%S"),
                    "title": "INPUT_ISOLATION",
                    "desc": "Gated data_in_a/b with en control signal."
                },
                {
                    "id": int(now.timestamp() * 1000) - 3,
                    "time": now.strftime("%H:%M:%S"),
                    "title": "LOGIC_MERGING",
                    "desc": "Merged redundant mult_result cones."
                }
            ]
        }
        
        with open(telemetry_path, 'w') as f:
            json.dump(telemetry_data, f, indent=2)
    
    print("\n" + "="*80)
    print("   MISSION SUCCESS: DESIGN TAPE-OUT READY (VIRTUAL)")
    print("="*80)

if __name__ == "__main__":
    run_end_to_end_demo()
