"""
Factory Demonstration - The Brain Leap (Synthetic Data Generation)
Demonstrates the stochastic generation of a silicon design corpus.
Branded by Street Heart Technologies.
"""

from core.synthetic_factory import SyntheticFactory
from utils.authority_dashboard import AuthorityDashboard
import os

def run_factory_demonstration():
    print("Initializing Street Heart Synthetic Design Factory...")
    dashboard = AuthorityDashboard()
    factory = SyntheticFactory()
    
    # 1. Generate Batch
    count = 10
    print(f"Starting batch generation of {count} unique designs...")
    paths = factory.generate_batch(count)
    
    for i, path in enumerate(paths):
        # Log to dashboard
        module_name = os.path.basename(path).replace(".v", "")
        dashboard.log_optimization("FACTORY_GEN", module_name, "READY")
        if (i+1) % 2 == 0:
            print(f" Generated {i+1}/{count} designs...")

    # 2. Display Resulting Telemetry
    print("\n" + "="*50)
    dashboard.display_full_report()
    print("="*50)
    
    print(f"\nBatch Generation Complete. {count} designs located in 'synthetic_data/'.")
    print("These designs represent the high-entropy training data for our professional GNN models.")

if __name__ == "__main__":
    run_factory_demonstration()
