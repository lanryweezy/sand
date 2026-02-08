
import sys
import os

# Ensure we can import core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.chiplet_architect import ChipletArchitect

def test_3d_stack_stress():
    print("="*80)
    print("   SILICON INTELLIGENCE: 3D-IC THERMAL STRESS TEST")
    print("="*80)
    
    architect = ChipletArchitect()
    
    # Base Die: High power computation
    architect.add_die("BASE_COMPUTE", 10000, 5000.0, 5) # 5W Base
    
    # Layering 5 dies on top to stress the thermal resistance
    for i in range(1, 6):
        die_id = f"LAYER_{i}_MEMORY"
        architect.add_die(die_id, 2000, 1000.0, 7) # 1W each
        architect.stack_dies(f"LAYER_{i-1}_MEMORY" if i > 1 else "BASE_COMPUTE", die_id)
        
    print("\n--- Running Package PPA Analysis ---")
    results = architect.calculate_package_ppa()
    
    print(f"\nFinal Results for 6-Layer Stack:")
    print(f"Total Power: {results['total_power_W']:.2f} W")
    print(f"Peak Temperature: {results['peak_temperature_C']:.1f} C")
    print(f"Thermal Headroom: {results['thermal_headroom_C']:.1f} C")
    
    if results['peak_temperature_C'] > 105.0:
        print("\n[ALERT] THERMAL FAILURE DETECTED! Package exceeds 105C limit.")
        print("[AI RECOMMENDATION] Re-distribute power or increase die area for better heat spreading.")
    else:
        print("\n[OK] Package within safe thermal operating range.")

    print("\n" + "="*80)

if __name__ == "__main__":
    test_3d_stack_stress()
