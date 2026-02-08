"""
Chiplet & UCIe Demonstration - The 3D Sovereign.
Demonstrates multi-die integration and high-speed interconnect optimization.
Street Heart Technologies Proprietary.
"""

from core.chiplet_architect import ChipletArchitect
from utils.authority_dashboard import AuthorityDashboard

def run_chiplet_demo():
    print("="*60)
    print("STREET HEART TECHNOLOGIES: CHIPLET & UCIe DEMO")
    print("="*60)
    
    dashboard = AuthorityDashboard()
    architect = ChipletArchitect()
    
    # 1. Define Multi-Die Configuration
    print("\n[CONFIG] Integrating Heterogeneous Dies: 3nm CPU + 2nm Neural Engine")
    architect.add_die("VORTEX_CPU", 12400, 1.45, 3)
    architect.add_die("NEURAL_ENGINE_V2", 8200, 1.12, 2)
    
    # 2. Perform 3D Stacking
    print("\n[3D-IC] Executing High-Density Vertical Stacking...")
    architect.stack_dies("VORTEX_CPU", "NEURAL_ENGINE_V2")
    
    # 3. Optimize High-Speed UCIe Link
    print("\n[UCIe] Routing 512Gbps Interconnect Fabric...")
    link_results = architect.optimize_ucie_link("VORTEX_CPU", "NEURAL_ENGINE_V2", 512)
    print(f"Configuration: {link_results['config']}")
    print(f"Topology: {link_results['type']}")
    print(f"Latency: {link_results['latency_ps']}ps (Flight Time: {link_results['flight_time_ps']}ps)")
    
    # 4. Final Package PPA Analysis
    print("\n[ANALYSIS] Calculating Global Package Sovereignty...")
    ppa = architect.calculate_package_ppa()
    print(f"Package Area: {ppa['package_area_mm2']:.2f} mm2")
    print(f"Total Power: {ppa['total_power_W']:.2f} W")
    print(f"Peak Temperature: {ppa['peak_temperature_C']:.1f} C (Headroom: {ppa['thermal_headroom_C']} C)")

    # 5. Update Authority
    dashboard.log_optimization("CHIPLET_INTEGRATION", "vortex_neural_stack", "SUCCESS")
    dashboard.log_optimization("UCIe_OPT", "link_512gbps", "OPTIMAL")
    
    print("\n" + "="*60)
    print("CHIPLET MISSION COMPLETE: The 3D Silicon Empire is Rising.")

if __name__ == "__main__":
    run_chiplet_demo()
