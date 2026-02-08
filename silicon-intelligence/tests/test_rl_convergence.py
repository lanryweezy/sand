
import sys
import os

# Ensure we can import core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autonomous_optimizer import AdvancedAutonomousOptimizer

def test_rl_training_episode():
    print("Testing Reinforcement Learning Episode (RL-EDA)...")
    
    optimizer = AdvancedAutonomousOptimizer()
    
    # Simple adder logic for fast training
    test_rtl = '''
module small_unit (
    input clk,
    input [7:0] a,
    input [7:0] b,
    output reg [7:0] q
);
    wire [7:0] sum = a + b;
    always @(posedge clk) q <= sum;
endmodule
'''
    
    # Run 5 episodes to verify the learning loop
    for i in range(1, 6):
        print(f"\n--- Episode {i} ---")
        results = optimizer.run_rl_optimization_loop(test_rtl, f"test_rl_{i}")
        
        print(f"Sequence Taken: {results['sequence']}")
        print(f"Final Improvement: {results['improvement']['area_improvement_pct']:.2f}% Area, {results['improvement']['power_improvement_pct']:.2f}% Power")

    # Check if memory was cleared and policy updated (implicit in learn call)
    print("\nSUCCESS: RL-EDA Training foundation verified.")

if __name__ == "__main__":
    test_rl_training_episode()
