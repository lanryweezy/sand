"""
Main training script for the RTL sequencing agent.
"""
import os
import sys

# Add project root to path for correct module resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.rl_environment import SiliconRLEnvironment
from networks.rl_trainer import SiliconRLTrainer

def create_test_design_file():
    """Creates a simple Verilog file for training."""
    rtl_content = """
module test_engine (
    input clk,
    input rst_n,
    input [7:0] data_in_a,
    input [7:0] data_in_b,
    output [7:0] data_out
);
    wire [7:0] processed_data_a;
    wire [7:0] processed_data_b;

    assign processed_data_a = data_in_a ^ 8'hFF;
    assign processed_data_b = data_in_b + 8'hA5;
    assign data_out = processed_data_a & processed_data_b;

endmodule
"""
    with open("test_design.v", "w") as f:
        f.write(rtl_content)
    return "test_design.v"

def main():
    """Main training function."""
    num_episodes = 20
    max_steps_per_episode = 5

    # 1. Create a representative Verilog design
    rtl_file = create_test_design_file()
    module_name = "test_engine"
    gnn_model_path = "models/trained/gnn_pre_trained.pth"

    # 2. Initialize the RL Environment
    env = SiliconRLEnvironment(rtl_file, module_name, gnn_model_path)

    # Restrict the action space to the first 5 available transformations for simplicity
    action_space_size = min(5, env.get_action_space_size())
    if action_space_size == 0:
        print("Error: No actions available in the environment. Exiting.")
        return

    # 3. Instantiate the RL Trainer
    # The action targets are now indices, not strings
    trainer = SiliconRLTrainer(action_targets=[str(i) for i in range(action_space_size)])

    # 4. Run the Training Loop
    print("--- Starting RL Training for RTL Sequencing ---")
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps_per_episode):
            action_data = trainer.choose_action(state)
            action_idx = int(action_data['target'])

            # Ensure the chosen action is within the valid range for the environment
            if action_idx >= action_space_size:
                print(f"  Warning: Action index {action_idx} is out of bounds. Skipping step.")
                continue

            next_state, reward, done, info = env.step(action_idx)

            trainer.update_policy(state, action_idx, reward, next_state)

            state = next_state
            total_reward += reward
            if done:
                break

        print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward:.4f}")

    print("\n--- RL Training Finished ---")
    print("Final Q-table:")
    for state_key, actions in trainer.q_table.items():
        print(f"  State: {state_key} -> Actions: {actions}")

    # Clean up the test file
    os.remove(rtl_file)

if __name__ == "__main__":
    main()
