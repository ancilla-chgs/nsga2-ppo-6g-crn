from matplotlib import pyplot as plt
from scipy.stats import ttest_rel

from benchmarks import benchmark_nsga2_configs
from visualisations import (
    plot_reward_comparison_bar,
    plot_allocation_comparison_heatmap,
    plot_allocation_usage_comparison,
    plot_strategy_comparison_bar_chart,
    plot_channel_breakdown, run_random_policy,
    run_greedy_policy,
    generate_trained_dataset_aware_allocations,
    evaluate_strategy,
    export_allocation_to_csv, plot_entropy_curve,
)
from evaluations import (
    evaluate_policy_with_fairness, evaluate_baseline_policy, random_action_fn, 
    greedy_action_fn, compare_algorithms
)
from spectrum_env import SpectrumEnv
from ppo_agent import PPOAgent
from utils import create_crn_envs_static, compute_and_log_hypervolume
import torch
import pandas as pd
import numpy as np
import random
import os
import argparse
from visualisations import plot_fairness_comparison_bar_chart, plot_radar_chart, plot_su_pu_grouped_bars

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Argument parser
parser = argparse.ArgumentParser(description="Evaluate trained model on a selected dataset")
parser.add_argument('--dataset', type=str, default='val.csv', help='Name of the evaluation CSV file in /data/')
parser.add_argument('--episodes', type=int, default=30, help='Number of episodes to run evaluation')
args = parser.parse_args()

# Dataset path
dataset_path = os.path.join('data', args.dataset)


def compute_channel_usage(allocation_matrix, user_type=1):
    # allocation_matrix: shape (timesteps, action_dim)
    channel_usage = np.sum(allocation_matrix == user_type, axis=0)  # average usage per channel
    return channel_usage


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

df = pd.read_csv(dataset_path)
#df.columns = [c.lower() for c in df.columns]
test_env = SpectrumEnv(df, num_channels=5, num_sus=3, max_steps=512)

seeds = [42, 123, 999]
all_results = []
for seed in seeds:
    print(f"\n Running evaluation with seed{seed}")
    set_seed(seed)

    # Recreate environments and agents
    ppo_agent_alone = PPOAgent(test_env)
    ppo_agent_nsga2 = PPOAgent(test_env)

   
    # Evaluate PPO and Hybrid
  
    ppo_alone_results = evaluate_policy_with_fairness(ppo_agent_alone, test_env, name="PPO Alone (Seed)", episodes=30)
    ppo_alone_eval_reward = ppo_alone_results["reward"]
    ppo_alone_fairness = ppo_alone_results["fairness"]

    nsga2_ppo_results = evaluate_policy_with_fairness(ppo_agent_nsga2, test_env, name="NSGA + PPO  (Seed)", episodes=30)
    nsga2_ppo_eval_reward = nsga2_ppo_results["reward"]
    nsga2_ppo_fairness = nsga2_ppo_results["fairness"]


    all_results.append({
        "Seed": seed,
        "PPO Reward": ppo_alone_eval_reward,
        "NSGA2+PPO Reward": nsga2_ppo_eval_reward,
        "PPO Fairness": ppo_alone_fairness,
        "NSGA2+PPO Fairness": nsga2_ppo_fairness
    })

# Load environment for benchmarking
df = pd.read_csv(dataset_path)
test_env = SpectrumEnv(df, num_channels=5, num_sus=3, max_steps=512)

# === Evaluate All Algorithms ===
# === Load trained PPO agents ===
ppo_agent_alone = PPOAgent(test_env)
ppo_agent_nsga2 = PPOAgent(test_env)

# Load models from saved .pth files
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_ppo = torch.load("models/best_ppo_model.pth",map_location=device )
ppo_agent_alone.policy.load_state_dict(checkpoint_ppo["policy_state_dict"])
ppo_agent_alone.policy.to(device)
checkpoint_nsga2 = torch.load("models/best_nsga2_ppo_model.pth", map_location=device)
ppo_agent_nsga2.policy.load_state_dict(checkpoint_nsga2["policy_state_dict"])
ppo_agent_nsga2.policy.to(device)
# nsga2_ppo_eval_reward = evaluate_policy(agent=ppo_agent_nsga2, name="NSGA-II + PPO (Test)", env=test_env)

# TESTING EVALUATIONS WITH FAIRNESS

nsga2ppo_result = evaluate_policy_with_fairness(agent=ppo_agent_nsga2, env=test_env,
                                                name="NSGA-II + PPO (Test)", return_energy=True)
nsga2_ppo_eval_reward = nsga2ppo_result["reward"]
nsga2_ppo_fairness = nsga2ppo_result["fairness"]
nsga2_energy_eff = nsga2ppo_result["energy"]
nsga2_interference = nsga2ppo_result["interference_percent"]
nsga2_sue = nsga2ppo_result["spectrum_utilization"]
nsga2_qos = nsga2ppo_result["qos_violation_rate"]
nsga2_collision = nsga2ppo_result["pu_collision_percent"]

ppo_result = evaluate_policy_with_fairness(agent=ppo_agent_alone, env=test_env,
                                           name="PPO Alone (Test)", return_energy=True)
ppo_alone_eval_reward = ppo_result["reward"]
ppo_alone_fairness = ppo_result["fairness"]
ppo_energy_eff = ppo_result["energy"]
ppo_interference = ppo_result["interference_percent"]
ppo_sue = ppo_result["spectrum_utilization"]
ppo_qos = ppo_result["qos_violation_rate"]
ppo_collision = ppo_result["pu_collision_percent"]

random_result = evaluate_baseline_policy(test_env, random_action_fn, name="Random Policy")
random_reward = random_result["reward"]
random_fairness = random_result["fairness"]
random_energy_eff = random_result["energy"]
random_interference = random_result["interference_percent"]
random_sue = random_result["spectrum_utilization"]
random_qos = random_result["qos_violation_rate"]
random_collision = random_result["pu_collision_percent"]

greedy_result = evaluate_baseline_policy(test_env, greedy_action_fn, name="Greedy Policy")
greedy_reward = greedy_result["reward"]
greedy_fairness = greedy_result["fairness"]
greedy_energy_eff = greedy_result["energy"]
greedy_interference = greedy_result["interference_percent"]
greedy_sue = greedy_result["spectrum_utilization"]
greedy_qos = greedy_result["qos_violation_rate"]
greedy_collision = greedy_result["pu_collision_percent"]

# Optional: NSGA-II proxy reward
pareto_fitness = torch.load("models/nsga2_pareto_front.pt", weights_only=False, map_location=device)
nsga2_eval_reward = np.mean([f[0] for f in pareto_fitness])
print(f"[NSGA-II] Average Proxy Reward (SE): {nsga2_eval_reward:.2f}\n")
# Compute and log hypervolume from final Pareto front
hypervolume_value = compute_and_log_hypervolume(pareto_fitness, results_dir=RESULTS_DIR)

# === Export Results to CSV ===
results_summary = pd.DataFrame({
    "Algorithm": ["Random", "Greedy", "PPO Alone", "NSGA-II + PPO"],
    "Average Reward": [random_reward, greedy_reward, ppo_alone_eval_reward, nsga2_ppo_eval_reward],
    "Fairness (Jain Index)": [random_fairness, greedy_fairness, ppo_alone_fairness, nsga2_ppo_fairness],
    "Energy Efficiency": [random_energy_eff, greedy_energy_eff, ppo_energy_eff, nsga2_energy_eff],
    "Interference (%)": [random_interference, greedy_interference, ppo_interference, nsga2_interference],
    "Spectrum Utilization (%)": [random_sue, greedy_sue, ppo_sue, nsga2_sue],
   "QoS Violation Rate (%)": [random_qos, greedy_qos, ppo_qos, nsga2_qos],
    "PU Collision (%)": [random_collision, greedy_collision, ppo_collision, nsga2_collision]


})
results_summary["Hypervolume"] = ["-", "-", "-", f"{hypervolume_value:.6f}"]

#results_summary.to_csv(os.path.join(RESULTS_DIR, "final_evaluation_results.csv"), index=False)
dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
output_file = f"final_results_{dataset_name}.csv"
results_summary.to_csv(os.path.join(RESULTS_DIR, output_file), index=False)

plot_fairness_comparison_bar_chart(
    labels=["Random", "Greedy", "PPO Alone", "NSGA-II + PPO"],
    fairness_values=[random_fairness, greedy_fairness, ppo_alone_fairness, nsga2_ppo_fairness],
    filename="fairness_comparison_bar_chart.png"
)

# === Plot Reward Comparison Bar Chart ===

labels = ["Greedy", "Random", "NSGA-II", "PPO", "NSGA-II + PPO"]
values = [greedy_reward, random_reward, nsga2_eval_reward, ppo_alone_eval_reward, nsga2_ppo_eval_reward]
plot_reward_comparison_bar(labels, values)

# === spectrum allocation comparisons  ===

print("Generating allocation matrices...")
time_steps = len(test_env.episode_time_slots)

# Step 1: Generate allocations
alloc_random, _ = run_random_policy(test_env, time_steps)
alloc_greedy, _ = run_greedy_policy(test_env, time_steps)
alloc_ppo = generate_trained_dataset_aware_allocations(ppo_agent_alone, test_env, time_steps)
alloc_nsga2_ppo = generate_trained_dataset_aware_allocations(ppo_agent_nsga2, test_env, time_steps)

if np.shape(alloc_random) != np.shape(alloc_ppo):
    print("A shape mismatch between allo_random and alloc_ppo. skipping plots")
    exit()

usage_df = pd.DataFrame({
    "Channel": np.arange(alloc_random.shape[1]),

    # Random
    "SU Random": compute_channel_usage(alloc_random, user_type=1),
    "PU Random": compute_channel_usage(alloc_random, user_type=2),

    # Greedy
    "SU Greedy": compute_channel_usage(alloc_greedy, user_type=1),
    "PU Greedy": compute_channel_usage(alloc_greedy, user_type=2),

    # PPO
    "SU PPO": compute_channel_usage(alloc_ppo, user_type=1),
    "PU PPO": compute_channel_usage(alloc_ppo, user_type=2),

    # NSGA-II + PPO
    "SU NSGA-II + PPO": compute_channel_usage(alloc_nsga2_ppo, user_type=1),
    "PU NSGA-II + PPO": compute_channel_usage(alloc_nsga2_ppo, user_type=2)
})

usage_df.to_csv(os.path.join(RESULTS_DIR, "channel_usage_comparison.csv"), index=False)
print("Channel usage comparison saved to channel_usage_comparison.csv")
plot_su_pu_grouped_bars(usage_df)

plot_allocation_comparison_heatmap(
    alloc_ppo, alloc_nsga2_ppo,
    title="Time vs Allocation Difference: PPO vs NSGA-II + PPO",
    label_a="PPO Alone",
    label_b="NSGA-II + PPO",
    save_path="heatmap_ppo_vs_nsga2ppo.png"
)
plot_allocation_comparison_heatmap(
    alloc_greedy, alloc_nsga2_ppo,
    title="Time vs Allocation Difference: Greedy vs NSGA-II + PPO",
    label_a="Greedy",
    label_b="NSGA-II + PPO",
    save_path="heatmap_greedy_vs_nsga2ppo.png"
)
plot_allocation_comparison_heatmap(
    alloc_random, alloc_nsga2_ppo,
    title="Time vs Allocation Difference: Random vs NSGA-II + PPO",
    label_a="Random",
    label_b="NSGA-II + PPO",
    save_path="heatmap_random_vs_nsga2ppo.png"
)

plot_allocation_usage_comparison(alloc_random, alloc_nsga2_ppo, label_a="Random", label_b="NSGA-II + PPO", filename="usage_random_vs_nsga2ppo.png")
plot_allocation_usage_comparison(alloc_ppo, alloc_nsga2_ppo, label_a="PPO Alone", label_b="NSGA-II + PPO", filename="usage_ppo_vs_nsga2ppo.png")
plot_allocation_usage_comparison(alloc_greedy, alloc_nsga2_ppo, label_a="Greedy", label_b="NSGA-II + PPO", filename="usage_greedy_vs_nsga2ppo.png")

# Step 3: Export allocation matrices
export_allocation_to_csv(alloc_random, "alloc_random.csv")
export_allocation_to_csv(alloc_greedy, "alloc_greedy.csv")
export_allocation_to_csv(alloc_ppo, "alloc_ppo.csv")
export_allocation_to_csv(alloc_nsga2_ppo, "alloc_nsga2ppo.csv")

# Step 4: Evaluate strategies
df = pd.read_csv("data/val_dataset_sinr.csv")
summary_random = evaluate_strategy("Random", alloc_random, df)
summary_greedy = evaluate_strategy("Greedy", alloc_greedy, df)
summary_ppo = evaluate_strategy("PPO", alloc_ppo, df)
summary_nsga2ppo = evaluate_strategy("NSGA-II + PPO", alloc_nsga2_ppo, df)

# Step 5: Combine and export summary
strategy_comparison = pd.concat([
    summary_random,
    summary_greedy,
    summary_ppo,
    summary_nsga2ppo
], ignore_index=True)

strategy_comparison.to_csv(os.path.join(RESULTS_DIR, "strategy_comparison_summary.csv"), index=False)
print("Final evaluation summary exported as strategy_comparison_summary_full.csv")
print(strategy_comparison.head())

# Step 6: Visual summary

plot_radar_chart(
    results_summary,
    title="Strategy Comparison Across Key Metrics",
    save_path="results/strategy_radar_plot.png"
)

plot_strategy_comparison_bar_chart(strategy_comparison, save_path="strategy_comparison_bar_chart.png")
plot_channel_breakdown(strategy_comparison, save_path="strategy_channel_breakdown.png")

# Final summary values (already computed previously)
algorithms = ["Random", "Greedy", "PPO", "NSGA-II + PPO"]
rewards = [random_reward, greedy_reward, ppo_alone_eval_reward, nsga2_ppo_eval_reward]
fairnesses = [random_fairness, greedy_fairness, ppo_alone_fairness, nsga2_ppo_fairness]

# Append to strategy_comparison DataFrame
strategy_comparison["Average Reward"] = strategy_comparison["Algorithm"].map(dict(zip(algorithms, rewards)))
strategy_comparison["Jain Fairness Index"] = strategy_comparison["Algorithm"].map(dict(zip(algorithms, fairnesses)))

# Reorder columns for readability
cols = ['Algorithm', 'Average Reward', 'Jain Fairness Index', 'PU Collision %', 'SU Usage %', 'Channel']
strategy_comparison = strategy_comparison[cols]

# Export clean version
strategy_comparison.to_csv(os.path.join(RESULTS_DIR,"strategy_comparison_summary_full.csv"), index=False)
print("Final evaluation summary exported as strategy_comparison_summary_full.csv")



