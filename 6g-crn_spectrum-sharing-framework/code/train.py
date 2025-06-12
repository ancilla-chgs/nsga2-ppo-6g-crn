import numpy as np
import pandas as pd
import torch
import time
import os
from scipy.spatial.distance import euclidean
from matplotlib import pyplot as plt
from nsga2 import optimize_spectrum_allocation, extract_strict_pareto_front
from ppo_agent import PPOAgent
from utils import create_crn_envs_static,  compute_and_log_hypervolume
from sklearn.preprocessing import MinMaxScaler
from visualisations import plot_pareto_front, plot_3d_pareto_front, plot_nsga2_benchmark_results, \
    plot_dual_learning_curves, plot_hypervolume_curve, plot_scalar_reward_convergence, \
    plot_reward_moving_avg, plot_combined_reward_moving_avg, plot_entropy_curve

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Step 1: Load environment
train_env, val_env, test_env = create_crn_envs_static(max_steps=512)

# Step 2: Load dataset and preprocess for NSGA-II
#dataset = pd.read_csv("train.csv")
dataset = pd.read_csv("train_dataset_sinr.csv")
#dataset.columns = [c.lower() for c in dataset.columns]

population_size = 200
num_generations = 100
sampled_data = dataset.sample(n=population_size, random_state=42).reset_index(drop=True)

# extract parameters form sampled data
temp_bandwidth = 10  #MHz
B = np.full((population_size,1), fill_value=temp_bandwidth)
SINR = sampled_data['SINR'].values.reshape(-1, 1)
I = sampled_data['interference_level'].values.reshape(-1, 1)
P = sampled_data['transmit_power'].values.reshape(-1, 1)
R = sampled_data['throughput'].values.reshape(-1, 1)


num_channels = 5

# Step 3: Run NSGA-II Optimization
print("Running NSGA-II to generate Pareto-optimal spectrum allocations...")
start_nsga2 = time.time()
pareto_fitness,convergence_data, nsga2_scalar_reward, hv_per_gen, final_population = optimize_spectrum_allocation(
    train_env, B, SINR, I, P, R, population_size, num_generations
)
end_nsga2 = time.time()
nsga2_runtime = end_nsga2-start_nsga2
print(f"NSGA-II Runtime: {nsga2_runtime:.2f} seconds")
torch.save(pareto_fitness,"nsga2_pareto_front.pt")
print("[NSGA-II] Average Weighted Reward:", nsga2_scalar_reward)

# === Filter high-SE Pareto solutions for PPO pretraining ===
pareto_fitness = np.array(pareto_fitness)

# Spectrum Efficiency is the first column (already normalized)
se_threshold = 0.7  # adjust if too strict or too loose
filtered_indices = np.where(pareto_fitness[:, 0] > se_threshold)[0]

# Filtered solutions and fitness values
filtered_solutions = final_population[filtered_indices]
filtered_fitness = pareto_fitness[filtered_indices]


# Prepare training data for PPO from NSGA-II filtered results
train_env.reset()

B, SINR, I, P, R = train_env.get_averaged_nsga2_inputs()  # Use averaged values
B = B.reshape(-1, train_env.num_channels)
SINR = SINR.reshape(-1, train_env.num_channels)
I = I.reshape(-1, train_env.num_channels)
P = P.reshape(-1, train_env.num_channels)
R = R.reshape(-1, train_env.num_channels)

# Generate observation-action pairs
ppo_agent = PPOAgent(env=train_env)
ppo_agent_nsga2 = PPOAgent(env = train_env)
train_env_1, _, _ = create_crn_envs_static(max_steps=512)
train_env_2, _, _ = create_crn_envs_static(max_steps=512)
pareto_data = ppo_agent.generate_pareto_training_data(B, SINR, I, P, R, filtered_solutions, filtered_indices)
ppo_agent_nsga2.pretrain_from_pareto(pareto_data, epochs=10)

print(f"[NSGA-II] Selected {len(filtered_solutions)} high-SE solutions for PPO pretraining.")


hv_df = pd.DataFrame({
    "Generation": list(range(len(hv_per_gen))),
    "Hypervolume": hv_per_gen
})
hv_df.to_csv("results/hypervolume_curve.csv", index=False)
plot_hypervolume_curve(csv_path="results/hypervolume_curve.csv", save_path="results/hypervolume_convergence.png")



# === Per-objective analysis ===
se_values = [f[0] for f in pareto_fitness]
print("Sample fitness entry:", pareto_fitness[0])
il_values = [f[1] for f in pareto_fitness]
ec_values = [f[2] for f in pareto_fitness]

df_objectives = pd.DataFrame({
    "Spectrum Efficiency": se_values,
    "Interference": il_values,
    "Energy Consumption": ec_values
})
# Save full Pareto fitness values
df_objectives.to_csv(os.path.join(RESULTS_DIR, "nsga2_pareto_objectives.csv"), index=False)
# Print summary
print("\n NSGA-II Per-Objective Summary:")
print(df_objectives.describe().round(4))

# === Compute and save Hypervolume ===
# hypervolume_value = compute_hypervolume(pareto_fitness, reference_point=[1.1, 1.1, 1.1])

# print(f"[NSGA-II] Hypervolume = {hypervolume_value:.6f}")

# Step 4: Extract strict Pareto front and normalize
strict_pareto_front = extract_strict_pareto_front(pareto_fitness)
print(f"Strict Pareto Front has {len(strict_pareto_front)} solutions (non-dominated)")
pareto_results = np.array(pareto_fitness)
scaler = MinMaxScaler()
normalized_pareto = scaler.fit_transform(pareto_results)

# Step 5: Save top 5 solutions by SE
top5_idx = np.argsort(pareto_results[:, 0])[::-1][:5]
top5_solutions = pareto_results[top5_idx]

top5_df = pd.DataFrame(top5_solutions, columns=[
    'Spectrum Efficiency', 'Interference', 'Energy Consumption'
])
top5_df.to_csv(os.path.join(RESULTS_DIR, "top5_pareto_solutions.csv"), index=False)
# print("Top 5 Pareto solutions saved to 'top5_pareto_solutions.csv'")



# Step 6: Save optimal single-objective solutions
# ==== SELECT AND SAVE OPTIMAL SOLUTIONS ======
optimal_SE_idx = np.argmax(normalized_pareto[:, 0])
minimal_Interference_idx = np.argmin(normalized_pareto[:, 1])
minimal_Energy_idx = np.argmin(normalized_pareto[:, 2])
optimal_solutions = pd.DataFrame({
    "Optimal_SE": pareto_results[optimal_SE_idx],
    "Minimal_Interference": pareto_results[minimal_Interference_idx],
    "Minimal_Energy": pareto_results[minimal_Energy_idx]
}, index=["Spectrum Efficiency", "Interference", "Energy Consumption"])
optimal_solutions.to_csv(os.path.join(RESULTS_DIR, "final_nsga2_optimal_solutions.csv"))
print("Optimal single-objective solutions saved to 'final_nsga2_optimal_solutions.csv'")
print(optimal_solutions)
# ===  NSGA-II Pareto Front (3D Scatter) with log scalling for clear visualisations ===
print("PLOTING PARETO FRONTS AND NSGA2 BENCHMARKS ")
plot_pareto_front(pareto_results)
plot_scalar_reward_convergence(convergence_data)
# lot_3d_pareto_front(top5_solutions)

# plot_nsga2_benchmark_results(os.path.join(RESULTS_DIR,"nsga2_benchmark_results.csv"))
# plot_3d_pareto_front(strict_pareto_front)  # strick pareto with non-dominated solutions only

    # === 3. NSGA-II + PPO  TRAINING ===============================
print("\\n=== Pretraining PPO using NSGA-II Pareto data and then training (NSGA-II + PPO) ===")

# filtered_indices, filtered_fitness, filtered_population = filter_pareto_front(pareto_fitness, final_population)


# ppo_agent_nsga2 = PPOAgent(train_env)


# === PPO PreTraining ===


# Pretrain PPO agent using Pareto Data

# === fine-tune PPO after pretraining ===
print("\n=== Fine-tuning NSGA-II + PPO agent ===")
best_reward_nsga2_ppo = -float('inf')
best_model_path_nsga2_ppo = "best_nsga2_ppo_model.pth"


all_rewards = []
start_hybrid = time.time()
ppo_nsga2_rewards, ppo_nsga2_metrics = ppo_agent_nsga2.train(timesteps=200_000, num_steps=512, entropy_log_path="results/nsga2_ppo_entropy_log.csv" )
end_hybrid = time.time()
hybrid_runtime = end_hybrid-start_hybrid
print(f"\n NSGA-II + PPO Runtime: {hybrid_runtime:.2f} seconds")
print(f"NSGA-II + PPO Avg Reward (mean): {np.mean(ppo_nsga2_rewards):.2f}")

# Evaluate agent performance after training
ppo_nsga2_avg_reward = ppo_agent_nsga2.evaluate(episodes=30)
print("[NSGA-II + PPO] Average Reward:", ppo_nsga2_avg_reward)

# Save best NSGA+PPO model
if ppo_nsga2_avg_reward > best_reward_nsga2_ppo:
    best_reward_nsga2_ppo = ppo_nsga2_avg_reward
    torch.save({
        "policy_state_dict": ppo_agent_nsga2.policy.state_dict(),
        "value_state_dict": ppo_agent_nsga2.value_network.state_dict(),
        "policy_optimizer_state_dict": ppo_agent_nsga2.policy_optimizer.state_dict(),
        "value_optimizer_state_dict": ppo_agent_nsga2.value_optimizer.state_dict()
    }, best_model_path_nsga2_ppo)
    print(f"Best NSGA-II + PPO model saved to {best_model_path_nsga2_ppo}")

# print(f"NSGA-II + PPO Avg Reward (mean): {np.mean([r.item() for r in ppo_nsga2_rewards]):.2f}")


# === 2. PPO ALONE TRAINING========================================
print("\\n=== Training PPO agent without NSGA-II data (PPO Alone) ===")
ppo_agent_alone = PPOAgent(train_env_2)
# ppo_agent_alone = PPOAgent(train_env)
best_reward_ppo = -float('inf')
best_model_ppo_path = "best_ppo_model.pth"
all_ppo_rewards = []

# Train PPO Alone
start_ppo = time.time()
ppo_rewards_alone, _ = ppo_agent_alone.train(timesteps=200_000, num_steps=512, entropy_log_path="results/ppo_alone_entropy_log.csv" )
end_ppo = time.time()
ppo_runtime = end_ppo-start_ppo
print(f"PPO Alone Runtime: {ppo_runtime:.2f} seconds")
print(f"PPO alone Avg Reward (mean): {np.mean(ppo_rewards_alone):.2f}")

# === Plot reward moving averages ===
plot_combined_reward_moving_avg(
    ppo_rewards_alone,
    ppo_nsga2_rewards,
    window=100,
    save_path="results/combined_moving_avg.png"
)

plot_reward_moving_avg(ppo_nsga2_rewards, window=100, save_path="results/reward_moving_avg_nsga2ppo.png")
plot_reward_moving_avg(ppo_rewards_alone, window=100, save_path="results/reward_moving_avg_ppo.png")


ppo_alone_avg_reward = ppo_agent_alone.evaluate(episodes=30)
print("[PPO Alone] Average Reward:", ppo_alone_avg_reward)
# Save best PPO model
if ppo_alone_avg_reward > best_reward_ppo:
    best_reward_ppo = ppo_alone_avg_reward
    torch.save({
        "policy_state_dict": ppo_agent_alone.policy.state_dict(),
        "value_state_dict": ppo_agent_alone.value_network.state_dict(),
        "policy_optimizer_state_dict": ppo_agent_alone.policy_optimizer.state_dict(),
        "value_optimizer_state_dict": ppo_agent_alone.value_optimizer.state_dict()
    }, "best_ppo_model.pth")
    print(f"Best PPO model saved to {best_model_ppo_path}")

# Save runtime summary
runtime_df = pd.DataFrame([
    {"Method": "NSGA-II", "Runtime (s)": nsga2_runtime},
    {"Method": "PPO", "Runtime (s)": ppo_runtime},
    {"Method": "NSGA-II + PPO", "Runtime (s)": hybrid_runtime},
])
runtime_df.to_csv(os.path.join(RESULTS_DIR, "runtime_comparison.csv"), index=False)
print("\n Runtime comparison saved as 'runtime_comparison.csv'")


rewards_ppo_alone = ppo_rewards_alone
rewards_nsga2_ppo = ppo_nsga2_rewards

#  truncate to equal length
min_len = min(len(rewards_ppo_alone), len(rewards_nsga2_ppo))
rewards_ppo_alone = rewards_ppo_alone[:min_len]
rewards_nsga2_ppo = rewards_nsga2_ppo[:min_len]

plot_dual_learning_curves(rewards_ppo_alone, rewards_nsga2_ppo)

pd.DataFrame({
    "PPO Alone": rewards_ppo_alone,
    "NSGA-II + PPO": rewards_nsga2_ppo
}).to_csv(os.path.join(RESULTS_DIR, "ppo_vs_nsga2ppo_rewards.csv"), index=False)
# Plot combined entropy curves
plot_entropy_curve(
    ppo_csv="results/ppo_alone_entropy_log.csv",
    nsga2_csv="results/nsga2_ppo_entropy_log.csv",
    save_path="results/combined_entropy_curve.png"
)

print("Training complete. Models and learning curves saved.")

plot_entropy_curve(
    ppo_csv="results/ppo_alone_entropy_log.csv",
    nsga2_csv="results/nsga2_ppo_entropy_log.csv",
    save_path="results/combined_entropy_curve.png"
)



#=========== DUMP ==========
'''
from collections import Counter

pareto_training_data = []
obs_dim = train_env_1.observation_space.shape[0]
action_dim = int(np.prod(train_env_1.action_space.nvec))

for idx, i in enumerate(filtered_indices):
    obs = np.concatenate([B[i], SINR[i], I[i], P[i], R[i]])
    if obs.shape[0] > obs_dim:
        obs = obs[:obs_dim]
    elif obs.shape[0] < obs_dim:
        obs = np.pad(obs, (0, obs_dim - obs.shape[0]))

    binary_allocation = filtered_population[idx].astype(int)
    required_len = len(train_env_1.action_space.nvec)
    if binary_allocation.shape[0] < required_len:
        binary_allocation = np.pad(binary_allocation, (0, required_len - binary_allocation.shape[0]))
    elif binary_allocation.shape[0] > required_len:
        binary_allocation = binary_allocation[:required_len]

    action = np.ravel_multi_index(binary_allocation, train_env_1.action_space.nvec)
    pareto_training_data.append((obs, action))
'''

'''for i in range(min(len(pareto_fitness), len(sampled_data))):
    # === Step 1: Build observation vector ===
    obs = np.concatenate([B[i], SINR[i], I[i], P[i], R[i]])
    if obs.shape[0] > obs_dim:
        obs = obs[:obs_dim]
    elif obs.shape[0] < obs_dim:
        obs = np.pad(obs, (0, obs_dim - obs.shape[0]))


    # === Step 2: Use SE, IL, EC from fitness to calculate scalar reward (optional) ===
    se, interference, energy = pareto_fitness[i][0], -pareto_fitness[i][1], -pareto_fitness[i][2]
    reward = ppo_agent_nsga2.calculate_reward_from_objectives(se, interference, energy)

    # === Step 3: Generate binary allocation vector from SINR threshold (or use NSGA-II individuals if available) ===
    # binary_allocation = (SINR[i] > 1.0).astype(int).flatten()[:train_env.action_space.nvec.shape[0]]

    # binary_allocation = (SINR[i] > 1.0).astype(int).flatten()
    binary_allocation = final_population[i].astype(int)


    # Ensure it matches the exact number of action dimensions
    required_len = len(train_env_1.action_space.nvec)
    if binary_allocation.shape[0] < required_len:
        binary_allocation = np.pad(binary_allocation, (0, required_len - binary_allocation.shape[0]), mode='constant')
    elif binary_allocation.shape[0] > required_len:
        binary_allocation = binary_allocation[:required_len]

    # === Step 4: Convert binary allocation to flat action index ===
    action = np.ravel_multi_index(binary_allocation, train_env_1.action_space.nvec)

    # === Step 5: Add (obs, action) pair to pretraining data ===
    pareto_training_data.append((obs, action))
'''
# Print action distribution summary
# print("Pretraining Action Distribution:", Counter([a for _, a in pareto_training_data]))

'''
pareto_training_data = []
obs_dim = train_env.observation_space.shape[0]
action_dim = int(np.prod(train_env.action_space.nvec))

for i in range(min(len(pareto_fitness), len(sampled_data))):
    # Build observation vector
    obs = np.concatenate([B[i], SINR[i], I[i], P[i], R[i]])
    if obs.shape[0] > obs_dim:
        obs = obs[:obs_dim]
    elif obs.shape[0] < obs_dim:
        obs = np.pad(obs, (0, obs_dim - obs.shape[0]))

    se, interference, energy = pareto_fitness[i][0], -pareto_fitness[i][1], -pareto_fitness[i][2]
    reward = ppo_agent_nsga2.calculate_reward_from_objectives(se, interference, energy)
    flat_action_space_size = int(np.prod(train_env.action_space.nvec))
    
    scores = B[i] * np.log2(1 + np.clip(SINR[i], 1e-8, None))
    top_indices = np.argwhere(scores == np.max(scores)).flatten()
    selected = np.random.choice(top_indices)
    action = selected % flat_action_space_size
    # action = int(np.argmax(B[i] * np.log2(1 + SINR[i]))) % flat_action_space_size
    pareto_training_data.append((obs, action))
    from collections import Counter
    print("Pretraining Action Distribution:", Counter([a for _, a in pareto_training_data]))



    #FILTER TOP 50 PARETO SOLUTIONS ========================================
def filter_pareto_front(pareto_fitness, final_population, top_k=50, diversity_threshold=0.05):
    selected_indices = []

    # Step 1: Sort by Spectrum Efficiency (column 0)
    sorted_indices = np.argsort(pareto_fitness[:, 0])[::-1]
    top_se_indices = sorted_indices[:top_k]

    # Step 2: Apply diversity filter
    for i in top_se_indices:
        candidate = pareto_fitness[i]
        if all(euclidean(candidate, pareto_fitness[j]) > diversity_threshold for j in selected_indices):
            selected_indices.append(i)

    filtered_fitness = np.array([pareto_fitness[i] for i in selected_indices])
    filtered_population = np.array([final_population[i] for i in selected_indices])
    print(f" Selected {len(selected_indices)} filtered Pareto samples.")
    return selected_indices, filtered_fitness, filtered_population

pareto_fitness = np.array(pareto_fitness)
filtered_indices, filtered_fitness, filtered_population = filter_pareto_front(
    pareto_fitness, final_population, top_k=50, diversity_threshold=0.05
)


# === Convert Pareto results to (observation, action) training data ===

pareto_training_data = ppo_agent_nsga2.generate_pareto_training_data(
    B, SINR, I, P, R, filtered_indices, filtered_population
)
'''
