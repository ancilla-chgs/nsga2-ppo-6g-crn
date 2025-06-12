import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from utils import compute_jains_index, unflatten_action


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compare_algorithms(greedy, random, ppo, hybrid):
    data = {
        "Algorithm": ["Greedy", "Random", "PPO", "NSGA-II + PPO"],
        "Avg Reward": [
            greedy["reward"],
            random["reward"],
            ppo["reward"],
            hybrid["reward"]
        ],
        "Fairness": [
            greedy["fairness"],
            random["fairness"],
            ppo["fairness"],
            hybrid["fairness"]
        ],
        "Energy Eff.": [
            greedy["energy"],
            random["energy"],
            ppo["energy"],
            hybrid["energy"]
        ],
        "Interference %": [
            greedy["interference_percent"],
            random["interference_percent"],
            ppo["interference_percent"],
            hybrid["interference_percent"]
        ],
        "Spectrum Util. %": [
            greedy["spectrum_utilization"],
            random["spectrum_utilization"],
            ppo["spectrum_utilization"],
            hybrid["spectrum_utilization"]
        ],
        "QoS Violation %": [
            greedy["qos_violation_rate"],
            random["qos_violation_rate"],
            ppo["qos_violation_rate"],
            hybrid["qos_violation_rate"]
        ],
        "PU Collision %": [
            greedy["pu_collision_percent"],
            random["pu_collision_percent"],
            ppo["pu_collision_percent"],
            hybrid["pu_collision_percent"]
        ]

    }
    results_df = pd.DataFrame(data)
    print(results_df)
    return results_df

def visualize_results(results_df):
    # Bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Algorithm", y="Avg Reward", data=results_df,
        hue="Algorithm", palette="coolwarm", legend=False
    )
    plt.title("Average Reward Comparison Across Algorithms")
    plt.ylabel("Average Reward (Spectrum Efficiency)")
    plt.xlabel("Algorithm")
    plt.show()

    # Box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x="Algorithm", y="Avg Reward", data=results_df,
        hue="Algorithm", palette="viridis", legend=False
    )
    plt.title("Distribution of Rewards")
    plt.show()

def evaluate_policy(agent, name="Agent", episodes=30, max_steps=5000, env=None):
    rewards = []
    active_env = env or agent.env
    for ep in range(episodes):
        obs, _ = active_env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done and steps < max_steps:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action_probs = agent.policy(obs_tensor)
            # action = torch.argmax(action_probs).item()
            flat_action = torch.argmax(action_probs).item()
            action = unflatten_action(flat_action, active_env.action_space.nvec)
            obs, reward, done, _, _ = active_env.step(action)
            total_reward += reward
            steps += 1
        rewards.append(total_reward)
        # print(f"[{name} Eval Episode {ep + 1}] Total Reward: {total_reward:.2f}")
    avg_reward = np.mean(rewards)
    print(f"[{name}] Average Reward: {avg_reward:.2f}\n")
    return avg_reward

def evaluate_policy_with_fairness(agent, env, name="Agent", episodes=30, max_steps=5000, return_energy=False):
    total_throughput = 0
    total_energy = 0
    rewards = []
    per_su_throughputs = []
    total_slots = 0
    used_slots = 0
    total_interference = 0
    qos_violations = 0
    total_pu_collisions = 0

    for ep in range(episodes):
        obs, _ = env.reset()
        done, steps = False, 0
        total_reward = 0
        su_throughputs = np.zeros(env.num_sus)

        while not done and steps < max_steps:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action_probs = agent.policy(obs_tensor)
            flat_action = torch.argmax(action_probs).item()
            action = list(np.unravel_index(flat_action, env.action_space.nvec))
            obs, reward, done, _, info = env.step(action)

            total_slots += 1
            if info["interference"] > 0:
                total_interference += 1
            if info["throughput"] > 0:
                used_slots += 1
            if 'throughput' in info:
                total_throughput += np.sum(info['throughput'])  # bits or Mbps
            if 'energy_consumption' in info:
                total_energy += np.sum(info['energy_consumption'])  # Joules
            if 'QoS_violated' in info:
                qos_violations += int(info['QoS_violated'])
            if "pu_collisions" in info:
                total_pu_collisions += info["pu_collisions"]

            # Here we estimate throughput contribution
            su_throughputs += info.get("throughput", 0) / env.num_sus
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        per_su_throughputs.append(su_throughputs)

    avg_reward = np.mean(rewards)
    avg_su_throughputs = np.mean(per_su_throughputs, axis=0)
    fairness = compute_jains_index(avg_su_throughputs)
    energy_efficiency = total_throughput / total_energy if total_energy > 0 else 0
    interference_percent = 100 * total_interference / total_slots
    sue = 100 * used_slots / total_slots
    qos_violation_rate = 100 * qos_violations / total_slots
    collision_rate = 100 * total_pu_collisions / total_slots

    print(f"[{name}] Average Reward: {avg_reward:.2f}")
    print(f"[{name}] Jain's Fairness Index: {fairness:.4f}")

    return {
        "reward": avg_reward,
        "fairness": fairness,
        "energy": energy_efficiency,
        "interference_percent": interference_percent,
        "spectrum_utilization": sue,
        "qos_violation_rate": qos_violation_rate,
        "pu_collision_percent": collision_rate

    }

def evaluate_baseline_policy(env, policy_fn, name="Policy", time_steps=100):
    total_reward = 0
    su_throughput = np.zeros(env.num_sus)
    total_throughput = 0
    total_energy = 0
    used_slots = 0
    total_slots = 0
    total_interference = 0
    qos_violations = 0
    total_pu_collisions = 0

    obs, _ = env.reset()
    for t in range(time_steps):
        action = policy_fn(env)
        obs, reward, _, _, info = env.step(action)
        total_reward += reward
        total_slots += 1

        if "throughput" in info:
            throughput = np.array(info["throughput"])
            su_throughput += throughput / env.num_sus
            total_throughput += np.sum(throughput)
            if np.any(throughput > 0):
                used_slots += 1

        if "energy_consumption" in info:
            total_energy += np.sum(info["energy_consumption"])

        if "interference" in info:
            if info["interference"] > 0:
                total_interference += 1

        if "QoS_violated" in info:
            qos_violations += int(info["QoS_violated"])
        if "pu_collisions" in info:
            total_pu_collisions += info["pu_collisions"]

    avg_reward = total_reward / time_steps
    fairness = compute_jains_index(su_throughput)
    energy_efficiency = total_throughput / total_energy if total_energy > 0 else 0
    interference_percent = 100 * total_interference / total_slots
    spectrum_utilization = 100 * used_slots / total_slots
    qos_violation_rate = 100 * qos_violations / total_slots
    pu_collision_percent = 100 * total_pu_collisions / (time_steps * env.num_sus)

    print(f"[{name}] Avg Reward: {avg_reward:.2f} | Fairness: {fairness:.4f} | Energy Eff: {energy_efficiency:.2f} | Interference: {interference_percent:.2f}% | SU Util: {spectrum_utilization:.2f}% | QoS Violations: {qos_violation_rate:.2f}%")

    return {
        "reward": avg_reward,
        "fairness": fairness,
        "energy": energy_efficiency,
        "interference_percent": interference_percent,
        "spectrum_utilization": spectrum_utilization,
        "qos_violation_rate": qos_violation_rate,
        "pu_collision_percent": pu_collision_percent
    }

def random_action_fn(env):
    return [np.random.randint(n) for n in env.action_space.nvec]


def greedy_action_fn(env):
    if hasattr(env, 'current_data') and 'SNR' in env.current_data:
        best_ch = int(np.argmax(env.current_data['SNR']))
        return [best_ch] * env.num_sus
    else:
        return [np.random.randint(n) for n in env.action_space.nvec]

