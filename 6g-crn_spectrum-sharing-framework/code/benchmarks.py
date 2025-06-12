import numpy as np
import time
import pandas as pd
from itertools import product
from nsga2 import NSGAII
from spectrum_env import SpectrumEnv


def benchmark_nsga2_configs(env, pop_sizes, generations_list, output_path="nsga2_benchmark_results.csv"):
    results = []

    for pop_size, generations in product(pop_sizes, generations_list):
        #print(f"Running NSGA-II: pop_size={pop_size}, generations={generations}")
        start_time = time.time()

        # Initialize and run NSGA-II
        B, SINR, I, P, R = env.get_averaged_nsga2_inputs()
        num_vars = B.shape[1]
        nsga = NSGAII(pop_size=pop_size, num_gen=generations)
        # pareto_front = nsga.optimize()
        pareto_front = nsga.optimize(num_vars, B, SINR, I, P, R)
        duration = time.time() - start_time

        # Evaluate final generation metrics
        print(f"Example entry from pareto_front: {pareto_front[0]}")
        print(f"Length of pareto_front: {len(pareto_front)}")

        for i, pf in enumerate(pareto_front):
            print(f"Entry {i} shape: {np.shape(pf)}")


        # Check and clean if needed
        # filtered = [entry for entry in pareto_front if len(entry) == 3]
        # pareto_array = np.vstack(filtered)
        clean_fronts = [entry for entry in pareto_front if isinstance(entry, np.ndarray) and entry.ndim == 2 and entry.shape[1] == 3]

        if len(clean_fronts) == 0:
            raise ValueError("No valid 2D Pareto fronts with 3 objectives found.")
        
        pareto_array = np.vstack(clean_fronts)
        avg_se = pareto_array[:, 0].mean()
        avg_intf = pareto_array[:, 1].mean()
        avg_energy = pareto_array[:, 2].mean()

        results.append({
            "Population Size": pop_size,
            "Generations": generations,
            "Avg Spectrum Efficiency": avg_se,
            "Avg Interference": avg_intf,
            "Avg Energy": avg_energy,
            "Runtime (s)": duration
        })

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Benchmark results saved to {output_path}")


def greedy_allocation(env, B, SINR, I, P, R):
    """
    Perform greedy allocation of spectrum resources based on maximum SINR.
    
    Args:
        env: The environment (not used directly here but included for consistency).
        B: Bandwidth matrix (num_samples x num_channels).
        SINR: Signal-to-Interference Ratio matrix (num_samples x num_channels).
        I: Interference matrix (num_samples x num_channels).
        P: Power matrix (num_samples x num_channels).
        R: Data rate matrix (num_samples x num_channels).
    
    Returns:
        rewards: Array of rewards for each sample.
        avg_reward: Average reward across all samples.
    """
    rewards = []
    for i in range(B.shape[0]):
        # Greedy allocation: Select the channel with the highest SINR
        allocation = np.argmax(SINR[i])

        # Calculate reward: Spectrum efficiency minus interference cost
        spectrum_efficiency = B[i, allocation] * np.log2(1 + SINR[i, allocation])
        interference_cost = I[i, allocation] * P[i, allocation]
        reward = spectrum_efficiency - interference_cost

        # Add constraints to prevent unrealistic rewards
        reward = np.clip(reward, 0, 20)  # Clip reward to a reasonable range (e.g., 0 to 100)

        # Debugging: Print intermediate values
        # print(f"Sample {i}:")
        # print(f"  Allocation: {allocation}")
        # print(f"  Bandwidth (B): {B[i, allocation]}")
        # print(f"  SINR: {SINR[i, allocation]}")
        # print(f"  Interference (I): {I[i, allocation]}")
        # print(f"  Power (P): {P[i, allocation]}")
        # print(f"  Spectrum Efficiency: {spectrum_efficiency}")
        # print(f"  Interference Cost: {interference_cost}")
        # print(f"  Reward: {reward}")

        rewards.append(reward)

    rewards = np.array(rewards)
    avg_reward = np.mean(rewards)

    # Debugging: Print final rewards
    # print("\nFinal Rewards:")
    # print(rewards)

    # Return rewards and average reward
    return rewards, avg_reward


def random_drl(env, B, SINR, I, P, R):
    """
    Perform random allocation of spectrum resources.
    
    Args:
        env: The environment (not used directly here but included for consistency).
        B: Bandwidth matrix (num_samples x num_channels).
        SINR: Signal-to-Interference Ratio matrix (num_samples x num_channels).
        I: Interference matrix (num_samples x num_channels).
        P: Power matrix (num_samples x num_channels).
        R: Data rate matrix (num_samples x num_channels).
    
    Returns:
        rewards: Array of rewards for each sample.
        avg_reward: Average reward across all samples.
    """
    rewards = []
    for i in range(B.shape[0]):
        # Random allocation: Select a random channel
        allocation = np.random.randint(0, B.shape[1])

        # Calculate reward: Spectrum efficiency minus interference cost
        reward = B[i, allocation] * np.log2(1 + SINR[i, allocation]) - I[i, allocation] * P[i, allocation]
        rewards.append(reward)

    rewards = np.array(rewards)
    avg_reward = np.mean(rewards)

    # Return rewards and average reward
    return rewards, avg_reward
