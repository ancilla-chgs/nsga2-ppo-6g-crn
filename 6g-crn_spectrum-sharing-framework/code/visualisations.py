import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_pareto_front(pareto_results, filename="log_scaled_normalised_pareto_front.png"):
    # Logarithmic scaling to handle scale differences
    pareto_results_scaled = np.zeros_like(pareto_results)
    pareto_results_scaled[:, 0] = np.log1p(pareto_results[:, 0])  # SE (large values)
    pareto_results_scaled[:, 1] = np.log1p(pareto_results[:, 1])  # Interference
    pareto_results_scaled[:, 2] = np.log1p(pareto_results[:, 2])  # Energy Consumption
    # Normalize after log scaling
    scaler = MinMaxScaler()
    normalized_pareto = scaler.fit_transform(pareto_results_scaled)

    se_vals = normalized_pareto[:, 0]
    intf_vals = normalized_pareto[:, 1]
    energy_vals = normalized_pareto[:, 2]

    # Visualize normalized Pareto Front clearly
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

        # Main scatter
    scatter = ax.scatter(
        se_vals, intf_vals, energy_vals,
        c=se_vals, cmap='viridis', s=50, alpha=0.8
    )

        # Projection lines (shadows)
    for i in range(len(se_vals)):
        ax.plot(
            [se_vals[i], se_vals[i]],
            [intf_vals[i], intf_vals[i]],
            [0, energy_vals[i]],
            color='gray', alpha=0.2, linewidth=0.5
        )

    # Highlight Top 3 Spectrum-Efficient points
    top_idxs = np.argsort(se_vals)[-3:]
    for i in top_idxs:
        ax.scatter(se_vals[i], intf_vals[i], energy_vals[i], c='red', s=60, edgecolors='black')
        ax.text(se_vals[i], intf_vals[i], energy_vals[i], f" P{i+1}", fontsize=9, color='black')

    ax.set_title("Log-Scaled and Normalised NSGA-II Pareto Front", fontsize=14)
    ax.set_xlabel("Spectrum Efficiency (log-scaled)", fontsize=12)
    ax.set_ylabel("Interference (log-scaled)", fontsize=12)
    ax.set_zlabel("Energy Consumption (log-scaled)", fontsize=12)

    ax.view_init(elev=25, azim=135)

    # ax.yaxis.set_label_position('left')
    ax.yaxis.labelpad = 20
    ax.yaxis._axinfo['label']['va'] = 'center'
    ax.yaxis._axinfo['label']['ha'] = 'left'
    fig.colorbar(scatter, ax=ax, label='Normalised SE')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_learning_curve(rewards, filename="ppo_learning_curve.pdf", title="PPO Learning Curve", window=10):
    if len(rewards) >= window:
        smoothed_rewards = np.convolve(rewards, np.ones(window) / window, mode='valid')
    else:
        smoothed_rewards = rewards  # skip smoothing for short lists
    plt.figure(figsize=(10, 5))
    plt.plot(smoothed_rewards, linewidth=2, label=" Smoothed  Reward")
    plt.title(title, fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Avg Episode Reward', fontsize=12)
    # plt.xticks(fontsize=10)
    # plt.yticks(fontsize=10)
    # plt.legend(loc='best', fontsize=10)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()


def plot_reward_comparison_bar(labels, values, filename="reward_comparison_bar_chart.png"):
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color='cornflowerblue', edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 2, f"{height:.1f}", ha='center', va='bottom',
                 fontsize=10)

    plt.title("Average Reward Comparison Across Algorithms", fontsize=14)
    plt.ylabel("Average Reward", fontsize=12)
    plt.xticks(rotation=15, fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(RESULTS_DIR, filename))


def plot_dual_learning_curves(rewards_ppo_alone, rewards_nsga2_ppo, filename="dual_learning_curve.png", window=10):
    # Apply smoothing
    def smooth(rewards, w):
        return np.convolve(rewards, np.ones(w) / w, mode='valid') if len(rewards) >= w else rewards

    smoothed_ppo = smooth(rewards_ppo_alone, window)
    smoothed_nsga2_ppo = smooth(rewards_nsga2_ppo, window)

    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_ppo, label="PPO Alone", linewidth=2)
    plt.plot(smoothed_nsga2_ppo, label="NSGA-II + PPO", linewidth=2)
    plt.title("Learning Curves: PPO Alone vs NSGA-II + PPO")
    plt.xlabel("Episode")
    plt.ylabel("Average Episode Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()


def plot_allocation_usage_comparison(alloc_a, alloc_b, label_a="Baseline", label_b="Hybrid", filename="channel_usage_comparison.png"):
    def count_usage(mat):
        return (
            (mat == 0).sum(axis=0),  # free
            (mat == 1).sum(axis=0),  # SU
            (mat == 2).sum(axis=0),  # PU
        )
    free_b, su_b, pu_b = count_usage(alloc_a)
    free_a, su_a, pu_a = count_usage(alloc_b)
    channels = np.arange(alloc_a.shape[1])

    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(channels - width, su_b, width, label=f"SU {label_a}", color='skyblue')
    ax.bar(channels - width, pu_b, width, bottom=su_b, color='lightcoral', label=f"PU {label_a}")

    ax.bar(channels + width, su_a, width, label=f"SU {label_b}", color='dodgerblue')
    ax.bar(channels + width, pu_a, width, bottom=su_a, color='firebrick', label=f"PU {label_b}")

    ax.set_xlabel("Channel")
    ax.set_ylabel("Usage (Time Steps)")
    ax.set_title(f"Channel Usage: {label_a} vs {label_b}")
    #ax.set_title("Channel Usage Before vs After Training")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()


def generate_random_allocations(time_steps, num_channels):
    # Random spectrum usage: 0 = free, 1 = SU, 2 = PU
    return np.random.choice([0, 1, 2], size=(time_steps, num_channels), p=[0.2, 0.5, 0.3])

'''
def plot_allocation_heatmap(alloc_matrix, title="Channel Allocation Heatmap", save_path="heatmap_debug.png"):
    """
    Visualizes the allocation matrix as a heatmap.
    0 = Free, 1 = SU, 2 = PU
    """
    plt.figure(figsize=(10, 6))
    cmap = sns.color_palette(["black", "purple", "gold"])  # 0=Free, 1=SU, 2=PU

    sns.heatmap(
        alloc_matrix,
        cmap=cmap,
        cbar_kws={
            "ticks": [0.5, 1.5, 2.5],
            "label": "Channel State"
        },
        linewidths=0.1,
        linecolor="gray"
    )

    plt.title(title)
    plt.ylabel("Time Step")
    plt.xlabel("Channel Index")
    plt.xticks(rotation=0)
    plt.yticks([], [])  # optional for large steps
    plt.tight_layout()

    # Save in your results folder
    if not os.path.exists("results"):
        os.makedirs("results")
    plt.savefig(os.path.join("results", save_path))
    plt.close()
    print(f"✅ Saved heatmap to results/{save_path}")
  '''
def plot_allocation_comparison_heatmap( alloc_a, alloc_b,
    title="Time vs Allocation Difference",
    label_a="PPO Alone",
    label_b="NSGA-II + PPO",
    save_path=None
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, constrained_layout=True)
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=0, vmax=2)

    # === LEFT HEATMAP: BASELINE ===
    ax0 = axes[0]
    ax0.imshow(alloc_a, aspect='auto', cmap=cmap, norm=norm)
    ax0.set_title(label_a, fontsize=13)
    ax0.set_xlabel("Channel Index")
    ax0.set_ylabel("Time Step")

    # === RIGHT HEATMAP: HYBRID ===
    ax1 = axes[1]
    im1 = ax1.imshow(alloc_b, aspect='auto', cmap=cmap, norm=norm)
    ax1.set_title(label_b, fontsize=13)
    ax1.set_xlabel("Channel Index")

    fig.suptitle(title, fontsize=16)

    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['Free', 'SU', 'PU'])
    cbar.set_label("Channel State")

    if save_path:
        plt.savefig(os.path.join(RESULTS_DIR, save_path), dpi=300)
    plt.close()


def summarise_channel_usage(alloc_matrix):
    usage = {
        'SU': np.sum(alloc_matrix == 1, axis=0),
        'PU': np.sum(alloc_matrix == 2, axis=0),
        'Free': np.sum(alloc_matrix == 0, axis=0),
    }
    return pd.DataFrame(usage)


def evaluate_strategy(strategy_name, alloc_matrix, dataset_df ):
    df = summarise_channel_usage(alloc_matrix)
    total_steps = alloc_matrix.shape[0]
    num_channels = alloc_matrix.shape[1]

     # Create PU activity matrix from dataset
    pu_activity_matrix = np.zeros_like(alloc_matrix)
    unique_time_slots = sorted(dataset_df['time_slot'].unique())

    for t_idx, ts in enumerate(unique_time_slots[:total_steps]):
        frame = dataset_df[dataset_df['time_slot'] == ts]
        for ch in range(num_channels):
            if frame[frame['channel_id'] == ch]['PU_active'].any():
                pu_activity_matrix[t_idx, ch] = 1

    
    # Count collisions: where alloc_matrix == 1 (SU) and also == 2 (PU)
    collisions = []
    for ch in range(num_channels):
        #channel_alloc = alloc_matrix[:, ch]
        su_used =(alloc_matrix[:, ch] == 1)
        pu_active =(pu_activity_matrix[:, ch] == 1)
        collision_count = np.sum(su_used & pu_active)
        collision_percent = 100 * collision_count / total_steps
        collisions.append(round(collision_percent, 2))

    df['PU Collision %'] = collisions
    df['SU Usage %'] = (df['SU'] / total_steps * 100).round(2)
    df['Algorithm'] = strategy_name
    df['Channel'] = df.index
    return df.reset_index(drop=True)


def export_allocation_to_csv(allocation_matrix, filename):
    df = pd.DataFrame(allocation_matrix)
    df.columns = [f"Channel_{i}" for i in df.columns]
    df.index.name = "Time_Step"
    df.to_csv(os.path.join(RESULTS_DIR, filename))


def generate_trained_dataset_aware_allocations(agent, env, time_steps=100):
    if isinstance(env.action_space, gym.spaces.MultiDiscrete):
        action_dim = int(np.prod(env.action_space.nvec))
    else:
        action_dim = env.action_space.n

    alloc_matrix = np.zeros((time_steps, env.num_channels), dtype=int)

    # alloc_matrix = np.zeros((time_steps, env.action_space.n), dtype=int)
    obs, _ = env.reset()
    for t in range(time_steps):
        # action, _ = agent.select_action(obs)
        action = agent.predict(obs)

         # Step 1: Mark PUs using dataset info
        if hasattr(env, 'data') and hasattr(env, 'episode_time_slots'):
            current_time = env.episode_time_slots[t]
            frame = env.data[env.data['time_slot'] == current_time]
            for ch_index in range(env.num_channels):
                pu_row = frame[frame['channel_id'] == ch_index]
                if not pu_row.empty:
                    pu_active = pu_row['PU_active'].values[0]
                    if pu_active == 1:
                        alloc_matrix[t, ch_index] = 2
            if np.any(alloc_matrix[t] == 2):
                pu_channels = np.where(alloc_matrix[t] == 2)[0]
                print(f"[STEP {t}] PU Active on Channels: {pu_channels}")
        # Step 2: Mark SUs only on free channels
        for ch in action:
            if alloc_matrix[t, ch] != 2:  # Avoid overwriting PU
                alloc_matrix[t, ch] = 1

        # step through environment
        obs, _, _, _, _ = env.step(action)

    #plot_allocation_heatmap(alloc_matrix, title="NSGA-II + PPO Allocation", save_path="alloc_heatmap_hybrid.png")

    return alloc_matrix


def run_random_policy(env, time_steps=100):
    # For MultiDiscrete, use total number of flattened actions
    if isinstance(env.action_space, gym.spaces.MultiDiscrete):
        nvec = env.action_space.nvec
        action_dim = int(np.prod(nvec))  # flatten total action space size
    else:
        action_dim = env.action_space.n
    # alloc = np.zeros((time_steps, action_dim), dtype=int)
    alloc = np.zeros((time_steps, env.num_channels), dtype=int)

    num_sus = env.num_sus
    su_throughput = np.zeros(num_sus)
    obs, _ = env.reset()
    for t in range(time_steps):
        if isinstance(env.action_space, gym.spaces.MultiDiscrete):
            action = [np.random.randint(n) for n in env.action_space.nvec]
            # flat_index = np.ravel_multi_index(action, env.action_space.nvec)
            # alloc[t, flat_index] = 1
        else:
            action = np.random.choice(env.action_space.n)

        #assert isinstance(env.current_data['PU_active'], (list, np.ndarray)), "PU_active should be list/array"
        if hasattr(env, 'data') and hasattr(env, 'episode_time_slots'):
            current_time = env.episode_time_slots[t]
            frame = env.data[env.data['time_slot'] == current_time]
            for ch_index in range(env.num_channels):
                pu_row = frame[frame['channel_id'] == ch_index]
                if not pu_row.empty:
                    pu_active = pu_row['PU_active'].values[0]
                    if pu_active == 1:
                        alloc[t, ch_index] = 2

            if np.any(alloc[t] == 2):
                pu_channels = np.where(alloc[t] == 2)[0]
                print(f"[STEP {t}] PU Active on Channels: {pu_channels}")

        # Step 2: Mark SUs only on free channels
        for ch in action:
            if alloc[t, ch] != 2:
                alloc[t, ch] = 1

        obs, _, _, _, info = env.step(action)
        if isinstance(info, dict) and 'throughput' in info:
            throughput = info['throughput']
            su_throughput += np.array(throughput) / num_sus  # assume equal contribution
    
    #plot_allocation_heatmap(alloc, title="Random Allocation", save_path="alloc_heatmap_random.png")
    
    return alloc, su_throughput


def run_greedy_policy(env, time_steps=100):
    # For MultiDiscrete, use total number of flattened actions
    if isinstance(env.action_space, gym.spaces.MultiDiscrete):
        action_dim = int(np.prod(env.action_space.nvec))
    else:
        action_dim = env.action_space.n
    alloc = np.zeros((time_steps, env.num_channels), dtype=int)
    # alloc = np.zeros((time_steps, env.action_space.n), dtype=int)
    num_sus = env.num_sus
    su_throughput = np.zeros(num_sus)
    obs, _ = env.reset()
    for t in range(time_steps):
        # ==== Action selection ====
        if hasattr(env, 'data') and hasattr(env, 'episode_time_slots'):
            current_time = env.episode_time_slots[t]
            frame = env.data[env.data['time_slot'] == current_time]

            if 'SNR' in frame.columns:
                snrs = []
                for i in range(env.num_channels):
                    snr_vals = frame[frame['channel_id'] == i]['SNR'].values
                    snrs.append(snr_vals[0] if len(snr_vals) > 0 else -np.inf)
                best_channel = int(np.argmax(snrs))
                action = [best_channel] * num_sus
            else:
                action = [np.random.randint(n) for n in env.action_space.nvec]
        else:
            action = [np.random.randint(n) for n in env.action_space.nvec]

        # ==== Mark PUs ====
        if hasattr(env, 'data') and hasattr(env, 'episode_time_slots'):
            current_time = env.episode_time_slots[t]
            frame = env.data[env.data['time_slot'] == current_time]
            for ch_index in range(env.num_channels):
                pu_row = frame[frame['channel_id'] == ch_index]
                if not pu_row.empty:
                    pu_active = pu_row['PU_active'].values[0]
                    if pu_active == 1:
                        alloc[t, ch_index] = 2
            if np.any(alloc[t] == 2):
                pu_channels = np.where(alloc[t] == 2)[0]
                print(f"[STEP {t}] PU Active on Channels: {pu_channels}")
        # Step 2: Mark SUs only on free channels
        for ch in action:
            if alloc[t, ch] != 2:
                alloc[t, ch] = 1

        obs, _, _, _, info = env.step(action)

        if 'throughput' in info:
            throughput = info['throughput']
            su_throughput += throughput / num_sus

    #plot_allocation_heatmap(alloc, title="Greedy Allocation", save_path="alloc_heatmap_greedy.png")

    return alloc, su_throughput


def plot_strategy_comparison_bar_chart(strategy_comparison_df, save_path="strategy_comparison_bar_chart.png"):
    grouped = strategy_comparison_df.groupby("Algorithm").agg({
        "PU Collision %": "mean",
        "SU Usage %": "mean"
    }).reset_index()

    x = np.arange(len(grouped["Algorithm"]))
    width = 0.35

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width / 2, grouped["PU Collision %"], width, label="PU Collision %", color="crimson")
    bars2 = plt.bar(x + width / 2, grouped["SU Usage %"], width, label="SU Usage %", color="steelblue")

      # Add percentage labels on top of each bar
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f"{height:.1f}%", ha="center", va="bottom", fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f"{height:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.xticks(x, grouped["Algorithm"])
    plt.ylabel("Percentage %")
    plt.xticks(rotation=45)
    plt.title("PU Collision vs SU Usage Comparison by Strategy")
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, save_path))
    plt.close()


def plot_channel_breakdown(strategy_comparison_df, save_path="strategy_channel_breakdown.png"):
    plt.figure(figsize=(12, 6))
    melted = strategy_comparison_df.melt(id_vars=["Algorithm", "Channel"],
                                         value_vars=["PU Collision %", "SU Usage %"],
                                         var_name="Metric", value_name="Percentage")

    sns.barplot(data=melted, x="Channel", y="Percentage", hue="Algorithm", palette="Set2", errorbar=None)
    plt.title("Per-Channel Average Comparison Across Strategies")
    plt.ylabel("Percentage")
    plt.xlabel("Channel")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, save_path))
    plt.close()


def plot_nsga2_benchmark_results(csv_path="nsga2_benchmark_results.csv"):
    df = pd.read_csv(csv_path)

    # Plot: Average Spectrum Efficiency vs Population Size and Generations
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Generations", y="Avg Spectrum Efficiency", hue="Population Size", marker="o")
    plt.title("NSGA-II: Spectrum Efficiency vs Generations")
    plt.ylabel("Average Spectrum Efficiency")
    plt.xlabel("Number of Generations")
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "nsga2_benchmark_spectrum_efficiency.png"), dpi=300)
    # plt.show()
    plt.close()

    # Plot: Runtime vs Generations
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Generations", y="Runtime (s)", hue="Population Size", marker="o")
    plt.title("NSGA-II: Runtime vs Generations")
    plt.ylabel("Runtime (seconds)")
    plt.xlabel("Number of Generations")
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "nsga2_benchmark_runtime.png"), dpi=300)
    plt.close()

'''
def plot_strict_3d_pareto_front(fitness_array, save_path=os.path.join(RESULTS_DIR, "strict_pareto_front_3d.png")):
    fitness_array = np.array(fitness_array)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(fitness_array[:, 0], fitness_array[:, 1], fitness_array[:, 2], c='r', marker='o')
    ax.set_xlabel('Spectrum Efficiency')
    ax.set_ylabel('Interference')
    ax.set_zlabel('Energy Consumption')
    ax.set_title("Strict 3D Pareto Front")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

'''
def plot_fairness_comparison_bar_chart(labels, fairness_values, filename="fairness_comparison_bar_chart.png"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, fairness_values, color='mediumseagreen', edgecolor='black')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f"{height:.2f}", ha='center', va='bottom',
                 fontsize=10)

    plt.title("Jain’s Fairness Index Comparison", fontsize=14)
    plt.ylabel("Fairness (0 to 1)", fontsize=12)
    plt.xticks(rotation=15, fontsize=11)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
    print(f"Fairness bar chart saved as {filename}")
    plt.close()


def plot_hypervolume_curve(csv_path="results/hypervolume_curve.csv", save_path="results/hypervolume_convergence.png"):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 5))
    plt.plot(df["Generation"], df["Hypervolume"], marker='o')
    plt.title("Hypervolume Convergence Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f" Hypervolume curve saved to: {save_path}")


def plot_entropy_curve( ppo_csv="results/ppo_alone_entropy_log.csv",
    nsga2_csv="results/nsga2_ppo_entropy_log.csv",
    save_path="results/combined_entropy_curve.png"):
    """
    Plots entropy curves of PPO Alone and NSGA-II + PPO on the same axis.
    """
    try:
        df_ppo = pd.read_csv(ppo_csv)
        df_nsga2 = pd.read_csv(nsga2_csv)

        plt.figure(figsize=(10, 6))
        # plt.plot(df["Step"], df["Entropy"], color="blue", marker='o', markersize=3)
        plt.plot(df_ppo["Step"], df_ppo["Entropy"], label="PPO Alone", color="steelblue", linewidth=2)
        plt.plot(df_nsga2["Step"], df_nsga2["Entropy"], label="NSGA-II + PPO", color="darkorange", linewidth=2)
        plt.title("Entropy Comparison During PPO Alone vs NSGA2 + PPO Training")
        plt.xlabel("Training Step", fontsize=12)
        plt.ylabel("Policy Entropy", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f" Entropy curve saved to: {save_path}")
    except Exception as e:
        print(f" Failed to plot entropy curve: {e}")
        
def plot_scalar_reward_convergence(convergence_data, filename="nsga2_convergence_curve.png"):
    if not convergence_data or len(convergence_data) == 0:
        print("Warning: Empty convergence data passed. Skipping plot.")
        return

    plt.figure(figsize=(8, 4))
    plt.plot(convergence_data, marker='o', color='blue', label="Scalarised Reward")
    plt.title("NSGA-II Convergence Curve")
    plt.xlabel("Generation")
    plt.ylabel("Best Scalarised Reward")
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join("results", filename)
    plt.savefig(save_path)
    plt.close()
   # print(f"[Plot Saved] Scalar reward convergence plot saved to: {save_path}")

def plot_reward_moving_avg(reward_list, window=100, save_path="reward_moving_average.png"):
    """
    Plot the reward per episode and its moving average.
    :param reward_list: List of episode rewards.
    :param window: Window size for moving average.
    :param save_path: Path to save the figure.
    """
    moving_avg = pd.Series(reward_list).rolling(window).mean()
    plt.figure(figsize=(10, 4))
    plt.plot(reward_list, label="Episode Reward", alpha=0.6)
    plt.plot(moving_avg, label=f"{window}-Episode Moving Average", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward vs Episode with Moving Average")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_combined_reward_moving_avg(ppo_rewards, hybrid_rewards, window=100, save_path="results/combined_moving_avg.png"):
    ppo_avg = pd.Series(ppo_rewards).rolling(window).mean()
    hybrid_avg = pd.Series(hybrid_rewards).rolling(window).mean()
    plt.figure(figsize=(10, 4))
    plt.plot(ppo_avg, label="PPO Alone (Moving Avg)", linewidth=2)
    plt.plot(hybrid_avg, label="NSGA-II + PPO (Moving Avg)", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Moving Average Comparison (Window={window})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_radar_chart(results_df, title="Strategy Comparison Radar Plot", save_path="results/strategy_radar_plot.png"):
    # Metrics to include in the radar plot

    metrics = [
        "Average Reward",
        "PU Collision (%)",
        "Energy Efficiency",
        "Spectrum Utilization (%)",
        "Interference (%)",
        "QoS Violation Rate (%)"
    ]
   
    algorithms = results_df["Algorithm"].tolist()

    # Extract and convert metric values
    print("Available columns in results_df:", results_df.columns.tolist())

    metric_values = results_df[metrics].copy()
    metric_values = metric_values.replace("-", 0).fillna(0).astype(float)
    #metric_values = metric_values.fillna(0).astype(float)

    print(" Metrics going into radar chart:")
    print(metric_values)


    # Invert bad metrics: lower is better → invert so higher is better for plotting
    for col in ["Interference (%)", "QoS Violation Rate (%)", "PU Collision (%)"]:
        max_val = metric_values[col].max()
        min_val = metric_values[col].min()
        metric_values[col] = max_val - metric_values[col] + min_val

    # Normalize data between 0 and 1
    #metric_values = results_df[metrics].copy()
    #metric_values = metric_values.fillna(0)
    norm_values = (metric_values - metric_values.min()) / (metric_values.max() - metric_values.min() + 1e-8)

    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # complete the circle

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    for i, row in norm_values.iterrows():
        values = row.tolist()
        values += values[:1]  # repeat first value to close the radar
        ax.plot(angles, values, label=algorithms[i], linewidth=2)
        ax.fill(angles, values, alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
    ax.set_title(title, size=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Radar chart saved to: {save_path}")

def plot_su_pu_grouped_bars(usage_df, save_path="channel_usage_grouped.png"):
    channels = usage_df["Channel"]
    bar_width = 0.1
    x = np.arange(len(channels))

    plt.figure(figsize=(16, 6))

    # SU usage bars (left group)
    plt.bar(x - 1.5*bar_width, usage_df["SU Random"], width=bar_width, label="SU Random", color="skyblue")
    plt.bar(x - 0.5*bar_width, usage_df["SU Greedy"], width=bar_width, label="SU Greedy", color="lightgreen")
    plt.bar(x + 0.5*bar_width, usage_df["SU PPO"], width=bar_width, label="SU PPO", color="dodgerblue")
    plt.bar(x + 1.5*bar_width, usage_df["SU NSGA-II + PPO"], width=bar_width, label="SU NSGA-II + PPO", color="navy")

    # PU usage bars (right group)
    plt.bar(x + 2.5*bar_width, usage_df["PU Random"], width=bar_width, label="PU Random", color="salmon")
    plt.bar(x + 3.5*bar_width, usage_df["PU Greedy"], width=bar_width, label="PU Greedy", color="orange")
    plt.bar(x + 4.5*bar_width, usage_df["PU PPO"], width=bar_width, label="PU PPO", color="tomato")
    plt.bar(x + 5.5*bar_width, usage_df["PU NSGA-II + PPO"], width=bar_width, label="PU NSGA-II + PPO", color="firebrick")

    plt.xticks(x, [f"Ch {c}" for c in channels])
    plt.ylabel("Usage Frequency (Time Steps)")
    plt.xlabel("Channel")
    plt.title("Grouped PU and SU Usage per Channel Across Strategies")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=4)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, save_path))
    plt.close()
    print(f"✅ Grouped bar chart saved to: {os.path.join(RESULTS_DIR, save_path)}")
