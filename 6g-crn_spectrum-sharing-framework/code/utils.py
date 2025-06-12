# Helper functions
import gym
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from spectrum_env import SpectrumEnv
from pymoo.indicators.hv import HV

def create_crn_envs_static(path_dir="./", num_channels=5, num_sus=3, max_steps=512):
    
    '''
    train = pd.read_csv(f"{path_dir}/train.csv")
    val = pd.read_csv(f"{path_dir}/val.csv")
    test = val.copy()

    return (
        SpectrumEnv(train, num_channels, num_sus, max_steps=max_steps),
        SpectrumEnv(val, num_channels, num_sus, max_steps=max_steps),
        SpectrumEnv(test, num_channels, num_sus, max_steps=max_steps)
    )

    '''
    train = pd.read_csv(f"{path_dir}/train_dataset_sinr.csv")
    val = pd.read_csv(f"{path_dir}/val_dataset_sinr.csv")
    test = pd.read_csv(f"{path_dir}/test_dataset_sinr.csv")

    return (
        SpectrumEnv(train, num_channels, num_sus, max_steps=max_steps),
        SpectrumEnv(val, num_channels, num_sus, max_steps=max_steps),
        SpectrumEnv(test, num_channels, num_sus, max_steps=max_steps)
    )
    

'''
def create_single_env(dataset_path, num_channels=5, num_sus=3, max_steps=512):
    df = pd.read_csv(dataset_path)
    return SpectrumEnv(df, num_channels, num_sus, max_steps=max_steps)


def create_crn_envs_dynamic(path="ns3_crn_6g_dataset.csv", num_channels=5, num_sus=3):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    # Add renaming for compatibility with SpectrumEnv

    train, temp = train_test_split(df, test_size=0.3, shuffle=False)
    val, test = train_test_split(temp, test_size=0.5, shuffle=False)
    return (
        SpectrumEnv(train, num_channels, num_sus),
        SpectrumEnv(val, num_channels, num_sus),
        SpectrumEnv(test, num_channels, num_sus)
    )


def create_crn_envs(path="ns3_crn_6g_dataset.csv", num_channels=5, num_sus=3):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    train, temp = train_test_split(df, test_size=0.3, shuffle=False)
    val, test = train_test_split(temp, test_size=0.5, shuffle=False)
    return (
        SpectrumEnv(train, num_channels, num_sus),
        SpectrumEnv(val, num_channels, num_sus),
        SpectrumEnv(test, num_channels, num_sus)
    )

'''

def visualize_split_distributions(train_path="train.csv", val_path="val.csv", test_path="test.csv"):
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)

    def plot_distribution(df, name):
        pu_counts = df['PU_active'].value_counts()
        su_counts = df['SU_request'].value_counts()

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].bar(pu_counts.index.astype(str), pu_counts.values)
        axs[0].set_title(f"{name} - PU Activity")
        axs[0].set_xlabel("PU Active")
        axs[0].set_ylabel("Count")

        axs[1].bar(su_counts.index.astype(str), su_counts.values)
        axs[1].set_title(f"{name} - SU Request")
        axs[1].set_xlabel("SU Request")
        axs[1].set_ylabel("Count")

        plt.tight_layout()
        plt.savefig(f"{name.lower()}_distribution.png")
        print(f"{name} distribution plot saved as {name.lower()}_distribution.png")

    plot_distribution(train, "Train")
    plot_distribution(val, "Validation")
    plot_distribution(test, "Test")


def compute_jains_index(values):
    """
    Compute Jain's Fairness Index for a list of values (e.g., SU throughput).
    :param values: list or np.array of performance values per SU
    :return: fairness index (float between 0 and 1)
    """
    values = np.array(values)
    numerator = np.sum(values) ** 2
    denominator = len(values) * np.sum(values ** 2)
    if denominator == 0:
        return 0.0  # avoid divide by zero
    return numerator / denominator


def unflatten_action(flat_action, nvec):
    """
    Converts a flat action index into a structured action list for MultiDiscrete.
    """
    return list(np.unravel_index(flat_action, nvec))


def get_action_dim(space):
    if isinstance(space, gym.spaces.MultiDiscrete):
        return int(np.prod(space.nvec))
    return space.n


def compute_and_log_hypervolume(pareto_fitness, results_dir="results", filename="hypervolume_evaluation.txt"):
    """
  Computes and logs the hypervolume of a normalized 3-objective Pareto front.
    Objective 1 (SE): maximize
    Objective 2 (IL): minimize → transformed to (1 - IL)
    Objective 3 (EC): minimize → transformed to (1 - EC)
    """

    # Ensure its a Numpy array
    pareto_array = np.array(pareto_fitness)

    # Ensure valid shape (N, 3)
    #pareto_fitness = np.array([f for f in pareto_fitness if len(f) == 3])

    #if len(pareto_fitness) == 0:
        #print("Empty or invalid Pareto front. Cannot compute HV.")
        #return 0.0

    # Dynamic reference point: slightly worse than worst values
    #ref_point = np.max(pareto_fitness, axis=0) + 0.05
    #hv_calc = HV(ref_point=ref_point)
    #hv_score = hv_calc(pareto_fitness)

       # Step 1: Transform minimisation objectives for HV ( all maximised)
    fitness_transformed = np.copy(pareto_array) 
    fitness_transformed[:, 1]*= -1 #IL  (minise to maximise )
    fitness_transformed[:, 2]*= -1 #EC  (monimise to maximise)

    #fitness_transformed = np.clip(fitness_transformed, 0, 1)  # ensure all values are valid

    # Step 2: Define reference point slightly outside [1,1,1]
    #reference_point = [1.1, 1.1, 1.1]

    # Define reference point slightly worse than worst point
    reference_point = np.max(fitness_transformed, axis=0) + 0.1

    # Step 3: Compute hypervolume
    try:
        hv = HV(ref_point = reference_point)
        hv_score = hv.do(fitness_transformed)
    except Exception as e:
      print (f"[ERROR] Hypervolume calculation failed {e}")
      return 0.0

    
    # Save to file
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, filename), "w") as f:
        f.write(f"Reference Point: {reference_point.tolist()}\n")
        f.write(f" Computed Hypervolume: {hv_score:.6f}\n")

    print(f"Hypervolume: {hv_score:.6f} (saved to {filename})")
    return hv_score


'''def compute_hypervolume(pareto_front, reference_point=[1.1, 1.1, 1.1]):
    pareto_front = np.array(pareto_front, dtype=np.float64)
    if pareto_front.ndim != 2 or pareto_front.shape[1] != len(reference_point):
        raise ValueError(f"Expected shape (N, {len(reference_point)}), got {pareto_front.shape}")
    try:
        hv = hypervolume(reference_point)
        return hv(pareto_front)
    except Exception as e:
        print(f"Hypervolume computation failed: {e}")
        return 0

'''


'''def compute_hypervolume(pareto_csv_path, reference_point=[1.1, 1.1, 1.1], save_path="results/hypervolume.txt"):
    # Load normalized Pareto front
    df = pd.read_csv(pareto_csv_path)
    pareto_front = df.values  # assume columns = [SE, IL, EC]

    if pareto_front.shape[1] != len(reference_point):
        raise ValueError("Mismatch between objectives and reference point dimensions.")

    volume = hypervolume(pareto_front, reference_point)
    print(f"[Hypervolume] HV = {volume:.6f}")

    with open(save_path, "w") as f:
        f.write(f"Hypervolume: {volume:.6f}\n")
        print(f"[Saved] Hypervolume written to {save_path}")

    return volume
    '''
