import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import gymnasium as gym
from torch.distributions import Categorical
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

    def get_log_prob(self, observations, actions):
        action_probs = self.forward(observations)
        dist = Categorical(action_probs)
        return dist.log_prob(actions)


# Define Value Network
class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.fc(x)


# PPO Agent

class PPOAgent:
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.input_dim = env.observation_space.shape[0]
        # self.output_dim = env.action_space.n
        self.output_dim = int(np.prod(env.action_space.nvec))  # flatten multiDiscrete

        # Define networks
        self.policy = PolicyNetwork(self.input_dim, self.output_dim).to(device)
        self.value_network = ValueNetwork(self.input_dim).to(device)

        # Hyperparamters

        self.gamma = 0.99 # Increased discount factor to give more weight to future rewards
        self.lambda_ = 0.95  # Adjusted GAE parameter to reduce variance
        self.clip_epsilon = 0.1  # Reduced clipping range for more stable updates
        self.learning_rate = 2e-4  # can also try 5e-4 or 1e-4

        # Define optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.learning_rate)

        # For running mean/std normalisation
        self.reward_stats = {"se": [], "intf": [], "ec": []}  # se= spec eff etc

        # === Entropy Decay ===
        self.initial_entropy_weight = 0.05
        self.final_entropy_weight = 0.001
        self.entropy_decay_steps = 400_000
        self.entropy_step = 0

    def unflatten_action(self, flat_action):
        return list(np.unravel_index(flat_action, self.env.action_space.nvec))

    def normalize_metric(self, stats,value): # metric_list, new_value):
        """
         Min-Max normalisation with stability for small ranges.  
        """
        min_val = stats.get("min", 0)
        max_val = stats.get("max", 1)
        
        # metric_list.append(new_value)
        #if len(metric_list) > 1000:
         #   metric_list.pop(0)
        #mean = np.mean(metric_list)
        #std = np.std(metric_list) + 1e-8
        #return (new_value - mean) / std
        return (value-min_val) / (max_val - min_val + 1e-8)

    def calculate_reward_from_objectives(self, se, interference, energy):
        # Normalize each metric based on running stats
        norm_se = self.normalize_metric(self.reward_stats["se"], se)
        norm_intf = self.normalize_metric(self.reward_stats["intf"], interference)
        norm_ec = self.normalize_metric(self.reward_stats["ec"], energy)

        # Optional thresholds (e.g., bonus if all metrics are favorable)
        bonus = 0.0
        if norm_se > 1.0 and norm_intf < 0.0 and norm_ec < 0.0:
            bonus = 1.0

        # Balanced weights
        w1, w2, w3 = 0.7, 0.15, 0.15
        reward = w1 * norm_se - w2 * norm_intf - w3 * norm_ec + bonus

         # Debugging output
        print(f"[Reward] SE: {se:.2f} | IL: {interference:.2f} | EC: {energy:.5f} → "
          f"Norm: ({norm_se:.3f}, {norm_intf:.3f}, {norm_ec:.3f}) → Reward: {reward:.4f}")



        return reward

    def compute_entropy_weight(self):
        decay_ratio = min(self.entropy_step / self.entropy_decay_steps, 1.0)
        weight = (1 - decay_ratio) * self.initial_entropy_weight + decay_ratio * self.final_entropy_weight
        return weight

    def pretrain_from_pareto(self, pareto_data, epochs=10):
        """
        pareto_data: list of (observation, action) pairs derived from NSGA-II
        """
        print("[Pretraining PPO policy using NSGA-II Pareto data]")
        self.policy.train()
        for epoch in range(epochs):
            total_loss = 0
            valid_samples = 0
            for obs, action in pareto_data:
                if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                    continue  # skip invalid data
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                action_tensor = torch.tensor([action], dtype=torch.long).to(device)

                action_probs = self.policy(obs_tensor)
                action_probs = torch.clamp(action_probs, 1e-8, 1.0)
                action_probs = action_probs / torch.sum(action_probs)

                log_prob = torch.log(action_probs.squeeze(0)[action_tensor])
                loss = -log_prob.mean()  # Cross-entropy

                self.policy_optimizer.zero_grad()
                loss.backward()
                self.policy_optimizer.step()

                total_loss += loss.item()
                valid_samples += 1

            avg_loss = total_loss / max(valid_samples, 1)
            # avg_loss = total_loss / len(pareto_data)
            print(f"[Pretrain Epoch {epoch + 1}] Avg Loss: {avg_loss:.4f}")

    def compute_advantages(self, rewards, values):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            gae = delta + self.gamma * self.lambda_ * gae
            advantages.insert(0, gae)
        return torch.tensor(advantages, dtype=torch.float32).to(device)

    def compute_loss(self, observations, actions, advantages, old_log_probs):
        new_log_probs = self.policy.get_log_prob(observations, actions)
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        surrogate_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # added entropy bonus to encourage exploration
        action_probs = self.policy(observations)
        dist = Categorical(action_probs)
        entropy = dist.entropy().mean()
        self.last_entropy_value = entropy.item()

        entropy_weight = self.compute_entropy_weight()
        self.entropy_step += 1

        return surrogate_loss - entropy_weight * entropy

    def compute_value_loss(self, values, returns):
        value_loss = torch.mean((values - returns) ** 2)
        return value_loss

    def update_policy(self, observations, actions, advantages, old_log_probs):
        policy_loss = self.compute_loss(observations, actions, advantages, old_log_probs)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        return policy_loss.item(), self.last_entropy_value

    def update_value_function(self, observations, returns):
        values = self.value_network(observations)
        value_loss = self.compute_value_loss(values, returns)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        return value_loss.item()

    def calculate_reward(self, obs, env_reward, done):
        # Penalize if the agent is stuck in the same state
        # if self.last_obs is not None and np.array_equal(obs, self.last_obs):
        # return -0.1  # Small negative reward for no movement

        # Use environment reward, but scale it dynamically
        # reward = env_reward + np.random.uniform(-0.2, 0.2)  # Add randomness
        reward = env_reward
        # Encourage longer survival if episode continues
        if not done:
            reward += 0.1
        # self.last_obs = obs  # Track last state
        return reward

    def collect_trajectories(self, num_steps=128):
        print("Starting to collect trajectories...")
        observations, actions, rewards, values, old_log_probs = [], [], [], [], []

        obs, info = self.env.reset()
        self.last_obs = None
        episode_reward = 0  # to track environment reward per episode
        retry_count = 0
        max_retries = 10
        step_counter = 0

        for step_num  in range(num_steps):
            if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                retry_count += 1
                print(f"[Retry {retry_count}] Invalid obs at step {step_num}, resetting...")
                obs, info = self.env.reset()
                if retry_count >= max_retries:
                    print(" Too many invalid resets. Skipping step.")
                    break
                continue

            step_counter += 1
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

            action_probs = self.policy(obs_tensor)
            action_probs = torch.clamp(action_probs, 1e-8, 1.0)
            action_probs = action_probs / torch.sum(action_probs)

            dist = Categorical(action_probs)
            action = dist.sample()
            structured_action = self.unflatten_action(action.item())
            try:
                obs, env_reward, terminated, truncated, info = self.env.step(structured_action)
                
            except Exception as e:
                print(f" CRASH in env.step at step {step_num}: {e}")
                break
            done = terminated or truncated

            if done:
                    print(f" Episode ended at step {step_num}. Resetting env.")
                    obs, info = self.env.reset()

            reward = self.calculate_reward(obs, env_reward, done)

            observations.append(obs_tensor.cpu().numpy())
            actions.append(action)
            rewards.append(reward)
            values.append(self.value_network(obs_tensor).item())
            old_log_probs.append(dist.log_prob(action).detach())

            # progress log
            if step_num % 25 == 0:
                 print(f"Collected step {step_num}")

           
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        values.append(self.value_network(obs_tensor).item())

        observations = torch.tensor(np.array(observations), dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        values = torch.tensor(values, dtype=torch.float32).to(device)
        # old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
        old_log_probs = torch.stack(old_log_probs)

        print(f" Finished collecting {len(rewards)} steps.")

        return observations, actions, rewards, values, old_log_probs

    def compute_returns(self, rewards):
        returns = []
        discounted_sum = 0
        for reward in reversed(rewards):
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        return torch.tensor(returns, dtype=torch.float32).to(device)

    def train(self, timesteps=3_000, num_steps=128, entropy_log_path="results/ppo_entropy_curve.csv"):
        import time
        start_time = time.time()

        episode_rewards = []  # store cumulative episode rewards
        policy_losses = []
        value_losses = []
        entropy_values = []
        entropy_steps = []


        total_steps_collected = 0
        episode_reward = 0
        obs, _ = self.env.reset()

        while total_steps_collected < timesteps:
            # Collect trajectories
            observations, actions, rewards, values, old_log_probs = self.collect_trajectories(num_steps)
            total_steps_collected += len(rewards)

           # print(f" Collected {total_steps_collected}/{timesteps} steps")

            # Compute advantages
            advantages = self.compute_advantages(rewards, values)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # Compute returns
            returns = self.compute_returns(rewards)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Update policy
            policy_loss,entropy_val = self.update_policy(observations, actions, advantages, old_log_probs)
            policy_losses.append(policy_loss)
            entropy_values.append(entropy_val)
            entropy_steps.append(total_steps_collected)

            # Update value function
            value_loss = self.update_value_function(observations, returns)
            value_losses.append(value_loss)

            # Track rewards for learning curve
            for reward in rewards:
                episode_reward += reward
                if reward == rewards[-1]:
                    episode_rewards.append(episode_reward)
                    episode_reward = 0  # reset after episode ends

            # Optional: log every ~10 rounds
            if total_steps_collected % (num_steps * 2) == 0:
                recent_rewards = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
                avg_reward = np.mean([r if not isinstance(r, torch.Tensor) else r.item() for r in recent_rewards])
                # print(f" PPO Progress — Steps: {total_steps_collected}, Avg Reward (last 10): {avg_reward:.2f}")

        # === Summary and Save ===
        total_time = time.time() - start_time
        print(f"\nPPO training completed in {total_time / 60:.2f} minutes.")

        entropy_df = pd.DataFrame({
            "Step": list(range(len(entropy_values))),
            "Entropy": entropy_values
        })
        entropy_df.to_csv(entropy_log_path, index=False)
        # Return average rewards and metrics
        rewards_cpu = [r.cpu().item() if isinstance(r, torch.Tensor) else r for r in episode_rewards]
        avg_reward = np.mean(rewards_cpu)

        # avg_reward = np.mean(episode_rewards)
        avg_policy_loss = np.mean(policy_losses)
        avg_value_loss = np.mean(value_losses)
        metrics = {"policy_loss": avg_policy_loss, "value_loss": avg_value_loss}
        
        return rewards_cpu, metrics

    def evaluate(self, episodes=30):
        self.policy.eval()
        rewards = []
        for i in range(episodes):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            max_steps = self.env.max_steps

            while not done and steps < max_steps:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
                with torch.no_grad():
                    action_probs = self.policy(obs_tensor)
                    flat_action = torch.argmax(action_probs).item()
                    action = self.unflatten_action(flat_action)
                # action = torch.argmax(action_probs).item()
                obs, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                steps += 1
                # if steps % 50 == 0:
                    # print(f"Step {steps}: current reward {total_reward:.2f}")
                done = terminated or truncated

            # print(f"[Eval Episode {i + 1}] Total Reward: {total_reward: .2f}")
            rewards.append(total_reward)
        avg_reward = np.mean(rewards)
        print(f"Average Reward over {episodes} episodes: {avg_reward:.2f}")   
        return avg_reward  # scalar avg reward

    def predict(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        action_probs = self.policy(obs_tensor)
        dist = Categorical(action_probs)
        # action = dist.sample().item()
        flat_action = dist.sample().item()

        # Unflatten the flat action into a structured list of channel selections
        # structured_action = np.unravel_index(flat_action, self.env.action_space.nvec)
        return self.unflatten_action(flat_action)

    # Pareto training data is generated from the Pareto Front each bentry is a tuple\
    
    def generate_pareto_training_data(self, B, SINR, I, P, R, final_population, filtered_indices):
        pareto_training_data = []
        obs_dim = self.input_dim
        action_dim = self.output_dim

        # Use the same averaged obs for all (B.shape[0] == 1)
        obs_template = np.concatenate([B[0], SINR[0], I[0], P[0], R[0]])

        for idx in range(len(filtered_indices)): #i in enumerate(filtered_indices):
            obs = obs_template.copy()
            if obs.shape[0] > obs_dim:
                obs = obs[:obs_dim]
            elif obs.shape[0] < obs_dim:
                obs = np.pad(obs, (0, obs_dim - obs.shape[0]))

            #binary_allocation = final_population[i].astype(int)
            binary_allocation = np.array(final_population[idx], dtype = int).flatten()    #.astype(int)

            required_len = len(self.env.action_space.nvec)
            if binary_allocation.shape[0] < required_len:
                binary_allocation = np.pad(binary_allocation, (0, required_len - binary_allocation.shape[0]))
            elif binary_allocation.shape[0] > required_len:
                binary_allocation = binary_allocation[:required_len]

            nvec = self.env.action_space.nvec
            binary_allocation = np.clip(binary_allocation, 0, nvec - 1)
            action = np.ravel_multi_index(binary_allocation, nvec)
            pareto_training_data.append((obs, action))

        return pareto_training_data

