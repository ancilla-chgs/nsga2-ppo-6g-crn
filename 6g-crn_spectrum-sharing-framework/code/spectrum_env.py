import gym
import numpy as np
import gymnasium as gym
import pandas as pd
from gym.spaces import Box, MultiDiscrete


class SpectrumEnv(gym.Env):

    def __init__(self, data, num_channels=5, num_sus=3, max_steps=4000):
        super(SpectrumEnv, self).__init__()

        self.data = data.copy()
        # self.data.columns = [c.lower() for c in self.data.columns]

        self.num_channels = num_channels
        self.num_sus = num_sus
        self.max_steps = max_steps

        self.current_step = 0
        self.time_slots = sorted(self.data['time_slot'].unique())

        # Observation space: [PU_active, SU_request, channel_gain, SNR, interference_level] per channel
        self.obs_dim = 5 * num_channels
        self.observation_space = Box(low=0, high=1, shape=(self.obs_dim,), dtype=np.float32)

        # Action space: each SU chooses a channel (0 to num_channels-1)
        self.action_space = MultiDiscrete([num_channels] * num_sus)

        self.state = None

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        super().reset(seed=seed)

        max_start_index = len(self.time_slots) - self.max_steps

        if max_start_index <=0:
            start_idx = 0  # fallback to start
        else:
            start_idx = np.random.randint(0, max_start_index)
            
        self.episode_time_slots = self.time_slots[start_idx : start_idx + self.max_steps]

        self.state = self._get_state()
        return self.state, {}


    def step(self, action):
        qos_threshold = 280.0  ## Mbps QoS requirement
        qos_violations = 0
        qos_checks = 0
        pu_collisions = 0

        if self.current_step >= self.max_steps:
            if not hasattr(self, "_warned"):
                print(f" self.current_step={self.current_step}, max={len(self.time_slots) - 1}")
                self._warned = True
            obs = np.zeros(self.obs_dim, dtype=np.float32)
            reward = 0.0
            terminated = True
            truncated = True
            info = {
                "throughput": 0.0,
                "interference": 0.0,
                "energy_consumption": 0.0,
                "QoS_violated": 0.0,
                "pu_collisions": 0
            }
            return obs, reward, terminated, truncated, info

       
        current_time = self.episode_time_slots[self.current_step]
        frame = self.data[self.data['time_slot'] == current_time]

        throughput = 0
        interference_penalty = 0
        energy_cost = 0

        for i, ch in enumerate(action):
            su_row = frame[(frame['channel_id'] == ch) & (frame['SU_id'] == i)]

            if not su_row.empty and su_row['SU_request'].values[0] == 1:
                print(f"SU {i} requested channel {ch}? {su_row['SU_request'].values[0]}")
                pu_active = su_row['PU_active'].values[0]
                gain = su_row['channel_gain'].values[0]
                snr = su_row['SINR'].values[0]
                interference = su_row['interference_level'].values[0]
                power = su_row['transmit_power'].values[0]
                throughput_val = su_row['throughput'].values[0]
                energy = su_row['energy_consumption'].values[0]

                if pu_active:
                    interference_penalty += interference
                    pu_collisions += 1  # <-- Count this as a PU collision

                throughput += throughput_val
                energy_cost += energy

                # QoS check
                qos_checks += 1
                print(f"[STEP {self.current_step}] SU {i}, Channel {ch} → Throughput: {throughput_val:.2f} Mbps (QoS Threshold: {qos_threshold})")

                if throughput_val < qos_threshold:
                    print(f"❌ QoS Violation Detected! Throughput {throughput_val:.2f} < {qos_threshold}")
                    qos_violations += 1
                    #print(f"[DEBUG] SU {i} on CH {ch}: throughput = {throughput_val:.4f} Mbps; QoS Threshold = {qos_threshold} Mbps")


        qos_violation_rate = qos_violations / qos_checks if qos_checks > 0 else 0.0
        print(f"[STEP {self.current_step}] QoS Checks: {qos_checks}, Violations: {qos_violations}, Violation Rate: {qos_violation_rate:.2f}")


        # Normalize metrics
        norm_throughput = throughput / (self.num_sus * self.num_channels)
        norm_interference = interference_penalty / self.num_sus
        norm_energy = energy_cost / self.num_sus

        reward = norm_throughput - 0.5 * norm_interference - 0.2 * norm_energy

        self.current_step += 1
        terminated = self.current_step >= len(self.time_slots) or self.current_step >= self.max_steps
        truncated = False

        self.state = self._get_state()

        info = {
            "throughput": norm_throughput,
            "interference": norm_interference,
            "energy_consumption": norm_energy,
            "QoS_violated": qos_violation_rate,
            "pu_collisions": pu_collisions
        }

        return self.state, reward, terminated, truncated, info

    def _get_state(self):
        if self.current_step >= len(self.episode_time_slots):
            self.current_step = len(self.episode_time_slots) - 1 
            return np.zeros(self.obs_dim, dtype=np.float32)

        current_time = self.episode_time_slots[self.current_step]
        frame = self.data[self.data['time_slot'] == current_time]

        
        obs = []
        for ch in range(self.num_channels):
            row = frame[frame['channel_id'] == ch]
            if not row.empty:
                obs.extend([
                    row['PU_active'].values[0],
                    row['SU_request'].values[0],
                    row['channel_gain'].values[0],
                    row['SINR'].values[0],
                    row['interference_level'].values[0]
                ])
            else:
                obs.extend([0, 0, 0, 0, 0])

        return np.array(obs, dtype=np.float32)

    def get_nsga2_inputs(self):
        # Filter for all time slots up to max_steps (or full data)
        frame = self.data[self.data['time_slot'].isin(self.time_slots[:self.max_steps])]

        # Sort by time_slot and channel_id to keep things aligned
        frame = frame.sort_values(by=["time_slot", "channel_id"])

        # Extract required fields as arrays
        B = frame['channel_gain'].values.reshape(-1, self.num_channels)
        SINR = frame['SINR'].values.reshape(-1, self.num_channels)
        I = frame['interference_level'].values.reshape(-1, self.num_channels)
        P = frame['transmit_power'].values.reshape(-1, self.num_channels)
        R = frame['throughput'].values.reshape(-1, self.num_channels)

        return B, SINR, I, P, R

    def get_averaged_nsga2_inputs(self):
        frame = self.data[self.data['time_slot'].isin(self.episode_time_slots[:self.max_steps])]
        # Ensure frame length is divisible by num_channels
        total_rows = (frame.shape[0] // self.num_channels) * self.num_channels
        frame = frame.iloc[:total_rows]

        frame = frame.sort_values(by=["time_slot", "channel_id"])

        B = frame['channel_gain'].values.reshape(-1, self.num_channels)
        SINR = frame['SINR'].values.reshape(-1, self.num_channels)
        I = frame['interference_level'].values.reshape(-1, self.num_channels)
        P = frame['transmit_power'].values.reshape(-1, self.num_channels)
        R = frame['throughput'].values.reshape(-1, self.num_channels)

        # Average across time steps → shape (1, num_channels)
        B_avg = B.mean(axis=0, keepdims=True)
        SINR_avg = SINR.mean(axis=0, keepdims=True)
        I_avg = I.mean(axis=0, keepdims=True)
        P_avg = P.mean(axis=0, keepdims=True)
        R_avg = R.mean(axis=0, keepdims=True)

        return B_avg, SINR_avg, I_avg, P_avg, R_avg
   