import numpy as np
import pandas as pd
import csv
import random
import math

'''
def generate_6g_crn_dataset(filename="crn_dataset_6g_with_pu_and_datarate.csv", num_time_steps=5000, num_channels=5):
    np.random.seed(42)

    frequency_bands = [3.5, 28, 60, 100, 140]  # GHz
    modulation_schemes = ['QPSK', '16QAM', '64QAM', '256QAM']
    modulation_map = {'QPSK': 2, '16QAM': 4, '64QAM': 6, '256QAM': 8}
    coding_rate = 0.8

    data = []

    for t in range(num_time_steps):
        for ch in range(num_channels):
            freq = frequency_bands[ch % len(frequency_bands)]
            bandwidth = np.random.uniform(10, 100)  # MHz
            tx_power_dbm = np.random.uniform(0, 30)
            tx_power_w = 10 ** ((tx_power_dbm - 30) / 10)

            channel_gain = np.random.rayleigh(scale=1.0)
            noise_power_dbm = -100
            noise_power_w = 10 ** ((noise_power_dbm - 30) / 10)

            snr_linear = (tx_power_w * channel_gain) / noise_power_w
            interference = np.random.uniform(0.01, 0.2) * tx_power_w
            sinr_linear = snr_linear / (1 + interference / noise_power_w)
            sinr_db = 10 * np.log10(sinr_linear)

            pu_active = np.random.choice([0, 1], p=[0.8, 0.2])
            su_request = np.random.choice([0, 1], p=[0.7, 0.3])
            modulation = np.random.choice(modulation_schemes)
            bits_per_symbol = modulation_map[modulation]

            if su_request and pu_active == 0:
                throughput = bandwidth * np.log2(1 + sinr_linear)
            else:
                throughput = 0

            energy_consumption = tx_power_w / (throughput + 1e-3)
            data_rate = bandwidth * bits_per_symbol * coding_rate

            data.append({
                "time_slot": t,
                "channel_id": ch,
                "Frequency_Band_GHz": freq,
                "Bandwidth_MHz": bandwidth,
                "Transmit_Power_dBm": tx_power_dbm,
                "Channel_Gain": channel_gain,
                "SNR_dB": 10 * np.log10(snr_linear),
                "SINR_dB": sinr_db,
                "PU_active": pu_active,
                "SU_request": su_request,
                "Interference_Level": interference,
                "Throughput_Mbps": throughput,
                "Energy_Consumption": energy_consumption,
                "Modulation": modulation,
                "Bits_per_Symbol": bits_per_symbol,
                "Data_Rate_Mbps": data_rate,
                "Noise_Power_dBm": noise_power_dbm
            })

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return df
'''
# Parameters
NUM_CHANNELS = 5
NUM_SUS = 3
TIME_SLOTS = 3000
OUTPUT_CSV = "ns3_crn_6g_dataset.csv"

def dbm_to_watt(dbm):
    return 10 ** ((dbm - 30) / 10)

def snr_to_capacity(sinr_linear, bandwidth_hz):
    return bandwidth_hz * math.log2(1 + sinr_linear) / 1e6  # in Mbps

def generate_dataset():
    with open(OUTPUT_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "time_slot", "channel_id", "pu_active", "su_request",
            "bandwidth_mhz", "channel_gain", "sinr_db",
            "interference_level", "transmit_power_dbm",
            "throughput_mbps", "data_rate_mbps", "energy_consumption"
        ])
        for t in range(TIME_SLOTS):
            for ch in range(NUM_CHANNELS):
                pu_active = random.randint(0, 1)
                for su in range(NUM_SUS):
                    su_request = random.randint(0, 1)
                    bandwidth_mhz = random.choice([5, 10, 15, 20])
                    bandwidth_hz = bandwidth_mhz * 1e6
                    channel_gain = round(random.uniform(0.1, 1.0), 4)
                    tx_power_dbm = random.uniform(10, 23)
                    tx_power_watt = dbm_to_watt(tx_power_dbm)
                    interference_mw = random.uniform(0, 5) if pu_active else 0
                    interference_watt = interference_mw / 1000
                    noise_watt = 1e-13
                    sinr_linear = (tx_power_watt * channel_gain) / (interference_watt + noise_watt)
                    sinr_db = 10 * math.log10(sinr_linear)
                    data_rate = snr_to_capacity(sinr_linear, bandwidth_hz)
                    energy = tx_power_watt * 0.001  # 1 ms

                    writer.writerow([
                        t, ch, pu_active, su_request,
                        bandwidth_mhz, channel_gain, round(sinr_db, 2),
                        round(interference_mw, 2), round(tx_power_dbm, 2),
                        round(data_rate, 4), round(data_rate, 4), round(energy, 6)
                    ])
    print(f"Dataset generated and saved to: {OUTPUT_CSV}")

generate_dataset()