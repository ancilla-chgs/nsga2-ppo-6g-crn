# SINR calculations
import numpy as np

def compute_sinr(P_i, G_i, P_k, G_k, N0):
    interference = np.sum(P_k * G_k)
    sinr = (P_i * G_i) / (N0 + interference)
    return sinr
