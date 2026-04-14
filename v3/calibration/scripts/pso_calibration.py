import numpy as np
import pandas as pd
import os
import re
from scipy.optimize import lsq_linear

# --- CONSTANTS ---
# Use the precise speed of light in air
SPEED_OF_LIGHT = 299792458 / 1.00027 
TIME_RES = 1.0 / (128 * 499.2e6) 
KNOWN_DISTANCE = 2.0955 # 82.5 inches in meters
DATA_DIR = "calibration/calibration_data"

# PHYSICAL BOUNDS (Ticks for a single antenna)
# We expand these to see where the math naturally wants to go.
LB = 14000 
UB = 18000

def calculate_fitness(candidate_total_delay, timestamps, real_dist):
    tr1, tr2, tp1, tp2 = timestamps[:,0], timestamps[:,1], timestamps[:,2], timestamps[:,3]
    
    # 1. Raw ToF in ticks
    num = (tr1 * tr2) - (tp1 * tp2)
    den = tr1 + tr2 + tp1 + tp2
    raw_tof_ticks = num / den
    
    # 2. Subtract the candidate delay
    # candidate_total_delay = (Antenna_A + Antenna_B)
    # We subtract the TOTAL delay from the raw round-trip calculation
    corrected_tof = raw_tof_ticks - candidate_total_delay
    
    dist_est = corrected_tof * TIME_RES * SPEED_OF_LIGHT
    return np.mean(np.abs(real_dist - dist_est))

def run_pso(data, real_dist):
    particles = 100
    iterations = 300
    # Searching for a SUM (approx 16450 * 2 = 32900)
    pos = np.random.uniform(32000, 34000, particles) 
    vel = np.zeros(particles)
    p_best_pos = np.copy(pos)
    p_best_val = np.array([float('inf')] * particles)
    g_best_pos, g_best_val = 0, float('inf')

    for _ in range(iterations):
        for i in range(particles):
            score = calculate_fitness(pos[i], data, real_dist)
            if score < p_best_val[i]:
                p_best_val[i], p_best_pos[i] = score, pos[i]
            if score < g_best_val:
                g_best_val, g_best_pos = score, pos[i]
        
        w, c1, c2 = 0.5, 1.4, 1.4
        vel = w*vel + c1*np.random.rand()*(p_best_pos - pos) + c2*np.random.rand()*(g_best_pos - pos)
        pos += vel
    return g_best_pos

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory {DATA_DIR} not found.")
        return

    pairs_found = []
    for filename in os.listdir(DATA_DIR):
        match = re.search(r'data_(\w+)_(\w+)\.csv', filename)
        if match:
            pairs_found.append({'dev1': match.group(1), 'dev2': match.group(2), 'path': os.path.join(DATA_DIR, filename)})

    if not pairs_found:
        print("No valid CSV files found.")
        return

    device_names = sorted(list(set([p['dev1'] for p in pairs_found] + [p['dev2'] for p in pairs_found])))
    name_to_idx = {name: i for i, name in enumerate(device_names)}
    num_devices = len(device_names)
    num_pairs = len(pairs_found)

    print(f"Calibrating {num_devices} devices across {num_pairs} pairs...")

    total_delays_b = []
    for pair in pairs_found:
        data = pd.read_csv(pair['path']).values
        best_total = run_pso(data, KNOWN_DISTANCE)
        print(f"DEBUG: Pair {pair['dev1']}-{pair['dev2']} Sum: {best_total:.2f}")
        total_delays_b.append(best_total)

    # Solve System
    A = np.zeros((len(pairs_found), len(device_names)))
    for i, pair in enumerate(pairs_found):
        A[i, name_to_idx[pair['dev1']]] = 1
        A[i, name_to_idx[pair['dev2']]] = 1

    # Solve with physical bounds for ONE antenna
    res = lsq_linear(A, total_delays_b, bounds=(LB, UB))
    indiv_delays = res.x

    results_df = pd.DataFrame({
        'Device': device_names,
        'Antenna_Delay_Ticks': indiv_delays.round(2),
        'Register_Value_Hex': [hex(int(d)) for d in indiv_delays]
    })

    print("\n--- Corrected Results ---")
    print(results_df)

if __name__ == "__main__":
    main()