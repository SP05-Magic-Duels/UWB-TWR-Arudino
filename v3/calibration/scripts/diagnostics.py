import numpy as np
import pandas as pd
import os
import re

# Constants
SPEED_OF_LIGHT = 299792458
TIME_RES = 1.0 / (128 * 499.2e6) # 15.65 picoseconds
KNOWN_DIST = 2.0955 # Your ground truth
DATA_DIR = "calibration/calibration_data" # Change to your data folder

def analyze_raw_data():
    summary = []
    
    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".csv"): continue
        
        # Extract Pair Name
        match = re.search(r'data_(\w+)_(\w+)\.csv', filename)
        pair_name = f"{match.group(1)}-{match.group(2)}" if match else filename
        
        # Load Data
        df = pd.read_csv(os.path.join(DATA_DIR, filename), header=None)
        # Assuming format: TR1, TR2, TP1, TP2
        tr1, tr2, tp1, tp2 = np.array(df[0].values[1:], float), np.array(df[1].values[1:], float), \
                                np.array(df[2].values[1:], float), np.array(df[3].values[1:], float)
        
        # 1. Calculate Raw ToF (No Antenna Delay subtracted)
        # ADS-TWR Formula
        num = (tr1 * tr2) - (tp1 * tp2)
        den = tr1 + tr2 + tp1 + tp2
        tof_ticks = num / den
        
        # 2. Calculate Distance
        dist_raw = tof_ticks * TIME_RES * SPEED_OF_LIGHT
        
        # 3. Stats
        tof_mean = np.mean(tof_ticks)
        tof_var = np.var(tof_ticks)
        dist_mean = np.mean(dist_raw)
        dist_bias = dist_mean - KNOWN_DIST
        dist_var = np.var(dist_raw)
        
        # "Expected" Total Antenna Delay in ticks (Total for the pair)
        implied_delay = dist_bias / (TIME_RES * SPEED_OF_LIGHT)
        
        summary.append({
            "Pair": pair_name,
            "Samples": len(df),
            "ToF_Var (ticks²)": round(tof_var, 2),
            "Dist_Mean (m)": round(dist_mean, 3),
            "Dist_Bias (m)": round(dist_bias, 3),
            "Dist_Var (m²)": f"{dist_var:.6f}",
            "Implied_Total_Delay": int(implied_delay)
        })

    result_df = pd.DataFrame(summary)
    print("\n--- RAW DATA DIAGNOSTICS (Before Calibration) ---")
    print(result_df.to_string(index=False))
    return result_df

if __name__ == "__main__":
    analyze_raw_data()