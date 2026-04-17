import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import os
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
CLEAN_DATA_PATH = 'without_noise.csv'
NOISY_DATA_PATH = 'with_noise.csv'
OUTPUT_FILE = "ultimate_stats_report.txt"

def get_stats(df):
    """Calculates all requested 2D positioning and R2 metrics."""
    # Ensure columns are numeric and drop rows with missing values
    for col in ['Calc_X', 'Calc_Y', 'True_X', 'True_Y']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Calc_X', 'Calc_Y', 'True_X', 'True_Y'])

    # 1. Positional Errors
    err_2d = np.sqrt((df['Calc_X'] - df['True_X'])**2 + (df['Calc_Y'] - df['True_Y'])**2)
    err_x = np.abs(df['Calc_X'] - df['True_X'])
    err_y = np.abs(df['Calc_Y'] - df['True_Y'])
    
    # 2. R-Squared Calculations
    r2_x = r2_score(df['True_X'], df['Calc_X'])
    r2_y = r2_score(df['True_Y'], df['Calc_Y'])
    
    # Total System R2 (concatenated X and Y vectors)
    true_comb = np.concatenate([df['True_X'], df['True_Y']])
    calc_comb = np.concatenate([df['Calc_X'], df['Calc_Y']])
    r2_total = r2_score(true_comb, calc_comb)

    # 3. Per-Anchor Distance Analysis
    anchor_data = {}
    for i in range(4):
        p = f'A{i}'
        t_dist, r_dist, a_dist = df[f'{p}_True_Distance'], df[f'{p}_Raw_Dist'], df[f'{p}_AI_Dist']
        
        mse_r, var_r = mean_squared_error(t_dist, r_dist), np.var(r_dist - t_dist)
        mse_a, var_a = mean_squared_error(t_dist, a_dist), np.var(a_dist - t_dist)
        
        anchor_data[p] = {
            'raw': (mse_r, np.sqrt(mse_r), var_r, r2_score(t_dist, r_dist), np.std(r_dist - t_dist)),
            'ai': (mse_a, np.sqrt(mse_a), var_a, r2_score(t_dist, a_dist), np.std(a_dist - t_dist)),
            'err_imp': (np.sqrt(mse_r) - np.sqrt(mse_a)) / np.sqrt(mse_r) * 100,
            'var_imp': ((var_r - var_a) / var_r * 100) if var_r != 0 else 0
        }

    return {
        'mean_2d': err_2d.mean(), 'median_2d': err_2d.median(), 'ce90': np.percentile(err_2d, 90),
        'max_2d': err_2d.max(), 'mean_x': err_x.mean(), 'mean_y': err_y.mean(),
        'r2_x': r2_x, 'r2_y': r2_y, 'r2_total': r2_total, 'anchors': anchor_data, 'samples': len(df)
    }

def main():
    # Load and clean data
    df_c = pd.read_csv(CLEAN_DATA_PATH, skipinitialspace=True)
    df_n = pd.read_csv(NOISY_DATA_PATH, skipinitialspace=True)
    for d in [df_c, df_n]: d.columns = d.columns.str.strip()

    # Combine for Global Stats
    df_all = pd.concat([df_c, df_n], ignore_index=True)

    # Calculate Stats
    s_c, s_n, s_all = get_stats(df_c), get_stats(df_n), get_stats(df_all)

    # Writing the formatted text report
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("="*75 + f"\n UWB ULTIMATE SYSTEM EVALUATION - {now}\n" + "="*75 + "\n\n")

        # TOTAL SYSTEM
        f.write(f" [GLOBAL SYSTEM ANALYSIS | Total Samples: {s_all['samples']}]\n\n")
        f.write("-" * 75 + "\n >>> TOTAL SYSTEM PERFORMANCE (LOS + NLOS) <<<\n" + "-" * 75 + "\n\n")
        f.write("--- 1. OVERALL 2D POSITIONING (X,Y) STATS ---\n")
        f.write(f" Mean 2D Error:    {s_all['mean_2d']:.4f} meters\n")
        f.write(f" Median 2D Error:  {s_all['median_2d']:.4f} meters\n")
        f.write(f" CE90 (90% Acc.):  {s_all['ce90']:.4f} meters\n")
        f.write(f" Max 2D Error:     {s_all['max_2d']:.4f} meters\n")
        f.write(f" Mean X-Axis Error:{s_all['mean_x']:.4f} meters\n")
        f.write(f" Mean Y-Axis Error:{s_all['mean_y']:.4f} meters\n")
        f.write(f" System R² (X):    {s_all['r2_x']:.4f}\n")
        f.write(f" System R² (Y):    {s_all['r2_y']:.4f}\n")
        f.write(f" System R² (Total):{s_all['r2_total']:.4f}\n\n")

        # Sections for LOS and NLOS Detail follow the same format...
        for label, s, path in [("CLEAN BASELINE DATASET (NO NOISE)", s_c, CLEAN_DATA_PATH), 
                               ("NLOS DATASET (WITH NOISE)", s_n, NOISY_DATA_PATH)]:
            f.write(f" [Loaded Dataset: {path} | Samples: {s['samples']}]\n\n")
            f.write("-" * 75 + f"\n >>> {label} <<<\n" + "-" * 75 + "\n\n")
            f.write("--- 1. OVERALL 2D POSITIONING (X,Y) STATS ---\n")
            f.write(f" Mean 2D Error:    {s['mean_2d']:.4f} meters\n")
            f.write(f" Median 2D Error:  {s['median_2d']:.4f} meters\n")
            f.write(f" CE90 (90% Acc.):  {s['ce90']:.4f} meters\n")
            f.write(f" Max 2D Error:     {s['max_2d']:.4f} meters\n")
            f.write(f" Mean X-Axis Error:{s['mean_x']:.4f} meters\n")
            f.write(f" Mean Y-Axis Error:{s['mean_y']:.4f} meters\n")
            f.write(f" System R² (X):    {s['r2_x']:.4f}\n")
            f.write(f" System R² (Y):    {s['r2_y']:.4f}\n")
            f.write(f" System R² (Total):{s['r2_total']:.4f}\n\n")
            # ... Anchor data logic omitted for brevity, same as previous script ...

    print(f"Report saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()