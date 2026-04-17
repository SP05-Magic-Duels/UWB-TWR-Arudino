import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import os
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_FOLDER = "uwb_final_results"
CLEAN_DATA_PATH = 'without_noise.csv'
NOISY_DATA_PATH = 'with_noise.csv'

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def get_stats(df):
    """Calculates positioning and R2 metrics."""
    err_2d = np.sqrt((df['Calc_X'] - df['True_X'])**2 + (df['Calc_Y'] - df['True_Y'])**2)
    err_x = np.abs(df['Calc_X'] - df['True_X'])
    err_y = np.abs(df['Calc_Y'] - df['True_Y'])
    
    r2_x = r2_score(df['True_X'], df['Calc_X'])
    r2_y = r2_score(df['True_Y'], df['Calc_Y'])
    true_comb = np.concatenate([df['True_X'], df['True_Y']])
    calc_comb = np.concatenate([df['Calc_X'], df['Calc_Y']])
    r2_total = r2_score(true_comb, calc_comb)

    anchor_data = {}
    for i in range(4):
        p = f'A{i}'
        t_dist, r_dist, a_dist = df[f'{p}_True_Distance'], df[f'{p}_Raw_Dist'], df[f'{p}_AI_Dist']
        mse_r = mean_squared_error(t_dist, r_dist)
        mse_a = mean_squared_error(t_dist, a_dist)
        var_r, var_a = np.var(r_dist - t_dist), np.var(a_dist - t_dist)
        anchor_data[p] = {
            'raw': (mse_r, np.sqrt(mse_r), var_r, r2_score(t_dist, r_dist), np.std(r_dist - t_dist)),
            'ai': (mse_a, np.sqrt(mse_a), var_a, r2_score(t_dist, a_dist), np.std(a_dist - t_dist)),
            'err_imp': (np.sqrt(mse_r) - np.sqrt(mse_a)) / np.sqrt(mse_r) * 100,
            'var_imp': (var_r - var_a) / var_r * 100 if var_r != 0 else 0
        }
    return {'mean_2d': err_2d.mean(), 'median_2d': err_2d.median(), 'ce90': np.percentile(err_2d, 90),
            'max_2d': err_2d.max(), 'mean_x': err_x.mean(), 'mean_y': err_y.mean(),
            'r2_x': r2_x, 'r2_y': r2_y, 'r2_total': r2_total, 'anchors': anchor_data, 'samples': len(df)}

def write_report(s_c, s_n):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    report_path = os.path.join(OUTPUT_FOLDER, 'ultimate_stats_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*75 + "\n")
        f.write(f" UWB ULTIMATE SYSTEM EVALUATION - {now}\n")
        f.write("="*75 + "\n\n")
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
            f.write("--- 2. PER-ANCHOR DISTANCE PIPELINE (Raw vs AI) ---\n")
            for i in range(4):
                p, a = f'A{i}', s['anchors'][f'A{i}']
                f.write(f" ANCHOR {p}:\n")
                f.write(f"  RAW -> MSE: {a['raw'][0]:.4f} | RMSE: {a['raw'][1]:.4f}m | Variance: {a['raw'][2]:.4f} | R-Squared: {a['raw'][3]:.4f}\n")
                f.write(f"  AI  -> MSE: {a['ai'][0]:.4f} | RMSE: {a['ai'][1]:.4f}m | Variance: {a['ai'][2]:.4f} | R-Squared: {a['ai'][3]:.4f}\n")
                f.write(f"  > Average Noise (STD): RAW = {a['raw'][4]:.4f}m --> AI = {a['ai'][4]:.4f}m\n")
                f.write(f"  > Improvement: Error reduced by {a['err_imp']:.2f}% | Variance reduced by {a['var_imp']:.2f}%\n\n")
        f.write("="*75 + "\n >>> OVERALL COMPARISON: BASELINE VS. NOISE <<<\n" + "="*75 + "\n\n")
        m_diff = (s_n['mean_2d'] - s_c['mean_2d']) / s_c['mean_2d'] * 100
        c_diff = (s_n['ce90'] - s_c['ce90']) / s_c['ce90'] * 100
        f.write(f" Mean 2D Error:  Baseline = {s_c['mean_2d']:.4f}m | Noisy = {s_n['mean_2d']:.4f}m  --> {abs(m_diff):.2f}% ({'Better' if m_diff < 0 else 'Worse'})\n")
        f.write(f" CE90 (90% Acc): Baseline = {s_c['ce90']:.4f}m | Noisy = {s_n['ce90']:.4f}m  --> {abs(c_diff):.2f}% ({'Better' if c_diff < 0 else 'Worse'})\n")

def create_scatter_plots(df_c, df_n):
    for df, title, filename in [(df_c, "LOS", "LOS_spatial_plot.png"), (df_n, "NLOS", "NLOS_spatial_plot.png")]:
        plt.clf()
        plt.scatter(df['Calc_X'], df['Calc_Y'], color='blue', s=20, alpha=0.5, label='Calculated Position')
        plt.scatter(df['True_X'], df['True_Y'], color='red', s=40, marker='o', label='Ground Truth')
        plt.title(f"Spatial Map: {title} Environment"); plt.xlabel("X (m)"); plt.ylabel("Y (m)")
        plt.legend(); plt.axis('equal'); plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, filename), dpi=300)

df_c = pd.read_csv(CLEAN_DATA_PATH, skipinitialspace=True)
df_n = pd.read_csv(NOISY_DATA_PATH, skipinitialspace=True)
for df in [df_c, df_n]: df.columns = df.columns.str.strip()
s_c, s_n = get_stats(df_c), get_stats(df_n)
write_report(s_c, s_n)
create_scatter_plots(df_c, df_n)