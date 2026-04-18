import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error

# ==========================================
# CONFIGURATION - ADJUST THESE VARIABLES
# ==========================================
OUTPUT_FOLDER = "3D_DATA_MODELS/EVALUATION/large-room"  # Folder for all outputs
CLEAN_DATA_PATH = '3D_DATA_MODELS/DATA/eval_WOUT_NOISE_large_room_8A_100sam.csv'
NOISY_DATA_PATH = '3D_DATA_MODELS/DATA/eval_WITH_NOISE_large_room_8A_100sam.csv'
# ==========================================

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def get_anchor_stats(df, anchor_prefix):
    """Calculates MSE, RMSE, Var, R2, and Noise for a specific anchor."""
    true_dist = df[f'{anchor_prefix}_True_Distance']
    raw_dist = df[f'{anchor_prefix}_Raw_Dist']
    ai_dist = df[f'{anchor_prefix}_AI_Dist']

    def metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        var = np.var(y_pred - y_true)
        r2 = r2_score(y_true, y_pred)
        std = np.std(y_pred - y_true)
        return mse, rmse, var, r2, std

    m_raw = metrics(true_dist, raw_dist)
    m_ai = metrics(true_dist, ai_dist)
    
    return {
        'raw': m_raw, 'ai': m_ai,
        'err_imp': (m_raw[1] - m_ai[1]) / m_raw[1] * 100,
        'var_imp': (m_raw[2] - m_ai[2]) / m_raw[2] * 100
    }

def get_overall_stats(df):
    """Calculates 2D/3D error stats and system-level R2 values."""
    err_2d = np.sqrt((df['Calc_X'] - df['True_X'])**2 + (df['Calc_Y'] - df['True_Y'])**2)
    err_x = np.abs(df['Calc_X'] - df['True_X'])
    err_y = np.abs(df['Calc_Y'] - df['True_Y'])
    
    # Z-Axis Calculations (with fallback to 0 if column is missing)
    calc_z = df['Calc_Z'] if 'Calc_Z' in df.columns else np.zeros(len(df))
    true_z = df['True_Z'] if 'True_Z' in df.columns else np.zeros(len(df))
    err_z = np.abs(calc_z - true_z)
    
    # R2 Calculations
    r2_x = r2_score(df['True_X'], df['Calc_X'])
    r2_y = r2_score(df['True_Y'], df['Calc_Y'])
    
    # Handle Z R2 specifically (r2_score can throw warnings/errors if True_Z is all zeros)
    if np.var(true_z) == 0:
        r2_z = 0.0 # If the tag never moved on the Z axis, R2 is undefined/0
    else:
        r2_z = r2_score(true_z, calc_z)
        
    true_comb = np.concatenate([df['True_X'], df['True_Y'], true_z])
    calc_comb = np.concatenate([df['Calc_X'], df['Calc_Y'], calc_z])
    r2_comb = r2_score(true_comb, calc_comb)

    return {
        'mean_2d': err_2d.mean(), 'median_2d': err_2d.median(),
        'ce90': np.percentile(err_2d, 90), 'max_2d': err_2d.max(),
        'mean_x': err_x.mean(), 'mean_y': err_y.mean(), 'mean_z': err_z.mean(),
        'r2_x': r2_x, 'r2_y': r2_y, 'r2_z': r2_z, 'r2_comb': r2_comb,
        'samples': len(df)
    }

def generate_ultimate_report(df_c, df_n):
    """Writes the text report using the exact user-requested format."""
    s_c = get_overall_stats(df_c)
    s_n = get_overall_stats(df_n)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Dynamically find how many anchors exist in the dataset
    num_anchors = len([col for col in df_c.columns if col.endswith('_True_Distance')])
    
    path = os.path.join(OUTPUT_FOLDER, 'ultimate_stats_report.txt')
    with open(path, 'w') as f:
        f.write("="*75 + "\n")
        f.write(f" UWB ULTIMATE SYSTEM EVALUATION - {now}\n")
        f.write("="*75 + "\n\n")

        f.write(f" [Loaded Baseline Dataset: {CLEAN_DATA_PATH} | Samples: {s_c['samples']}]\n\n")
        f.write("-" * 75 + "\n")
        f.write(" >>> CLEAN BASELINE DATASET (NO NOISE) <<<\n")
        f.write("-" * 75 + "\n\n")
        
        f.write("--- 1. OVERALL 3D POSITIONING (X,Y,Z) STATS ---\n")
        f.write(f" Mean 2D Error:    {s_c['mean_2d']:.4f} meters\n")
        f.write(f" Median 2D Error:  {s_c['median_2d']:.4f} meters\n")
        f.write(f" CE90 (90% Acc.):  {s_c['ce90']:.4f} meters\n")
        f.write(f" Max 2D Error:     {s_c['max_2d']:.4f} meters\n")
        f.write(f" Mean X-Axis Error:{s_c['mean_x']:.4f} meters\n")
        f.write(f" Mean Y-Axis Error:{s_c['mean_y']:.4f} meters\n")
        f.write(f" Mean Z-Axis Error:{s_c['mean_z']:.4f} meters\n")
        f.write(f" System R² (X):    {s_c['r2_x']:.4f}\n")
        f.write(f" System R² (Y):    {s_c['r2_y']:.4f}\n")
        f.write(f" System R² (Z):    {s_c['r2_z']:.4f}\n")
        f.write(f" System R² (Total):{s_c['r2_comb']:.4f}\n\n")

        f.write("--- 2. PER-ANCHOR DISTANCE PIPELINE (Raw vs AI) ---\n")
        for i in range(num_anchors):
            a = get_anchor_stats(df_c, f'A{i}')
            f.write(f" ANCHOR A{i}:\n")
            f.write(f"  RAW -> MSE: {a['raw'][0]:.4f} | RMSE: {a['raw'][1]:.4f}m | Variance: {a['raw'][2]:.4f} | R-Squared: {a['raw'][3]:.4f}\n")
            f.write(f"  AI  -> MSE: {a['ai'][0]:.4f} | RMSE: {a['ai'][1]:.4f}m | Variance: {a['ai'][2]:.4f} | R-Squared: {a['ai'][3]:.4f}\n")
            f.write(f"  > Average Noise (STD): RAW = {a['raw'][4]:.4f}m --> AI = {a['ai'][4]:.4f}m\n")
            f.write(f"  > Improvement: Error reduced by {a['err_imp']:.2f}% | Variance reduced by {a['var_imp']:.2f}%\n\n")

        f.write(f" [Loaded Noisy Dataset: {NOISY_DATA_PATH} | Samples: {s_n['samples']}]\n\n")
        f.write("-" * 75 + "\n")
        f.write(" >>> NLOS DATASET (WITH NOISE) <<<\n")
        f.write("-" * 75 + "\n\n")

        f.write("--- 1. OVERALL 3D POSITIONING (X,Y,Z) STATS ---\n")
        f.write(f" Mean 2D Error:    {s_n['mean_2d']:.4f} meters\n")
        f.write(f" Median 2D Error:  {s_n['median_2d']:.4f} meters\n")
        f.write(f" CE90 (90% Acc.):  {s_n['ce90']:.4f} meters\n")
        f.write(f" Max 2D Error:     {s_n['max_2d']:.4f} meters\n")
        f.write(f" Mean X-Axis Error:{s_n['mean_x']:.4f} meters\n")
        f.write(f" Mean Y-Axis Error:{s_n['mean_y']:.4f} meters\n")
        f.write(f" Mean Z-Axis Error:{s_n['mean_z']:.4f} meters\n")
        f.write(f" System R² (X):    {s_n['r2_x']:.4f}\n")
        f.write(f" System R² (Y):    {s_n['r2_y']:.4f}\n")
        f.write(f" System R² (Z):    {s_n['r2_z']:.4f}\n")
        f.write(f" System R² (Total):{s_n['r2_comb']:.4f}\n\n")

        f.write("--- 2. PER-ANCHOR DISTANCE PIPELINE (Raw vs AI) ---\n")
        for i in range(num_anchors):
            a = get_anchor_stats(df_n, f'A{i}')
            f.write(f" ANCHOR A{i}:\n")
            f.write(f"  RAW -> MSE: {a['raw'][0]:.4f} | RMSE: {a['raw'][1]:.4f}m | Variance: {a['raw'][2]:.4f} | R-Squared: {a['raw'][3]:.4f}\n")
            f.write(f"  AI  -> MSE: {a['ai'][0]:.4f} | RMSE: {a['ai'][1]:.4f}m | Variance: {a['ai'][2]:.4f} | R-Squared: {a['ai'][3]:.4f}\n")
            f.write(f"  > Average Noise (STD): RAW = {a['raw'][4]:.4f}m --> AI = {a['ai'][4]:.4f}m\n")
            f.write(f"  > Improvement: Error reduced by {a['err_imp']:.2f}% | Variance reduced by {a['var_imp']:.2f}%\n\n")

        f.write("="*75 + "\n")
        f.write(" >>> OVERALL COMPARISON: BASELINE VS. NOISE <<<\n")
        f.write("="*75 + "\n\n")
        diff_mean = (s_n['mean_2d'] - s_c['mean_2d']) / s_c['mean_2d'] * 100
        diff_ce90 = (s_n['ce90'] - s_c['ce90']) / s_c['ce90'] * 100
        f.write(f" Mean 2D Error:  Baseline = {s_c['mean_2d']:.4f}m | Noisy = {s_n['mean_2d']:.4f}m  --> {abs(diff_mean):.2f}% ({'Better' if diff_mean < 0 else 'Worse'})\n")
        f.write(f" CE90 (90% Acc): Baseline = {s_c['ce90']:.4f}m | Noisy = {s_n['ce90']:.4f}m  --> {abs(diff_ce90):.2f}% ({'Better' if diff_ce90 < 0 else 'Worse'})\n")

    print(f"[SUCCESS] Ultimate report generated at {path}")


def generate_visuals(df_c, df_n):
    """Generates graphs matching the requested style and formatting."""
    # Set the global style to match the user's images (gray background, white gridlines)
    plt.style.use('ggplot')
    
    # Calculate time vectors (seconds relative to the start)
    t_n = df_n['Timestamp'] - df_n['Timestamp'].iloc[0]
    
    # ---------------------------------------------------------
    # 01. Temporal XYZ Stacked Plot
    # ---------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # X Position
    axes[0].plot(t_n, df_n['Calc_X'], label='Calculated X', color='blue', alpha=0.7)
    axes[0].plot(t_n, df_n['True_X'], label='True X', color='black', linestyle='--', linewidth=2)
    axes[0].set_title('X-Axis Trilateration Position Over Time')
    axes[0].set_ylabel('X Position (meters)')
    axes[0].legend(loc='upper right')

    # Y Position
    axes[1].plot(t_n, df_n['Calc_Y'], label='Calculated Y', color='orange', alpha=0.8)
    axes[1].plot(t_n, df_n['True_Y'], label='True Y', color='black', linestyle='--', linewidth=2)
    axes[1].set_title('Y-Axis Trilateration Position Over Time')
    axes[1].set_ylabel('Y Position (meters)')
    axes[1].legend(loc='upper right')

    # Z Position (Falls back to zero for calculated Z if missing from CSV)
    calc_z = df_n['Calc_Z'] if 'Calc_Z' in df_n.columns else np.zeros(len(df_n))
    true_z = df_n['True_Z'] if 'True_Z' in df_n.columns else np.zeros(len(df_n))
    axes[2].plot(t_n, calc_z, label='Calculated Z', color='forestgreen', alpha=0.8)
    axes[2].plot(t_n, true_z, label='True Z', color='black', linestyle='--', linewidth=2)
    axes[2].set_title('Z-Axis Trilateration Position Over Time')
    axes[2].set_ylabel('Z Position (meters)')
    axes[2].set_xlabel('Time (Seconds)')
    axes[2].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, '06_temporal_xyz_stacked.png'), dpi=300)
    plt.close()


    # ---------------------------------------------------------
    # 02. Temporal Overlay per Anchor (Dynamically count anchors)
    # ---------------------------------------------------------
    anchors = [col.split('_')[0] for col in df_n.columns if col.endswith('_Raw_Dist')]
    for a in anchors:
        plt.figure(figsize=(12, 5))
        plt.plot(t_n, df_n[f'{a}_Raw_Dist'], label='Raw Distance', color='#ff5a50', alpha=0.8) # Light red
        plt.plot(t_n, df_n[f'{a}_AI_Dist'], label='AI Corrected Distance', color='blue', alpha=0.8)
        plt.plot(t_n, df_n[f'{a}_True_Distance'], label='True Distance', color='black', linestyle='--', linewidth=2)
        
        plt.title(f'Temporal Overlay: Anchor {a[-1]} Raw vs AI Distance (Noisy Data)')
        plt.xlabel('Time (Seconds)')
        plt.ylabel('Distance (meters)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, f'03_temporal_overlay_{a}.png'), dpi=300)
        plt.close()


    # ---------------------------------------------------------
    # 03. Baseline Error Comparison Boxplot
    # ---------------------------------------------------------
    err_c = np.sqrt((df_c['Calc_X'] - df_c['True_X'])**2 + (df_c['Calc_Y'] - df_c['True_Y'])**2)
    err_n = np.sqrt((df_n['Calc_X'] - df_n['True_X'])**2 + (df_n['Calc_Y'] - df_n['True_Y'])**2)
    
    plt.figure(figsize=(10, 6))
    bplot = plt.boxplot([err_c, err_n], labels=['Baseline (Clean)', 'With Noise (NLOS)'], patch_artist=True)
    
    # Custom Box Colors
    colors = ['lightgreen', 'salmon']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    
    # Custom Median Line Color
    for median in bplot['medians']:
        median.set_color('steelblue')

    plt.title('Overall 2D Error Comparison: Baseline vs Noisy Environment')
    plt.ylabel('2D Positional Error (meters)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, '04_baseline_comparison.png'), dpi=300)
    plt.close()


    # ---------------------------------------------------------
    # 04. Spatial Scatter Map (Green vs Blue with Red Targets)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 10))
    
    # Plot Baseline & Noisy scatter points
    plt.scatter(df_c['Calc_X'], df_c['Calc_Y'], color='forestgreen', alpha=0.3, s=15, label='Clean Baseline Calc')
    plt.scatter(df_n['Calc_X'], df_n['Calc_Y'], color='blue', alpha=0.3, s=15, label='Noisy (NLOS) Calc')
    
    # Plot unique Target (True) Locations
    true_pos = df_n[['True_X', 'True_Y']].drop_duplicates()
    plt.scatter(true_pos['True_X'], true_pos['True_Y'], marker='*', s=400, color='red', edgecolor='black', label='True Location', zorder=5)
    
    plt.title('2D Positioning Scatter Map')
    plt.xlabel('X Coordinate (meters)')
    plt.ylabel('Y Coordinate (meters)')
    plt.legend()
    plt.axis('equal') # Ensures realistic spatial scaling (grid remains square)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, '05_spatial_scatter_map.png'), dpi=300)
    plt.close()


def main():
    df_n = pd.read_csv(NOISY_DATA_PATH, skipinitialspace=True)
    df_c = pd.read_csv(CLEAN_DATA_PATH, skipinitialspace=True)
    for d in [df_n, df_c]: d.columns = d.columns.str.strip()
    
    generate_ultimate_report(df_c, df_n)
    generate_visuals(df_c, df_n)
    print(f"\n[DONE] All analysis files are in '{OUTPUT_FOLDER}/'")

if __name__ == "__main__":
    main()