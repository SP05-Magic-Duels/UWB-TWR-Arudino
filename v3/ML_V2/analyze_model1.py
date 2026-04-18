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
OUTPUT_FOLDER = "3D_DATA_MODELS/EVALUATION/large-room_8A_100sam"
CLEAN_DATA_PATH = '3D_DATA_MODELS/DATA/eval_WOUT_NOISE_large_room_8A_100sam.csv'
NOISY_DATA_PATH = '3D_DATA_MODELS/DATA/eval_WITH_NOISE_large_room_8A_100sam.csv'

# Set this to True to enable the outlier removal engine
REMOVE_OUTLIERS = True 
# ==========================================

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def filter_outliers(df, label=""):
    """
    Identifies and removes huge positional jumps using the IQR method.
    Calculates error first, then filters the whole dataframe based on that error.
    """
    if not REMOVE_OUTLIERS:
        return df

    # Calculate 2D Error for filtering
    err_2d = np.sqrt((df['Calc_X'] - df['True_X'])**2 + (df['Calc_Y'] - df['True_Y'])**2)
    
    Q1 = np.percentile(err_2d, 25)
    Q3 = np.percentile(err_2d, 75)
    IQR = Q3 - Q1
    
    # Define upper bound (Points > Q3 + 1.5*IQR are standard outliers)
    # We use 3.0 * IQR to only catch "Huge" outliers/glitches
    upper_bound = Q3 + (3.0 * IQR) 
    
    initial_count = len(df)
    df_filtered = df[err_2d <= upper_bound].copy()
    final_count = len(df_filtered)
    
    if initial_count != final_count:
        print(f" [IQR Engine] Removed {initial_count - final_count} outliers from {label} (Threshold: {upper_bound:.2f}m)")
    
    return df_filtered.reset_index(drop=True)

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
    
    calc_z = df['Calc_Z'] if 'Calc_Z' in df.columns else np.zeros(len(df))
    true_z = df['True_Z'] if 'True_Z' in df.columns else np.zeros(len(df))
    err_z = np.abs(calc_z - true_z)
    
    r2_x = r2_score(df['True_X'], df['Calc_X'])
    r2_y = r2_score(df['True_Y'], df['Calc_Y'])
    
    if np.var(true_z) == 0:
        r2_z = 0.0 
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

def generate_ultimate_report(df_c, df_n, raw_counts):
    """Writes report and includes notes about filtered outliers."""
    s_c = get_overall_stats(df_c)
    s_n = get_overall_stats(df_n)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    num_anchors = len([col for col in df_c.columns if col.endswith('_True_Distance')])
    
    path = os.path.join(OUTPUT_FOLDER, 'ultimate_stats_report.txt')
    with open(path, 'w') as f:
        f.write("="*75 + "\n")
        f.write(f" UWB ULTIMATE SYSTEM EVALUATION (OUTLIERS REMOVED) - {now}\n")
        f.write("="*75 + "\n\n")

        f.write(f" [Dataset Integrity]\n")
        f.write(f"  Baseline: {s_c['samples']} used / {raw_counts['c']} original\n")
        f.write(f"  Noisy:    {s_n['samples']} used / {raw_counts['n']} original\n\n")

        f.write("-" * 75 + "\n")
        f.write(" >>> CLEAN BASELINE DATASET (NO NOISE) <<<\n")
        f.write("-" * 75 + "\n\n")
        
        f.write(f" Mean 2D Error:    {s_c['mean_2d']:.4f} meters\n")
        f.write(f" CE90 (90% Acc.):  {s_c['ce90']:.4f} meters\n")
        f.write(f" System R² (Total):{s_c['r2_comb']:.4f}\n\n")

        f.write("-" * 75 + "\n")
        f.write(" >>> NLOS DATASET (WITH NOISE) <<<\n")
        f.write("-" * 75 + "\n\n")

        f.write(f" Mean 2D Error:    {s_n['mean_2d']:.4f} meters\n")
        f.write(f" CE90 (90% Acc.):  {s_n['ce90']:.4f} meters\n")
        f.write(f" System R² (Total):{s_n['r2_comb']:.4f}\n\n")

    print(f"[SUCCESS] Ultimate report generated at {path}")

def generate_visuals(df_c, df_n):
    plt.style.use('ggplot')
    
    # Calculate relative time in seconds (assuming Timestamp is in seconds or similar)
    t_n = df_n['Timestamp'] - df_n['Timestamp'].iloc[0]
    
    # ==========================================
    # 01. Temporal XYZ Stacked Plot
    # ==========================================
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Temporal Tracking Accuracy: Calculated vs. True Position', fontsize=16, fontweight='bold')

    # X-Axis Subplot
    axes[0].plot(t_n, df_n['Calc_X'], label='Calculated X', color='blue', alpha=0.6)
    axes[0].plot(t_n, df_n['True_X'], label='True X (Ground Truth)', color='black', linestyle='--', linewidth=2)
    axes[0].set_ylabel('X Position (m)', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Y-Axis Subplot
    axes[1].plot(t_n, df_n['Calc_Y'], label='Calculated Y', color='orange', alpha=0.7)
    axes[1].plot(t_n, df_n['True_Y'], label='True Y (Ground Truth)', color='black', linestyle='--', linewidth=2)
    axes[1].set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # Z-Axis Subplot
    calc_z = df_n['Calc_Z'] if 'Calc_Z' in df_n.columns else np.zeros(len(df_n))
    true_z = df_n['True_Z'] if 'True_Z' in df_n.columns else np.zeros(len(df_n))
    axes[2].plot(t_n, calc_z, label='Calculated Z', color='forestgreen', alpha=0.7)
    axes[2].plot(t_n, true_z, label='True Z (Ground Truth)', color='black', linestyle='--', linewidth=2)
    axes[2].set_ylabel('Z Position (m)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time Elapsed (s)', fontsize=12, fontweight='bold')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    plt.savefig(os.path.join(OUTPUT_FOLDER, '06_temporal_xyz_stacked.png'), dpi=300)
    plt.close()

    # ==========================================
    # 04. Spatial Scatter Map (Unchanged logic, added labels)
    # ==========================================
    plt.figure(figsize=(10, 10))
    plt.title('2D Spatial Error Distribution: Clean vs. Noisy Data', fontsize=14)
    plt.scatter(df_c['Calc_X'], df_c['Calc_Y'], color='forestgreen', alpha=0.2, s=15, label='Clean Samples')
    plt.scatter(df_n['Calc_X'], df_n['Calc_Y'], color='blue', alpha=0.2, s=15, label='Noisy Samples')
    
    true_pos = df_n[['True_X', 'True_Y']].drop_duplicates()
    plt.scatter(true_pos['True_X'], true_pos['True_Y'], marker='*', s=300, color='red', 
                edgecolor='black', label='Target Ground Truth', zorder=5)
    
    plt.xlabel('X Coordinate (meters)', fontsize=12)
    plt.ylabel('Y Coordinate (meters)', fontsize=12)
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, shadow=True)
    plt.savefig(os.path.join(OUTPUT_FOLDER, '05_spatial_scatter_map.png'), dpi=300)
    plt.close()

def main():
    df_n_raw = pd.read_csv(NOISY_DATA_PATH, skipinitialspace=True)
    df_c_raw = pd.read_csv(CLEAN_DATA_PATH, skipinitialspace=True)
    
    raw_counts = {'n': len(df_n_raw), 'c': len(df_c_raw)}

    for d in [df_n_raw, df_c_raw]: d.columns = d.columns.str.strip()
    
    # APPLY FILTERING
    df_n = filter_outliers(df_n_raw, "Noisy Dataset")
    df_c = filter_outliers(df_c_raw, "Clean Dataset")
    
    generate_ultimate_report(df_c, df_n, raw_counts)
    generate_visuals(df_c, df_n)
    print(f"\n[DONE] All analysis files are in '{OUTPUT_FOLDER}/'")

if __name__ == "__main__":
    main()