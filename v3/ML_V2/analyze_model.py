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
OUTPUT_FOLDER = "uwb_analysis_results"  # Folder for all outputs
CLEAN_DATA_PATH = 'without_noise.csv'
NOISY_DATA_PATH = 'with_noise.csv'
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
    """Calculates 2D error stats and system-level R2 values."""
    err_2d = np.sqrt((df['Calc_X'] - df['True_X'])**2 + (df['Calc_Y'] - df['True_Y'])**2)
    err_x = np.abs(df['Calc_X'] - df['True_X'])
    err_y = np.abs(df['Calc_Y'] - df['True_Y'])
    
    # R2 Calculations
    r2_x = r2_score(df['True_X'], df['Calc_X'])
    r2_y = r2_score(df['True_Y'], df['Calc_Y'])
    true_comb = np.concatenate([df['True_X'], df['True_Y']])
    calc_comb = np.concatenate([df['Calc_X'], df['Calc_Y']])
    r2_comb = r2_score(true_comb, calc_comb)

    return {
        'mean_2d': err_2d.mean(), 'median_2d': err_2d.median(),
        'ce90': np.percentile(err_2d, 90), 'max_2d': err_2d.max(),
        'mean_x': err_x.mean(), 'mean_y': err_y.mean(),
        'r2_x': r2_x, 'r2_y': r2_y, 'r2_comb': r2_comb,
        'samples': len(df)
    }

def generate_ultimate_report(df_c, df_n):
    """Writes the text report using the exact user-requested format."""
    s_c = get_overall_stats(df_c)
    s_n = get_overall_stats(df_n)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    path = os.path.join(OUTPUT_FOLDER, 'ultimate_stats_report.txt')
    with open(path, 'w') as f:
        f.write("="*75 + "\n")
        f.write(f" UWB ULTIMATE SYSTEM EVALUATION - {now}\n")
        f.write("="*75 + "\n\n")

        f.write(f" [Loaded Baseline Dataset: {CLEAN_DATA_PATH} | Samples: {s_c['samples']}]\n\n")
        f.write("-" * 75 + "\n")
        f.write(" >>> CLEAN BASELINE DATASET (NO NOISE) <<<\n")
        f.write("-" * 75 + "\n\n")
        
        f.write("--- 1. OVERALL 2D POSITIONING (X,Y) STATS ---\n")
        f.write(f" Mean 2D Error:    {s_c['mean_2d']:.4f} meters\n")
        f.write(f" Median 2D Error:  {s_c['median_2d']:.4f} meters\n")
        f.write(f" CE90 (90% Acc.):  {s_c['ce90']:.4f} meters\n")
        f.write(f" Max 2D Error:     {s_c['max_2d']:.4f} meters\n")
        f.write(f" Mean X-Axis Error:{s_c['mean_x']:.4f} meters\n")
        f.write(f" Mean Y-Axis Error:{s_c['mean_y']:.4f} meters\n")
        f.write(f" System R² (X):    {s_c['r2_x']:.4f}\n")
        f.write(f" System R² (Y):    {s_c['r2_y']:.4f}\n")
        f.write(f" System R² (Total):{s_c['r2_comb']:.4f}\n\n")

        f.write("--- 2. PER-ANCHOR DISTANCE PIPELINE (Raw vs AI) ---\n")
        for i in range(4):
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

        f.write("--- 1. OVERALL 2D POSITIONING (X,Y) STATS ---\n")
        f.write(f" Mean 2D Error:    {s_n['mean_2d']:.4f} meters\n")
        f.write(f" Median 2D Error:  {s_n['median_2d']:.4f} meters\n")
        f.write(f" CE90 (90% Acc.):  {s_n['ce90']:.4f} meters\n")
        f.write(f" Max 2D Error:     {s_n['max_2d']:.4f} meters\n")
        f.write(f" Mean X-Axis Error:{s_n['mean_x']:.4f} meters\n")
        f.write(f" Mean Y-Axis Error:{s_n['mean_y']:.4f} meters\n")
        f.write(f" System R² (X):    {s_n['r2_x']:.4f}\n")
        f.write(f" System R² (Y):    {s_n['r2_y']:.4f}\n")
        f.write(f" System R² (Total):{s_n['r2_comb']:.4f}\n\n")

        f.write("--- 2. PER-ANCHOR DISTANCE PIPELINE (Raw vs AI) ---\n")
        for i in range(4):
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
    """Generates the 5 standard plots with the Green/Red heatmap color scheme."""
    sns.set_theme(style="whitegrid")
    
    # 01. Spatial Heatmap (Green/Red Scheme)
    plt.figure(figsize=(10, 10))
    sns.kdeplot(x=df_n['Calc_X'], y=df_n['Calc_Y'], fill=True, levels=12, cmap="Reds", alpha=0.4)
    plt.scatter(df_n['Calc_X'], df_n['Calc_Y'], color='crimson', s=10, alpha=0.2, label='NLOS (Noisy)')
    
    sns.kdeplot(x=df_c['Calc_X'], y=df_c['Calc_Y'], fill=True, levels=12, cmap="Greens", alpha=0.5)
    plt.scatter(df_c['Calc_X'], df_c['Calc_Y'], color='seagreen', s=10, alpha=0.2, label='LOS (Clean)')
    
    t_pos = df_n[['True_X', 'True_Y']].drop_duplicates()
    plt.scatter(t_pos['True_X'], t_pos['True_Y'], marker='*', s=500, color='gold', edgecolor='black', label='Target Truth', zorder=5)
    
    plt.title("Spatial Precision: LOS (Green) vs NLOS (Red)"); plt.legend(); plt.axis('equal'); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, '01_spatial_heatmap.png'), dpi=300); plt.close()

    # 02. Blockage Diagnostic (Keep Existing)
    plt.figure()
    all_err, all_gap = [], []
    for i in range(4):
        all_err.extend(np.abs(df_n[f'A{i}_Raw_Dist'] - df_n[f'A{i}_True_Distance']))
        all_gap.extend(df_n[f'A{i}_RX_Power'] - df_n[f'A{i}_FP_Power'])
    sns.regplot(x=all_gap, y=all_err, scatter_kws={'alpha':0.1}, line_kws={'color':'crimson'})
    plt.title("NLOS Diagnostic: Error vs Power Gap"); plt.xlabel("Power Gap (dB)"); plt.ylabel("Error (m)")
    plt.savefig(os.path.join(OUTPUT_FOLDER, '02_blockage_diagnostic.png'), dpi=300); plt.close()

    # 03. Signal Quality (Keep Existing)
    plt.figure()
    q_data = pd.concat([
        pd.concat([df_c[f'A{i}_Quality'] for i in range(4)]).to_frame('Q').assign(Env='LOS'),
        pd.concat([df_n[f'A{i}_Quality'] for i in range(4)]).to_frame('Q').assign(Env='NLOS')
    ])
    sns.boxplot(x='Env', y='Q', data=q_data, palette=['seagreen', 'crimson'])
    plt.title("Signal Quality Metrics"); plt.savefig(os.path.join(OUTPUT_FOLDER, '03_quality_boxplot.png'), dpi=300); plt.close()

    # 04. Pipeline Densities (Keep Existing)
    plt.figure()
    for col, lab, clr in [('Raw_Dist', 'Raw', 'gray'), ('AI_Dist', 'AI Corrected', 'blue'), ('Final_Filtered_Dist', 'Filtered', 'green')]:
        errs = pd.concat([np.abs(df_n[f'A{i}_{col}'] - df_n[f'A{i}_True_Distance']) for i in range(4)])
        sns.kdeplot(errs, fill=True, label=lab, color=clr, alpha=0.3)
    plt.title("Error Evolution: Pipeline Accuracy"); plt.xlabel("Error (m)"); plt.xlim(0, 1.2); plt.legend()
    plt.savefig(os.path.join(OUTPUT_FOLDER, '04_pipeline_evolution.png'), dpi=300); plt.close()

    # 05. Temporal Stability (Keep Existing)
    plt.figure(figsize=(12, 5))
    df_s = df_n.sort_values('Timestamp')
    t = df_s['Timestamp'] - df_s['Timestamp'].iloc[0]
    e = np.sqrt((df_s['Calc_X'] - df_s['True_X'])**2 + (df_s['Calc_Y'] - df_s['True_Y'])**2)
    plt.plot(t, e, alpha=0.3, color='gray')
    plt.plot(t, e.rolling(20).mean(), color='navy', label='MA(20)')
    plt.title("System Stability: Error Over Time"); plt.ylabel("2D Error (m)"); plt.xlabel("Time (s)"); plt.legend()
    plt.savefig(os.path.join(OUTPUT_FOLDER, '05_temporal_stability.png'), dpi=300); plt.close()

def main():
    df_n = pd.read_csv(NOISY_DATA_PATH, skipinitialspace=True)
    df_c = pd.read_csv(CLEAN_DATA_PATH, skipinitialspace=True)
    for d in [df_n, df_c]: d.columns = d.columns.str.strip()
    
    generate_ultimate_report(df_c, df_n)
    generate_visuals(df_c, df_n)
    print(f"\n[DONE] All analysis files are in '{OUTPUT_FOLDER}/'")

if __name__ == "__main__":
    main()