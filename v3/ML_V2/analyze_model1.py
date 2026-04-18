import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_FOLDER = "3D_DATA_MODELS/EVALUATION/single_file_analysis"
DATA_PATH     = '3D_DATA_MODELS/DATA/RECALCULATED_random_forest_8A_100sam_WITH_NOISE.csv'

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

plt.style.use('ggplot')

# ============================================================
# ANALYSIS ENGINE
# ============================================================

def _metrics(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    bias = np.mean(np.array(y_pred) - np.array(y_true))
    r2   = r2_score(y_true, y_pred)
    return dict(mse=mse, rmse=rmse, bias=bias, r2=r2)

def get_sections(df):
    pos_cols = ['True_X', 'True_Y', 'True_Z']
    # Filter pos_cols to only those that exist in the dataframe
    pos_cols = [c for c in pos_cols if c in df.columns]
    
    changes  = df[pos_cols].ne(df[pos_cols].shift()).any(axis=1)
    boundaries = df.index[changes].tolist() + [len(df)]

    sections = []
    for k, start in enumerate(boundaries[:-1]):
        end    = boundaries[k + 1]
        sub    = df.iloc[start:end].reset_index(drop=True)
        x      = sub['True_X'].iloc[0]
        y      = sub['True_Y'].iloc[0]
        z      = sub['True_Z'].iloc[0] if 'True_Z' in sub.columns else 0.0
        label  = f"P{k} (x={x:.3f}, y={y:.3f}, z={z:.3f})"
        sections.append((label, sub))
    return sections

def get_overall_stats(df):
    err_2d = np.sqrt((df['Calc_X'] - df['True_X'])**2 + (df['Calc_Y'] - df['True_Y'])**2)
    calc_z = df['Calc_Z'] if 'Calc_Z' in df.columns else np.zeros(len(df))
    true_z = df['True_Z'] if 'True_Z' in df.columns else np.zeros(len(df))
    
    err_3d = np.sqrt((df['Calc_X'] - df['True_X'])**2 +
                     (df['Calc_Y'] - df['True_Y'])**2 +
                     (calc_z - true_z)**2)

    return {
        'mean_2d': err_2d.mean(),
        'median_2d': err_2d.median(),
        'ce90': np.percentile(err_2d, 90),
        'mean_3d': err_3d.mean(),
        'r2_total': r2_score(df['True_X'], df['Calc_X']),
        'samples': len(df)
    }

def generate_simple_report(df, sections):
    stats = get_overall_stats(df)
    num_anchors = len([c for c in df.columns if c.endswith('_True_Distance')])
    path = os.path.join(OUTPUT_FOLDER, 'single_file_stats.txt')
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"UWB SINGLE FILE ANALYSIS — {datetime.now()}\n")
        f.write(f"Source: {DATA_PATH}\n")
        f.write("="*60 + "\n\n")
        
        f.write("--- OVERALL POSITIONING ---\n")
        f.write(f"Samples:         {stats['samples']}\n")
        f.write(f"Mean 2D Error:   {stats['mean_2d']:.4f} m\n")
        f.write(f"Median 2D Error: {stats['median_2d']:.4f} m\n")
        f.write(f"CE90 Accuracy:   {stats['ce90']:.4f} m\n")
        f.write(f"Mean 3D Error:   {stats['mean_3d']:.4f} m\n\n")

        f.write("--- PER-ANCHOR DISTANCE PERFORMANCE ---\n")
        for i in range(num_anchors):
            true_d = df[f'A{i}_True_Distance']
            ai_d   = df[f'A{i}_AI_Dist']
            m = _metrics(true_d, ai_d)
            f.write(f"Anchor A{i}: RMSE={m['rmse']:.4f}m | Bias={m['bias']:+.4f}m | R2={m['r2']:.4f}\n")

    print(f" [OK] Report saved to {path}")

# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Loading: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    
    sections = get_sections(df)
    
    # Text summary
    generate_simple_report(df, sections)
    
    # Simple Scatter Plot
    plt.figure(figsize=(8,8))
    plt.scatter(df['Calc_X'], df['Calc_Y'], alpha=0.3, label='Calculated', color='royalblue')
    true_pos = df[['True_X', 'True_Y']].drop_duplicates()
    plt.scatter(true_pos['True_X'], true_pos['True_Y'], marker='*', s=200, color='red', label='True')
    plt.title("Spatial Positioning")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    plt.legend()
    plt.axis('equal')
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'spatial_scatter.png'))
    
    print(f"[DONE] Analysis complete.")

if __name__ == "__main__":
    main()