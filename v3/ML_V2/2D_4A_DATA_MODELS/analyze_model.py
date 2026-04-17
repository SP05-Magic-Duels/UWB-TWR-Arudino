import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def analyze(csv_file='with_noise.csv', output_folder='eval_results_with_noise'):
    # 1. CREATE THE OUTPUT FOLDER
    if not os.path.exists(output_folder): 
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}/")
        
    if not os.path.exists(csv_file): 
        print(f"Data file '{csv_file}' not found. Run the data collector first.")
        return
    
    df = pd.read_csv(csv_file)
    num_anchors = 4
    
    # 2. GENERATE THE TEXT REPORT WITH LABELS
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = "="*70 + "\n"
    report += f" UWB PER-ANCHOR AI PERFORMANCE REPORT - {timestamp}\n"
    report += "="*70 + "\n\n"

    all_raw_errors = []
    all_ai_errors = []
    all_true_dists = []

    # Process Stats
    for i in range(num_anchors):
        if f'A{i}_Raw_Dist' not in df.columns:
            continue
            
        raw_err = df[f'A{i}_Raw_Dist'] - df[f'A{i}_True_Distance']
        ai_err = df[f'A{i}_AI_Dist'] - df[f'A{i}_True_Distance']
        
        all_raw_errors.extend(raw_err.tolist())
        all_ai_errors.extend(ai_err.tolist())
        all_true_dists.extend(df[f'A{i}_True_Distance'].tolist())
        
        # Calculate Metrics
        rmse_raw = np.sqrt(np.mean(raw_err**2))
        rmse_ai = np.sqrt(np.mean(ai_err**2))
        mae_raw = np.mean(np.abs(raw_err))
        mae_ai = np.mean(np.abs(ai_err))
        std_raw = np.std(raw_err)
        std_ai = np.std(ai_err)
        imp_rmse = ((rmse_raw - rmse_ai) / rmse_raw) * 100 if rmse_raw != 0 else 0
        
        report += f"--- ANCHOR A{i} STATS ---\n"
        report += f"A{i}_Raw_RMSE_meters: {rmse_raw:.4f}\n"
        report += f"A{i}_Raw_MAE_meters:  {mae_raw:.4f}\n"
        report += f"A{i}_Raw_STD_meters:  {std_raw:.4f}\n"
        report += f"A{i}_AI_RMSE_meters:  {rmse_ai:.4f}\n"
        report += f"A{i}_AI_MAE_meters:   {mae_ai:.4f}\n"
        report += f"A{i}_AI_STD_meters:   {std_ai:.4f}\n"
        report += f"A{i}_RMSE_Improvement_Percent: {imp_rmse:.2f}%\n\n"

    # Global Stats
    all_raw_errors = np.array(all_raw_errors)
    all_ai_errors = np.array(all_ai_errors)
    
    total_rmse_raw = np.sqrt(np.mean(all_raw_errors**2))
    total_rmse_ai = np.sqrt(np.mean(all_ai_errors**2))
    total_imp = ((total_rmse_raw - total_rmse_ai) / total_rmse_raw) * 100 if total_rmse_raw != 0 else 0
                 
    report += "--- GLOBAL SYSTEM STATS ---\n"
    report += f"Global_Raw_RMSE_meters: {total_rmse_raw:.4f}\n"
    report += f"Global_AI_RMSE_meters:  {total_rmse_ai:.4f}\n"
    report += f"Global_Error_Reduction_Percent: {total_imp:.2f}%\n"
    report += "="*70 + "\n"

    # SAVE THE REPORT TO TXT
    txt_path = os.path.join(output_folder, 'stats_report.txt')
    with open(txt_path, 'w') as f:
        f.write(report)
    print(f"Saved numerical statistics to: {txt_path}")
    print(report)

    # ==========================================
    # 3. GENERATE AND SAVE PLOTS AS PNGs
    # ==========================================
    
    # 1. Error Distribution Boxplot
    plt.figure(figsize=(14, 6))
    data_to_plot = []
    labels = []
    colors = []
    
    for i in range(num_anchors):
        if f'A{i}_Raw_Dist' in df.columns:
            data_to_plot.append(df[f'A{i}_Raw_Dist'] - df[f'A{i}_True_Distance'])
            data_to_plot.append(df[f'A{i}_AI_Dist'] - df[f'A{i}_True_Distance'])
            labels.extend([f'A{i} Raw', f'A{i} AI'])
            colors.extend(['lightcoral', 'lightblue'])
    
    box = plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        
    plt.axhline(0, color='black', linestyle='--', alpha=0.7)
    plt.title('Distance Error Distribution per Anchor (Raw vs AI)')
    plt.ylabel('Error (meters)')
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    # Save PNG
    plot1_path = os.path.join(output_folder, '01_anchor_boxplot.png')
    plt.savefig(plot1_path)
    print(f"Saved plot: {plot1_path}")
    plt.close() # Close plot to prevent memory leaks

    # 2. Cumulative Distribution Function (CDF)
    plt.figure(figsize=(10, 6))
    abs_raw_err = np.abs(all_raw_errors)
    abs_ai_err = np.abs(all_ai_errors)
    
    x_raw, y_raw = np.sort(abs_raw_err), np.arange(1, len(abs_raw_err)+1) / len(abs_raw_err)
    x_ai, y_ai = np.sort(abs_ai_err), np.arange(1, len(abs_ai_err)+1) / len(abs_ai_err)

    plt.plot(x_raw, y_raw, label='Raw UWB', color='red', linewidth=2)
    plt.plot(x_ai, y_ai, label='AI Corrected', color='blue', linewidth=2)
    
    plt.title('Cumulative Distribution Function (CDF) of Absolute Errors')
    plt.xlabel('Absolute Error (meters)')
    plt.ylabel('Cumulative Probability')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    # Save PNG
    plot2_path = os.path.join(output_folder, '02_error_cdf.png')
    plt.savefig(plot2_path)
    print(f"Saved plot: {plot2_path}")
    plt.close()

    # 3. Error vs. True Distance Scatter Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(all_true_dists, all_raw_errors, alpha=0.3, label='Raw Error', color='red', s=10)
    plt.scatter(all_true_dists, all_ai_errors, alpha=0.3, label='AI Error', color='blue', s=10)
    
    # Trendlines
    if len(all_true_dists) > 1:
        z_raw = np.polyfit(all_true_dists, all_raw_errors, 1)
        p_raw = np.poly1d(z_raw)
        plt.plot(np.unique(all_true_dists), p_raw(np.unique(all_true_dists)), "r--", linewidth=2)
        
        z_ai = np.polyfit(all_true_dists, all_ai_errors, 1)
        p_ai = np.poly1d(z_ai)
        plt.plot(np.unique(all_true_dists), p_ai(np.unique(all_true_dists)), "b--", linewidth=2)

    plt.axhline(0, color='black', linewidth=1)
    plt.title('Measurement Error vs. True Physical Distance')
    plt.xlabel('True Distance (meters)')
    plt.ylabel('Error (meters)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    # Save PNG
    plot3_path = os.path.join(output_folder, '03_error_vs_distance.png')
    plt.savefig(plot3_path)
    print(f"Saved plot: {plot3_path}")
    plt.close()

    print(f"\nAll files successfully generated and saved in the '{output_folder}' directory!")

if __name__ == "__main__":
    analyze()