import warnings
import os
import joblib
import numpy as np
import csv
from scipy.optimize import least_squares

# 1. SILENCE WARNINGS
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['PYTHONWARNINGS'] = 'ignore'

# --- CONFIGURATION ---
INPUT_CSV = '3D_DATA_MODELS/DATA/random_forest_8A_100sam_WITH_NOISE.csv'
OUTPUT_CSV = '3D_DATA_MODELS/DATA/RECALCULATED_random_forest_8A_100sam_WITH_NOISE.csv'
MODEL_FILENAME = '3D_DATA_MODELS/MODELS/WITH_NOISE_8A_100sam.pkl'

NUM_ANCHORS = 8
SOLVER_TYPE = "nonlinear"  # "linear" or "nonlinear"
FILTER_TYPE = "kf"         # "kf" or "ekf"
EMA_ALPHA = 0.5
PROCESS_NOISE = 0.1  
MEASURE_NOISE = 0.05 

# ANCHOR POSITIONS (x, y, z) in meters
ANCHOR_POSITIONS = np.array([
    [3.048,     0.43815,    0.7493],    # A0
    [1.66624,   0.43815,    0.74935],   # A1
    [0.2032,    0.4064,     0.7366],    # A2
    [0.2032,    1.6002,     0.7493],    # A3
    [3.048,     0.43815,    1.3716],    # A4
    [1.6662,    0.43815,    1.5113],    # A5
    [0.2032,    0.4064,     1.07315],   # A6
    [0.2032,    1.6002,     1.42875],   # A7
])

# --- MODIFIED FILTERS FOR POST-PROCESSING (Takes CSV Timestamps) ---

class EMAFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.current_value = None

    def update(self, new_value):
        if self.current_value is None:
            self.current_value = new_value
        else:
            self.current_value = (self.alpha * new_value) + ((1 - self.alpha) * self.current_value)
        return self.current_value

class KalmanAnchor:
    def __init__(self, q, r):
        self.x = np.array([[0.0], [0.0]]) 
        self.P = np.eye(2) * 1.0
        self.Q = np.eye(2) * q
        self.R = np.array([[r]])
        self.H = np.array([[1.0, 0.0]])
        self.last_time = None

    def update(self, measured_dist, current_time):
        if not np.isfinite(measured_dist) or abs(measured_dist) > 50:
            return self.x[0, 0]
        
        dt = max(current_time - self.last_time, 0.001) if self.last_time else 0.001
        self.last_time = current_time
        
        F = np.array([[1.0, dt], [0.0, 1.0]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        y = np.array([[measured_dist]]) - (self.H @ self.x)
        S = (self.H @ self.P @ self.H.T + self.R)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = (np.eye(2) - (K @ self.H)) @ self.P
        return self.x[0, 0]

class ExtendedKalmanFilter:
    def __init__(self, q, r, anchor_pos):
        self.x = np.array([[1.0], [1.0], [1.0], [0.0], [0.0], [0.0]]) 
        self.P = np.eye(6) * 1.0
        self.Q = np.eye(6) * q
        self.R = np.eye(len(anchor_pos)) * r
        self.anchor_pos = anchor_pos
        self.last_time = None

    def predict(self, current_time):
        dt = max(current_time - self.last_time, 0.001) if self.last_time else 0.001
        self.last_time = current_time
        
        F = np.eye(6)
        F[0, 3] = dt  
        F[1, 4] = dt  
        F[2, 5] = dt  
        
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurements):
        z = np.array(measurements).reshape(-1, 1)
        hx = []
        for i in range(len(self.anchor_pos)):
            dist = np.sqrt((self.x[0,0] - self.anchor_pos[i,0])**2 + 
                           (self.x[1,0] - self.anchor_pos[i,1])**2 + 
                           (self.x[2,0] - self.anchor_pos[i,2])**2)
            hx.append(dist)
        hx = np.array(hx).reshape(-1, 1)
        
        H = []
        for i in range(len(self.anchor_pos)):
            dist = max(hx[i, 0], 0.01)
            dx = (self.x[0,0] - self.anchor_pos[i,0]) / dist
            dy = (self.x[1,0] - self.anchor_pos[i,1]) / dist
            dz = (self.x[2,0] - self.anchor_pos[i,2]) / dist
            H.append([dx, dy, dz, 0, 0, 0])
            
        H = np.array(H)
        y = z - hx
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = (np.eye(6) - (K @ H)) @ self.P
        
        return self.x[0:3, 0]

# --- 3D SOLVERS ---
def trilaterate_linear(anchor_pos, distances):
    x0, y0, z0 = anchor_pos[0]
    r0 = distances[0]
    A, B = [], []
    for i in range(1, len(anchor_pos)):
        xi, yi, zi = anchor_pos[i]
        ri = distances[i]
        A.append([2*(xi - x0), 2*(yi - y0), 2*(zi - z0)])
        B.append(ri**2 - r0**2 - xi**2 - yi**2 - zi**2 + x0**2 + y0**2 + z0**2)
        
    pos, _, _, _ = np.linalg.lstsq(np.array(A), -np.array(B), rcond=None)
    return pos

def trilaterate_nonlinear(anchor_pos, distances, last_guess):
    def residuals(guess, anchor_pos, distances):
        return np.linalg.norm(anchor_pos - guess, axis=1) - distances
    res = least_squares(residuals, last_guess, args=(anchor_pos, distances), method='lm')
    return res.x

# --- MAIN POST-PROCESSING LOOP ---
def process_data():
    print("Loading AI Model...")
    ai_model = joblib.load(MODEL_FILENAME)

    print(f"Reading from: {INPUT_CSV}")
    print(f"Writing to:   {OUTPUT_CSV}")

    # Set up filters and trackers
    prev_true_pos = None
    ema_filters = None
    kf_filters = None
    ekf = None
    current_pos = np.mean(ANCHOR_POSITIONS, axis=0)
    
    rows_processed = 0

    with open(INPUT_CSV, 'r') as infile, open(OUTPUT_CSV, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        
        # Build the new header matching the old one, but ensuring Calc_Z is present
        out_header = ['Timestamp', 'True_X', 'True_Y', 'True_Z', 'Calc_X', 'Calc_Y', 'Calc_Z']
        for i in range(NUM_ANCHORS):
            out_header.extend([f'A{i}_True_Distance', f'A{i}_Raw_Dist', 
                               f'A{i}_RX_Power', f'A{i}_FP_Power', f'A{i}_Quality', 
                               f'A{i}_AI_Dist', f'A{i}_Final_Filtered_Dist'])
        
        writer = csv.DictWriter(outfile, fieldnames=out_header)
        writer.writeheader()

        for row in reader:
            try:
                timestamp = float(row['Timestamp'])
                true_pos = (float(row['True_X']), float(row['True_Y']), float(row['True_Z']))
                
                # Check if we moved to a new measurement point
                if true_pos != prev_true_pos:
                    print(f"New measurement point detected at True Pos: {true_pos}. Resetting filters...")
                    ema_filters = {i: EMAFilter(EMA_ALPHA) for i in range(NUM_ANCHORS)}
                    kf_filters = {i: KalmanAnchor(PROCESS_NOISE, MEASURE_NOISE) for i in range(NUM_ANCHORS)}
                    ekf = ExtendedKalmanFilter(PROCESS_NOISE, MEASURE_NOISE, ANCHOR_POSITIONS)
                    current_pos = np.mean(ANCHOR_POSITIONS, axis=0)
                    prev_true_pos = true_pos

                dists_for_solver = []
                out_row = {
                    'Timestamp': timestamp,
                    'True_X': true_pos[0],
                    'True_Y': true_pos[1],
                    'True_Z': true_pos[2],
                }

                # Process each anchor
                for i in range(NUM_ANCHORS):
                    raw = float(row[f'A{i}_Raw_Dist'])
                    rx_pwr = float(row[f'A{i}_RX_Power'])
                    fp_pwr = float(row[f'A{i}_FP_Power'])
                    qual = float(row[f'A{i}_Quality'])
                    true_dist = float(row[f'A{i}_True_Distance'])

                    # 1. EMA
                    smoothed_raw = ema_filters[i].update(raw)
                    
                    # 2. AI Correction
                    feat = np.array([[i, smoothed_raw, rx_pwr, fp_pwr, abs(rx_pwr-fp_pwr), qual]])
                    ai_corr = smoothed_raw + ai_model.predict(feat)[0]
                    
                    # 3. Kalman Filter
                    dist_final = ai_corr
                    if FILTER_TYPE == "kf":
                        dist_final = kf_filters[i].update(dist_final, timestamp)
                        
                    dists_for_solver.append(dist_final)
                    
                    # Store anchor stats for output
                    out_row[f'A{i}_True_Distance'] = true_dist
                    out_row[f'A{i}_Raw_Dist'] = raw
                    out_row[f'A{i}_RX_Power'] = rx_pwr
                    out_row[f'A{i}_FP_Power'] = fp_pwr
                    out_row[f'A{i}_Quality'] = qual
                    out_row[f'A{i}_AI_Dist'] = ai_corr
                    out_row[f'A{i}_Final_Filtered_Dist'] = dist_final

                # 4. Positioning Solver
                if not any(np.isnan(dists_for_solver)):
                    if FILTER_TYPE == "ekf":
                        ekf.predict(timestamp)
                        current_pos = ekf.update(dists_for_solver)
                    else:
                        if SOLVER_TYPE == "linear":
                            current_pos = trilaterate_linear(ANCHOR_POSITIONS, np.array(dists_for_solver))
                        else:
                            current_pos = trilaterate_nonlinear(ANCHOR_POSITIONS, np.array(dists_for_solver), current_pos)

                out_row['Calc_X'] = current_pos[0]
                out_row['Calc_Y'] = current_pos[1]
                out_row['Calc_Z'] = current_pos[2]

                writer.writerow(out_row)
                rows_processed += 1
                
                if rows_processed % 100 == 0:
                    print(f"Processed {rows_processed} rows...", end="\r")

            except ValueError:
                # Skip rows with corrupted data/headers
                continue

    print(f"\n[SUCCESS] Post-processing complete! Processed {rows_processed} lines.")
    print(f"New 3D data saved to: {OUTPUT_CSV}")

if __name__ == '__main__':
    process_data()