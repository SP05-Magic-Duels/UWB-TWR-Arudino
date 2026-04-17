import warnings
import os
import serial
import joblib
import numpy as np
import time
import csv
import sys
from scipy.optimize import least_squares

# 1. SILENCE WARNINGS
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['PYTHONWARNINGS'] = 'ignore'

# --- CONFIGURATION ---
SERIAL_PORT = 'COM3' 
BAUD_RATE = 115200
MODEL_FILENAME = '3D_DATA_MODELS/MODELS/random_forest_8A_100sam.pkl'
CSV_FILENAME = '3D_DATA_MODELS/DATA/random_forest_8A_100sam.csv'
NUM_ANCHORS = 8
SAMPLES_PER_POINT = 100  # <--- Set exactly how many samples to take per location

# OPTIONS:
SOLVER_TYPE = "nonlinear"  # "linear" or "nonlinear"
FILTER_TYPE = "kf"         # "kf" (filters distance) or "ekf" (filters x,y position)

# EMA CONFIGURATION
EMA_ALPHA = 0.5

# ANCHOR POSITIONS (x, y) in meters
ANCHOR_POSITIONS = np.array([
    [3.048,     0.43815,    0.7493],    # A0
    [1.66624,   0.43815,    0.74935],   # A1
    [0.2032,    0.4064,     0.7366],           # A2
    [0.2032,    1.6002,     0.7493],         # A3
    [3.048,     0.43815,    1.3716],      # A4
    [1.6662,    0.43815,    1.5113],   # A5
    [0.2032,    0.4064,     1.07315],        # A6
    [0.2032,    1.6002,     1.42875],        # A7
])

PROCESS_NOISE = 0.1  
MEASURE_NOISE = 0.05 
# ---------------------

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
        self.last_time = time.time()

    def update(self, measured_dist):
        if not np.isfinite(measured_dist) or abs(measured_dist) > 50:
            return self.x[0, 0]
        now = time.time()
        dt = max(now - self.last_time, 0.001)
        self.last_time = now
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
        self.x = np.array([[1.0], [1.0], [0.0], [0.0]]) 
        self.P = np.eye(4) * 1.0
        self.Q = np.eye(4) * q
        self.R = np.eye(len(anchor_pos)) * r
        self.anchor_pos = anchor_pos
        self.last_time = time.time()

    def predict(self):
        now = time.time()
        dt = max(now - self.last_time, 0.001)
        self.last_time = now
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurements):
        z = np.array(measurements).reshape(-1, 1)
        hx = []
        for i in range(len(self.anchor_pos)):
            dist = np.sqrt((self.x[0,0] - self.anchor_pos[i,0])**2 + (self.x[1,0] - self.anchor_pos[i,1])**2)
            hx.append(dist)
        hx = np.array(hx).reshape(-1, 1)
        H = []
        for i in range(len(self.anchor_pos)):
            dist = max(hx[i, 0], 0.01)
            dx = (self.x[0,0] - self.anchor_pos[i,0]) / dist
            dy = (self.x[1,0] - self.anchor_pos[i,1]) / dist
            H.append([dx, dy, 0, 0])
        H = np.array(H)
        y = z - hx
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = (np.eye(4) - (K @ H)) @ self.P
        return self.x[0:2, 0]

def trilaterate_linear(anchor_pos, distances):
    x0, y0 = anchor_pos[0]; r0 = distances[0]
    A, B = [], []
    for i in range(1, len(anchor_pos)):
        xi, yi = anchor_pos[i]; ri = distances[i]
        A.append([2*(xi - x0), 2*(yi - y0)])
        B.append(ri**2 - r0**2 - xi**2 - yi**2 + x0**2 + y0**2)
    pos, _, _, _ = np.linalg.lstsq(np.array(A), -np.array(B), rcond=None)
    return pos

def trilaterate_nonlinear(anchor_pos, distances, last_guess):
    def residuals(guess, anchor_pos, distances):
        return np.linalg.norm(anchor_pos - guess, axis=1) - distances
    res = least_squares(residuals, last_guess, args=(anchor_pos, distances), method='lm')
    return res.x

def main_collection_loop():
    print("="*50)
    print("     UWB MULTI-POINT DATA LOGGER")
    print(f"     Target: {SAMPLES_PER_POINT} samples per location")
    print("="*50)
    
    # 1. Initialize CSV Header if new file
    file_exists = os.path.exists(CSV_FILENAME)
    if not file_exists:
        with open(CSV_FILENAME, 'w', newline='') as csv_f:
            writer = csv.writer(csv_f)
            header = ['Timestamp', 'True_X', 'True_Y', 'True_Z', 'Calc_X', 'Calc_Y']
            for i in range(NUM_ANCHORS):
                header.extend([f'A{i}_True_Distance', f'A{i}_Raw_Dist', 
                               f'A{i}_RX_Power', f'A{i}_FP_Power', f'A{i}_Quality', 
                               f'A{i}_AI_Dist', f'A{i}_Final_Filtered_Dist'])
            writer.writerow(header)

    # 2. Load AI Model & Serial
    try:
        ai_model = joblib.load(MODEL_FILENAME)
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
        ser.reset_input_buffer()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return

    try:
        while True:
            print("\n" + "-"*50)
            print(" NEW MEASUREMENT POINT (Enter 'q' to quit)")
            print("-"*50)
            
            # --- USER INPUT ---
            val_x = input("Enter True X coordinate (inches): ")
            if val_x.strip().lower() == 'q': break
            true_x = float(val_x) * 0.0254
            
            val_y = input("Enter True Y coordinate (inches): ")
            if val_y.strip().lower() == 'q': break
            true_y = float(val_y) * 0.0254
            
            val_z = input("Enter True Z coordinate (inches) [Press Enter for 0]: ")
            if val_z.strip().lower() == 'q': break
            true_z = float(val_z) * 0.0254 if val_z.strip() != "" else 0.0

            user_true_pos = (true_x, true_y, true_z)
            
            user_true_dists = []
            print("\n--- Enter True Distances ---")
            for i in range(NUM_ANCHORS):
                val = input(f"True distance to Anchor {i} (inches): ")
                if val.strip().lower() == 'q': 
                    sys.exit()
                user_true_dists.append(float(val) * 0.0254)
                
            # --- RESET FILTERS FOR NEW POINT ---
            ema_filters = {i: EMAFilter(EMA_ALPHA) for i in range(NUM_ANCHORS)}
            kf_filters = {i: KalmanAnchor(PROCESS_NOISE, MEASURE_NOISE) for i in range(NUM_ANCHORS)}
            ekf = ExtendedKalmanFilter(PROCESS_NOISE, MEASURE_NOISE, ANCHOR_POSITIONS)
            current_pos = np.mean(ANCHOR_POSITIONS, axis=0)

            # --- START LOGGING SAMPLES ---
            packets_logged = 0
            print(f"\n>> Collecting {SAMPLES_PER_POINT} packets...")
            
            # Flush serial to prevent reading stale buffer data from while user was typing
            time.sleep(0.1)
            ser.reset_input_buffer()

            with open(CSV_FILENAME, 'a', newline='') as csv_f:
                writer = csv.writer(csv_f)
                
                while packets_logged < SAMPLES_PER_POINT:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if not line: continue
                    
                    parts = line.split(",")
                    if len(parts) == NUM_ANCHORS * 4:
                        dists_for_solver = []
                        anchor_stats = {} 
                        
                        for i in range(NUM_ANCHORS):
                            idx = i * 4
                            try:
                                chunk = parts[idx : idx+4]
                                raw = float(chunk[0])
                                rx_pwr = float(chunk[1])
                                fp_pwr = float(chunk[2])
                                qual = float(chunk[3])
                                
                                # 1. EMA
                                smoothed_raw = ema_filters[i].update(raw)
                                
                                # 2. AI Correction
                                feat = np.array([[i, smoothed_raw, rx_pwr, fp_pwr, abs(rx_pwr-fp_pwr), qual]])
                                ai_corr = smoothed_raw + ai_model.predict(feat)[0]
                                
                                # 3. Second Stage (Kalman)
                                dist_final = ai_corr
                                if FILTER_TYPE == "kf":
                                    dist_final = kf_filters[i].update(dist_final)
                                    
                                dists_for_solver.append(dist_final)
                                anchor_stats[i] = [user_true_dists[i], raw, rx_pwr, fp_pwr, qual, ai_corr, dist_final]

                            except: dists_for_solver.append(np.nan)

                        # Check if packet is fully valid
                        if not any(np.isnan(dists_for_solver)):
                            # Solve positioning
                            if FILTER_TYPE == "ekf":
                                ekf.predict()
                                current_pos = ekf.update(dists_for_solver)
                            else:
                                if SOLVER_TYPE == "linear":
                                    current_pos = trilaterate_linear(ANCHOR_POSITIONS, np.array(dists_for_solver))
                                else:
                                    current_pos = trilaterate_nonlinear(ANCHOR_POSITIONS, np.array(dists_for_solver), current_pos)

                            # Save to CSV
                            row = [
                                time.time(), 
                                user_true_pos[0], user_true_pos[1], user_true_pos[2], 
                                current_pos[0], current_pos[1]
                            ]
                            for i in range(NUM_ANCHORS):
                                row.extend(anchor_stats[i])
                            
                            writer.writerow(row)
                            csv_f.flush()
                            
                            packets_logged += 1
                            if packets_logged % 25 == 0:
                                print(f"Logged {packets_logged}/{SAMPLES_PER_POINT} frames...", end="\r")

            print(f"\n[SUCCESS] Saved {SAMPLES_PER_POINT} samples to {CSV_FILENAME}.")

    except KeyboardInterrupt:
        print("\n\nSession terminated by user.")
    except ValueError:
        print("\nInvalid coordinate or distance input. Please enter valid numbers.")
    except Exception as e: 
        print(f"\nSystem Error: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == '__main__':
    main_collection_loop()