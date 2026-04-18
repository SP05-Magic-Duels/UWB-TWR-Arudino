import warnings
import os
import serial
import joblib
import numpy as np
import time
import threading
from scipy.optimize import least_squares
from visualize import LocationVisualizer

# 1. SILENCE WARNINGS
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['PYTHONWARNINGS'] = 'ignore'

# --- CONFIGURATION ---
SERIAL_PORT = 'COM3' 
BAUD_RATE = 115200
MODEL_FILENAME = '3D_DATA_MODELS/MODELS/NOISE_large_room_data_8A_100sam.pkl'
NUM_ANCHORS = 8

# OPTIONS:
SOLVER_TYPE = "nonlinear"  # "linear" or "nonlinear"
FILTER_TYPE = "kf"         # "kf" (filters distance) or "ekf" (filters x,y position)
VIZ_MODE = "3d"            # NEW OPTION: "2d" or "3d"

# EMA CONFIGURATION
EMA_ALPHA = 0.6

# ANCHOR POSITIONS (x, y, z) in meters
ANCHOR_POSITIONS = np.array([
    [1.92024,     4.2164,    0.4699],    # A0
    [4.3434,     4.2037,    0.46736],   # A1
    [4.3942,      4.2037,    1.66624],           # A2
    [1.9685,      4.2291,    1.9177],         # A3
    [1.78435,     1.2319,    0.47625],      # A4
    [4.29895,     1.2065,    0.4826],   # A5
    [4.29926,     1.2065,    1.88595],        # A6
    [1.7907,      1.2319,    1.75895],        # A7
])

# For the visualizer, we maintain the Z from ANCHOR_POSITIONS
VIZ_ANCHORS = {str(i): tuple(ANCHOR_POSITIONS[i]) for i in range(NUM_ANCHORS)}

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
        if not np.isfinite(measured_dist) or abs(measured_dist) > 50: return self.x[0, 0]
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
        self.x = np.array([[1.0], [1.0], [1.0], [0.0], [0.0], [0.0]]) # x, y, z, vx, vy, vz
        self.P = np.eye(6) * 1.0
        self.Q = np.eye(6) * q
        self.R = np.eye(len(anchor_pos)) * r
        self.anchor_pos = anchor_pos
        self.last_time = time.time()
    def predict(self):
        now = time.time()
        dt = max(now - self.last_time, 0.001)
        self.last_time = now
        F = np.eye(6); F[0,3] = F[1,4] = F[2,5] = dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
    def update(self, measurements):
        z = np.array(measurements).reshape(-1, 1)
        hx = np.sqrt(np.sum((self.x[:3].T - self.anchor_pos)**2, axis=1)).reshape(-1, 1)
        H = []
        for i in range(len(self.anchor_pos)):
            dist = max(hx[i, 0], 0.01)
            diff = (self.x[:3, 0] - self.anchor_pos[i]) / dist
            H.append([diff[0], diff[1], diff[2], 0, 0, 0])
        H = np.array(H)
        y = z - hx
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = (np.eye(6) - (K @ H)) @ self.P
        return self.x[0:3, 0]

def trilaterate_linear(anchor_pos, distances):
    x0, y0, z0 = anchor_pos[0]; r0 = distances[0]
    A, B = [], []
    for i in range(1, len(anchor_pos)):
        xi, yi, zi = anchor_pos[i]; ri = distances[i]
        A.append([2*(xi - x0), 2*(yi - y0), 2*(zi - z0)])
        B.append(ri**2 - r0**2 - xi**2 - yi**2 - zi**2 + x0**2 + y0**2 + z0**2)
    pos, _, _, _ = np.linalg.lstsq(np.array(A), -np.array(B), rcond=None)
    return pos

def trilaterate_nonlinear(anchor_pos, distances, last_guess):
    def residuals(guess, anchor_pos, distances):
        return np.linalg.norm(anchor_pos - guess, axis=1) - distances
    res = least_squares(residuals, last_guess, args=(anchor_pos, distances), method='lm')
    return res.x

def tracking_thread(viz_obj):
    print(f"--- UWB AI Tracker: {FILTER_TYPE.upper()} + {SOLVER_TYPE.upper()} + EMA + {VIZ_MODE.upper()} ---")
    try:
        ai_model = joblib.load(MODEL_FILENAME)
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
        ser.reset_input_buffer()
        
        ema_filters = {i: EMAFilter(EMA_ALPHA) for i in range(NUM_ANCHORS)}
        kf_filters = {i: KalmanAnchor(PROCESS_NOISE, MEASURE_NOISE) for i in range(NUM_ANCHORS)}
        ekf = ExtendedKalmanFilter(PROCESS_NOISE, MEASURE_NOISE, ANCHOR_POSITIONS)
        
        current_pos = np.mean(ANCHOR_POSITIONS, axis=0)

        while True:
            if ser.in_waiting > 100: ser.reset_input_buffer()
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line: continue
            
            parts = line.split(",")
            if len(parts) == NUM_ANCHORS * 4:
                dists_for_solver = []
                display_ranges = {}
                
                for i in range(NUM_ANCHORS):
                    idx = i * 4
                    try:
                        chunk = parts[idx : idx+4]
                        raw = float(chunk[0])
                        smoothed_raw = ema_filters[i].update(raw)
                        feat = np.array([[i, smoothed_raw, float(chunk[1]), float(chunk[2]), abs(float(chunk[1])-float(chunk[2])), float(chunk[3])]])
                        ai_corr = smoothed_raw + ai_model.predict(feat)[0]
                        
                        dist_final = ai_corr
                        if FILTER_TYPE == "kf":
                            dist_final = kf_filters[i].update(dist_final)
                            
                        dists_for_solver.append(dist_final)
                        display_ranges[str(i)] = dist_final
                    except: dists_for_solver.append(np.nan)

                if not any(np.isnan(dists_for_solver)):
                    if FILTER_TYPE == "ekf":
                        ekf.predict()
                        current_pos = ekf.update(dists_for_solver)
                    else:
                        if SOLVER_TYPE == "linear":
                            current_pos = trilaterate_linear(ANCHOR_POSITIONS, np.array(dists_for_solver))
                        else:
                            current_pos = trilaterate_nonlinear(ANCHOR_POSITIONS, np.array(dists_for_solver), current_pos)
                    
                    # Pass the Z coordinate to the visualizer instead of hardcoded 0.0
                    viz_obj.update_position(x=current_pos[0], y=current_pos[1], z=current_pos[2], ranges=display_ranges)

    except Exception as e: print(f"Tracking Error: {e}")

if __name__ == '__main__':
    dims = 3 if VIZ_MODE.lower() == "3d" else 2
    
    # Define your specific room bounds here
    # Based on your anchor data: X goes to 120, Y to 63, Z to ~60
    X_BOUNDS = (0, 5)
    Y_BOUNDS = (0, 5)
    Z_BOUNDS = (0, 5)
    
    viz = LocationVisualizer(
        dimensions=dims, 
        anchor_positions=VIZ_ANCHORS, 
        x_lim=X_BOUNDS, 
        y_lim=Y_BOUNDS, 
        z_lim=Z_BOUNDS
    )
    
    threading.Thread(target=tracking_thread, args=(viz,), daemon=True).start()
    viz.start()