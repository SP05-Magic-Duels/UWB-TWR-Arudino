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
MODEL_FILENAME = 'uwb_random_forest_WIDER_TWIST_NEG_4A_100sam.pkl'
NUM_ANCHORS = 4

# OPTIONS:
SOLVER_TYPE = "nonlinear"  # "linear" or "nonlinear"
FILTER_TYPE = "kf"         # "kf" (filters distance) or "ekf" (filters x,y position)

# EMA CONFIGURATION
# Alpha (0.0 to 1.0): Lower is smoother but laggier, higher is more responsive.
EMA_ALPHA = 0.5

# ANCHOR POSITIONS (x, y) in meters
ANCHOR_POSITIONS = np.array([
    [0.0, 0.0],      # A0
    [0.0, 1.9812],   # A1
    [2.5654, 0.0],   # A2
    [2.54, 1.9812]   # A3
])

VIZ_ANCHORS = {str(i): tuple(list(ANCHOR_POSITIONS[i]) + [0.0]) for i in range(NUM_ANCHORS)}

PROCESS_NOISE = 0.1  
MEASURE_NOISE = 0.05 
# ---------------------

class EMAFilter:
    """Exponential Moving Average Filter for raw distances."""
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
    """Standard KF: Filters a single scalar (distance)"""
    def __init__(self, q, r):
        self.x = np.array([[0.0], [0.0]]) # [Dist, Velocity]
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
    """EKF: Filters (x, y) coordinates and (vx, vy) velocities"""
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

def tracking_thread(viz_obj):
    print(f"--- UWB AI Tracker: {FILTER_TYPE.upper()} + {SOLVER_TYPE.upper()} + EMA ---")
    try:
        ai_model = joblib.load(MODEL_FILENAME)
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
        ser.reset_input_buffer()
        
        # Initialize filters
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
                        
                        # --- 1. Apply EMA Filter to RAW distance FIRST ---
                        smoothed_raw = ema_filters[i].update(raw)
                        
                        # --- 2. Calculate AI Correction using the smoothed value ---
                        feat = np.array([[i, smoothed_raw, float(chunk[1]), float(chunk[2]), abs(float(chunk[1])-float(chunk[2])), float(chunk[3])]])
                        ai_corr = smoothed_raw + ai_model.predict(feat)[0]
                        
                        # --- 3. Apply Second Stage (optional Kalman) ---
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
                    
                    viz_obj.update_position(x=current_pos[0], y=current_pos[1], z=0.0, ranges=display_ranges)

    except Exception as e: print(f"Tracking Error: {e}")

if __name__ == '__main__':
    viz = LocationVisualizer(dimensions=2, anchor_positions=VIZ_ANCHORS, plot_limits=(-1.0, 4.0))
    threading.Thread(target=tracking_thread, args=(viz,), daemon=True).start()
    viz.start()