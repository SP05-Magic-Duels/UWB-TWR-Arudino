import warnings
import os
import serial
import joblib
import numpy as np
import time
from scipy.optimize import least_squares

# 1. SILENCE WARNINGS
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['PYTHONWARNINGS'] = 'ignore'

# --- CONFIGURATION ---
SERIAL_PORT = 'COM3' 
BAUD_RATE = 115200
MODEL_FILENAME = 'uwb_random_forest_WIDER_4A_100sam.pkl'
NUM_ANCHORS = 4

# CHOOSE YOUR SOLVER HERE:
SOLVER_TYPE = "linear"
# SOLVER_TYPE = "nonlinear"

# ANCHOR POSITIONS (x, y) in meters
ANCHOR_POSITIONS = np.array([
    [0.0, 0.0],    # A0
    [0.0, 0.7112], # A1
    [2.54, 0.0],   # A2
    [2.54, 0.7112] # A3
])

PROCESS_NOISE = 0.5  
MEASURE_NOISE = 0.1  
# ---------------------

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
        z = np.array([[measured_dist]])
        y = z - (self.H @ self.x)
        S = (self.H @ self.P @ self.H.T + self.R)[0,0]
        K = self.P @ self.H.T / S
        self.x = self.x + (K * y)
        self.P = (np.eye(2) - (K @ self.H)) @ self.P
        return self.x[0, 0]

def trilaterate_linear(anchor_pos, distances):
    """
    Linear Least Squares (LLS).
    Transforms circles into a linear system of equations (Ax = B).
    """
    # Use the first anchor as a reference to linearize
    x0, y0 = anchor_pos[0]
    r0 = distances[0]
    
    A = []
    B = []
    
    for i in range(1, len(anchor_pos)):
        xi, yi = anchor_pos[i]
        ri = distances[i]
        
        A.append([2*(xi - x0), 2*(yi - y0)])
        B.append(ri**2 - r0**2 - xi**2 - yi**2 + x0**2 + y0**2)
        
    A = np.array(A)
    B = np.array(B)
    
    # Solve using pseudo-inverse for best fit
    pos, residuals, rank, s = np.linalg.lstsq(A, -B, rcond=None)
    return pos

def trilaterate_nonlinear(anchor_pos, distances, last_guess):
    """
    Non-Linear Least Squares (NLLS).
    Iteratively minimizes the distance error using Levenberg-Marquardt.
    """
    def residuals(guess, anchor_pos, distances):
        return np.linalg.norm(anchor_pos - guess, axis=1) - distances

    res = least_squares(residuals, last_guess, args=(anchor_pos, distances), method='lm')
    return res.x

def main():
    print(f"--- UWB AI Tracker ({SOLVER_TYPE.upper()}) ---")
    
    try:
        ai_model = joblib.load(MODEL_FILENAME)
        if hasattr(ai_model, 'n_jobs'): ai_model.n_jobs = 1 
    except Exception as e:
        print(f"Model Error: {e}"); return

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
        ser.reset_input_buffer()
        
        filters = {i: KalmanAnchor(PROCESS_NOISE, MEASURE_NOISE) for i in range(NUM_ANCHORS)}
        current_pos = np.mean(ANCHOR_POSITIONS, axis=0) # Initial guess
        last_print = 0

        while True:
            if ser.in_waiting > 100:
                ser.reset_input_buffer()
                continue

            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line: continue
            
            parts = line.split(",")
            if len(parts) == NUM_ANCHORS * 4:
                dists = []
                for i in range(NUM_ANCHORS):
                    idx = i * 4
                    try:
                        chunk = parts[idx : idx+4]
                        if any(v.strip() == "" for v in chunk): dists.append(np.nan); continue
                        
                        raw = float(chunk[0])
                        feat = np.array([[i, raw, float(chunk[1]), float(chunk[2]), abs(float(chunk[1])-float(chunk[2])), float(chunk[3])]])
                        ai_corr = raw + ai_model.predict(feat)[0]
                        dists.append(filters[i].update(ai_corr))
                    except: dists.append(np.nan)

                if not any(np.isnan(dists)):
                    # Solver Selection
                    if SOLVER_TYPE == "linear":
                        current_pos = trilaterate_linear(ANCHOR_POSITIONS, np.array(dists))
                    else:
                        current_pos = trilaterate_nonlinear(ANCHOR_POSITIONS, np.array(dists), current_pos)
                    
                    if time.time() - last_print > 0.05:
                        print("\033[H\033[J", end="") # Clear terminal
                        print(f"SOLVER: {SOLVER_TYPE.upper()}")
                        print(f"Distances (in): " + " | ".join([f"A{i}: {d*39.37:>5.1f}" for i, d in enumerate(dists)]))
                        print(f"Coordinates (in): X: {current_pos[0]*39.37:>7.2f}   Y: {current_pos[1]*39.37:>7.2f}")
                        last_print = time.time()

    except Exception as e: print(f"Runtime Error: {e}")
    finally:
        if 'ser' in locals(): ser.close()

if __name__ == '__main__':
    main()