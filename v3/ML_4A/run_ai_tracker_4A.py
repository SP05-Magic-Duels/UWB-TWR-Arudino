import warnings
import os

# 1. SILENCE WARNINGS
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="`sklearn.utils.parallel.delayed` should be used")
os.environ['PYTHONWARNINGS'] = 'ignore'

import serial
import joblib
import numpy as np
import time
from scipy.optimize import least_squares

# --- CONFIGURATION ---
SERIAL_PORT = 'COM3' 
BAUD_RATE = 115200
MODEL_FILENAME = 'uwb_random_forest_WIDER_4A_100sam.pkl'
NUM_ANCHORS = 4

# ANCHOR POSITIONS (x, y) in meters
# Adjust these to match your actual room setup!
ANCHOR_POSITIONS = np.array([
    [0.0, 0.0],    # A0
    [0.0, 1.9812], # A1
    [2.5654, 0.0],   # A2
    [2.54, 1.9812] # A3
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

def trilaterate(anchor_pos, distances):
    """
    Finds (x, y) coordinates that best fit the measured distances.
    """
    def equations(guess, anchor_pos, distances):
        return np.linalg.norm(anchor_pos - guess, axis=1) - distances

    # Start guess at the center of the anchors
    initial_guess = np.mean(anchor_pos, axis=0)
    result = least_squares(equations, initial_guess, args=(anchor_pos, distances))
    return result.x

def main():
    print(f"--- UWB AI Tracker + Trilateration ---")
    
    try:
        ai_model = joblib.load(MODEL_FILENAME)
        if hasattr(ai_model, 'n_jobs'):
            ai_model.n_jobs = 1 
        print(f"Model loaded: {MODEL_FILENAME}")
    except Exception as e:
        print(f"Model Error: {e}")
        return

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
        ser.reset_input_buffer()
        
        filters = {i: KalmanAnchor(PROCESS_NOISE, MEASURE_NOISE) for i in range(NUM_ANCHORS)}
        last_print_time = 0

        while True:
            if ser.in_waiting > 100:
                ser.reset_input_buffer()
                continue

            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line: continue
            
            parts = line.split(",")
            if len(parts) == NUM_ANCHORS * 4:
                current_distances = []
                
                for i in range(NUM_ANCHORS):
                    idx = i * 4
                    try:
                        chunk = parts[idx : idx+4]
                        if any(val.strip() == "" for val in chunk): 
                            current_distances.append(np.nan)
                            continue
                        
                        raw_dist = float(chunk[0])
                        rx_pwr, fp_pwr, qual = float(chunk[1]), float(chunk[2]), float(chunk[3])
                        
                        # AI Correction
                        feat = np.array([[i, raw_dist, rx_pwr, fp_pwr, abs(rx_pwr-fp_pwr), qual]])
                        ai_corrected = raw_dist + ai_model.predict(feat)[0]
                        
                        # Kalman filter per anchor
                        final_dist = filters[i].update(ai_corrected)
                        current_distances.append(final_dist)
                    except:
                        current_distances.append(np.nan)

                # Only proceed if we have valid data for all anchors
                if not any(np.isnan(current_distances)):
                    # Perform Trilateration
                    pos_m = trilaterate(ANCHOR_POSITIONS, np.array(current_distances))
                    
                    # Convert to inches for display
                    dist_str = " | ".join([f"A{i}: {d*39.3701:>6.2f}\"" for i, d in enumerate(current_distances)])
                    pos_str = f"X: {pos_m[0]*39.3701:>7.2f}\"   Y: {pos_m[1]*39.3701:>7.2f}\""
                    
                    # Throttled printing to keep terminal clean
                    if time.time() - last_print_time > 0.05:
                        # Clear screen effect using ANSI escape sequences
                        print("\033[H\033[J", end="") 
                        print(f"--- UWB AI LIVE TRACKING ---")
                        print(f"Distances: {dist_str}")
                        print(f"Coordinates: {pos_str}")
                        print(f"-----------------------------")
                        last_print_time = time.time()

    except Exception as e:
        print(f"\nRuntime Error: {e}")
    finally:
        if 'ser' in locals(): ser.close()

if __name__ == '__main__':
    main()