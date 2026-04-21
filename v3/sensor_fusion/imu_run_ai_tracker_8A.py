import warnings
import os
import serial
import joblib
import numpy as np
import time
import threading
import sys
from scipy.optimize import least_squares
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from imu_visualize import LocationVisualizer

# 1. SILENCE WARNINGS
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['PYTHONWARNINGS'] = 'ignore'

# --- CONFIGURATION ---
SERIAL_PORT = '/dev/ttyUSB0' 
BAUD_RATE = 115200
MODEL_FILENAME = 'NOISE_large_room_data_8A_100sam.pkl'
NUM_ANCHORS = 8

SOLVER_TYPE = "nonlinear" 
FILTER_TYPE = "kf"         
VIZ_MODE = "3d"            
DEBUG_MODE = True          # Debug prints are fully restored
USE_IMU_DATA = True        

EMA_ALPHA = 0.6

ANCHOR_POSITIONS = np.array([
    [1.92024,     4.2164,    0.4699], [4.3434,      4.2037,    0.46736],
    [4.3942,      4.2037,    1.66624],[1.9685,      4.2291,    1.9177],
    [1.78435,     1.2319,    0.47625],[4.29895,     1.2065,    0.4826],
    [4.29926,     1.2065,    1.88595],[1.7907,      1.2319,    1.75895],
])
VIZ_ANCHORS = {str(i): tuple(ANCHOR_POSITIONS[i]) for i in range(NUM_ANCHORS)}

PROCESS_NOISE = 0.1  
MEASURE_NOISE = 0.05 

# --- FILTERS ---
class EMAFilter:
    def __init__(self, alpha):
        self.alpha = alpha; self.current_value = None
    def update(self, new_value):
        if self.current_value is None: self.current_value = new_value
        else: self.current_value = (self.alpha * new_value) + ((1 - self.alpha) * self.current_value)
        return self.current_value

class KalmanAnchor:
    def __init__(self, q, r):
        self.x = np.array([[0.0], [0.0]]); self.P = np.eye(2) * 1.0
        self.Q = np.eye(2) * q; self.R = np.array([[r]]); self.H = np.array([[1.0, 0.0]])
        self.last_time = time.time()
    def update(self, measured_dist):
        if not np.isfinite(measured_dist) or abs(measured_dist) > 50: return self.x[0, 0]
        now = time.time(); dt = max(now - self.last_time, 0.001); self.last_time = now
        F = np.array([[1.0, dt], [0.0, 1.0]])
        self.x = F @ self.x; self.P = F @ self.P @ F.T + self.Q
        y = np.array([[measured_dist]]) - (self.H @ self.x)
        S = (self.H @ self.P @ self.H.T + self.R)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y); self.P = (np.eye(2) - (K @ self.H)) @ self.P
        return self.x[0, 0]

class ExtendedKalmanFilter:
    def __init__(self, q, r, anchor_pos):
        self.x = np.array([[1.0], [1.0], [1.0], [0.0], [0.0], [0.0]]) 
        self.P = np.eye(6) * 1.0; self.Q = np.eye(6) * q; self.R = np.eye(len(anchor_pos)) * r
        self.anchor_pos = anchor_pos; self.last_time = time.time()
    def predict(self):
        now = time.time(); dt = max(now - self.last_time, 0.001); self.last_time = now
        F = np.eye(6); F[0,3] = F[1,4] = F[2,5] = dt
        self.x = F @ self.x; self.P = F @ self.P @ F.T + self.Q
    def update(self, measurements):
        z = np.array(measurements).reshape(-1, 1)
        hx = np.sqrt(np.sum((self.x[:3].T - self.anchor_pos)**2, axis=1)).reshape(-1, 1)
        H = []
        for i in range(len(self.anchor_pos)):
            dist = max(hx[i, 0], 0.01)
            diff = (self.x[:3, 0] - self.anchor_pos[i]) / dist
            H.append([diff[0], diff[1], diff[2], 0, 0, 0])
        H = np.array(H); y = z - hx
        S = H @ self.P @ H.T + self.R; K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y); self.P = (np.eye(6) - (K @ H)) @ self.P
        return self.x[0:3, 0]

class IMUKalmanFilter:
    def __init__(self, process_variance, measurement_variance, initial_value=0.0):
        self.Q = process_variance
        self.R = measurement_variance
        self.x = initial_value
        self.P = 1.0
    def update(self, measurement):
        self.P = self.P + self.Q
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * self.P
        return self.x

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
        
        # IMU Kalman Filters
        kf_ax = IMUKalmanFilter(0.001, 0.1, 0.0); kf_ay = IMUKalmanFilter(0.001, 0.1, 0.0); kf_az = IMUKalmanFilter(0.001, 0.1, 1.0)
        kf_gx = IMUKalmanFilter(0.1, 10.0, 0.0);  kf_gy = IMUKalmanFilter(0.1, 10.0, 0.0);  kf_gz = IMUKalmanFilter(0.1, 10.0, 0.0)
        
        current_pos = np.mean(ANCHOR_POSITIONS, axis=0)

        while True:
            if ser.in_waiting > 100: ser.reset_input_buffer()
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line: continue
            
            if DEBUG_MODE:
                print(f"\n[DEBUG] RAW SERIAL: {line}")
            
            parts = line.split(",")
            if len(parts) >= (NUM_ANCHORS * 4):
                imu_accel = None
                imu_gyro = None
                
                # PARSE & FILTER IMU DATA
                if USE_IMU_DATA and len(parts) >= (NUM_ANCHORS * 4) + 6:
                    imu_offset = NUM_ANCHORS * 4
                    try:
                        ax = kf_ax.update(float(parts[imu_offset]))
                        ay = kf_ay.update(float(parts[imu_offset+1]))
                        az = kf_az.update(float(parts[imu_offset+2]))
                        gx = kf_gx.update(float(parts[imu_offset+3]))
                        gy = kf_gy.update(float(parts[imu_offset+4]))
                        gz = kf_gz.update(float(parts[imu_offset+5]))
                        
                        imu_accel = np.array([ax, ay, az])
                        imu_gyro = np.array([gx, gy, gz])
                        
                        if DEBUG_MODE:
                            print(f"  [DEBUG] IMU (Filtered) -> Accel: ({ax:.2f}, {ay:.2f}, {az:.2f}) | Gyro: ({gx:.2f}, {gy:.2f}, {gz:.2f})")
                    except ValueError:
                        if DEBUG_MODE: print("  [DEBUG] Failed to parse IMU data.")

                # PARSE UWB DATA
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
                        dist_final = kf_filters[i].update(ai_corr) if FILTER_TYPE == "kf" else ai_corr
                        
                        dists_for_solver.append(dist_final)
                        display_ranges[str(i)] = dist_final
                        
                        if DEBUG_MODE:
                            print(f"  [DEBUG] A{i}: Raw={raw:5.2f}m -> AI={ai_corr:5.2f}m -> Final={dist_final:5.2f}m")
                    except Exception as e: 
                        dists_for_solver.append(np.nan)
                        if DEBUG_MODE: print(f"  [DEBUG] A{i} Error: {e}")

                # COMPUTE POS
                if not any(np.isnan(dists_for_solver)):
                    ekf.predict()
                    current_pos = ekf.update(dists_for_solver) if FILTER_TYPE == "ekf" else trilaterate_nonlinear(ANCHOR_POSITIONS, np.array(dists_for_solver), current_pos)
                    
                    if DEBUG_MODE:
                        print(f"  [DEBUG] COMPUTED POS -> X: {current_pos[0]:.2f}, Y: {current_pos[1]:.2f}, Z: {current_pos[2]:.2f}")
                    
                    viz_obj.update_position(x=current_pos[0], y=current_pos[1], z=current_pos[2], ranges=display_ranges, imu_accel=imu_accel, imu_gyro=imu_gyro)
                else:
                    if DEBUG_MODE: print("  [DEBUG] Skipping position computation due to NaN values.")

    except Exception as e: print(f"Tracking Error: {e}")

if __name__ == '__main__':
    viz = LocationVisualizer(dimensions=3 if VIZ_MODE.lower() == "3d" else 2, anchor_positions=VIZ_ANCHORS, x_lim=(0, 5), y_lim=(0, 5), z_lim=(0, 5))
    threading.Thread(target=tracking_thread, args=(viz,), daemon=True).start()
    viz.start()