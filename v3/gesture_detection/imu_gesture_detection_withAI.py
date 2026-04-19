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

# OPTIONS:
SOLVER_TYPE = "nonlinear"  # "linear" or "nonlinear"
FILTER_TYPE = "kf"         # "kf" (filters distance) or "ekf" (filters x,y position)
VIZ_MODE = "3d"            # "2d" or "3d"
DEBUG_MODE = False         # True to see print statements, False to hide them
USE_IMU_DATA = False       # <-- TOGGLE: Set to False to ignore IMU data entirely

# EMA CONFIGURATION
EMA_ALPHA = 0.6

# ANCHOR POSITIONS (x, y, z) in meters
ANCHOR_POSITIONS = np.array([
    [1.92024,     4.2164,    0.4699],    # A0
    [4.3434,      4.2037,    0.46736],   # A1
    [4.3942,      4.2037,    1.66624],   # A2
    [1.9685,      4.2291,    1.9177],    # A3
    [1.78435,     1.2319,    0.47625],   # A4
    [4.29895,     1.2065,    0.4826],    # A5
    [4.29926,     1.2065,    1.88595],   # A6
    [1.7907,      1.2319,    1.75895],   # A7
])

# For the visualizer, we maintain the Z from ANCHOR_POSITIONS
VIZ_ANCHORS = {str(i): tuple(ANCHOR_POSITIONS[i]) for i in range(NUM_ANCHORS)}

PROCESS_NOISE = 0.1  
MEASURE_NOISE = 0.05 

# --- SPELL DETECTION CONFIGURATION ---
TEMPLATE_DIR    = "templates/"
NORM_STATS_FILE = "templates/norm_stats.npy"

# --- SHARED STATE ---
_is_recording      = False
_active_spell_data = []
_recording_lock    = threading.Lock()

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

# --- SPELL DETECTION LOGIC ---

def extract_features(timed_coords):
    positions  = np.array([pt for pt, _ in timed_coords])
    timestamps = np.array([t  for _, t  in timed_coords])

    dts = np.diff(timestamps)
    dts = np.where(dts < 1e-6, 1e-6, dts)

    velocities = np.diff(positions, axis=0) / dts[:, None]

    features = []
    speeds = []

    for i in range(1, len(velocities)):
        v_prev, v_curr = velocities[i-1], velocities[i]
        speed = np.linalg.norm(v_curr)
        mag_p = np.linalg.norm(v_prev)
        mag_c = np.linalg.norm(v_curr)

        curvature = 0.0
        if mag_p > 1e-5 and mag_c > 1e-5:
            cos_theta = np.dot(v_prev, v_curr) / (mag_p * mag_c)
            curvature = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        verticality = v_curr[2] / (speed + 1e-5)
        turn_magnitude = np.linalg.norm(np.cross(v_prev, v_curr))

        speeds.append(speed)
        features.append([speed, curvature, verticality, turn_magnitude])

    features = np.array(features)
    mean_speed = np.mean(speeds)
    if mean_speed > 1e-5:
        features[:, 0] /= mean_speed

    return features

def load_norm_stats():
    if os.path.exists(NORM_STATS_FILE):
        stats = np.load(NORM_STATS_FILE, allow_pickle=True).item()
        return stats['mean'], stats['std']
    return None, None

def apply_feature_normalization(features, mean, std):
    return (features - mean) / (std + 1e-6)

def classify_spell(user_features):
    if not os.path.exists(TEMPLATE_DIR):
        return "No Templates Found (Directory missing)"

    norm_mean, norm_std = load_norm_stats()
    if norm_mean is None:
        pass # Optional: Print warning
    else:
        user_features = apply_feature_normalization(user_features, norm_mean, norm_std)

    spell_best = {}

    for file in os.listdir(TEMPLATE_DIR):
        if not file.endswith(".npy") or file == "norm_stats.npy":
            continue

        template = np.load(os.path.join(TEMPLATE_DIR, file))
        if norm_mean is not None:
            template = apply_feature_normalization(template, norm_mean, norm_std)

        dist, path = fastdtw(user_features, template, dist=euclidean)
        avg_len = (len(user_features) + len(template)) / 2
        normalized_dist = dist / avg_len

        spell_name = "_".join(file.replace(".npy", "").split("_")[:-1]) or file.replace(".npy", "")

        if spell_name not in spell_best or normalized_dist < spell_best[spell_name]:
            spell_best[spell_name] = normalized_dist

    if not spell_best:
        return "No Templates Found"

    print("\n--- Match Results ---")
    for spell_name, d in sorted(spell_best.items(), key=lambda x: x[1]):
        indicator = " <--- WINNER" if d == min(spell_best.values()) else ""
        print(f"  Spell: {spell_name:15} | Distance: {d:8.2f}{indicator}")

    best_spell, min_dist = min(spell_best.items(), key=lambda x: x[1])

    sorted_dists = sorted(spell_best.values())
    if len(sorted_dists) > 1:
        margin = sorted_dists[1] - sorted_dists[0]
        if margin < 5.0:
            print(f"⚠️  Ambiguous result (margin: {margin:.2f}), returning Unknown")
            return "Unknown"

    return best_spell if min_dist < 100 else "Unknown"

# --- KEY-PRESS THREAD ---

def keypress_thread():
    global _is_recording, _active_spell_data
    while True:
        input("\n[GESTURE] Press ENTER to start recording...")
        with _recording_lock:
            _active_spell_data.clear()
            _is_recording = True
        print("[GESTURE] Recording... (press ENTER to stop and classify)")

        input()
        with _recording_lock:
            _is_recording = False
            data_snapshot = list(_active_spell_data)

        try:
            import termios
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except Exception:
            time.sleep(0.3)

        if len(data_snapshot) > 1:
            duration = data_snapshot[-1][1] - data_snapshot[0][1]
            rate = len(data_snapshot) / duration if duration > 0 else 0
            print(f"\n[GESTURE] Stopped. Captured {len(data_snapshot)} fixes over {duration:.1f}s ({rate:.1f} fixes/sec)")
        else:
            print(f"\n[GESTURE] Stopped. Captured {len(data_snapshot)} fixes.")

        if len(data_snapshot) > 5:
            feats = extract_features(data_snapshot)
            spell = classify_spell(feats)
            print(f">>> FINAL CLASSIFICATION: {spell} <<<")
        else:
            print("[GESTURE] Recording too short - no classification attempted.")

# --- TRACKING THREAD ---

def tracking_thread(viz_obj):
    global _is_recording, _active_spell_data
    
    print(f"--- UWB AI Tracker: {FILTER_TYPE.upper()} + {SOLVER_TYPE.upper()} + EMA + {VIZ_MODE.upper()} ---")
    print(f"--- IMU DATA: {'ENABLED' if USE_IMU_DATA else 'DISABLED'} ---")
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
            
            if DEBUG_MODE:
                print(f"\n[DEBUG] RAW SERIAL INPUT: {line}")
            
            parts = line.split(",")
            
            # Check if we at least have standard UWB data
            if len(parts) >= (NUM_ANCHORS * 4):
                
                imu_accel = None
                imu_gyro = None
                
                # 1. PARSE IMU DATA (If toggle is ON and data is present)
                if USE_IMU_DATA and len(parts) >= (NUM_ANCHORS * 4) + 6:
                    imu_offset = NUM_ANCHORS * 4
                    try:
                        ax = float(parts[imu_offset])
                        ay = float(parts[imu_offset+1])
                        az = float(parts[imu_offset+2])
                        gx = float(parts[imu_offset+3])
                        gy = float(parts[imu_offset+4])
                        gz = float(parts[imu_offset+5])
                        
                        imu_accel = np.array([ax, ay, az])
                        imu_gyro = np.array([gx, gy, gz])
                        
                        if DEBUG_MODE:
                            print(f"  [DEBUG] IMU Parsed -> Accel: ({ax:.2f}, {ay:.2f}, {az:.2f}) | Gyro: ({gx:.2f}, {gy:.2f}, {gz:.2f})")
                    except ValueError:
                        if DEBUG_MODE:
                            print("  [DEBUG] Failed to parse IMU data. Ignoring IMU for this frame.")

                # 2. PARSE UWB DATA
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
                        
                        if DEBUG_MODE:
                            print(f"  [DEBUG] A{i}: Raw={raw:5.2f}m -> EMA={smoothed_raw:5.2f}m -> AI={ai_corr:5.2f}m -> Final={dist_final:5.2f}m")
                            
                    except Exception as e: 
                        dists_for_solver.append(np.nan)
                        if DEBUG_MODE:
                            print(f"  [DEBUG] A{i}: ERROR parsing UWB chunk: {e}")

                # 3. COMPUTE POSITION
                if not any(np.isnan(dists_for_solver)):
                    if FILTER_TYPE == "ekf":
                        ekf.predict()
                        current_pos = ekf.update(dists_for_solver)
                    else:
                        if SOLVER_TYPE == "linear":
                            current_pos = trilaterate_linear(ANCHOR_POSITIONS, np.array(dists_for_solver))
                        else:
                            current_pos = trilaterate_nonlinear(ANCHOR_POSITIONS, np.array(dists_for_solver), current_pos)
                    
                    if DEBUG_MODE:
                        print(f"  [DEBUG] COMPUTED POS -> X: {current_pos[0]:.2f}, Y: {current_pos[1]:.2f}, Z: {current_pos[2]:.2f}")
                    
                    # ---> RECORD GESTURE DATA <---
                    with _recording_lock:
                        if _is_recording:
                            _active_spell_data.append((current_pos.copy(), time.time()))

                    # Update Visualizer
                    viz_obj.update_position(x=current_pos[0], y=current_pos[1], z=current_pos[2], ranges=display_ranges)
                else:
                    if DEBUG_MODE:
                        print("  [DEBUG] Skipping position computation due to NaN values in distance array.")

    except Exception as e: print(f"Tracking Error: {e}")

# --- MAIN ---

if __name__ == '__main__':
    dims = 3 if VIZ_MODE.lower() == "3d" else 2
    
    # Define your specific room bounds here
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
    
    print("Initializing components...")
    
    # 1. Start Tracker (Also handles spell recording locally)
    threading.Thread(target=tracking_thread, args=(viz,), daemon=True).start()
    
    # 2. Start Keypress Listener for Gestures
    threading.Thread(target=keypress_thread, daemon=True).start()
    
    # 3. Start Visualizer (MUST be in main thread)
    viz.start()