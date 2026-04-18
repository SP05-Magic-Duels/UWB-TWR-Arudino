import warnings
import os
import serial
import joblib
import numpy as np
import time
import threading
from scipy.optimize import least_squares

# 1. SILENCE WARNINGS
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['PYTHONWARNINGS'] = 'ignore'

# --- TRACKER CONFIGURATION (must match run_spell_tracker.py) ---
SERIAL_PORT    = 'COM5'
BAUD_RATE      = 115200
MODEL_FILENAME = 'NOISE_large_room_data_8A_100sam.pkl'
NUM_ANCHORS    = 8

SOLVER_TYPE  = "nonlinear"  # "linear" or "nonlinear"
FILTER_TYPE  = "kf"         # "kf" or "ekf"
EMA_ALPHA    = 0.5

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

PROCESS_NOISE = 0.1
MEASURE_NOISE = 0.05

# --- TEMPLATE CONFIGURATION ---
TEMPLATE_NAME = "heal"   # <-- Change this for each spell you record
TEMPLATE_DIR  = "templates/"

if not os.path.exists(TEMPLATE_DIR):
    os.makedirs(TEMPLATE_DIR)

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
        self.x = np.array([[1.0], [1.0], [1.0], [0.0], [0.0], [0.0]])
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

# --- FEATURE EXTRACTION (must match run_spell_tracker.py exactly) ---

def extract_features(timed_coords):
    """
    Features: speed (gesture-normalized), curvature, verticality, turn_magnitude
    timed_coords: list of (position_array, timestamp) tuples
    """
    positions  = np.array([pt for pt, _ in timed_coords])
    timestamps = np.array([t  for _, t  in timed_coords])

    if len(positions) < 3:
        return None

    dts = np.diff(timestamps)
    dts = np.where(dts < 1e-6, 1e-6, dts)

    velocities = np.diff(positions, axis=0) / dts[:, None]

    features = []
    speeds = []

    for i in range(1, len(velocities)):
        v_prev, v_curr = velocities[i-1], velocities[i]
        speed  = np.linalg.norm(v_curr)
        mag_p  = np.linalg.norm(v_prev)
        mag_c  = np.linalg.norm(v_curr)

        curvature = 0.0
        if mag_p > 1e-5 and mag_c > 1e-5:
            cos_theta = np.dot(v_prev, v_curr) / (mag_p * mag_c)
            curvature = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        verticality    = v_curr[2] / (speed + 1e-5)
        turn_magnitude = np.linalg.norm(np.cross(v_prev, v_curr))

        speeds.append(speed)
        features.append([speed, curvature, verticality, turn_magnitude])

    features = np.array(features)

    mean_speed = np.mean(speeds)
    if mean_speed > 1e-5:
        features[:, 0] /= mean_speed

    return features

# --- TEMPLATE HELPERS ---

def get_next_template_number(spell_name):
    existing = [
        f for f in os.listdir(TEMPLATE_DIR)
        if f.startswith(spell_name + "_") and f.endswith(".npy")
    ]
    if not existing:
        return 1
    numbers = []
    for f in existing:
        try:
            numbers.append(int(f.replace(".npy", "").split("_")[-1]))
        except ValueError:
            continue
    return max(numbers) + 1 if numbers else 1

def recompute_norm_stats():
    """Recompute per-feature mean and std across all templates and save to norm_stats.npy."""
    all_features = []
    for file in os.listdir(TEMPLATE_DIR):
        if file.endswith(".npy") and file != "norm_stats.npy":
            all_features.append(np.load(os.path.join(TEMPLATE_DIR, file)))

    if not all_features:
        print("No templates found, skipping norm stats computation.")
        return

    combined = np.vstack(all_features)
    mean = combined.mean(axis=0)
    std  = combined.std(axis=0)
    std[std < 1e-6] = 1.0

    np.save(os.path.join(TEMPLATE_DIR, "norm_stats.npy"), {'mean': mean, 'std': std})
    print(f"\n📊 Norm stats recomputed from {len(all_features)} template(s):")
    print(f"   Mean (speed, curvature, verticality, turn_magnitude): {mean}")
    print(f"   Std  (speed, curvature, verticality, turn_magnitude): {std}")

# --- SHARED STATE (tracking thread -> main thread) ---
_latest_pos  = None
_latest_time = None
_pos_lock    = threading.Lock()

def tracking_thread():
    global _latest_pos, _latest_time
    try:
        ai_model = joblib.load(MODEL_FILENAME)
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
        ser.reset_input_buffer()

        ema_filters = {i: EMAFilter(EMA_ALPHA) for i in range(NUM_ANCHORS)}
        kf_filters  = {i: KalmanAnchor(PROCESS_NOISE, MEASURE_NOISE) for i in range(NUM_ANCHORS)}
        ekf = ExtendedKalmanFilter(PROCESS_NOISE, MEASURE_NOISE, ANCHOR_POSITIONS)
        current_pos = np.mean(ANCHOR_POSITIONS, axis=0)

        while True:
            if ser.in_waiting > 100:
                ser.reset_input_buffer()

            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) == NUM_ANCHORS * 4:
                dists_for_solver = []
                for i in range(NUM_ANCHORS):
                    idx = i * 4
                    try:
                        chunk = parts[idx : idx+4]
                        raw = float(chunk[0])
                        smoothed_raw = ema_filters[i].update(raw)
                        feat = np.array([[i, smoothed_raw, float(chunk[1]), float(chunk[2]),
                                          abs(float(chunk[1])-float(chunk[2])), float(chunk[3])]])
                        ai_corr = smoothed_raw + ai_model.predict(feat)[0]
                        dist_final = ai_corr
                        if FILTER_TYPE == "kf":
                            dist_final = kf_filters[i].update(dist_final)
                        dists_for_solver.append(dist_final)
                    except:
                        dists_for_solver.append(np.nan)

                if not any(np.isnan(dists_for_solver)):
                    if FILTER_TYPE == "ekf":
                        ekf.predict()
                        current_pos = ekf.update(dists_for_solver)
                    else:
                        if SOLVER_TYPE == "linear":
                            current_pos = trilaterate_linear(ANCHOR_POSITIONS, np.array(dists_for_solver))
                        else:
                            current_pos = trilaterate_nonlinear(ANCHOR_POSITIONS, np.array(dists_for_solver), current_pos)

                    with _pos_lock:
                        _latest_pos  = current_pos.copy()
                        _latest_time = time.time()

    except Exception as e:
        print(f"Tracking error: {e}")

# --- MAIN ---

if __name__ == '__main__':
    next_num = get_next_template_number(TEMPLATE_NAME)
    print(f"\n--- INSTRUCTIONS ---")
    print(f"1. Wait for 'Tracker ready' before pressing ENTER.")
    print(f"2. Press ENTER to start recording the '{TEMPLATE_NAME}' spell.")
    print(f"3. Perform the gesture.")
    print(f"4. Press CTRL+C to stop and save.")
    print(f"\nWill save as: {TEMPLATE_DIR}{TEMPLATE_NAME}_{next_num}.npy\n")

    # Start tracker BEFORE the ENTER prompt so model load and serial init
    # happen in the background while the user is reading the instructions.
    t = threading.Thread(target=tracking_thread, daemon=True)
    t.start()

    # Block until the tracker has produced at least one valid position fix
    # so we know the model loaded, serial opened, and data is flowing.
    print("Initializing tracker...", end="", flush=True)
    while True:
        with _pos_lock:
            ready = _latest_pos is not None
        if ready:
            break
        time.sleep(0.05)
    print(" Tracker ready.")

    input("\nPress ENTER to arm the recorder...")
    print("RECORDING... (Perform your motion now, then press CTRL+C)")

    recording = []  # list of (position, timestamp) tuples

    try:
        while True:
            with _pos_lock:
                pos = _latest_pos
                ts  = _latest_time
                if pos is not None:
                    _latest_pos = None  # consume so we don't double-append

            if pos is not None:
                recording.append((pos, ts))

            time.sleep(0.005)

    except KeyboardInterrupt:
        print(f"\nRecording stopped. Captured {len(recording)} position fixes.")

        if len(recording) > 1:
            feature_array = extract_features(recording)
            if feature_array is not None:
                next_num  = get_next_template_number(TEMPLATE_NAME)
                file_path = f"{TEMPLATE_DIR}{TEMPLATE_NAME}_{next_num}.npy"
                np.save(file_path, feature_array)

                print(f"✅ Saved {len(feature_array)} feature frames to {file_path}")
                print(f"   Array shape: {feature_array.shape}")
                print("   Preview of first 3 frames (speed, curvature, verticality, turn_magnitude):")
                print(feature_array[:3])

                recompute_norm_stats()
            else:
                print("Error: Not enough points to extract features.")
        else:
            print(f"Error: Recording too short ({len(recording)} fixes). No file saved.")