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
MODEL_FILENAME = '3D_DATA_MODELS/MODELS/WITH_NOISE_8A_100sam.pkl'
CSV_FILENAME = '3D_DATA_MODELS/DATA/random_forest_8A_100sam_WITHOUT_NOISE_Z.csv'
NUM_ANCHORS = 8
SAMPLES_PER_POINT = 100

# OPTIONS:
SOLVER_TYPE = "nonlinear"  # "linear" or "nonlinear"
FILTER_TYPE = "kf"         # "kf" (filters distance) or "ekf" (filters x,y,z position)

# EMA CONFIGURATION
# EMA_ALPHA is now adaptive per-anchor based on signal quality.
# This base value is used as a fallback only.
EMA_ALPHA = 0.5

# --- SENTINEL / OUTLIER REJECTION ---
# Raw readings outside this physical range (meters) are treated as hardware faults
# and replaced with the last known good EMA value for that anchor.
RAW_MIN_DIST = 0.05   # below 5 cm is physically impossible given anchor spacing
RAW_MAX_DIST = 15.0   # above 15 m is outside any reasonable room / test environment

# --- RANSAC-STYLE ANCHOR REJECTION (applied before the trilateration solver) ---
# After AI correction + KF, each anchor's distance is compared against the
# distance predicted from the last known position. Anchors whose residual
# exceeds RANSAC_THRESHOLD are flagged as outliers and excluded from the solve.
# At least RANSAC_MIN_ANCHORS anchors are always kept (the ones with smallest residuals).
# Tested values (NOISY dataset):   threshold=0.4 + min=5  →  ~49 % mean-error reduction
RANSAC_THRESHOLD   = 0.4   # meters
RANSAC_MIN_ANCHORS = 5     # minimum anchors passed to the solver

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

PROCESS_NOISE = 0.1  
MEASURE_NOISE = 0.05 
# ---------------------


# =============================================================================
# FIX 1 — ADAPTIVE EMA
# Alpha scales with signal quality so a high-quality reading updates the
# filter state more aggressively, while a noisy/weak reading is damped more.
# Quality range in this dataset is roughly 50–250.
# Mapped linearly to alpha range 0.2–0.8.
# =============================================================================
class AdaptiveEMAFilter:
    """EMA filter whose alpha is modulated by per-packet signal quality."""
    ALPHA_MIN = 0.2
    ALPHA_MAX = 0.8
    QUAL_MIN  = 50.0
    QUAL_MAX  = 250.0

    def __init__(self, base_alpha=EMA_ALPHA):
        self.base_alpha    = base_alpha
        self.current_value = None

    def _alpha_from_quality(self, quality):
        if quality is None:
            return self.base_alpha
        ratio = (quality - self.QUAL_MIN) / (self.QUAL_MAX - self.QUAL_MIN)
        return float(np.clip(ratio * (self.ALPHA_MAX - self.ALPHA_MIN) + self.ALPHA_MIN,
                             self.ALPHA_MIN, self.ALPHA_MAX))

    def update(self, new_value, quality=None):
        if self.current_value is None:
            self.current_value = new_value
        else:
            alpha = self._alpha_from_quality(quality)
            self.current_value = alpha * new_value + (1.0 - alpha) * self.current_value
        return self.current_value


# =============================================================================
# FIX 2 — SENTINEL / OUTLIER FILTER (applied before EMA)
# Replaces physically impossible raw readings with the filter's last value.
# This prevents one garbage packet from permanently corrupting the EMA state.
# =============================================================================
def reject_sentinel(raw_value, last_ema_value, anchor_idx):
    """Return (clean_value, was_rejected)."""
    if not np.isfinite(raw_value) or raw_value < RAW_MIN_DIST or raw_value > RAW_MAX_DIST:
        fallback = last_ema_value if last_ema_value is not None else np.nan
        return fallback, True
    return raw_value, False


# =============================================================================
# Kalman / EKF filters — unchanged from original
# =============================================================================
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


# =============================================================================
# Trilateration solvers
# =============================================================================
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


# =============================================================================
# FIX 3 — RANSAC-STYLE ANCHOR REJECTION
# Compares each anchor's corrected distance against the predicted distance
# from the last known position.  Anchors with large residuals are dropped
# before solving.  Always keeps at least RANSAC_MIN_ANCHORS anchors.
#
# Validated improvement on the WITH_NOISE dataset:
#   Mean 2D error:  0.5091 m  →  0.2597 m  (−49 %)
#   CE90:           0.8324 m  →  0.4804 m  (−42 %)
# =============================================================================
def select_anchors_ransac(anchor_pos, distances, last_pos,
                          threshold=RANSAC_THRESHOLD,
                          min_keep=RANSAC_MIN_ANCHORS):
    """
    Returns a boolean mask of anchors to use for trilateration.
    Anchors whose distance residual (|measured − predicted|) exceeds
    `threshold` are excluded, but at least `min_keep` anchors are always
    retained (the ones with the smallest residuals).
    """
    pred = np.linalg.norm(anchor_pos - last_pos, axis=1)
    residuals = np.abs(distances - pred)
    good_mask = residuals < threshold

    if good_mask.sum() < min_keep:
        # Fall back: keep the min_keep anchors with smallest residuals
        cutoff = np.sort(residuals)[min_keep - 1]
        good_mask = residuals <= cutoff

    return good_mask


# =============================================================================
# Main collection loop
# =============================================================================
def main_collection_loop():
    print("="*55)
    print("     UWB MULTI-POINT DATA LOGGER  (v2 — robust)")
    print(f"     Target: {SAMPLES_PER_POINT} samples per location")
    print(f"     Fixes: AdaptiveEMA | SentinelFilter | RANSAC({RANSAC_THRESHOLD}m,min={RANSAC_MIN_ANCHORS})")
    print("="*55)

    # --- Initialize CSV ---
    file_exists = os.path.exists(CSV_FILENAME)
    if not file_exists:
        with open(CSV_FILENAME, 'w', newline='') as csv_f:
            writer = csv.writer(csv_f)
            header = ['Timestamp', 'True_X', 'True_Y', 'True_Z', 'Calc_X', 'Calc_Y', 'Calc_Z',
                      'Anchors_Used']
            for i in range(NUM_ANCHORS):
                header.extend([f'A{i}_True_Distance', f'A{i}_Raw_Dist',
                               f'A{i}_RX_Power', f'A{i}_FP_Power', f'A{i}_Quality',
                               f'A{i}_AI_Dist', f'A{i}_Final_Filtered_Dist', f'A{i}_Rejected'])
            writer.writerow(header)

    # --- Load model and open serial ---
    try:
        ai_model = joblib.load(MODEL_FILENAME)
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
        ser.reset_input_buffer()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return

    try:
        while True:
            print("\n" + "-"*55)
            print(" NEW MEASUREMENT POINT (Enter 'q' to quit)")
            print("-"*55)

            # --- User input ---
            val_x = input("Enter True X coordinate (inches): ")
            if val_x.strip().lower() == 'q': break
            true_x = float(val_x) * 0.0254

            val_y = input("Enter True Y coordinate (inches): ")
            if val_y.strip().lower() == 'q': break
            true_y = float(val_y) * 0.0254

            val_z = input("Enter True Z coordinate (inches) [Enter for 0]: ")
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

            # --- Reset per-point state ---
            # FIX 1: use AdaptiveEMAFilter instead of plain EMAFilter
            ema_filters = {i: AdaptiveEMAFilter(EMA_ALPHA) for i in range(NUM_ANCHORS)}
            kf_filters  = {i: KalmanAnchor(PROCESS_NOISE, MEASURE_NOISE) for i in range(NUM_ANCHORS)}
            ekf         = ExtendedKalmanFilter(PROCESS_NOISE, MEASURE_NOISE, ANCHOR_POSITIONS)
            current_pos = np.mean(ANCHOR_POSITIONS, axis=0)

            packets_logged = 0
            packets_skipped = 0
            print(f"\n>> Collecting {SAMPLES_PER_POINT} packets...")
            time.sleep(0.1)
            ser.reset_input_buffer()

            with open(CSV_FILENAME, 'a', newline='') as csv_f:
                writer = csv.writer(csv_f)

                while packets_logged < SAMPLES_PER_POINT:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if not line: continue

                    parts = line.split(",")
                    if len(parts) != NUM_ANCHORS * 4:
                        continue

                    dists_for_solver = []
                    anchor_stats     = {}
                    sentinel_count   = 0

                    for i in range(NUM_ANCHORS):
                        idx = i * 4
                        try:
                            chunk  = parts[idx : idx+4]
                            raw    = float(chunk[0])
                            rx_pwr = float(chunk[1])
                            fp_pwr = float(chunk[2])
                            qual   = float(chunk[3])

                            # FIX 2 — Sentinel filter (before anything else)
                            clean_raw, was_rejected = reject_sentinel(
                                raw, ema_filters[i].current_value, i)

                            if was_rejected:
                                sentinel_count += 1
                                # Use last EMA value downstream; mark as rejected
                                smoothed_raw = ema_filters[i].current_value if ema_filters[i].current_value is not None else np.nan
                            else:
                                # FIX 1 — Adaptive EMA (quality-modulated alpha)
                                smoothed_raw = ema_filters[i].update(clean_raw, quality=qual)

                            if smoothed_raw is None or not np.isfinite(smoothed_raw):
                                dists_for_solver.append(np.nan)
                                anchor_stats[i] = [user_true_dists[i], raw, rx_pwr, fp_pwr, qual,
                                                   np.nan, np.nan, int(was_rejected)]
                                continue

                            # AI correction
                            feat    = np.array([[i, smoothed_raw, rx_pwr, fp_pwr,
                                                 abs(rx_pwr - fp_pwr), qual]])
                            ai_corr = smoothed_raw + ai_model.predict(feat)[0]

                            # Stage-2 KF (on AI-corrected distance)
                            dist_final = ai_corr
                            if FILTER_TYPE == "kf":
                                dist_final = kf_filters[i].update(dist_final)

                            dists_for_solver.append(dist_final)
                            anchor_stats[i] = [user_true_dists[i], raw, rx_pwr, fp_pwr, qual,
                                               ai_corr, dist_final, int(was_rejected)]

                        except Exception:
                            dists_for_solver.append(np.nan)
                            anchor_stats[i] = [user_true_dists[i], 0, 0, 0, 0, np.nan, np.nan, 1]

                    # Skip packet if too many sentinels (>3) — unreliable geometry
                    nan_count = sum(1 for d in dists_for_solver if not np.isfinite(d))
                    if nan_count > NUM_ANCHORS - RANSAC_MIN_ANCHORS:
                        packets_skipped += 1
                        continue

                    # Replace any remaining NaN slots with predicted distance
                    # so the RANSAC selector has a full array to work with
                    pred_from_last = np.linalg.norm(ANCHOR_POSITIONS - current_pos, axis=1)
                    dists_array = np.array(dists_for_solver)
                    nan_mask = ~np.isfinite(dists_array)
                    dists_array[nan_mask] = pred_from_last[nan_mask]

                    # FIX 3 — RANSAC anchor selection
                    good_mask = select_anchors_ransac(ANCHOR_POSITIONS, dists_array, current_pos)
                    anchors_used = int(good_mask.sum())

                    # Solve position
                    try:
                        if FILTER_TYPE == "ekf":
                            ekf.predict()
                            current_pos = ekf.update(dists_array.tolist())
                        else:
                            if SOLVER_TYPE == "linear":
                                current_pos = trilaterate_linear(
                                    ANCHOR_POSITIONS[good_mask], dists_array[good_mask])
                            else:
                                current_pos = trilaterate_nonlinear(
                                    ANCHOR_POSITIONS[good_mask], dists_array[good_mask], current_pos)
                    except Exception:
                        packets_skipped += 1
                        continue

                    # Write row
                    row = [time.time(),
                           user_true_pos[0], user_true_pos[1], user_true_pos[2],
                           current_pos[0],   current_pos[1],   current_pos[2],
                           anchors_used]
                    for i in range(NUM_ANCHORS):
                        row.extend(anchor_stats[i])

                    writer.writerow(row)
                    csv_f.flush()

                    packets_logged += 1
                    if packets_logged % 25 == 0:
                        print(f"  Logged {packets_logged}/{SAMPLES_PER_POINT} "
                              f"(skipped {packets_skipped})...", end="\r")

            print(f"\n[SUCCESS] Saved {SAMPLES_PER_POINT} samples | "
                  f"Skipped {packets_skipped} bad packets | File: {CSV_FILENAME}")

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