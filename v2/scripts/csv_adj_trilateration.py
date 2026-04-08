import serial
import time
import threading
import math
from collections import deque
import yaml

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3

# ==========================================
# LOAD CONFIGURATION
# ==========================================
try:
    with open("csv_in_viz.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("ERROR: csv_in_viz.yaml not found! Please create it in the same directory.")
    exit()

USE_MOCK_DATA = config["settings"].get("use_mock_data", False)
SERIAL_PORT = config["settings"].get("serial_port", "COM3")
BAUD_RATE = config["settings"].get("baud_rate", 115200)
DT = config["settings"].get("dt", 0.05) 

# Set to 2 or 3 to toggle the solver and visualization mode
PLOT_DIMENSIONS = config["settings"].get("plot_dimensions", 2) 
TAG_Z_2D = config["settings"].get("fixed_tag_z_2d", 0.0) 

ANCHORS = config["anchors"]
ANCHOR_IDS = list(ANCHORS.keys())
ANCHOR_COORDS = np.array([ANCHORS[aid] for aid in ANCHOR_IDS])
NUM_ANCHORS = len(ANCHOR_IDS)

MAX_QUEUE = 1 
# Dynamic plotting limits based on anchor spread
min_xy = np.min(ANCHOR_COORDS[:, :2]) - 1.0
max_xy = np.max(ANCHOR_COORDS[:, :2]) + 1.0
PLOT_LIMITS = (min_xy, max_xy)

EMA_ALPHA = 0.20 # Low alpha means more smoothing

# ==========================================
# PRODUCTION UWB SOLVER (Bancroft + LM/TRF NLLS)
# ==========================================

def _bancroft_algorithm(anchor_coords, ranges):
    """
    Bancroft's algorithm for initial position estimation.
    Returns a 4D vector where the first 3 elements are [x, y, z].
    """
    n = len(ranges)
    # Lorentz inner product matrix
    e = np.ones((n, 1))
    y = anchor_coords
    r = ranges.reshape(-1, 1)
    
    # Compute alpha_i = 0.5 * (x_i^2 + y_i^2 + z_i^2 - r_i^2)
    alpha = 0.5 * (np.sum(np.square(y), axis=1).reshape(-1, 1) - np.square(r))
    
    # SVD-based pseudo-inverse for stability
    B = np.hstack((y, e))
    try:
        B_inv = np.linalg.pinv(B)
        u = B_inv @ alpha
        v = B_inv @ e
        
        # Solve quadratic equation: lambda^2 * <v,v> + 2*lambda*(<u,v> - 1) + <u,u> = 0
        # where <a,b> is the Minkowski/Lorentz inner product
        def lorentz_inner(a, b):
            return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] - a[3]*b[3]

        a_q = lorentz_inner(v, v)
        b_q = 2 * (lorentz_inner(u, v) - 1)
        c_q = lorentz_inner(u, u)
        
        discriminant = b_q**2 - 4*a_q*c_q
        if discriminant < 0:
            return u[:3].flatten() # Fallback to linear part
            
        l1 = (-b_q + np.sqrt(discriminant)) / (2 * a_q)
        l2 = (-b_q - np.sqrt(discriminant)) / (2 * a_q)
        
        pos1 = u + l1 * v
        pos2 = u + l2 * v
        
        # Typically the solution with a positive lambda or the one closer to anchors is correct
        # Here we return the one that minimizes the residual error
        res1 = np.sum(np.square(np.linalg.norm(y - pos1[:3].T, axis=1) - ranges))
        res2 = np.sum(np.square(np.linalg.norm(y - pos2[:3].T, axis=1) - ranges))
        
        return pos1[:3].flatten() if res1 < res2 else pos2[:3].flatten()
    except np.linalg.LinAlgError:
        return np.mean(anchor_coords, axis=0)

# ==========================================
# EXTENDED KALMAN FILTER (EKF) STATE
# ==========================================
class UWB_EKF:
    def __init__(self, dt, dim=2):
        self.dim = dim
        # State: [x, y, z, vx, vy, vz] if 3D else [x, y, vx, vy]
        size = 6 if dim == 3 else 4
        self.x = np.zeros((size, 1))
        self.P = np.eye(size) * 1.0  # Initial uncertainty
        
        # Process Noise (Q): How much we trust our "constant velocity" model
        q_val = 0.1
        self.Q = np.eye(size) * q_val
        
        # Measurement Noise (R): How much we trust the UWB ranges (meters)
        self.R_std = 0.15 
        
        self.dt = dt

    def predict(self):
        # State Transition Matrix (F)
        if self.dim == 3:
            F = np.eye(6)
            F[0, 3] = F[1, 4] = F[2, 5] = self.dt
        else:
            F = np.eye(4)
            F[0, 2] = F[1, 3] = self.dt
            
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, ranges_dict, anchor_coords, fixed_z):
        valid_aids = list(ranges_dict.keys())
        if not valid_aids: return

        z_meas = np.array([ranges_dict[aid] for aid in valid_aids]).reshape(-1, 1)
        h = np.zeros((len(valid_aids), 1))
        H = np.zeros((len(valid_aids), self.x.shape[0]))

        for i, aid in enumerate(valid_aids):
            a_pos = ANCHORS[aid]
            # Current estimated position
            tx, ty = self.x[0, 0], self.x[1, 0]
            tz = self.x[2, 0] if self.dim == 3 else fixed_z
            
            dist = math.sqrt((tx - a_pos[0])**2 + (ty - a_pos[1])**2 + (tz - a_pos[2])**2) + 1e-6
            h[i, 0] = dist
            
            # Jacobian H
            H[i, 0] = (tx - a_pos[0]) / dist
            H[i, 1] = (ty - a_pos[1]) / dist
            if self.dim == 3:
                H[i, 2] = (tz - a_pos[2]) / dist

        R = np.eye(len(valid_aids)) * (self.R_std**2)
        y = z_meas - h # Innovation
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.pinv(S) # Kalman Gain

        self.x = self.x + K @ y
        self.P = (np.eye(self.x.shape[0]) - K @ H) @ self.P

# Initialize EKF global
ekf = UWB_EKF(DT, dim=PLOT_DIMENSIONS)
ekf_initialized = False

def solve_position(valid_ranges_dict, dimensions=3, fixed_z=0.0):
    global ekf_initialized
    
    # Use Bancroft only for the very first frame to seed the EKF
    if not ekf_initialized:
        initial_guess = _bancroft_algorithm(
            np.array([ANCHORS[aid] for aid in valid_ranges_dict.keys()]), 
            np.array(list(valid_ranges_dict.values()))
        )
        ekf.x[0, 0] = initial_guess[0]
        ekf.x[1, 0] = initial_guess[1]
        if dimensions == 3: ekf.x[2, 0] = initial_guess[2]
        ekf_initialized = True
        return initial_guess

    ekf.predict()
    ekf.update(valid_ranges_dict, ANCHOR_COORDS, fixed_z)
    
    if dimensions == 3:
        return ekf.x[:3, 0]
    else:
        return np.array([ekf.x[0, 0], ekf.x[1, 0], fixed_z])

# ==========================================
# SERIAL / DATA READER
# ==========================================
loc_queue = deque(maxlen=MAX_QUEUE)
stop_flag = False
last_smoothed_ranges = {}

def data_reader():
    global stop_flag
    if USE_MOCK_DATA:
        t_start = time.perf_counter()
        while not stop_flag:
            t = time.perf_counter() - t_start
            mock_x = 0.5 + 0.3 * math.sin(t * 1.0)
            mock_y = 0.5 + 0.3 * math.cos(t * 1.0)
            mock_z = TAG_Z_2D if PLOT_DIMENSIONS == 2 else (0.3 + 0.1 * math.sin(t * 2.0))
            
            mock_ranges = []
            for ax, ay, az in ANCHOR_COORDS:
                dist = math.sqrt((mock_x - ax)**2 + (mock_y - ay)**2 + (mock_z - az)**2)
                mock_ranges.append(dist + 0.36 + np.random.uniform(-0.15, 0.15)) 
            
            # Occasionally simulate a missing/negative anchor signal
            if np.random.rand() > 0.95:
                mock_ranges[1] = -1.0 
                
            csv_line = ",".join([f"{r:.3f}" for r in mock_ranges])
            process_csv_line(csv_line)
            time.sleep(DT)
    else:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        try:
            while not stop_flag:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    process_csv_line(line)
        finally:
            ser.close()

def process_csv_line(line):
    global last_smoothed_ranges
    parts = line.split(",")
    if len(parts) == NUM_ANCHORS:
        try:
            raw_floats = [float(p) for p in parts]
            
            # 1. Filter out missing/negative ranges
            ranges_raw = {}
            for i, val in enumerate(raw_floats):
                if val >= 0:
                    # r = (1.1 * val) - 0.36
                    # if r > 0: # Only accept physically possible distances after offset
                    #     ranges_raw[ANCHOR_IDS[i]] = r
                    ranges_raw[ANCHOR_IDS[i]] = val
                        
            # Need at least 3 valid anchors to perform a proper solve
            if len(ranges_raw) < 3:
                return  

            # 2. Apply EMA Low-Pass Filter dynamically
            ranges_smooth = {}
            current_valid_aids = set(ranges_raw.keys())
            
            # CRITICAL: Throw out (reset) EMA history for any anchor that dropped out this cycle.
            # This prevents interpolating against a stale value when the anchor comes back online.
            stale_aids = set(last_smoothed_ranges.keys()) - current_valid_aids
            for aid in stale_aids:
                del last_smoothed_ranges[aid]

            for aid, r_raw in ranges_raw.items():
                if aid in last_smoothed_ranges:
                    ranges_smooth[aid] = (EMA_ALPHA * r_raw) + ((1.0 - EMA_ALPHA) * last_smoothed_ranges[aid])
                else:
                    ranges_smooth[aid] = r_raw
                last_smoothed_ranges[aid] = ranges_smooth[aid]
            
            # 3. Solve using NLLS on subset of valid anchors
            tag_pos_raw = solve_position(ranges_raw, PLOT_DIMENSIONS, TAG_Z_2D)
            tag_pos_smooth = solve_position(ranges_smooth, PLOT_DIMENSIONS, TAG_Z_2D)
            
            loc_queue.append((
                time.perf_counter(), 
                tag_pos_raw, tag_pos_smooth, 
                ranges_raw, ranges_smooth
            ))
        except ValueError:
            pass 

# ==========================================
# MAIN VISUALIZATION
# ==========================================
def setup_axis(ax, title):
    ax.set_title(title)
    ax.set_xlim(PLOT_LIMITS)
    ax.set_ylim(PLOT_LIMITS)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    
    if PLOT_DIMENSIONS == 3:
        max_z = np.max(ANCHOR_COORDS[:, 2]) if np.max(ANCHOR_COORDS[:, 2]) > 0 else 3.0
        ax.set_zlim(0, max_z + 1.0)
        ax.set_zlabel("Z (meters)")
    else:
        ax.set_aspect('equal', adjustable='box')
        
    ax.grid(True, linestyle=':', alpha=0.4)

    for aid, pos in ANCHORS.items():
        if PLOT_DIMENSIONS == 3:
            ax.scatter(pos[0], pos[1], pos[2], color='red', marker='^', s=120, zorder=5)
            ax.text(pos[0], pos[1], pos[2] + 0.1, aid, color='red', weight='bold', ha='center')
        else:
            ax.scatter(pos[0], pos[1], color='red', marker='^', s=120, zorder=5)
            ax.text(pos[0], pos[1] + 0.1, aid, color='red', weight='bold', ha='center')

    if PLOT_DIMENSIONS == 3:
        tag_scatter, = ax.plot([], [], [], marker='o', color='blue', markersize=12, linestyle='None', zorder=10)
        info_text = ax.text2D(0.02, 0.02, "Waiting for data...", fontsize=9, family="monospace",
                              transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                              verticalalignment='bottom')
    else:
        tag_scatter, = ax.plot([], [], marker='o', color='blue', markersize=12, linestyle='None', zorder=10)
        info_text = ax.text(0.02, 0.02, "Waiting for data...", fontsize=9, family="monospace",
                            transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                            verticalalignment='bottom')
    
    # Pre-populate empty lines for all anchors
    lines_dict = {}
    for aid in ANCHOR_IDS:
        if PLOT_DIMENSIONS == 3:
            lines_dict[aid], = ax.plot([], [], [], linestyle='--', alpha=0.4, zorder=3)
        else:
            lines_dict[aid], = ax.plot([], [], linestyle='--', alpha=0.4, zorder=3)

    return tag_scatter, info_text, lines_dict

def main():
    global stop_flag

    threading.Thread(target=data_reader, daemon=True).start()

    fig = plt.figure(figsize=(16, 8))
    
    if PLOT_DIMENSIONS == 3:
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        title1 = "Raw 3D + Unbounded NLLS"
        title2 = f"Smoothed 3D + NLLS (EMA α={EMA_ALPHA})"
    else:
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        title1 = f"Raw 2D + Unbounded NLLS (Z={TAG_Z_2D}m)"
        title2 = f"Smoothed 2D + NLLS (EMA α={EMA_ALPHA})"

    tag_raw, text_raw, lines_raw = setup_axis(ax1, title1)
    tag_smooth, text_smooth, lines_smooth = setup_axis(ax2, title2)

    def update(_):
        if not loc_queue:
            return [tag_raw, text_raw, tag_smooth, text_smooth] + list(lines_raw.values()) + list(lines_smooth.values())

        latest = loc_queue.pop()
        t, pos_raw, pos_smooth, ranges_raw, ranges_smooth = latest
        
        # --- RAW PANEL ---
        dist_text_raw = f"Raw Position:\n X: {pos_raw[0]:6.3f}\n Y: {pos_raw[1]:6.3f}\n Z: {pos_raw[2]:6.3f}\n\nRaw Ranges:\n"
        if PLOT_DIMENSIONS == 3:
            tag_raw.set_data_3d([pos_raw[0]], [pos_raw[1]], [pos_raw[2]])
        else:
            tag_raw.set_data([pos_raw[0]], [pos_raw[1]])
            
        for aid in ANCHOR_IDS:
            lines_raw[aid].set_color('gray')
            if aid in ranges_raw:
                dist_text_raw += f" {aid}: {ranges_raw[aid]:6.3f} m\n"
                ax_p, ay_p, az_p = ANCHORS[aid]
                if PLOT_DIMENSIONS == 3:
                    lines_raw[aid].set_data_3d([ax_p, pos_raw[0]], [ay_p, pos_raw[1]], [az_p, pos_raw[2]])
                else:
                    lines_raw[aid].set_data([ax_p, pos_raw[0]], [ay_p, pos_raw[1]])
            else:
                dist_text_raw += f" {aid}: MISSING (Negative)\n"
                if PLOT_DIMENSIONS == 3:
                    lines_raw[aid].set_data_3d([], [], [])
                else:
                    lines_raw[aid].set_data([], [])
                    
        text_raw.set_text(dist_text_raw)

        # --- SMOOTH PANEL ---
        dist_text_smooth = f"Smoothed Position:\n X: {pos_smooth[0]:6.3f}\n Y: {pos_smooth[1]:6.3f}\n Z: {pos_smooth[2]:6.3f}\n\nFiltered Ranges:\n"
        if PLOT_DIMENSIONS == 3:
            tag_smooth.set_data_3d([pos_smooth[0]], [pos_smooth[1]], [pos_smooth[2]])
        else:
            tag_smooth.set_data([pos_smooth[0]], [pos_smooth[1]])
            
        for aid in ANCHOR_IDS:
            lines_smooth[aid].set_color('green')
            if aid in ranges_smooth:
                dist_text_smooth += f" {aid}: {ranges_smooth[aid]:6.3f} m\n"
                ax_p, ay_p, az_p = ANCHORS[aid]
                if PLOT_DIMENSIONS == 3:
                    lines_smooth[aid].set_data_3d([ax_p, pos_smooth[0]], [ay_p, pos_smooth[1]], [az_p, pos_smooth[2]])
                else:
                    lines_smooth[aid].set_data([ax_p, pos_smooth[0]], [ay_p, pos_smooth[1]])
            else:
                dist_text_smooth += f" {aid}: MISSING (Negative)\n"
                if PLOT_DIMENSIONS == 3:
                    lines_smooth[aid].set_data_3d([], [], [])
                else:
                    lines_smooth[aid].set_data([], [])
                    
        text_smooth.set_text(dist_text_smooth)

        return [tag_raw, text_raw, tag_smooth, text_smooth] + list(lines_raw.values()) + list(lines_smooth.values())

    ani = FuncAnimation(fig, update, interval=10, cache_frame_data=False, blit=(PLOT_DIMENSIONS == 2))

    try:
        plt.tight_layout()
        plt.show()
    finally:
        stop_flag = True

if __name__ == "__main__":
    main()