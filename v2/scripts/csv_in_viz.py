import serial
import time
import threading
import math
from collections import deque
import yaml

import numpy as np
from scipy.optimize import minimize
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
PLOT_DIMENSIONS = config["settings"].get("plot_dimensions", 3)

# EKF Settings
USE_KALMAN = config["settings"].get("use_kalman_filter", True)
DT = config["settings"].get("dt", 0.05)
Q_VAR = config["settings"].get("process_noise", 0.1)
R_VAR = config["settings"].get("measurement_noise", 0.5)

ANCHORS = config["anchors"]
ANCHOR_IDS = list(ANCHORS.keys())
ANCHOR_COORDS = np.array([ANCHORS[aid] for aid in ANCHOR_IDS])
NUM_ANCHORS = len(ANCHOR_IDS)

MAX_QUEUE = 200
PLOT_LIMITS = (-0.5, 2.0)

# ==========================================
# EXTENDED KALMAN FILTER (EKF)
# ==========================================
class UWB_EKF:
    def __init__(self, anchor_coords, dt, q_variance, r_variance):
        self.anchors = anchor_coords
        self.num_anchors = len(anchor_coords)
        
        # State vector: [x, y, z, vx, vy, vz]^T
        self.x = np.zeros((6, 1)) 
        self.P = np.eye(6) * 1.0  # Covariance matrix

        # State transition matrix F (Physics model)
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        self.Q = np.eye(6) * q_variance # Process noise
        self.R = np.eye(self.num_anchors) * r_variance # Measurement noise
        self.initialized = False

    def initialize(self, initial_pos):
        self.x[0:3, 0] = initial_pos
        self.x[3:6, 0] = 0.0 # Assume starting at rest
        self.initialized = True

    def predict(self):
        # Predict next state and uncertainty
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurements):
        z = np.array(measurements).reshape(-1, 1)

        # Expected measurements h(x) and Jacobian H
        h = np.zeros((self.num_anchors, 1))
        H = np.zeros((self.num_anchors, 6))
        pos = self.x[0:3, 0]

        for i, anc in enumerate(self.anchors):
            dist = np.linalg.norm(pos - anc)
            h[i, 0] = dist
            if dist > 1e-6: # Prevent division by zero
                dx = (pos[0] - anc[0]) / dist
                dy = (pos[1] - anc[1]) / dist
                dz = (pos[2] - anc[2]) / dist
                H[i, 0:3] = [dx, dy, dz]

        # Calculate Kalman Gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update State with measurements
        y = z - h
        self.x = self.x + K @ y

        # Update Covariance
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P

    def get_position(self):
        return self.x[0:3, 0]

    def get_velocity(self):
        return self.x[3:6, 0]

# Initialize the EKF Instance
ekf = UWB_EKF(ANCHOR_COORDS, DT, Q_VAR, R_VAR)

# ==========================================
# TRILATERATION SOLVER (Fallback/Initializer)
# ==========================================
def solve_position(ranges, anchor_coords):
    initial_guess = np.mean(anchor_coords, axis=0)
    def error_function(guess):
        error = 0
        for i, anchor in enumerate(anchor_coords):
            dist = np.linalg.norm(guess - anchor)
            error += (dist - ranges[i])**2
        return error
    result = minimize(error_function, initial_guess, method='L-BFGS-B')
    return result.x

# ==========================================
# SERIAL / DATA READER
# ==========================================
loc_queue = deque(maxlen=MAX_QUEUE)
stop_flag = False

def data_reader():
    global stop_flag
    if USE_MOCK_DATA:
        t_start = time.perf_counter()
        while not stop_flag:
            t = time.perf_counter() - t_start
            mock_x = 0.5 + 0.3 * math.sin(t * 1.0)
            mock_y = 0.5 + 0.3 * math.cos(t * 1.0)
            mock_z = 0.3 + 0.1 * math.sin(t * 2.0)
            
            mock_ranges = []
            for ax, ay, az in ANCHOR_COORDS:
                dist = math.sqrt((mock_x - ax)**2 + (mock_y - ay)**2 + (mock_z - az)**2)
                mock_ranges.append(dist + np.random.uniform(-0.05, 0.05))
            
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
    parts = line.split(",")
    if len(parts) == NUM_ANCHORS:
        try:
            ranges = [float(p) for p in parts]
            
            if USE_KALMAN:
                if not ekf.initialized:
                    # Use Least Squares just once to find the starting point
                    initial_pos = solve_position(ranges, ANCHOR_COORDS)
                    ekf.initialize(initial_pos)
                    tag_pos = initial_pos
                else:
                    ekf.predict()
                    ekf.update(ranges)
                    tag_pos = ekf.get_position()
            else:
                # Raw Least Squares (Jittery)
                tag_pos = solve_position(ranges, ANCHOR_COORDS)

            range_dict = {ANCHOR_IDS[i]: ranges[i] for i in range(NUM_ANCHORS)}
            loc_queue.append((time.perf_counter(), tag_pos[0], tag_pos[1], tag_pos[2], range_dict))
            
        except ValueError:
            pass 

# ==========================================
# MAIN VISUALIZATION
# ==========================================
def main():
    global stop_flag

    threading.Thread(target=data_reader, daemon=True).start()

    fig = plt.figure(figsize=(10, 8))
    if PLOT_DIMENSIONS == 3:
        ax = fig.add_subplot(projection='3d')
        ax.set_zlim(0, 1.5)
        ax.set_zlabel("Z (meters)")
        ax.set_title(f"Real-Time 3D UWB Tracking {'(EKF Filtered)' if USE_KALMAN else '(Raw Least Squares)'}")
    else:
        ax = fig.add_subplot()
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"Real-Time 2D UWB Tracking {'(EKF Filtered)' if USE_KALMAN else '(Raw Least Squares)'}")

    ax.set_xlim(PLOT_LIMITS)
    ax.set_ylim(PLOT_LIMITS)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.grid(True, linestyle=':', alpha=0.4)

    for aid, pos in ANCHORS.items():
        if PLOT_DIMENSIONS == 3:
            ax.scatter(pos[0], pos[1], pos[2], color='red', marker='^', s=120, zorder=5)
            ax.text(pos[0], pos[1], pos[2] + 0.05, aid, color='red', weight='bold', ha='center')
        else:
            ax.scatter(pos[0], pos[1], color='red', marker='^', s=120, zorder=5)
            ax.text(pos[0], pos[1] + 0.05, aid, color='red', weight='bold', ha='center')

    # 2. Initialize Tag
    if PLOT_DIMENSIONS == 3:
        tag_scatter, = ax.plot([], [], [], marker='o', color='blue', markersize=14, linestyle='None', label='Tag', zorder=10)
    else:
        tag_scatter, = ax.plot([], [], marker='o', color='blue', markersize=14, linestyle='None', label='Tag', zorder=10)
        
    dynamic_lines = {}
    ax.legend(loc="upper right")

    info_text = ax.text2D(0.02, 0.02, "Waiting for data...", fontsize=10, family="monospace",
                          transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                          verticalalignment='bottom') if PLOT_DIMENSIONS == 3 else \
                ax.text(0.02, 0.02, "Waiting for data...", fontsize=10, family="monospace",
                        transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                        verticalalignment='bottom')

    def update(_):
        if not loc_queue:
            return [tag_scatter, info_text] + list(dynamic_lines.values())

        latest = loc_queue.pop()
        loc_queue.clear()
        
        t, tag_x, tag_y, tag_z, ranges = latest
        
        # Display velocity if Kalman is on
        vel_text = ""
        if USE_KALMAN and ekf.initialized:
            vx, vy, vz = ekf.get_velocity()
            vel_text = f"Velocity:\n Vx: {vx:6.3f} m/s\n Vy: {vy:6.3f} m/s\n Vz: {vz:6.3f} m/s\n\n"

        dist_text = f"Calculated Position:\n X: {tag_x:6.3f} m\n Y: {tag_y:6.3f} m\n Z: {tag_z:6.3f} m\n\n{vel_text}Reported Ranges:\n"
        
        if PLOT_DIMENSIONS == 3:
            tag_scatter.set_data_3d([tag_x], [tag_y], [tag_z])
        else:
            tag_scatter.set_data([tag_x], [tag_y])
        
        for aid, reported_range in ranges.items():
            dist_text += f" {aid}: {reported_range:6.3f} m\n"
            ax_pos, ay_pos, az_pos = ANCHORS[aid]
            
            if aid not in dynamic_lines:
                dynamic_lines[aid], = ax.plot([], [], [], linestyle='--', color='gray', alpha=0.5, zorder=3) if PLOT_DIMENSIONS == 3 else ax.plot([], [], linestyle='--', color='gray', alpha=0.5, zorder=3)
            
            if PLOT_DIMENSIONS == 3:
                dynamic_lines[aid].set_data_3d([ax_pos, tag_x], [ay_pos, tag_y], [az_pos, tag_z])
            else:
                dynamic_lines[aid].set_data([ax_pos, tag_x], [ay_pos, tag_y])

        info_text.set_text(dist_text)
        return [tag_scatter, info_text] + list(dynamic_lines.values())

    ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False, blit=(PLOT_DIMENSIONS == 2))

    try:
        plt.show()
    finally:
        stop_flag = True

if __name__ == "__main__":
    main()