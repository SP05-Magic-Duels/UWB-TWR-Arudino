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
SERIAL_PORT   = config["settings"].get("serial_port", "COM7")
BAUD_RATE     = config["settings"].get("baud_rate", 115200)

# EKF Settings
USE_KALMAN = config["settings"].get("use_kalman_filter", True)
DT         = config["settings"].get("dt", 0.01)
Q_VAR      = config["settings"].get("process_noise", 0.2)
R_VAR      = config["settings"].get("measurement_noise", 0.5)

# Anchors — only x,y used; z is ignored if present in the YAML
ANCHORS      = config["anchors"]
ANCHOR_IDS   = list(ANCHORS.keys())
ANCHOR_COORDS = np.array([ANCHORS[aid][:2] for aid in ANCHOR_IDS])  # shape (N, 2)
NUM_ANCHORS  = len(ANCHOR_IDS)

MAX_QUEUE  = 200
PLOT_LIMITS = (-0.5, 2.0)

# ==========================================
# EXTENDED KALMAN FILTER (EKF) — 2D
# ==========================================
class UWB_EKF_2D:
    """
    2D EKF — State vector: [px, py, vx, vy]

    Motion model (constant velocity):
        px' = px + vx*dt,  py' = py + vy*dt
        vx' = vx,          vy' = vy

    Measurement model per anchor i at (aix, aiy):
        h_i = sqrt((px-aix)^2 + (py-aiy)^2)

    Jacobian row i:
        [∂h/∂px, ∂h/∂py, 0, 0] = [(px-aix)/h_i, (py-aiy)/h_i, 0, 0]
    """

    def __init__(self, anchor_coords, dt, q_variance, r_variance):
        self.anchors     = anchor_coords        # shape (N, 2)
        self.num_anchors = len(anchor_coords)

        # State [px, py, vx, vy]
        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 1.0

        # State transition matrix F
        self.F = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ], dtype=float)

        self.Q = np.eye(4) * q_variance         # Process noise
        self.R = np.eye(self.num_anchors) * r_variance  # Measurement noise
        self.initialized = False

    def initialize(self, initial_pos):
        """initial_pos: (x, y)"""
        self.x[0, 0] = initial_pos[0]
        self.x[1, 0] = initial_pos[1]
        self.x[2, 0] = 0.0
        self.x[3, 0] = 0.0
        self.initialized = True

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurements):
        z   = np.array(measurements).reshape(-1, 1)
        h   = np.zeros((self.num_anchors, 1))
        H   = np.zeros((self.num_anchors, 4))
        pos = self.x[0:2, 0]                    # (px, py)

        for i, anc in enumerate(self.anchors):
            dist = np.linalg.norm(pos - anc)
            h[i, 0] = dist
            if dist > 1e-6:
                H[i, 0] = (pos[0] - anc[0]) / dist   # ∂h/∂px
                H[i, 1] = (pos[1] - anc[1]) / dist   # ∂h/∂py
                # H[i, 2] and H[i, 3] remain 0 (no velocity dependence)

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ (z - h)
        self.P = (np.eye(4) - K @ H) @ self.P

    def get_position(self):
        return self.x[0:2, 0]                   # (px, py)

    def get_velocity(self):
        return self.x[2:4, 0]                   # (vx, vy)

    @property
    def speed(self):
        return float(np.linalg.norm(self.x[2:4, 0]))

    @property
    def heading(self):
        vx, vy = self.x[2, 0], self.x[3, 0]
        return math.degrees(math.atan2(vy, vx))


# Initialize EKF
ekf = UWB_EKF_2D(ANCHOR_COORDS, DT, Q_VAR, R_VAR)

# ==========================================
# TRILATERATION SOLVER (2D fallback / initialiser)
# ==========================================
def solve_position_2d(ranges, anchor_coords):
    """Least-squares 2D position from ranges to anchors."""
    initial_guess = np.mean(anchor_coords, axis=0)   # (x, y)

    def error_function(guess):
        error = 0
        for i, anchor in enumerate(anchor_coords):
            dist = np.linalg.norm(guess - anchor)
            error += (dist - ranges[i]) ** 2
        return error

    result = minimize(error_function, initial_guess, method='L-BFGS-B')
    return result.x                                   # (x, y)

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

            mock_ranges = []
            for ax, ay in ANCHOR_COORDS:
                dist = math.sqrt((mock_x - ax)**2 + (mock_y - ay)**2)
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
                    initial_pos = solve_position_2d(ranges, ANCHOR_COORDS)
                    ekf.initialize(initial_pos)
                    tag_pos = initial_pos
                else:
                    ekf.predict()
                    ekf.update(ranges)
                    tag_pos = ekf.get_position()
            else:
                tag_pos = solve_position_2d(ranges, ANCHOR_COORDS)

            range_dict = {ANCHOR_IDS[i]: ranges[i] for i in range(NUM_ANCHORS)}
            loc_queue.append((time.perf_counter(), tag_pos[0], tag_pos[1], range_dict))

        except ValueError:
            pass

# ==========================================
# MAIN VISUALIZATION — 2D only
# ==========================================
def main():
    global stop_flag

    threading.Thread(target=data_reader, daemon=True).start()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(PLOT_LIMITS)
    ax.set_ylim(PLOT_LIMITS)
    ax.set_xlabel("X (metres)")
    ax.set_ylabel("Y (metres)")
    ax.set_title(
        f"Real-Time 2D UWB Tracking "
        f"{'(EKF Filtered)' if USE_KALMAN else '(Raw Least Squares)'}"
    )
    ax.grid(True, linestyle=':', alpha=0.4)

    # Plot anchor markers
    for aid in ANCHOR_IDS:
        pos = ANCHOR_COORDS[ANCHOR_IDS.index(aid)]
        ax.scatter(pos[0], pos[1], color='red', marker='^', s=120, zorder=5)
        ax.text(pos[0], pos[1] + 0.05, aid, color='red',
                weight='bold', ha='center')

    tag_scatter, = ax.plot([], [], marker='o', color='blue',
                           markersize=14, linestyle='None',
                           label='Tag', zorder=10)

    # Heading arrow (quiver): updated each frame
    heading_arrow = ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy',
                               scale=1, color='blue', alpha=0.6, width=0.008)

    dynamic_lines = {}
    ax.legend(loc='upper right')

    info_text = ax.text(
        0.02, 0.02, "Waiting for data...",
        fontsize=10, family='monospace',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        verticalalignment='bottom'
    )

    def update(_):
        nonlocal heading_arrow
        if not loc_queue:
            return [tag_scatter, info_text, heading_arrow] + list(dynamic_lines.values())

        latest = loc_queue.pop()
        loc_queue.clear()

        t, tag_x, tag_y, ranges = latest

        tag_scatter.set_data([tag_x], [tag_y])

        # Build info text
        vel_text = ""
        arrow_dx, arrow_dy = 0.0, 0.0
        if USE_KALMAN and ekf.initialized:
            vx, vy  = ekf.get_velocity()
            speed   = ekf.speed
            heading = ekf.heading
            vel_text = (
                f"Velocity:\n"
                f"  Vx: {vx:6.3f} m/s\n"
                f"  Vy: {vy:6.3f} m/s\n"
                f"  Speed:   {speed:5.3f} m/s\n"
                f"  Heading: {heading:6.1f} deg\n\n"
            )
            # Scale arrow to 0.15 m for visibility
            mag = math.sqrt(vx**2 + vy**2)
            if mag > 1e-4:
                arrow_dx = (vx / mag) * 0.15
                arrow_dy = (vy / mag) * 0.15

        # Update heading arrow
        heading_arrow.remove()
        heading_arrow = ax.quiver(
            tag_x, tag_y, arrow_dx, arrow_dy,
            angles='xy', scale_units='xy', scale=1,
            color='blue', alpha=0.6, width=0.008
        )

        dist_text = (
            f"Position:\n"
            f"  X: {tag_x:6.3f} m\n"
            f"  Y: {tag_y:6.3f} m\n\n"
            f"{vel_text}"
            f"Ranges:\n"
        )

        for aid, reported_range in ranges.items():
            dist_text += f"  {aid}: {reported_range:6.3f} m\n"
            anc_idx = ANCHOR_IDS.index(aid)
            ax_pos, ay_pos = ANCHOR_COORDS[anc_idx]

            if aid not in dynamic_lines:
                dynamic_lines[aid], = ax.plot(
                    [], [], linestyle='--', color='gray', alpha=0.5, zorder=3
                )
            dynamic_lines[aid].set_data([ax_pos, tag_x], [ay_pos, tag_y])

        info_text.set_text(dist_text)
        return [tag_scatter, info_text, heading_arrow] + list(dynamic_lines.values())

    ani = FuncAnimation(fig, update, interval=50,
                        cache_frame_data=False, blit=False)

    try:
        plt.show()
    finally:
        stop_flag = True


if __name__ == "__main__":
    main()