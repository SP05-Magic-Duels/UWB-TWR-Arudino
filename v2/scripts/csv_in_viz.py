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

USE_MOCK_DATA = config["settings"]["use_mock_data"]
SERIAL_PORT = config["settings"]["serial_port"]
BAUD_RATE = config["settings"]["baud_rate"]
PLOT_DIMENSIONS = config["settings"]["plot_dimensions"]
ANCHORS = config["anchors"]

# Extract ordered anchor IDs and their coordinates
ANCHOR_IDS = list(ANCHORS.keys())
ANCHOR_COORDS = np.array([ANCHORS[aid] for aid in ANCHOR_IDS])
NUM_ANCHORS = len(ANCHOR_IDS)

MAX_QUEUE = 200
PLOT_LIMITS = (-1.0, 5.0)

# ==========================================
# TRILATERATION SOLVER
# ==========================================
def solve_position(ranges, anchor_coords):
    """Calculates X,Y,Z coordinates from raw distances using Least Squares"""
    # Initial guess: start in the mathematical center of all anchors
    initial_guess = np.mean(anchor_coords, axis=0)

    def error_function(guess):
        error = 0
        for i, anchor in enumerate(anchor_coords):
            # Calculate distance from guess to anchor, subtract real range, and square the error
            dist = np.linalg.norm(guess - anchor)
            error += (dist - ranges[i])**2
        return error

    # Run the optimizer to find the coordinate with the lowest error
    result = minimize(error_function, initial_guess, method='L-BFGS-B')
    return result.x

# ==========================================
# SERIAL / DATA READER
# ==========================================
loc_queue = deque(maxlen=MAX_QUEUE)
stop_flag = False

def data_reader():
    """Background thread to read CSV serial data or generate mock data."""
    global stop_flag
    
    if USE_MOCK_DATA:
        t_start = time.perf_counter()
        while not stop_flag:
            t = time.perf_counter() - t_start
            
            # Simulate a tag moving in a circle
            mock_x = 1.5 + 1.0 * math.sin(t * 1.0)
            mock_y = 1.25 + 1.0 * math.cos(t * 1.0)
            mock_z = 1.0 + 0.2 * math.sin(t * 2.0)
            
            # Generate perfect ranges and add a tiny bit of noise
            mock_ranges = []
            for ax, ay, az in ANCHOR_COORDS:
                dist = math.sqrt((mock_x - ax)**2 + (mock_y - ay)**2 + (mock_z - az)**2)
                mock_ranges.append(dist + np.random.uniform(-0.05, 0.05))
            
            # Simulate reading a CSV line
            csv_line = ",".join([f"{r:.2f}" for r in mock_ranges])
            process_csv_line(csv_line)
            
            time.sleep(0.05) # 20 Hz update rate
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
    """Parses a CSV string and calculates position"""
    parts = line.split(",")
    
    if len(parts) == NUM_ANCHORS:
        try:
            ranges = [float(p) for p in parts]
            # Calculate position using our PC's CPU
            tag_x, tag_y, tag_z = solve_position(ranges, ANCHOR_COORDS)
            
            # Map ranges back to their IDs for the UI
            range_dict = {ANCHOR_IDS[i]: ranges[i] for i in range(NUM_ANCHORS)}
            loc_queue.append((time.perf_counter(), tag_x, tag_y, tag_z, range_dict))
        except ValueError:
            pass # Ignore corrupted lines where a float conversion fails

# ==========================================
# MAIN VISUALIZATION
# ==========================================
def main():
    global stop_flag

    threading.Thread(target=data_reader, daemon=True).start()

    fig = plt.figure(figsize=(10, 8))
    
    if PLOT_DIMENSIONS == 3:
        ax = fig.add_subplot(projection='3d')
        ax.set_zlim(0, 3.0)
        ax.set_zlabel("Z (meters)")
        ax.set_title("Real-Time 3D UWB Tracking")
    else:
        ax = fig.add_subplot()
        ax.set_aspect('equal', adjustable='box')
        ax.set_title("Real-Time 2D UWB Tracking")

    ax.set_xlim(PLOT_LIMITS)
    ax.set_ylim(PLOT_LIMITS)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.grid(True, linestyle=':', alpha=0.4)

    # 1. Plot Known Anchors
    for aid, pos in ANCHORS.items():
        if PLOT_DIMENSIONS == 3:
            ax.scatter(pos[0], pos[1], pos[2], color='red', marker='^', s=120, zorder=5)
            ax.text(pos[0], pos[1], pos[2] + 0.15, aid, color='red', weight='bold', ha='center')
        else:
            ax.scatter(pos[0], pos[1], color='red', marker='^', s=120, zorder=5)
            ax.text(pos[0], pos[1] + 0.15, aid, color='red', weight='bold', ha='center')

    # 2. Initialize Tag
    if PLOT_DIMENSIONS == 3:
        tag_scatter, = ax.plot([], [], [], marker='o', color='blue', markersize=14, linestyle='None', label='Tag', zorder=10)
    else:
        tag_scatter, = ax.plot([], [], marker='o', color='blue', markersize=14, linestyle='None', label='Tag', zorder=10)

    # 3. Dynamic Distance Lines
    dynamic_lines = {}
    ax.legend(loc="upper right")

    # 4. Text Box Readout
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

        # Format Text
        dist_text = f"Calculated Position:\n X: {tag_x:6.3f} m\n Y: {tag_y:6.3f} m\n Z: {tag_z:6.3f} m\n\nReported Ranges:\n"
        
        # Update Tag Graphics
        if PLOT_DIMENSIONS == 3:
            tag_scatter.set_data_3d([tag_x], [tag_y], [tag_z])
        else:
            tag_scatter.set_data([tag_x], [tag_y])
        
        # Update Dynamic Lines
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