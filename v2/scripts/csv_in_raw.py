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
# PRODUCTION UWB SOLVER (NLLS + Soft-L1)
# ==========================================
# def solve_position(valid_ranges_dict, dimensions=3, fixed_z=0.0):
#     valid_aids = list(valid_ranges_dict.keys())
#     ranges = np.array([valid_ranges_dict[aid] for aid in valid_aids])
#     anchor_coords = np.array([ANCHORS[aid] for aid in valid_aids])

#     # --- STEP 1: Linear Least Squares Initial Guess ---
#     ref_anchor = anchor_coords[-1]
#     ref_range = ranges[-1]
    
#     A, b = [], []
#     for i in range(len(anchor_coords) - 1):
#         x_i, y_i, z_i = anchor_coords[i]
#         x_ref, y_ref, z_ref = ref_anchor
        
#         if dimensions == 3:
#             A.append([2 * (x_i - x_ref), 2 * (y_i - y_ref), 2 * (z_i - z_ref)])
#             b.append((ref_range**2 - ranges[i]**2) + (x_i**2 - x_ref**2) + (y_i**2 - y_ref**2) + (z_i**2 - z_ref**2))
#         elif dimensions == 2:
#             A.append([2 * (x_i - x_ref), 2 * (y_i - y_ref)])
#             b.append((ref_range**2 - ranges[i]**2) + (x_i**2 - x_ref**2) + (y_i**2 - y_ref**2) + (z_i**2 - z_ref**2) - (2 * (z_i - z_ref) * fixed_z))
            
#     tag_pos_guess, _, _, _ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    
#     # --- STEP 2: NLLS Refinement (Unbounded) ---
#     def residuals_3d(pos, anchors, measured_ranges):
#         return np.linalg.norm(anchors - pos, axis=1) - measured_ranges
        
#     def residuals_2d(pos_2d, anchors, measured_ranges, z):
#         pos_3d = np.array([pos_2d[0], pos_2d[1], z])
#         return np.linalg.norm(anchors - pos_3d, axis=1) - measured_ranges

#     if dimensions == 3:
#         # If A was underdetermined (e.g. only 3 anchors in 3D), NLLS will still try to find the best intersection
#         guess = tag_pos_guess[:3] if len(tag_pos_guess) >= 3 else np.zeros(3)
#         res = least_squares(
#             residuals_3d, guess, 
#             loss='soft_l1', f_scale=0.1, args=(anchor_coords, ranges)
#         )
#         return res.x
#     else:
#         guess = tag_pos_guess[:2] if len(tag_pos_guess) >= 2 else np.zeros(2)
#         res = least_squares(
#             residuals_2d, guess, 
#             loss='soft_l1', f_scale=0.1, args=(anchor_coords, ranges, fixed_z)
#         )
#         return np.array([res.x[0], res.x[1], fixed_z])

# ==========================================
# WEIGHTED CENTROID LOCALIZATION (WCL)
# ==========================================
# def solve_position(valid_ranges_dict, dimensions=3, fixed_z=0.0):
#     valid_aids = list(valid_ranges_dict.keys())
#     ranges = np.array([valid_ranges_dict[aid] for aid in valid_aids])
#     anchor_coords = np.array([ANCHORS[aid] for aid in valid_aids])

#     # We need at least 1 valid anchor to do this math
#     if len(ranges) == 0:
#         return np.zeros(3) if dimensions == 3 else np.array([0.0, 0.0, fixed_z])

#     # 'g' is the weighting exponent. 
#     # g = 1.0 means linear pull.
#     # g = 2.0 or 3.0 gives closer anchors a much more aggressive pull.
#     g = 2.0 
    
#     # Calculate the pull weight for each anchor (1 / distance^g)
#     # We add 1e-6 to prevent division by zero if a distance is exactly 0.0
#     weights = 1.0 / (ranges ** g + 1e-6)
    
#     # Calculate the sum of all weights
#     sum_weights = np.sum(weights)
    
#     if dimensions == 3:
#         # Multiply each anchor's 3D coordinate by its weight, sum them, and divide by total weight
#         pos_3d = np.sum(anchor_coords * weights[:, np.newaxis], axis=0) / sum_weights
#         return pos_3d
#     else:
#         # Isolate just the X and Y coordinates of the anchors
#         anchor_coords_2d = anchor_coords[:, :2]
        
#         # Multiply each anchor's 2D coordinate by its weight, sum them, and divide by total weight
#         pos_2d = np.sum(anchor_coords_2d * weights[:, np.newaxis], axis=0) / sum_weights
        
#         return np.array([pos_2d[0], pos_2d[1], fixed_z])

# ==========================================
# HYBRID UWB SOLVER (WCL Seed + NLLS + Blend)
# ==========================================
def solve_position(valid_ranges_dict, dimensions=3, fixed_z=0.0):
    valid_aids = list(valid_ranges_dict.keys())
    ranges = np.array([valid_ranges_dict[aid] for aid in valid_aids])
    anchor_coords = np.array([ANCHORS[aid] for aid in valid_aids])

    if len(ranges) < 3:
        return np.zeros(3) if dimensions == 3 else np.array([0.0, 0.0, fixed_z])

    # --- STEP 1: WCL (The Stable Center) ---
    # Lowered 'g' from 2.0 to 1.5 to make the center-pull less aggressive
    g = 1.5 
    weights = 1.0 / (ranges ** g + 1e-6)
    sum_weights = np.sum(weights)
    
    if dimensions == 3:
        pos_wcl = np.sum(anchor_coords * weights[:, np.newaxis], axis=0) / sum_weights
        wcl_guess = pos_wcl 
    else:
        anchor_coords_2d = anchor_coords[:, :2]
        pos_wcl_2d = np.sum(anchor_coords_2d * weights[:, np.newaxis], axis=0) / sum_weights
        pos_wcl = np.array([pos_wcl_2d[0], pos_wcl_2d[1], fixed_z])
        wcl_guess = pos_wcl_2d 

    # --- STEP 2: NLLS (The Responsive Tracker) ---
    def residuals_3d(pos, anchors, measured_ranges):
        return np.linalg.norm(anchors - pos, axis=1) - measured_ranges
        
    def residuals_2d(pos_2d, anchors, measured_ranges, z):
        pos_3d = np.array([pos_2d[0], pos_2d[1], z])
        return np.linalg.norm(anchors - pos_3d, axis=1) - measured_ranges

    BOUNDARY_MARGIN = 0.5
    
    if dimensions == 3:
        min_bounds = np.min(anchor_coords, axis=0) - BOUNDARY_MARGIN
        max_bounds = np.max(anchor_coords, axis=0) + BOUNDARY_MARGIN
        
        guess = np.clip(wcl_guess, min_bounds + 1e-5, max_bounds - 1e-5)
        
        res = least_squares(
            residuals_3d, guess, 
            bounds=(min_bounds, max_bounds),
            loss='soft_l1', f_scale=0.1, args=(anchor_coords, ranges)
        )
        pos_nlls = res.x
    else:
        min_bounds = np.min(anchor_coords[:, :2], axis=0) - BOUNDARY_MARGIN
        max_bounds = np.max(anchor_coords[:, :2], axis=0) + BOUNDARY_MARGIN
        
        guess = np.clip(wcl_guess, min_bounds + 1e-5, max_bounds - 1e-5)
        
        res = least_squares(
            residuals_2d, guess, 
            bounds=(min_bounds, max_bounds),
            loss='soft_l1', f_scale=0.1, args=(anchor_coords, ranges, fixed_z)
        )
        pos_nlls = np.array([res.x[0], res.x[1], fixed_z])

    # --- STEP 3: THE BLEND ---
    # Increased to 0.85 to heavily favor the accuracy and reactivity of NLLS
    BLEND_FACTOR = 0.55

    final_pos = (pos_wcl * (1.0 - BLEND_FACTOR)) + (pos_nlls * BLEND_FACTOR)
    
    return final_pos

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