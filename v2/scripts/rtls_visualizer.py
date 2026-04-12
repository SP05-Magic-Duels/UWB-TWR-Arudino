import serial
import time
import numpy as np
import csv
import re
import os
import threading
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ================= CONFIG =================
SERIAL_PORT = "COM14"
BAUD_RATE = 115200
NUM_ANCHORS = 5

CALIB_LIBRARY_FILE = "calibration_library.csv"

# ================= ANCHOR MODE =================
USE_MANUAL_ANCHORS = True  # ← switch this

MANUAL_ANCHORS = {
    "A1": [0.1143, 0.635, 0.7493],
    "A2": [0.2413, 3.2893, 0.9144],
    "A3": [1.3462, 3.429, 0.8763],
    "A4": [1.4732, 0.5334, 1.3843],
    "A5": [0.28575, 1.97485, 1.83515]  # example with height
}
# ===== ANCHOR DISTANCES =====
ANCHOR_DISTANCES = {
    "A1": {"A2": 2.6289, "A3": 3.0226, "A4": 1.5113, "A5": 1.7145},
    "A2": {"A3": 1.12395, "A4": 3.048, "A5": 1.6129},
    "A3": {"A4": 2.921, "A5": 2.032},
    "A4": {"A5": 1.9304}
}

# ================= PARSER =================
def parse_custom_line(line):
    try:
        parts = line.split(",")
        parsed_data = []
        for part in parts:
            match = re.search(r"([-+]?\d*\.\d+|\d+)m@(\d+)", part)
            if match:
                parsed_data.append((float(match.group(1)), float(match.group(2))))
        return parsed_data[:NUM_ANCHORS] if len(parsed_data) >= NUM_ANCHORS else None
    except:
        return None

# ================= ANCHOR SOLVER =================
def solve_anchor_positions(distance_dict, dim=3):
    anchors = set(distance_dict.keys())
    for neigh in distance_dict.values():
        anchors.update(neigh.keys())
    anchors = list(anchors)
    idx = {a:i for i,a in enumerate(anchors)}

    constraints = []
    for a1, neigh in distance_dict.items():
        for a2, d in neigh.items():
            constraints.append((a1, a2, d))

    x0 = np.random.rand(len(anchors), dim)

    def err(flat):
        pts = flat.reshape(len(anchors), dim)
        return sum((np.linalg.norm(pts[idx[a1]] - pts[idx[a2]]) - d)**2 for a1,a2,d in constraints)

    res = minimize(err, x0.flatten(), method='L-BFGS-B')
    pts = res.x.reshape(len(anchors), dim)
    pts -= pts[0]

    return {a: pts[idx[a]] for a in anchors}

if USE_MANUAL_ANCHORS:
    ANCHORS = {k: np.array(v) for k, v in MANUAL_ANCHORS.items()}
else:
    ANCHORS = solve_anchor_positions(ANCHOR_DISTANCES)

ANCHOR_IDS = ["A1", "A2", "A3", "A4", "A5"]
ANCHOR_COORDS = np.array([ANCHORS[a] for a in ANCHOR_IDS])

# ================= LOAD CALIBRATION =================
def load_library():
    lib = []
    if os.path.exists(CALIB_LIBRARY_FILE):
        with open(CALIB_LIBRARY_FILE, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lib.append({
                    "name": row["name"],
                    "biases": [float(row[f"bias_A{i+1}"]) for i in range(NUM_ANCHORS)],
                    "sig_signature": [float(row[f"sig_A{i+1}"]) for i in range(NUM_ANCHORS)]
                })
    return lib

# ================= WEIGHTED PROFILE MATCH =================
def compute_profile_weights(rssis, library):
    if not library:
        return np.ones(1)

    errors = []
    for p in library:
        err = sum((rssis[i] - p["sig_signature"][i])**2 for i in range(NUM_ANCHORS))
        errors.append(err)

    errors = np.array(errors)
    weights = np.exp(-errors / (np.mean(errors) + 1e-6))
    weights /= np.sum(weights)
    return weights

# ================= POSITION SOLVER =================
def solve_position_weighted(ranges, rssis):
    weights = np.array(rssis) / (np.sum(rssis) + 1e-6)

    def err(x):
        total = 0
        for i in range(len(ranges)):
            dist_err = (np.linalg.norm(x - ANCHOR_COORDS[i]) - ranges[i])
            total += weights[i] * dist_err**2
        return total

    return minimize(err, np.mean(ANCHOR_COORDS, axis=0), options={'maxiter': 20}).x

# ================= SHARED STATE =================
current_pos = np.mean(ANCHOR_COORDS, axis=0)
current_rssi = [0]*NUM_ANCHORS

# ================= TRACKING THREAD =================
def tracking_loop(ser, calibration_library):
    global current_pos, current_rssi

    while True:
        if ser.in_waiting:
            data = parse_custom_line(ser.readline().decode())
            if not data:
                continue

            raw = [d[0] for d in data]
            rssis = [d[1] for d in data]

            weights = compute_profile_weights(rssis, calibration_library)

            corrected = np.zeros(NUM_ANCHORS)
            for w, p in zip(weights, calibration_library):
                corrected += w * (np.array(raw) - np.array(p["biases"]))

            pos = solve_position_weighted(corrected, rssis)

            current_pos = pos
            current_rssi = rssis

        time.sleep(0.01)

# ================= VIS =================
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for aid,pos in ANCHORS.items():
    ax.scatter(*pos, color='red')
    ax.text(*pos, aid)

tag_plot, = ax.plot([], [], [], 'bo')
lines = {}

def color_rssi(r):
    # adjust these bounds to your system
    r_min = 0
    r_max = 300

    r_norm = (r - r_min) / (r_max - r_min)
    r_norm = max(0, min(1, r_norm))

    return (1 - r_norm, r_norm, 0)

def update(_):
    global current_pos, current_rssi

    pos = current_pos
    rssis = current_rssi

    tag_plot.set_data([pos[0]],[pos[1]])
    tag_plot.set_3d_properties([pos[2]])

    for i,aid in enumerate(ANCHOR_IDS):
        if aid not in lines:
            lines[aid], = ax.plot([],[],[], '--')

        a = ANCHOR_COORDS[i]
        lines[aid].set_data([a[0],pos[0]],[a[1],pos[1]])
        lines[aid].set_3d_properties([a[2],pos[2]])
        lines[aid].set_color(color_rssi(rssis[i]))

    return [tag_plot] + list(lines.values())

# ================= MAIN =================
def main():
    calibration_library = load_library()
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)

    thread = threading.Thread(target=tracking_loop, args=(ser, calibration_library), daemon=True)
    thread.start()

    ani = FuncAnimation(fig, update, interval=50)

    print("Visualizer running... close window to exit.")
    plt.show()

if __name__ == "__main__":
    main()