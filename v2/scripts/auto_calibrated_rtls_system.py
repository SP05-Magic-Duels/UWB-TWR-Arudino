import serial
import time
import numpy as np
import csv
import re
import os
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ================= CONFIG =================
MODE = "serial"
CSV_INPUT_FILE = "rtls_tracking_log.csv"

SERIAL_PORT = "COM14"
BAUD_RATE = 115200
NUM_ANCHORS = 5
SAMPLES_PER_POS = 100

LOG_TO_CSV = True
LOG_FILE = "rtls_tracking_log.csv"

CALIB_LIBRARY_FILE = "calibration_library.csv"

USE_VIS = True

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

ANCHORS = solve_anchor_positions(ANCHOR_DISTANCES)
ANCHOR_IDS = list(ANCHORS.keys())
ANCHOR_COORDS = np.array([ANCHORS[a] for a in ANCHOR_IDS])

# ================= CALIBRATION =================
def save_library(library):
    keys = ["name"] + [f"bias_A{i+1}" for i in range(NUM_ANCHORS)] + [f"sig_A{i+1}" for i in range(NUM_ANCHORS)]
    with open(CALIB_LIBRARY_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for prof in library:
            row = {"name": prof["name"]}
            for i in range(NUM_ANCHORS):
                row[f"bias_A{i+1}"] = prof["biases"][i]
                row[f"sig_A{i+1}"] = prof["sig_signature"][i]
            writer.writerow(row)


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
    """Soft weighting instead of hard nearest neighbor"""
    errors = []
    for p in library:
        err = sum((rssis[i] - p["sig_signature"][i])**2 for i in range(NUM_ANCHORS))
        errors.append(err)

    errors = np.array(errors)
    weights = np.exp(-errors / (np.mean(errors) + 1e-6)) # Change this to determine how much we change the node weightings
    weights /= np.sum(weights)
    return weights

# ================= POSITION SOLVER =================
def solve_position_weighted(ranges, rssis):
    """RSSI-weighted trilateration"""
    weights = np.array(rssis) / (np.sum(rssis) + 1e-6)

    def err(x):
        total = 0
        for i in range(len(ranges)):
            dist_err = (np.linalg.norm(x - ANCHOR_COORDS[i]) - ranges[i])
            total += weights[i] * dist_err**2
        return total

    return minimize(err, np.mean(ANCHOR_COORDS, axis=0)).x

# ================= VIS =================
if USE_VIS:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for aid,pos in ANCHORS.items():
        ax.scatter(*pos, color='red')
        ax.text(*pos, aid)

    tag_plot, = ax.plot([], [], [], 'bo')
    lines = {}

    def color_rssi(r):
        return (1-r/100, r/100, 0)

    def update(_):
        if not hasattr(update, "data"):
            return tag_plot,

        pos, rssis = update.data
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

    FuncAnimation(fig, update, interval=100)
    plt.show(block=False)

# ================= MAIN =================
def main():
    calibration_library = load_library()

    if MODE == "serial":
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)

    if LOG_TO_CSV and not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "x", "y", "z"] + [f"A{i+1}" for i in range(NUM_ANCHORS)])



    while True:
        cmd = input("\n[c] calibrate, [t] track, [l] list, [q] quit: ").lower()

        if cmd == 'q': return

        if cmd == 'l':
            for p in calibration_library:
                print(p['name'])

        if cmd == 'c':
            name = input("Location name: ")
            truths = [float(input(f"True dist A{i+1}: ")) for i in range(NUM_ANCHORS)]

            temp_d = [[] for _ in range(NUM_ANCHORS)]
            temp_r = [[] for _ in range(NUM_ANCHORS)]
            count = 0

            while count < SAMPLES_PER_POS:
                if ser.in_waiting:
                    
                    data = parse_custom_line(ser.readline().decode())
                    if data:
                        for i in range(NUM_ANCHORS):
                            temp_d[i].append(data[i][0])
                            temp_r[i].append(data[i][1])
                        count += 1
                        print(f"Samples: {count}/{SAMPLES_PER_POS}", end='\r', flush=True)

            profile = {
                "name": name,
                "biases": [np.median(temp_d[i]) - truths[i] for i in range(NUM_ANCHORS)],
                "sig_signature": [np.mean(temp_r[i]) for i in range(NUM_ANCHORS)]
            }

            calibration_library.append(profile)
            save_library(calibration_library)
            print("Saved")

        if cmd == 't':
            if MODE == "csv":
                with open(CSV_INPUT_FILE) as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        dists = list(map(float, row[2:2+NUM_ANCHORS]))
                        pos = solve_position_weighted(dists, [50]*NUM_ANCHORS)
                        if USE_VIS:
                            update.data = (pos, [50]*NUM_ANCHORS)
                        print(pos)
            else:
                while True:
                    if ser.in_waiting:
                        data = parse_custom_line(ser.readline().decode())
                        if not data:
                            continue

                        raw = [d[0] for d in data]
                        rssis = [d[1] for d in data]

                        weights = compute_profile_weights(rssis, calibration_library)

                        # Smooth blend of calibration profiles
                        corrected = np.zeros(NUM_ANCHORS)
                        for w, p in zip(weights, calibration_library):
                            corrected += w * (np.array(raw) - np.array(p["biases"]))

                        pos = solve_position_weighted(corrected, rssis)

                        if LOG_TO_CSV:
                            with open(LOG_FILE, "a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow(
                                    [time.time()] +
                                    list(pos) +
                                    list(corrected)
        )
                        if USE_VIS:
                            update.data = (pos, rssis)

                        print(f"Pos: {np.round(pos,2)}", end='\r')

if __name__ == "__main__":
    main()
