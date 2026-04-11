import serial
import time
import numpy as np
import csv
import re
import os

# --- CONFIGURATION ---
SERIAL_PORT = "COM13"
BAUD_RATE = 115200
NUM_ANCHORS = 5 
SAMPLES_PER_POS = 100 

# Files
CALIB_LIBRARY_FILE = "calibration_library.csv"
TRACKING_LOG_FILE = "rtls_tracking_log.csv"

def parse_custom_line(line):
    try:
        parts = line.split(",")
        parsed_data = []
        for part in parts:
            match = re.search(r"([-+]?\d*\.\d+|\d+)m@(\d+)", part)
            if match:
                parsed_data.append((float(match.group(1)), float(match.group(2))))
        
        if len(parsed_data) >= NUM_ANCHORS:
            return parsed_data[:NUM_ANCHORS]
        return None
    except:
        return None

def save_library_to_csv(library):
    """Saves the calibration profiles to a persistent CSV file."""
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

def load_library_from_csv():
    """Loads existing calibration profiles if the file exists."""
    library = []
    if os.path.exists(CALIB_LIBRARY_FILE):
        with open(CALIB_LIBRARY_FILE, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                profile = {
                    "name": row["name"],
                    "biases": [float(row[f"bias_A{i+1}"]) for i in range(NUM_ANCHORS)],
                    "sig_signature": [float(row[f"sig_A{i+1}"]) for i in range(NUM_ANCHORS)]
                }
                library.append(profile)
        print(f"--- Loaded {len(library)} profiles from {CALIB_LIBRARY_FILE} ---")
    return library

def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        print(f"--- RTLS System (Anchors: {NUM_ANCHORS}) ---")

        # Load existing data if available
        calibration_library = load_library_from_csv()
        
        while True:
            cmd = input("\n[Menu] 'c' calibrate, 't' track, 'l' list profiles, 'q' quit: ").lower()
            if cmd == 'q': return
            if cmd == 'l':
                for p in calibration_library: print(f" - {p['name']}")
                continue
            if cmd == 't': 
                if not calibration_library:
                    print("Error: No calibration data available.")
                    continue
                break
            
            if cmd == 'c':
                loc_name = input("Location Name: ")
                truths = [float(input(f"  True Dist A{i+1}: ")) for i in range(NUM_ANCHORS)]

                print(f"Sampling {SAMPLES_PER_POS} packets...")
                temp_dist, temp_rssi = [[] for _ in range(NUM_ANCHORS)], [[] for _ in range(NUM_ANCHORS)]
                count = 0
                ser.reset_input_buffer()
                
                while count < SAMPLES_PER_POS:
                    if ser.in_waiting > 0:
                        line = ser.readline().decode("utf-8", errors="ignore").strip()
                        parsed = parse_custom_line(line)
                        if parsed and all(p[0] > 0 for p in parsed):
                            for i in range(NUM_ANCHORS):
                                temp_dist[i].append(parsed[i][0])
                                temp_rssi[i].append(parsed[i][1])
                            count += 1
                            print(f"  Progress: {count}/{SAMPLES_PER_POS}", end='\r')

                profile = {
                    "name": loc_name,
                    "biases": [np.median(temp_dist[i]) - truths[i] for i in range(NUM_ANCHORS)],
                    "sig_signature": [np.mean(temp_rssi[i]) for i in range(NUM_ANCHORS)]
                }
                calibration_library.append(profile)
                save_library_to_csv(calibration_library)
                print(f"\nSaved and Logged Profile: {loc_name}")

        # --- TRACKING AND LOGGING ---
        print(f"\n--- Logging to {TRACKING_LOG_FILE} ---")
        header = ["Timestamp", "Matched_Loc"] + [f"A{i+1}_Corrected" for i in range(NUM_ANCHORS)]
        
        # Initialize the tracking log file with headers
        with open(TRACKING_LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(header)

        start_time = time.time()
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                live_data = parse_custom_line(line)
                
                if live_data:
                    current_rssis = [p[1] for p in live_data]
                    
                    # Find Best Match
                    best_match = min(calibration_library, key=lambda p: sum((current_rssis[i] - p["sig_signature"][i])**2 for i in range(NUM_ANCHORS)))

                    rel_time = round(time.time() - start_time, 3)
                    corrected_values = [round(live_data[i][0] - best_match["biases"][i], 3) for i in range(NUM_ANCHORS)]
                    
                    # Write to CSV
                    with open(TRACKING_LOG_FILE, "a", newline="") as f:
                        csv.writer(f).writerow([rel_time, best_match["name"]] + corrected_values)
                    
                    # Console Output
                    display = f"Time: {rel_time} | Loc: {best_match['name']} | Dists: {corrected_values}"
                    print(display, end='\r')

    except KeyboardInterrupt:
        print("\nLogging stopped.")
    finally:
        if 'ser' in locals(): ser.close()

if __name__ == "__main__":
    main()