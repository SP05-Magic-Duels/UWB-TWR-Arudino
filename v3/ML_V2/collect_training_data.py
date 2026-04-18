import serial
import csv
import os
import time

# --- CONFIGURATION ---
SERIAL_PORT = 'COM3'
BAUD_RATE = 115200
CSV_FILENAME = '3D_DATA_MODELS/DATA/training_data_8A_100sam.csv'
NUM_ANCHORS = 8  # Set this to 2, 4, etc.
SAMPLES_PER_POINT = 100  # Set how many samples to take at each location
# ---------------------

def main():
    print(f"--- UWB Wide-Format Collector ({NUM_ANCHORS} Anchors) ---")
    
    # 1. Dynamically ask for true distances based on NUM_ANCHORS
    true_distances = []
    try:
        for i in range(NUM_ANCHORS):
            dist_in = float(input(f"Enter TRUE physical distance for ANCHOR {i} (in inches): "))
            true_distances.append(dist_in * 0.0254) # Convert to meters
    except ValueError:
        print("Invalid input. Please enter numbers only.")
        return

    # Check if file exists and has content
    file_needs_header = not os.path.exists(CSV_FILENAME) or os.stat(CSV_FILENAME).st_size == 0

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        ser.reset_input_buffer() 
        time.sleep(2)
        
        # 2. Dynamically generate the CSV header
        header = []
        for i in range(NUM_ANCHORS):
            header.extend([
                f'A{i}_Raw_Dist', f'A{i}_RX_Power', f'A{i}_FP_Power', f'A{i}_Quality', f'A{i}_True_Distance'
            ])

        with open(CSV_FILENAME, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            if file_needs_header:
                writer.writerow(header)
                print(f"Created new file with headers in {CSV_FILENAME}")

            rows_saved = 0
            # Each anchor sends 4 values. Expected parts = NUM_ANCHORS * 4
            expected_parts = NUM_ANCHORS * 4
            
            print(f"\nCollecting {SAMPLES_PER_POINT} samples. Please hold the wand steady...")

            while rows_saved < SAMPLES_PER_POINT:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    parts = line.split(",")

                    if len(parts) == expected_parts:
                        try:
                            # 3. Dynamically construct the row
                            row = []
                            for i in range(NUM_ANCHORS):
                                # Get the 4 sensor values for this anchor
                                start_idx = i * 4
                                row.extend(parts[start_idx : start_idx + 4])
                                # Append the specific true distance for this anchor
                                row.append(true_distances[i])
                            
                            writer.writerow(row)
                            rows_saved += 1
                            
                            percent = (rows_saved / SAMPLES_PER_POINT) * 100
                            print(f"  Progress: [{rows_saved}/{SAMPLES_PER_POINT}] {percent:.1f}%", end='\r')
                        except ValueError:
                            continue

            print(f"\n\nDone! Successfully added {rows_saved} samples for this location.")

    except serial.SerialException:
        print(f"Error: Could not open {SERIAL_PORT}.")
    except KeyboardInterrupt:
        print(f"\n\nManual Stop. Saved {rows_saved} rows.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == '__main__':
    main()