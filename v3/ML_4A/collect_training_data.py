import serial
import csv
import os
import time

# --- CONFIGURATION ---
SERIAL_PORT = 'COM13'       # Change to match your Arduino's port
BAUD_RATE = 921600
CSV_FILENAME = 'training_data.csv'
NUM_ANCHORS = 4             # Number of UWB anchors in your system
# ---------------------

def main():
    print("--- UWB 4-Anchor Training Data Collector ---")
    print(f"Collecting data from {NUM_ANCHORS} anchors.\n")

    try:
        true_distance = 0.0254 * float(input("Enter the TRUE physical distance (in inches) for this session: "))
        anchor_id = int(input(f"Enter the ANCHOR ID you are calibrating (0 to {NUM_ANCHORS - 1}): "))
        if not (0 <= anchor_id < NUM_ANCHORS):
            print(f"Error: Anchor ID must be between 0 and {NUM_ANCHORS - 1}.")
            return
    except ValueError:
        print("Invalid input. Please enter a number.")
        return

    file_exists = os.path.isfile(CSV_FILENAME)

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"\nConnected to {SERIAL_PORT}. Recording for Anchor {anchor_id}...")
        print("Move the wand around! Press Ctrl+C to stop.\n")

        with open(CSV_FILENAME, mode='a', newline='') as file:
            writer = csv.writer(file)

            if not file_exists:
                # Anchor_ID column lets the model learn per-anchor characteristics
                writer.writerow(['Anchor_ID', 'Raw_Dist', 'RX_Power', 'FP_Power', 'Quality', 'True_Distance'])

            samples_collected = 0

            while True:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()

                    # Expected Arduino format: "ML_DATA,<anchor_id>,<raw_dist>,<rx_pwr>,<fp_pwr>,<quality>"
                    if line.startswith("ML_DATA"):
                        parts = line.split(",")

                        if len(parts) == 6:
                            try:
                                incoming_anchor = int(parts[1])

                                # Only record samples from the anchor we're calibrating
                                if incoming_anchor != anchor_id:
                                    continue

                                raw_dist = float(parts[2])
                                rx_power = float(parts[3])
                                fp_power = float(parts[4])
                                quality  = float(parts[5])

                                writer.writerow([anchor_id, raw_dist, rx_power, fp_power, quality, true_distance])
                                samples_collected += 1

                                if samples_collected % 10 == 0:
                                    print(f"  Anchor {anchor_id} | {samples_collected} samples @ {true_distance / 0.0254:.1f} in ({true_distance:.3f} m)...")

                            except ValueError:
                                pass  # Corrupted line — skip it

    except serial.SerialException:
        print(f"Error: Could not open {SERIAL_PORT}. Is it open in the Arduino IDE?")
    except KeyboardInterrupt:
        print(f"\nStopped! Saved {samples_collected} samples to '{CSV_FILENAME}'.")
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == '__main__':
    main()
