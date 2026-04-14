import serial
import csv
import os

# --- CONFIGURATION ---
PORT = '/dev/ttyUSB0'    # Change to your port (e.g., '/dev/ttyUSB0' on Linux/Mac)
BAUD_RATE = 115200       # Match your Serial.begin(115200)
# Naming convention required by your main() function: data_NAME1_NAME2.csv
FILENAME = "calibration/calibration_data/data_A5_T3.csv"
SAMPLES_TO_COLLECT = 1000 # Stop after this many valid samples

def main():
    print(f"--- UWB Calibration Data Collector ---")
    print(f"Connecting to {PORT}...")
    
    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
    except serial.SerialException as e:
        print(f"Error: Could not open port {PORT}. {e}")
        return

    # Ensure the directory exists
    os.makedirs(os.path.dirname(FILENAME) if os.path.dirname(FILENAME) else '.', exist_ok=True)

    with open(FILENAME, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header required by the PSO script
        writer.writerow(["TR1", "TR2", "TP1", "TP2"])
        
        count = 0
        print(f"Streaming data to {FILENAME}...")
        print("Waiting for valid UWB packets (TR1, TR2, TP1, TP2)...")
        print("-" * 50)

        while count < SAMPLES_TO_COLLECT:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            if not line:
                continue

            # Basic validation: check if line contains exactly 3 commas (4 values)
            # and consists of digits (and potentially commas)
            if line.count(',') == 3:
                try:
                    # Clean the line and split into values
                    parts = [p.strip() for p in line.split(',')]
                    # Verify they are all numbers (handles potential debug text in same line)
                    [float(p) for p in parts] 
                    
                    writer.writerow(parts)
                    count += 1
                    
                    # Print status update
                    print(f"[{count}/{SAMPLES_TO_COLLECT}] Data: {line}")
                    
                except ValueError:
                    # Skip lines that might be text like "Range: 2.50m"
                    continue
            else:
                # Print non-CSV lines (like debug info) so you know what the hardware is doing
                if line:
                    print(f"DEBUG: {line}")

    ser.close()
    print("-" * 50)
    print(f"Collection complete! Saved {count} samples to {FILENAME}.")

if __name__ == "__main__":
    main()