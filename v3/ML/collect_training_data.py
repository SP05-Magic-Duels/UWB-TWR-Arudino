import serial
import csv
import os
import time

# --- CONFIGURATION ---
# Change this to match your Arduino's serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Mac/Linux)
SERIAL_PORT = 'COM13' 
BAUD_RATE = 921600
CSV_FILENAME = 'training_data.csv'
# ---------------------

def main():
    print("--- UWB Spellcasting Data Collector ---")
    
    # Ask the user for the physical ground-truth distance for this session
    try:
        true_distance = 0.0254 * float(input("Enter the TRUE physical distance (in inches) for this recording session: "))
    except ValueError:
        print("Invalid input. Please enter a number.")
        return

    # Check if the file exists so we know whether to write the header row
    file_exists = os.path.isfile(CSV_FILENAME)

    try:
        # Open the serial port
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) # Give the Arduino a moment to reset upon connection
        print(f"\nConnected to {SERIAL_PORT}. Listening for 'ML_DATA'...")
        print("Move the wand around! Press Ctrl+C to stop recording.\n")

        with open(CSV_FILENAME, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # Write the header if this is a brand new file
            if not file_exists:
                writer.writerow(['Raw_Dist', 'RX_Power', 'FP_Power', 'Quality', 'True_Distance'])

            samples_collected = 0

            while True:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8').strip()
                    
                    # Look for our specific data string from the Arduino
                    if line.startswith("ML_DATA"):
                        parts = line.split(",")
                        
                        # Ensure we have exactly the right amount of data
                        if len(parts) == 5: 
                            try:
                                raw_dist = float(parts[1])
                                rx_power = float(parts[2])
                                fp_power = float(parts[3])
                                quality = float(parts[4])
                                
                                # Write to CSV with the true distance label
                                writer.writerow([raw_dist, rx_power, fp_power, quality, true_distance])
                                samples_collected += 1
                                
                                # Print a live update every 10 samples so you know it's working
                                if samples_collected % 10 == 0:
                                    print(f"Recorded {samples_collected} samples at {true_distance}m...")
                                    
                            except ValueError:
                                # Catch any corrupted serial data chunks
                                pass
                                
    except serial.SerialException:
        print(f"Error: Could not open {SERIAL_PORT}. Is it open in the Arduino IDE?")
    except KeyboardInterrupt:
        print(f"\nStopped! Successfully saved {samples_collected} samples to {CSV_FILENAME}.")
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == '__main__':
    main()