import serial
import struct
import csv
import time

# --- Configuration ---
PORT = '/dev/ttyUSB1'  # Change to your port (e.g., '/dev/ttyUSB0' on Linux)
BAUD = 921600
OUT_FILE = "cir_data.csv"
SYNC_HEADER = b'\xEF\xBE\xAD\xDE' # 0xDEADBEEF in Little Endian

# Define the number of samples you are sending (from your 128-window mod)
NUM_SAMPLES = 128

def process_serial():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        print(f"Connected to {PORT}. Waiting for SYNC...")

        with open(OUT_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            # Create header: Timestamp, Range, FP_Index, then Sample_0...Sample_127
            header = ["System_Time", "Range_M", "FP_Index"] + [f"S_{i}" for i in range(NUM_SAMPLES)]
            writer.writerow(header)

            while True:
                if ser.in_waiting >= 270: # Only try to read if a full packet is likely there
                    # 1. Search for the SYNC_HEADER
                    if ser.read(1) == b'\xEF':
                        if ser.read(3) == b'\xBE\xAD\xDE':
                            # Found a packet! 
                            # 2. Read Metadata: Range (4 bytes) + FP Index (4 bytes)
                            metadata_raw = ser.read(8)
                            if len(metadata_raw) < 8: continue
                            range_val, fp_idx = struct.unpack('<ff', metadata_raw)

                            # 3. Read Samples: 128 samples * 2 bytes each (uint16_t)
                            samples_raw = ser.read(NUM_SAMPLES * 2)
                            if len(samples_raw) < (NUM_SAMPLES * 2): continue
                            
                            # Unpack 'H' is unsigned short (2 bytes)
                            samples = struct.unpack(f'<{NUM_SAMPLES}H', samples_raw)

                            # 4. Save to CSV
                            row = [time.time(), round(range_val, 4), round(fp_idx, 2)] + list(samples)
                            writer.writerow(row)
                            f.flush() # Ensure data is written to disk
                            
                            print(f"Captured: Range={range_val:.3f}m | FP={fp_idx:.2f}")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if 'ser' in locals(): ser.close()

if __name__ == "__main__":
    process_serial()