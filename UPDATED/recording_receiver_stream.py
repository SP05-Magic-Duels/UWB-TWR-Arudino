import serial
import time
import csv
import os
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────
SERIAL_PORT = '/dev/ttyUSB0'   # Windows: 'COM3'
BAUD_RATE   = 115200

# Output CSV file — named with timestamp so you never overwrite old recordings
CSV_FILE = f"figure_eights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
# ─────────────────────────────────────────────────────────────────────────────

def parse_position(line):
    """
    Parses lines like:  P= 1.23,4.56,0.78
    Returns (x, y, z) as floats, or None if the line isn't a position line.
    """
    if not line.startswith("P= "):
        return None
    try:
        parts = line[3:].split(',')
        if len(parts) != 3:
            return None
        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        return x, y, z
    except ValueError:
        return None


def main():
    print(f"Connecting to {SERIAL_PORT} at {BAUD_RATE} baud...")

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except serial.SerialException as e:
        print(f"Could not open port: {e}")
        return

    print(f"Recording to: {CSV_FILE}")
    print("Press Ctrl+C to stop.\n")

    row_count = 0

    with open(CSV_FILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'x', 'y', 'z'])  # Header

        try:
            while True:
                if ser.in_waiting > 0:
                    raw = ser.readline()
                    try:
                        line = raw.decode('utf-8').rstrip()
                    except UnicodeDecodeError:
                        continue  # Skip garbled bytes on startup

                    print(f"  {line}")  # Echo everything to terminal

                    result = parse_position(line)
                    if result is not None:
                        x, y, z = result
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                        writer.writerow([timestamp, x, y, z])
                        csvfile.flush()  # Write to disk immediately — safe on crash
                        row_count += 1
                        print(f"  → Saved row {row_count}: t={timestamp}  x={x:.3f}  y={y:.3f}  z={z:.3f}")

                time.sleep(0.01)

        except KeyboardInterrupt:
            print(f"\nStopped. {row_count} position rows saved to '{CSV_FILE}'.")

        finally:
            ser.close()


if __name__ == '__main__':
    main()
