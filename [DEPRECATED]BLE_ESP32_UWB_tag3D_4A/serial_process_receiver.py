import serial
import time

# Update with your ESP32 serial port (e.g., 'COM3' on Windows or '/dev/ttyUSB0' on Linux)
SERIAL_PORT = '/dev/ttyUSB1'
BAUD_RATE = 115200

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
    print(f"Listening on {SERIAL_PORT}...")

    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').rstrip()
            print(f"Received: {line}")
        time.sleep(0.01) # Small delay to reduce CPU usage
except KeyboardInterrupt:
    print("Stopping...")
    ser.close()
except Exception as e:
    print(f"Error: {e}")
