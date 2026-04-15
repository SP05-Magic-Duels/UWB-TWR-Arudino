import serial
import joblib
import pandas as pd
import time
import warnings

# Suppress scikit-learn warnings about feature names (keeps the console clean)
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION ---
SERIAL_PORT = 'COM13' # Change to match your setup
BAUD_RATE = 921600
MODEL_FILENAME = 'uwb_spellcasting_model.pkl'

# Add this near the top of your script, before the while loop
SMOOTHING_FACTOR = 0.15  # Range from 0.0 (won't change) to 1.0 (no smoothing)
# ---------------------

def main():
    print("--- UWB AI Inference Engine ---")
    
    # 1. Load the trained Random Forest model
    try:
        print(f"Loading AI model '{MODEL_FILENAME}'...")
        ai_model = joblib.load(MODEL_FILENAME)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Could not find {MODEL_FILENAME}. Did you run the training script yet?")
        return

    # 2. Connect to the UWB Hardware
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"\nListening on {SERIAL_PORT}...")
        print("--------------------------------------------------")
        print(f"{'RAW DISTANCE':<15} | {'POWER DIFF':<15} | {'AI CORRECTED DISTANCE':<20}")
        print("--------------------------------------------------")

        smoothed_error = None

        while True:
            if ser.in_waiting > 0:
                # 2. DRAIN THE BUFFER: Rapidly read all lines until the buffer is empty
                while ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                    # 3. Now 'line' contains the absolute newest, real-time message!
                    if line.startswith("ML_DATA"):
                        parts = line.split(",")
                        
                        if len(parts) == 5:
                            try:
                                raw_dist = float(parts[1])
                                rx_pwr = float(parts[2])
                                fp_pwr = float(parts[3])
                                quality = float(parts[4])
                                
                                pwr_diff = abs(rx_pwr - fp_pwr)
                                live_data = [[raw_dist, rx_pwr, fp_pwr, pwr_diff, quality]]
                                
                                # 1. Ask the AI how wrong the raw distance is right now
                                predicted_error = ai_model.predict(live_data)[0]

                                # 2. Apply the Exponential Moving Average to the predicted error
                                if smoothed_error is None:
                                    smoothed_error = predicted_error # Initialize on first run
                                else:
                                    smoothed_error = (SMOOTHING_FACTOR * predicted_error) + ((1 - SMOOTHING_FACTOR) * smoothed_error)

                                # 2. Apply that correction to the raw, smoothly changing UWB distance
                                corrected_distance = raw_dist + smoothed_error

                                corrected_distance_in = corrected_distance * 39.3701
                                raw_distance_in = raw_dist * 39.3701
                                
                                # Print the comparison
                                print(f"{raw_distance_in:>10.2f} in   | {pwr_diff:>10.2f} dBm   | {corrected_distance_in:>18.2f} in")

                                # NOTE: In your final project, you would pass 'corrected_distance' 
                                # directly to your trilateration solver right here!

                            except ValueError:
                                pass # Ignore corrupted serial lines
               
                            
    except serial.SerialException:
        print(f"Error: Could not open {SERIAL_PORT}.")
    except KeyboardInterrupt:
        print("\nStopping inference engine.")
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == '__main__':
    main()