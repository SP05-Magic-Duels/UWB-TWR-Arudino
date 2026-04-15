import serial
import joblib
import pandas as pd
import time
import warnings

# Suppress scikit-learn warnings about feature names (keeps the console clean)
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION ---
SERIAL_PORT = 'COM3' # Change to match your setup
BAUD_RATE = 115200
MODEL_FILENAME = 'uwb_spellcasting_model.pkl'
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

        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                if line.startswith("ML_DATA"):
                    parts = line.split(",")
                    
                    if len(parts) == 5:
                        try:
                            # Parse the raw features
                            raw_dist = float(parts[1])
                            rx_pwr = float(parts[2])
                            fp_pwr = float(parts[3])
                            quality = float(parts[4])
                            
                            # Engineer the feature that detects the "meat shield"
                            pwr_diff = abs(rx_pwr - fp_pwr)
                            
                            # Format exactly as the model was trained: 
                            # ['Raw_Dist', 'RX_Power', 'FP_Power', 'Power_Diff', 'Quality']
                            live_data = pd.DataFrame([[raw_dist, rx_pwr, fp_pwr, pwr_diff, quality]], 
                                                     columns=['Raw_Dist', 'RX_Power', 'FP_Power', 'Power_Diff', 'Quality'])
                            
                            # 3. The Magic: Ask the AI for the true distance
                            corrected_distance = ai_model.predict(live_data)[0]
                            
                            # Print the comparison
                            print(f"{raw_dist:>10.2f} m   | {pwr_diff:>10.2f} dBm   | {corrected_distance:>18.2f} m")
                            
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