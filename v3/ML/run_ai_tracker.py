import serial
import joblib
import pandas as pd
import numpy as np
import time
import warnings

# Suppress scikit-learn warnings about feature names
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION ---
SERIAL_PORT = 'COM8' 
BAUD_RATE = 115200
MODEL_FILENAME = 'uwb_spellcasting_model.pkl'

# SET YOUR CURRENT ANCHOR COUNT HERE
NUM_ANCHORS = 2  

# Smoothing: 0.15 is responsive, 0.05 is very smooth but laggy
SMOOTHING_FACTOR = 0.15  
# ---------------------

def main():
    print(f"--- UWB AI Inference Engine ({NUM_ANCHORS} Anchors) ---")
    
    # 1. Load the trained model
    try:
        ai_model = joblib.load(MODEL_FILENAME)
        print(f"Model '{MODEL_FILENAME}' loaded successfully!")
    except FileNotFoundError:
        print(f"Error: {MODEL_FILENAME} not found. Ensure you have trained the model.")
        return

    # 2. Connect to Hardware
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        ser.reset_input_buffer()
        time.sleep(2)
        print(f"Listening on {SERIAL_PORT}...")

        # Dictionary to store independent smoothed errors for each anchor
        anchor_smoothed_errors = {i: None for i in range(NUM_ANCHORS)}

        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                parts = line.split(",")

                # Validate that the line has 4 values per anchor (Dist, RX, FP, Quality)
                expected_parts = NUM_ANCHORS * 4
                
                if len(parts) == expected_parts:
                    output_display = []
                    
                    for i in range(NUM_ANCHORS):
                        try:
                            # 1. Map the parts to the correct anchor
                            idx = i * 4
                            raw_dist = float(parts[idx])
                            rx_pwr   = float(parts[idx+1])
                            fp_pwr   = float(parts[idx+2])
                            quality  = float(parts[idx+3])
                            
                            # 2. Re-calculate the feature used during training
                            pwr_diff = abs(rx_pwr - fp_pwr)

                            # 3. Format features exactly as the model expects:
                            # [Anchor_ID, Raw_Dist, RX_Power, FP_Power, Power_Diff, Quality]
                            features = np.array([[i, raw_dist, rx_pwr, fp_pwr, pwr_diff, quality]])
                            
                            # 4. Predict correction (Error)
                            pred_error = ai_model.predict(features)[0]

                            # 5. Apply Smoothing
                            if anchor_smoothed_errors[i] is None:
                                anchor_smoothed_errors[i] = pred_error
                            else:
                                anchor_smoothed_errors[i] = (SMOOTHING_FACTOR * pred_error) + \
                                                            ((1 - SMOOTHING_FACTOR) * anchor_smoothed_errors[i])

                            # 6. Final Calculation
                            corrected_dist = raw_dist + anchor_smoothed_errors[i]
                            
                            # Display formatting (Inches)
                            corr_in = corrected_dist * 39.3701
                            output_display.append(f"A{i}: {corr_in:>6.2f} in")

                        except (ValueError, IndexError):
                            continue
                    
                    # Print all anchors on one line, updating in place
                    if output_display:
                        print(f"\r{' | '.join(output_display)}", end="")

    except serial.SerialException:
        print(f"Error: Could not open {SERIAL_PORT}.")
    except KeyboardInterrupt:
        print("\nStopping inference engine.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == '__main__':
    main()