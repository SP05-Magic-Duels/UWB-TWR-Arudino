import serial
import joblib
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
#  CONFIGURATION — edit these values to match your physical setup
# =============================================================================

SERIAL_PORT    = 'COM13'                    # Arduino serial port
BAUD_RATE      = 921600
MODEL_FILENAME = 'uwb_spellcasting_model.pkl'
NUM_ANCHORS    = 4

# Anchor positions in metres (x, y).
# Measure these from a shared origin point (e.g. bottom-left corner of the room).
#
#   Anchor 0 ──────────────── Anchor 1
#      │                          │
#      │          room            │
#      │                          │
#   Anchor 3 ──────────────── Anchor 2
#
ANCHOR_POSITIONS = {
    0: np.array([0.000, 0.000]),   # e.g. top-left
    1: np.array([3.000, 0.000]),   # e.g. top-right  (3 m apart)
    2: np.array([3.000, 3.000]),   # e.g. bottom-right
    3: np.array([0.000, 3.000]),   # e.g. bottom-left
}

# Exponential Moving Average — lower = smoother but slower to react
SMOOTHING_FACTOR = 0.15

# =============================================================================

def trilaterate(anchors: dict, distances: dict) -> np.ndarray | None:
    """
    Least-squares trilateration using all available anchors.

    anchors   — {id: np.array([x, y])}
    distances — {id: float (metres)}

    Returns the estimated (x, y) position, or None if not enough data.
    """
    ids = [i for i in anchors if i in distances]
    if len(ids) < 3:
        return None  # Need at least 3 ranges for 2-D position

    # Build the linear system: for each anchor i beyond the first,
    # subtract the circle equation of anchor 0 to eliminate the x²+y² term.
    ref_id = ids[0]
    ax, ay = anchors[ref_id]
    dr = distances[ref_id]

    A_rows, b_rows = [], []
    for i in ids[1:]:
        bx, by = anchors[i]
        di = distances[i]
        A_rows.append([2 * (bx - ax), 2 * (by - ay)])
        b_rows.append(dr**2 - di**2 - ax**2 + bx**2 - ay**2 + by**2)

    A = np.array(A_rows)
    b = np.array(b_rows)

    # Least-squares solution (overdetermined when 4 anchors available)
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return result  # [x, y]


def main():
    print("--- UWB 4-Anchor AI Tracker ---\n")

    # ── Load model ────────────────────────────────────────────────────────────
    try:
        print(f"Loading model '{MODEL_FILENAME}'...")
        ai_model = joblib.load(MODEL_FILENAME)
        print("Model loaded.\n")
    except FileNotFoundError:
        print(f"Error: '{MODEL_FILENAME}' not found. Run train_model.py first.")
        return

    # ── Connect to hardware ───────────────────────────────────────────────────
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"Listening on {SERIAL_PORT} ...\n")
    except serial.SerialException:
        print(f"Error: Could not open {SERIAL_PORT}.")
        return

    # ── Per-anchor state ──────────────────────────────────────────────────────
    smoothed_errors   = {i: None for i in range(NUM_ANCHORS)}
    corrected_dists   = {}      # latest AI-corrected distance per anchor
    last_seen         = {i: 0.0 for i in range(NUM_ANCHORS)}
    STALE_TIMEOUT_S   = 1.5     # drop an anchor's range if not refreshed in time

    print(f"{'ANC':<5} {'RAW (in)':<12} {'CORR (in)':<12} {'ΔERR (in)':<12} {'POS X (in)':<13} {'POS Y (in)'}")
    print("─" * 72)

    try:
        while True:
            # ── Buffer management ─────────────────────────────────────────────
            if ser.in_waiting > 200:
                ser.reset_input_buffer()
                ser.readline()   # discard the chopped partial line
                continue

            if ser.in_waiting == 0:
                continue

            line = ser.readline().decode('utf-8', errors='ignore').strip()

            # ── Parse incoming line ───────────────────────────────────────────
            # Expected Arduino format:
            #   ML_DATA,<anchor_id>,<raw_dist_m>,<rx_pwr>,<fp_pwr>,<quality>
            if not line.startswith("ML_DATA"):
                continue

            parts = line.split(",")
            if len(parts) != 6:
                continue

            try:
                anchor_id = int(parts[1])
                raw_dist  = float(parts[2])
                rx_pwr    = float(parts[3])
                fp_pwr    = float(parts[4])
                quality   = float(parts[5])
            except ValueError:
                continue

            if anchor_id not in ANCHOR_POSITIONS:
                continue

            # ── AI correction ─────────────────────────────────────────────────
            pwr_diff  = abs(rx_pwr - fp_pwr)
            features  = [[anchor_id, raw_dist, rx_pwr, fp_pwr, pwr_diff, quality]]
            pred_err  = ai_model.predict(features)[0]

            # Exponential Moving Average on the predicted error
            prev = smoothed_errors[anchor_id]
            if prev is None:
                smoothed_errors[anchor_id] = pred_err
            else:
                smoothed_errors[anchor_id] = (SMOOTHING_FACTOR * pred_err) + ((1 - SMOOTHING_FACTOR) * prev)

            corrected = raw_dist + smoothed_errors[anchor_id]
            corrected_dists[anchor_id] = corrected
            last_seen[anchor_id]       = time.time()

            # ── Drop stale anchors ────────────────────────────────────────────
            now = time.time()
            active_dists = {
                i: d for i, d in corrected_dists.items()
                if now - last_seen[i] < STALE_TIMEOUT_S
            }

            # ── Trilateration ─────────────────────────────────────────────────
            pos = trilaterate(ANCHOR_POSITIONS, active_dists)

            raw_in  = raw_dist * 39.3701
            corr_in = corrected * 39.3701
            err_in  = smoothed_errors[anchor_id] * 39.3701

            if pos is not None:
                pos_x_in = pos[0] * 39.3701
                pos_y_in = pos[1] * 39.3701
                pos_str  = f"{pos_x_in:>10.2f} in    {pos_y_in:>10.2f} in"
            else:
                anchors_needed = 3 - len(active_dists)
                pos_str = f"  (need {anchors_needed} more anchor{'s' if anchors_needed != 1 else ''})"

            print(
                f"  {anchor_id:<3} "
                f"{raw_in:>9.2f} in  "
                f"{corr_in:>9.2f} in  "
                f"{err_in:>+9.2f} in  "
                f"{pos_str}"
            )

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if ser.is_open:
            ser.close()


if __name__ == '__main__':
    main()
