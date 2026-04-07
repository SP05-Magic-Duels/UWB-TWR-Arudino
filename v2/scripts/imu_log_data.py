import serial
import time
import numpy as np
import quaternion

SERIAL_PORT = "COM9"
BAUD_RATE = 115200
CALIBRATION_SAMPLES = 100 

# KALMAN FILTER CLASS
class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, initial_value=0.0):
        self.Q = process_variance
        self.R = measurement_variance
        self.x = initial_value
        self.P = 1.0

    def update(self, measurement):
        self.P = self.P + self.Q
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * self.P
        return self.x

# Initialize Kalman Filters
kf_ax = KalmanFilter(0.001, 0.1, 0.0)
kf_ay = KalmanFilter(0.001, 0.1, 0.0)
kf_az = KalmanFilter(0.001, 0.1, 1.0)
kf_gx = KalmanFilter(0.1, 10.0, 0.0)
kf_gy = KalmanFilter(0.1, 10.0, 0.0)
kf_gz = KalmanFilter(0.1, 10.0, 0.0)

def parse_imu_line(line):
    parts = [x.strip() for x in line.strip().split(",")]
    if len(parts) != 6: raise ValueError
    return [float(x) for x in parts]

def main():
    q = np.quaternion(1, 0, 0, 0)
    last_t = None
    gyro_bias = np.array([0.0, 0.0, 0.0])
    is_calibrated = False
    calib_buffer = []
    
    # State for Low-Pass
    lp_acc = np.array([0.0, 0.0, 1.0])
    lp_gyro = np.array([0.0, 0.0, 0.0])

    print(f"Connecting to {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) # Wait for Arduino reset
        print("Starting Calibration... Keep IMU still.")

        while True:
            raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw_line: continue
            
            try:
                curr_t = time.perf_counter()
                imu_data = parse_imu_line(raw_line)
                ax_m, ay_m, az_m, gx, gy, gz = imu_data
            except ValueError:
                continue

            # 1. CALIBRATION
            if not is_calibrated:
                calib_buffer.append([gx, gy, gz])
                if len(calib_buffer) % 10 == 0:
                    print(f"Calibrating: {len(calib_buffer)}/{CALIBRATION_SAMPLES}", end='\r')
                
                if len(calib_buffer) >= CALIBRATION_SAMPLES:
                    gyro_bias = np.mean(calib_buffer, axis=0)
                    is_calibrated = True
                    print(f"\nCalibration Complete! Bias: {gyro_bias}")
                    print("Outputting Filtered Data (Format: AX, AY, AZ, GX, GY, GZ)")
                continue

            # 2. TIME DELTA
            if last_t is None:
                last_t = curr_t
                continue
            dt = curr_t - last_t
            last_t = curr_t

            raw_acc = np.array([ax_m, ay_m, az_m])
            raw_gyro = np.array([gx, gy, gz]) - gyro_bias

            # 3. FILTERING METHODS
            # UNCOMMENT the method you want to use for output
            
            # METHOD 1: KALMAN
            final_acc = np.array([kf_ax.update(raw_acc[0]), kf_ay.update(raw_acc[1]), kf_az.update(raw_acc[2])])
            final_gyro = np.array([kf_gx.update(raw_gyro[0]), kf_gy.update(raw_gyro[1]), kf_gz.update(raw_gyro[2])])

            # METHOD 2: LOW-PASS
            # alpha = 0.1
            # lp_acc = (1 - alpha) * lp_acc + alpha * raw_acc
            # lp_gyro = (1 - alpha) * lp_gyro + alpha * raw_gyro
            # final_acc, final_gyro = lp_acc, lp_gyro

            # 4. OPTIONAL: ORIENTATION FUSION (Complementary Filter)
            # (Keeping the math here in case you want to print Roll/Pitch/Yaw)
            omega = np.deg2rad(final_gyro)
            acc_norm = np.linalg.norm(final_acc)
            if 0.95 < acc_norm < 1.05:
                acc_n = final_acc / acc_norm
                g_pred = quaternion.rotate_vectors(q.conjugate(), np.array([0.0, 0.0, 1.0]))
                omega += 0.15 * np.cross(g_pred, acc_n)
            
            angle = np.linalg.norm(omega) * dt
            if angle > 1e-9:
                axis = omega / np.linalg.norm(omega)
                dq = np.quaternion(np.cos(angle/2), *(axis * np.sin(angle/2)))
                q = (q * dq).normalized()

            # 5. PRINT TO COM PORT
            # Format: ax, ay, az, gx, gy, gz
            output_str = f"{final_acc[0]:.3f},{final_acc[1]:.3f},{final_acc[2]:.3f},{final_gyro[0]:.2f},{final_gyro[1]:.2f},{final_gyro[2]:.2f}"
            print(output_str)

    except KeyboardInterrupt:
        print("\nClosing Serial Port...")
        ser.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()