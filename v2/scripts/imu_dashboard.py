import serial
import time
import threading
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import quaternion

SERIAL_PORT = "COM9"
BAUD_RATE = 115200
MAX_QUEUE = 500
PLOT_WINDOW = 200
AXIS_LEN = 0.8
CALIBRATION_SAMPLES = 100 

# DATA STORAGE
imu_queue = deque(maxlen=MAX_QUEUE)

# Deques for the 6 raw readings (used for Moving Average)
acc_x_raw = deque([0]*PLOT_WINDOW, maxlen=PLOT_WINDOW)
acc_y_raw = deque([0]*PLOT_WINDOW, maxlen=PLOT_WINDOW)
acc_z_raw = deque([0]*PLOT_WINDOW, maxlen=PLOT_WINDOW)

gyro_x_raw = deque([0]*PLOT_WINDOW, maxlen=PLOT_WINDOW)
gyro_y_raw = deque([0]*PLOT_WINDOW, maxlen=PLOT_WINDOW)
gyro_z_raw = deque([0]*PLOT_WINDOW, maxlen=PLOT_WINDOW)

stop_flag = False
gyro_bias = np.array([0.0, 0.0, 0.0])
is_calibrated = False
calib_buffer = []

# Filtering state variables for Low-Pass
lp_acc = np.array([0.0, 0.0, 1.0])
lp_gyro = np.array([0.0, 0.0, 0.0])

# KALMAN FILTER CLASS
class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, initial_value=0.0):
        self.Q = process_variance      # Environment noise
        self.R = measurement_variance # Sensor noise
        self.x = initial_value         # State estimate
        self.P = 1.0                   # Error covariance

    def update(self, measurement):
        self.P = self.P + self.Q
        K = self.P / (self.P + self.R) # Kalman Gain
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

# SERIAL
def parse_imu_line(line):
    parts = [x.strip() for x in line.strip().split(",")]
    if len(parts) != 6: raise ValueError
    return tuple(float(x) for x in parts)

def serial_reader():
    global stop_flag
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        while not stop_flag:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw: continue
            try:
                imu_queue.append((time.perf_counter(), *parse_imu_line(raw)))
            except: pass
        ser.close()
    except Exception as e: print(f"Serial Error: {e}")

# QUATERNION TO ROTATION MATRIX
def quat_to_rotmat(q):
    q = q.normalized()
    w, x, y, z = q.w, q.x, q.y, q.z
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])

# MAIN DASHBOARD
def main():
    global stop_flag, is_calibrated, gyro_bias, calib_buffer, lp_acc, lp_gyro
    q = np.quaternion(1, 0, 0, 0)
    last_t = None

    threading.Thread(target=serial_reader, daemon=True).start()

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2)
    ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax_gyro = fig.add_subplot(gs[0, 1])
    ax_accel = fig.add_subplot(gs[1, :]) 

    x_line, = ax_3d.plot([], [], [], 'r', lw=3, label="X")
    y_line, = ax_3d.plot([], [], [], 'g', lw=3, label="Y")
    z_line, = ax_3d.plot([], [], [], 'b', lw=3, label="Z")
    ax_3d.set_xlim(-1, 1); ax_3d.set_ylim(-1, 1); ax_3d.set_zlim(-1, 1)
    ax_3d.set_title("Orientation (Fused)")

    g_lx, = ax_gyro.plot([], [], 'r', label="gx")
    g_ly, = ax_gyro.plot([], [], 'g', label="gy")
    g_lz, = ax_gyro.plot([], [], 'b', label="gz")
    ax_gyro.set_title("Gyroscope (deg/s)")
    ax_gyro.set_xlim(0, PLOT_WINDOW); ax_gyro.legend(loc="upper right")

    a_lx, = ax_accel.plot([], [], 'r', label="ax")
    a_ly, = ax_accel.plot([], [], 'g', label="ay")
    a_lz, = ax_accel.plot([], [], 'b', label="az")
    ax_accel.set_title("Accelerometer (G)")
    ax_accel.set_xlim(0, PLOT_WINDOW); ax_accel.legend(loc="upper right")

    def update(_):
        global is_calibrated, gyro_bias, calib_buffer, lp_acc, lp_gyro
        nonlocal q, last_t
        
        if not imu_queue: return
        t, ax_m, ay_m, az_m, gx, gy, gz = imu_queue.pop()
        imu_queue.clear()

        # Pre-filter raw storage
        acc_x_raw.append(ax_m); acc_y_raw.append(ay_m); acc_z_raw.append(az_m)
        gyro_x_raw.append(gx); gyro_y_raw.append(gy); gyro_z_raw.append(gz)

        if not is_calibrated:
            calib_buffer.append([gx, gy, gz])
            if len(calib_buffer) >= CALIBRATION_SAMPLES:
                gyro_bias = np.mean(calib_buffer, axis=0)
                is_calibrated = True
            return

        if last_t is None:
            last_t = t
            return
        dt = t - last_t
        last_t = t

        raw_acc = np.array([ax_m, ay_m, az_m])
        raw_gyro = np.array([gx, gy, gz]) - gyro_bias

        # ==========================================================
        # FILTERING METHODS
        # ==========================================================
        
        # METHOD 1: KALMAN FILTER 
        final_acc = np.array([kf_ax.update(raw_acc[0]), kf_ay.update(raw_acc[1]), kf_az.update(raw_acc[2])])
        final_gyro = np.array([kf_gx.update(raw_gyro[0]), kf_gy.update(raw_gyro[1]), kf_gz.update(raw_gyro[2])])

        # METHOD 2: LOW-PASS FILTER 
        # alpha = 0.1 
        # lp_acc = (1 - alpha) * lp_acc + alpha * raw_acc
        # lp_gyro = (1 - alpha) * lp_gyro + alpha * raw_gyro
        # final_acc, final_gyro = lp_acc, lp_gyro

        # METHOD 3: MOVING AVERAGE 
        # final_acc = np.array([np.mean(acc_x_raw), np.mean(acc_y_raw), np.mean(acc_z_raw)])
        # final_gyro = np.array([np.mean(gyro_x_raw), np.mean(gyro_y_raw), np.mean(gyro_z_raw)])

        # ==========================================================

        # ORIENTATION FUSION
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

        # Update Graphs
        x_v = np.arange(PLOT_WINDOW)
        g_lx.set_data(x_v, list(gyro_x_raw)); g_ly.set_data(x_v, list(gyro_y_raw)); g_lz.set_data(x_v, list(gyro_z_raw))
        a_lx.set_data(x_v, list(acc_x_raw)); a_ly.set_data(x_v, list(acc_y_raw)); a_lz.set_data(x_v, list(acc_z_raw))
        
        # Dynamic Limits
        ax_gyro.set_ylim(min(list(gyro_x_raw)+list(gyro_y_raw)+list(gyro_z_raw))-5, max(list(gyro_x_raw)+list(gyro_y_raw)+list(gyro_z_raw))+5)
        ax_accel.set_ylim(min(list(acc_x_raw)+list(acc_y_raw)+list(acc_z_raw))-0.2, max(list(acc_x_raw)+list(acc_y_raw)+list(acc_z_raw))+0.2)

        # Update 3D Axes
        try:
            R = quat_to_rotmat(q)
            xb = R @ np.array([AXIS_LEN, 0, 0])
            yb = R @ np.array([0, AXIS_LEN, 0])
            zb = R @ np.array([0, 0, AXIS_LEN])
            
            x_line.set_data([0, xb[0]], [0, xb[1]]); x_line.set_3d_properties([0, xb[2]])
            y_line.set_data([0, yb[0]], [0, yb[1]]); y_line.set_3d_properties([0, yb[2]])
            z_line.set_data([0, zb[0]], [0, zb[1]]); z_line.set_3d_properties([0, zb[2]])
        except:
            pass

    ani = FuncAnimation(fig, update, interval=30, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    stop_flag = True

if __name__ == "__main__":
    main()