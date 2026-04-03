import serial
import time
import threading
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import quaternion  # numpy-quaternion


# SETTINGS
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
SERIAL_TIMEOUT = 1

ANGLE_OFFSETS = (0, 0, 0)
ACC_GAIN = 2.0
MAX_QUEUE = 200
AXIS_LEN = 0.8


# QUATERNION CONVERSIONS
def quat_to_elev_azim_roll(q, angle_offsets=(0, 0, 0)):
    q0, q1, q2, q3 = q.w, q.x, q.y, q.z
    phi = np.arctan2(-2*q1*q2 + 2*q0*q3, q1**2 + q0**2 - q3**2 - q2**2)
    theta = np.arcsin(np.clip(2*q1*q3 + 2*q0*q2, -1.0, 1.0))
    psi = np.arctan2(-2*q2*q3 + 2*q0*q1, q3**2 - q2**2 - q1**2 + q0**2)
    azim = np.rad2deg(phi) + angle_offsets[0]
    elev = np.rad2deg(-theta) + angle_offsets[1]
    roll = np.rad2deg(psi) + angle_offsets[2]
    return elev, azim, roll


def elev_azim_roll_to_quat(elev, azim, roll, angle_offsets=(0, 0, 0)):
    phi = np.deg2rad(azim) - angle_offsets[0]
    theta = np.deg2rad(-elev) - angle_offsets[1]
    psi = np.deg2rad(roll) - angle_offsets[2]
    q0 = np.cos(phi/2)*np.cos(theta/2)*np.cos(psi/2) - np.sin(phi/2)*np.sin(theta/2)*np.sin(psi/2)
    q1 = np.cos(phi/2)*np.cos(theta/2)*np.sin(psi/2) + np.sin(phi/2)*np.sin(theta/2)*np.cos(psi/2)
    q2 = np.cos(phi/2)*np.sin(theta/2)*np.cos(psi/2) - np.sin(phi/2)*np.cos(theta/2)*np.sin(psi/2)
    q3 = np.cos(phi/2)*np.sin(theta/2)*np.sin(psi/2) + np.sin(phi/2)*np.cos(theta/2)*np.cos(psi/2)
    return np.quaternion(q0, q1, q2, q3)


# SERIAL READER
imu_queue = deque(maxlen=MAX_QUEUE)
stop_flag = False


def parse_imu_line(line):
    parts = [x.strip() for x in line.strip().split(",")]
    if len(parts) != 6:
        raise ValueError(f"Expected 6 values, got {len(parts)}: {line!r}")
    return tuple(float(x) for x in parts)


def serial_reader():
    global stop_flag
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
    time.sleep(2)

    try:
        while not stop_flag:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw:
                continue

            try:
                imu_queue.append((time.perf_counter(), *parse_imu_line(raw)))
            except ValueError:
                pass
    finally:
        ser.close()


# QUATERNION UPDATE
def quat_from_gyro(omega, dt):
    angle = np.linalg.norm(omega) * dt
    if angle < 1e-9:
        return np.quaternion(1, 0, 0, 0)

    axis = omega / np.linalg.norm(omega)
    s = np.sin(angle / 2)
    return np.quaternion(np.cos(angle / 2), *(axis * s))


def accel_correction(q, acc, gain):
    norm = np.linalg.norm(acc)
    if norm < 1e-9:
        return np.zeros(3)

    acc = acc / norm

    # predicted gravity direction in body frame
    g_pred = quaternion.rotate_vectors(q.conjugate(), np.array([0.0, 0.0, 1.0]))
    g_pred = g_pred / np.linalg.norm(g_pred)

    error = np.cross(g_pred, acc)
    return gain * error


def quat_to_rotmat(q):
    q = q.normalized()
    w, x, y, z = q.w, q.x, q.y, q.z
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


# MAIN
def main():
    global stop_flag

    q = np.quaternion(1, 0, 0, 0)
    last_t = None

    latest_acc = np.zeros(3)
    latest_gyro = np.zeros(3)

    threading.Thread(target=serial_reader, daemon=True).start()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")

    # Fixed world axes
    ax.plot([0, 1], [0, 0], [0, 0], "--", linewidth=1)
    ax.plot([0, 0], [0, 1], [0, 0], "--", linewidth=1)
    ax.plot([0, 0], [0, 0], [0, 1], "--", linewidth=1)

    # Rotating body axes
    x_line, = ax.plot([0, AXIS_LEN], [0, 0], [0, 0], linewidth=3, label="Body X")
    y_line, = ax.plot([0, 0], [0, AXIS_LEN], [0, 0], linewidth=3, label="Body Y")
    z_line, = ax.plot([0, 0], [0, 0], [0, AXIS_LEN], linewidth=3, label="Body Z")

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper left")

    info_text = fig.text(
        0.02, 0.02, "", fontsize=10, family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    def update(_):
        nonlocal q, last_t, latest_acc, latest_gyro

        if not imu_queue:
            return x_line, y_line, z_line

        latest = imu_queue.pop()
        imu_queue.clear()

        t, ax_m, ay_m, az_m, gx, gy, gz = latest
        latest_acc = np.array([ax_m, ay_m, az_m], dtype=float)
        latest_gyro = np.array([gx, gy, gz], dtype=float)

        if last_t is None:
            last_t = t
            return x_line, y_line, z_line

        dt = t - last_t
        last_t = t

        if dt <= 0 or dt > 0.5:
            return x_line, y_line, z_line

        omega = np.deg2rad(latest_gyro)
        omega += accel_correction(q, latest_acc, ACC_GAIN)

        dq = quat_from_gyro(omega, dt)
        q = (q * dq).normalized()

        elev, azim, roll = quat_to_elev_azim_roll(q, ANGLE_OFFSETS)

        R = quat_to_rotmat(q)

        x_axis = R @ np.array([AXIS_LEN, 0.0, 0.0])
        y_axis = R @ np.array([0.0, AXIS_LEN, 0.0])
        z_axis = R @ np.array([0.0, 0.0, AXIS_LEN])

        x_line.set_data([0, x_axis[0]], [0, x_axis[1]])
        x_line.set_3d_properties([0, x_axis[2]])

        y_line.set_data([0, y_axis[0]], [0, y_axis[1]])
        y_line.set_3d_properties([0, y_axis[2]])

        z_line.set_data([0, z_axis[0]], [0, z_axis[1]])
        z_line.set_3d_properties([0, z_axis[2]])

        info_text.set_text(
            f"Roll : {roll:8.2f} deg\n"
            f"Elev : {elev:8.2f} deg\n"
            f"Azim : {azim:8.2f} deg\n\n"
            f"Accel x: {latest_acc[0]:8.3f} g\n"
            f"Accel y: {latest_acc[1]:8.3f} g\n"
            f"Accel z: {latest_acc[2]:8.3f} g\n\n"
            f"Gyro  x: {latest_gyro[0]:8.3f} dps\n"
            f"Gyro  y: {latest_gyro[1]:8.3f} dps\n"
            f"Gyro  z: {latest_gyro[2]:8.3f} dps"
        )

        return x_line, y_line, z_line

    ani = FuncAnimation(
        fig,
        update,
        interval=10,
        cache_frame_data=False,
        blit=False,
    )

    try:
        plt.show()
    finally:
        stop_flag = True


if __name__ == "__main__":
    main()