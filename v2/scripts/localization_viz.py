import serial
import time
import threading
import math
import re
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3

# ==========================================
# SETTINGS
# ==========================================
USE_MOCK_DATA = True  # Set to False to use real serial data
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
SERIAL_TIMEOUT = 1
MAX_QUEUE = 200

# Set to 2 for 2D visualization, or 3 for 3D visualization
PLOT_DIMENSIONS = 2

# Anchor mapping: Physical (X, Y, Z) coordinates in meters.
# Z is simply ignored during 2D plotting.
ANCHOR_POSITIONS = {
    "1": (0.0, 0.0, 2.5),   # Anchor A
    "2": (3.0, 0.0, 2.5),   # Anchor B
    "3": (3.0, 2.5, 2.5),   # Anchor C
}

PLOT_LIMITS = (-1.0, 6.0)

# ==========================================
# SERIAL / DATA READER
# ==========================================
loc_queue = deque(maxlen=MAX_QUEUE)
stop_flag = False

# Regex captures standard anchor report and optional position axes
REGEX_ANCHOR = re.compile(r"Anchor 0x([0-9A-Fa-f]+):\s*([\d.]+)\s*m")
REGEX_POS = re.compile(
    r"Position — x:\s*(?P<x>[-+\d.]+)"
    r"(?:\s+y:\s*(?P<y>[-+\d.]+))?"  
    r"(?:\s+z:\s*(?P<z>[-+\d.]+))?"  
)

def data_reader():
    """Background thread to read serial data or generate mock data."""
    global stop_flag
    
    if USE_MOCK_DATA:
        t_start = time.perf_counter()
        while not stop_flag:
            t = time.perf_counter() - t_start
            
            mock_x = 2.5 + 2.0 * math.sin(t * 0.5)
            mock_y = 2.0 + 1.5 * math.cos(t * 0.3)
            mock_z = 1.0 + 0.5 * math.sin(t * 1.5)
            
            # --- SIMULATE AXIS DROP ---
            current_time = time.perf_counter()
            if int(current_time) % 10 < 2: # Drop Z periodically
                mock_z = None 
            elif int(current_time) % 10 > 8: # Drop X and Y periodically
                mock_x = None
                mock_y = None
                
            mock_ranges = {}
            # Generate ranges if we have enough axes for the current mode
            if mock_x is not None and mock_y is not None and (mock_z is not None or PLOT_DIMENSIONS == 2):
                for aid, (ax, ay, az) in ANCHOR_POSITIONS.items():
                    if current_time % 5 < 4.5 or aid == "1":
                        dist = math.sqrt((mock_x - ax)**2 + (mock_y - ay)**2 + ((mock_z or 0) - az)**2)
                        mock_ranges[aid] = dist
            
            loc_queue.append((time.perf_counter(), mock_x, mock_y, mock_z, mock_ranges))
            time.sleep(0.1)
    else:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
        time.sleep(2)
        
        in_report = False
        current_ranges = {}
        
        try:
            while not stop_flag:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                
                if "--- Range Report ---" in line:
                    in_report = True
                    current_ranges = {}
                    continue
                
                if in_report:
                    match_anchor = REGEX_ANCHOR.search(line)
                    if match_anchor:
                        anchor_id = match_anchor.group(1).upper()
                        anchor_id = anchor_id.lstrip('0') if anchor_id.lstrip('0') else '0'
                        current_ranges[anchor_id] = float(match_anchor.group(2))
                        continue
                    
                    match_pos = REGEX_POS.search(line)
                    if match_pos:
                        try: tag_x = float(match_pos.group('x')) 
                        except (TypeError, ValueError): tag_x = None
                        
                        try: tag_y = float(match_pos.group('y'))
                        except (TypeError, ValueError): tag_y = None
                        
                        try: tag_z = float(match_pos.group('z'))
                        except (TypeError, ValueError): tag_z = None
                        
                        loc_queue.append((time.perf_counter(), tag_x, tag_y, tag_z, current_ranges))
                        in_report = False 
                        continue
                    
                    if "Ranging failed" in line:
                        in_report = False
                        
        finally:
            ser.close()

# ==========================================
# MAIN VISUALIZATION
# ==========================================
def main():
    global stop_flag

    threading.Thread(target=data_reader, daemon=True).start()

    fig = plt.figure(figsize=(10, 8))
    
    # ------------------------------------------
    # PLOT SETUP: Branching for 2D vs 3D
    # ------------------------------------------
    if PLOT_DIMENSIONS == 3:
        ax = fig.add_subplot(projection='3d')
        ax.set_zlim(0, 3.0)
        ax.set_zlabel("Z (meters)")
        ax.set_title("Real-Time 3D UWB Localization")
    else:
        ax = fig.add_subplot()
        ax.set_aspect('equal', adjustable='box')
        ax.set_title("Real-Time 2D UWB Localization")

    ax.set_xlim(PLOT_LIMITS)
    ax.set_ylim(PLOT_LIMITS)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.grid(True, linestyle=':', alpha=0.4)

    # 1. Plot Known Anchors
    for aid, pos in ANCHOR_POSITIONS.items():
        if PLOT_DIMENSIONS == 3:
            ax.scatter(pos[0], pos[1], pos[2], color='red', marker='^', s=120, zorder=5)
            ax.text(pos[0], pos[1], pos[2] + 0.15, f"A{aid}", color='red', weight='bold', ha='center')
        else:
            ax.scatter(pos[0], pos[1], color='red', marker='^', s=120, zorder=5)
            ax.text(pos[0], pos[1] + 0.15, f"A{aid}", color='red', weight='bold', ha='center')

    # 2. Initialize Tag
    if PLOT_DIMENSIONS == 3:
        tag_scatter, = ax.plot([], [], [], marker='o', color='blue', markersize=14, linestyle='None', label='Tag', zorder=10)
    else:
        tag_scatter, = ax.plot([], [], marker='o', color='blue', markersize=14, linestyle='None', label='Tag', zorder=10)

    # 3. Dynamic Distance Lines
    dynamic_lines = {}
    ax.legend(loc="upper right")

    # 4. Text Box Readout
    if PLOT_DIMENSIONS == 3:
        info_text = ax.text2D(
            0.02, 0.02, "Waiting for data...", fontsize=10, family="monospace",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            verticalalignment='bottom'
        )
    else:
        info_text = ax.text(
            0.02, 0.02, "Waiting for data...", fontsize=10, family="monospace",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            verticalalignment='bottom'
        )

    def update(_):
        if not loc_queue:
            return [tag_scatter, info_text] + list(dynamic_lines.values())

        latest = loc_queue.pop()
        loc_queue.clear()
        
        t, tag_x, tag_y, tag_z, ranges = latest

        # Hide old lines
        for line in dynamic_lines.values():
            if PLOT_DIMENSIONS == 3:
                line.set_data_3d([], [], [])
            else:
                line.set_data([], [])

        # Format Text
        dist_text = "Calculated Position:\n"
        str_x = f" X: {tag_x:6.3f} m\n" if tag_x is not None else " X: [DROPPED]\n"
        str_y = f" Y: {tag_y:6.3f} m\n" if tag_y is not None else " Y: [DROPPED]\n"
        str_z = f" Z: {tag_z:6.3f} m\n" if tag_z is not None else " Z: [DROPPED]\n"
        dist_text += str_x + str_y + str_z + "\n"

        # Determine validity based on required dimensions
        if PLOT_DIMENSIONS == 3:
            valid_pos = (tag_x is not None and tag_y is not None and tag_z is not None)
        else:
            valid_pos = (tag_x is not None and tag_y is not None)

        if valid_pos:
            # 1. Update Graphical Tag
            if PLOT_DIMENSIONS == 3:
                tag_scatter.set_data_3d([tag_x], [tag_y], [tag_z])
            else:
                tag_scatter.set_data([tag_x], [tag_y])
            
            # 2. Update Lines
            dist_text += "Reported Ranges:\n"
            for aid, reported_range in ranges.items():
                dist_text += f" Anchor 0x{aid}: {reported_range:6.3f} m\n"
                
                if aid in ANCHOR_POSITIONS:
                    ax_pos, ay_pos, az_pos = ANCHOR_POSITIONS[aid]
                    
                    if aid not in dynamic_lines:
                        if PLOT_DIMENSIONS == 3:
                            line, = ax.plot([], [], [], linestyle='--', color='gray', alpha=0.5, zorder=3)
                        else:
                            line, = ax.plot([], [], linestyle='--', color='gray', alpha=0.5, zorder=3)
                        dynamic_lines[aid] = line
                    
                    if PLOT_DIMENSIONS == 3:
                        dynamic_lines[aid].set_data_3d([ax_pos, tag_x], [ay_pos, tag_y], [az_pos, tag_z])
                    else:
                        dynamic_lines[aid].set_data([ax_pos, tag_x], [ay_pos, tag_y])
        else:
            # Missing required axes
            if PLOT_DIMENSIONS == 3:
                tag_scatter.set_data_3d([], [], [])
            else:
                tag_scatter.set_data([], [])
            
            dist_text += "!!! POSITION DATA INVALID !!!\n"
            req_axes = "(X,Y,Z)" if PLOT_DIMENSIONS == 3 else "(X,Y)"
            dist_text += f"Waiting for complete {req_axes} fix."

        info_text.set_text(dist_text)
        return [tag_scatter, info_text] + list(dynamic_lines.values())

    # Blitting is problematic in 3D but helpful in 2D
    use_blit = (PLOT_DIMENSIONS == 2)
    
    ani = FuncAnimation(
        fig,
        update,
        interval=50,
        cache_frame_data=False,
        blit=use_blit,
    )

    try:
        plt.show()
    finally:
        stop_flag = True

if __name__ == "__main__":
    main()