import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3
from collections import deque
import numpy as np
import time
import quaternion

class LocationVisualizer:
    def __init__(self, dimensions=2, anchor_positions=None, x_lim=(0, 150), y_lim=(0, 100), z_lim=(0, 60), history_length=150):
        self.dimensions = dimensions
        self.x_lim = x_lim; self.y_lim = y_lim; self.z_lim = z_lim
        self.anchor_positions = anchor_positions if anchor_positions else {}
        self.latest_data = None  
        self.history_length = history_length
        self.range_history = {str(aid): deque([0.0]*history_length, maxlen=history_length) for aid in self.anchor_positions.keys()}
        
        # --- IMU QUATERNION STATE ---
        self.q = np.quaternion(1.0, 0.0, 0.0, 0.0)
        self.last_imu_time = time.time()
        
        self.fig = plt.figure(figsize=(14, 7)) 
        self.dynamic_lines = {}; self.history_lines = {} 
        self._setup_plot()

    def _setup_plot(self):
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
        self.ax = self.fig.add_subplot(gs[0, 0], projection='3d') if self.dimensions == 3 else self.fig.add_subplot(gs[0, 0])
        if self.dimensions == 3:
            self.ax.set_zlim(self.z_lim); self.ax.set_zlabel("Z (meters)"); self.ax.view_init(elev=20, azim=-35)
        else: self.ax.set_aspect('equal', adjustable='box')
        
        self.ax.set_xlim(self.x_lim); self.ax.set_ylim(self.y_lim); self.ax.set_xlabel("X (meters)"); self.ax.set_ylabel("Y (meters)")

        for aid, pos in self.anchor_positions.items():
            if self.dimensions == 3:
                self.ax.scatter(pos[0], pos[1], pos[2], color='red', marker='^', s=100, edgecolors='black')
                self.ax.text(pos[0], pos[1], pos[2] + 0.3, f"A{aid}", color='red', weight='bold')
            else:
                self.ax.scatter(pos[0], pos[1], color='red', marker='^', s=100, edgecolors='black')

        if self.dimensions == 3:
            self.tag_scatter, = self.ax.plot([], [], [], marker='o', color='blue', ms=12, ls='None', label='Tag')
            self.tag_axis_x, = self.ax.plot([], [], [], color='red', lw=3, label='X-forward') 
            self.tag_axis_y, = self.ax.plot([], [], [], color='green', lw=3) 
            self.tag_axis_z, = self.ax.plot([], [], [], color='blue', lw=3) 
            self.info_text = self.ax.text2D(0.02, 0.98, "Waiting...", transform=self.ax.transAxes, bbox=dict(facecolor='white', alpha=0.8), va='top', family='monospace')
        else:
            self.tag_scatter, = self.ax.plot([], [], marker='o', color='blue', ms=12, ls='None', label='Tag')
            self.tag_axis_x, = self.ax.plot([], [], color='red', lw=3, label='X-forward') 
            self.tag_axis_y, = self.ax.plot([], [], color='green', lw=3) 
            self.tag_axis_z, = self.ax.plot([], [], color='blue', lw=3) 
            self.info_text = self.ax.text(0.02, 0.98, "Waiting...", transform=self.ax.transAxes, bbox=dict(facecolor='white', alpha=0.8), va='top', family='monospace')

        self.ax.legend(loc="lower right")
        self.ax_dist = self.fig.add_subplot(gs[0, 1])
        self.ax_dist.set_xlim(0, self.history_length); self.ax_dist.set_ylim(0, 10); self.ax_dist.grid(True)

        colors = plt.cm.tab10.colors
        for i, aid in enumerate(self.anchor_positions.keys()):
            line, = self.ax_dist.plot([], [], label=f"A{aid}", color=colors[i % len(colors)], linewidth=1.5)
            self.history_lines[str(aid)] = line
        self.ax_dist.legend(loc="upper right", ncol=2, fontsize='small')
        plt.tight_layout()

    def update_position(self, x, y, z=0.0, ranges=None, imu_accel=None, imu_gyro=None):
        self.latest_data = (x, y, z, ranges or {}, imu_accel, imu_gyro)

    def _process_imu(self, accel, gyro, dt):
        omega = np.deg2rad(gyro)
        acc_norm = np.linalg.norm(accel)
        
        if acc_norm > 0.1: 
            acc_n = accel / acc_norm
            g_pred = quaternion.rotate_vectors(self.q.conjugate(), np.array([0.0, 0.0, 1.0]))
            omega += 0.15 * np.cross(g_pred, acc_n)
        
        angle = np.linalg.norm(omega) * dt
        if angle > 1e-9:
            axis = omega / np.linalg.norm(omega)
            dq = np.quaternion(np.cos(angle/2), *(axis * np.sin(angle/2)))
            self.q = (self.q * dq).normalized()

    def _update_frame(self, _):
        if not self.latest_data: return []
        tag_x, tag_y, tag_z, ranges, imu_accel, imu_gyro = self.latest_data

        for aid in self.anchor_positions.keys():
            aid_str = str(aid)
            self.range_history[aid_str].append(ranges.get(aid_str, self.range_history[aid_str][-1] if self.range_history[aid_str] else 0))
            self.history_lines[aid_str].set_data(range(len(self.range_history[aid_str])), list(self.range_history[aid_str]))

        if tag_x is not None and tag_y is not None:
            now = time.time(); dt = min(now - self.last_imu_time, 0.1); self.last_imu_time = now
            
            if self.dimensions == 3:
                self.tag_scatter.set_data_3d([tag_x], [tag_y], [tag_z])
            else:
                self.tag_scatter.set_data([tag_x], [tag_y])
                
            if imu_accel is not None and imu_gyro is not None:
                self._process_imu(imu_accel, imu_gyro, dt)
                R = quaternion.as_rotation_matrix(self.q)
                axis_len = 0.5
                ax_x = R @ np.array([axis_len, 0, 0]); ax_y = R @ np.array([0, axis_len, 0]); ax_z = R @ np.array([0, 0, axis_len])
                
                if self.dimensions == 3:
                    self.tag_axis_x.set_data_3d([tag_x, tag_x + ax_x[0]], [tag_y, tag_y + ax_x[1]], [tag_z, tag_z + ax_x[2]])
                    self.tag_axis_y.set_data_3d([tag_x, tag_x + ax_y[0]], [tag_y, tag_y + ax_y[1]], [tag_z, tag_z + ax_y[2]])
                    self.tag_axis_z.set_data_3d([tag_x, tag_x + ax_z[0]], [tag_y, tag_y + ax_z[1]], [tag_z, tag_z + ax_z[2]])
                else:
                    self.tag_axis_x.set_data([tag_x, tag_x + ax_x[0]], [tag_y, tag_y + ax_x[1]])
                    self.tag_axis_y.set_data([tag_x, tag_x + ax_y[0]], [tag_y, tag_y + ax_y[1]])
                    self.tag_axis_z.set_data([tag_x, tag_x + ax_z[0]], [tag_y, tag_y + ax_z[1]])

            # Restore dynamic dotted lines connecting tag to anchors
            for aid, r_val in ranges.items():
                if aid in self.anchor_positions:
                    ap = self.anchor_positions[aid]
                    if aid not in self.dynamic_lines:
                        line, = self.ax.plot([], [], [] if self.dimensions==3 else [], ls=':', color='black', alpha=0.2)
                        self.dynamic_lines[aid] = line
                    
                    if self.dimensions == 3:
                        self.dynamic_lines[aid].set_data_3d([ap[0], tag_x], [ap[1], tag_y], [ap[2], tag_z])
                    else:
                        self.dynamic_lines[aid].set_data([ap[0], tag_x], [ap[1], tag_y])

            # Restore detailed text overlay with ranges
            info = f"TAG POS (m)\n-----------\nX: {tag_x:>5.2f}\nY: {tag_y:>5.2f}\nZ: {tag_z:>5.2f}\n"
            info += "\nRANGES (m)\n----------"
            for k, v in ranges.items():
                info += f"\nA{k}: {v:>5.2f}"
            self.info_text.set_text(info)

        return [self.tag_scatter, self.tag_axis_x, self.tag_axis_y, self.tag_axis_z, self.info_text] + list(self.history_lines.values()) + list(self.dynamic_lines.values())

    def start(self):
        self.ani = FuncAnimation(self.fig, self._update_frame, interval=50, blit=False, cache_frame_data=False)
        plt.show()