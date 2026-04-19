import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3
from collections import deque
import numpy as np

class LocationVisualizer:
    def __init__(self, dimensions=2, anchor_positions=None, x_lim=(0, 150), y_lim=(0, 100), z_lim=(0, 60), history_length=150):
        """
        Initializes the 2D/3D visualization object.
        
        :param dimensions: 2 for 2D (XY), 3 for 3D (XYZ)
        :param anchor_positions: Dict mapping IDs to (X, Y, Z) tuples
        :param x_lim: Tuple (min, max) for X axis
        :param y_lim: Tuple (min, max) for Y axis
        :param z_lim: Tuple (min, max) for Z axis
        :param history_length: Number of frames to keep in the range plot
        """
        self.dimensions = dimensions
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.z_lim = z_lim
        self.anchor_positions = anchor_positions if anchor_positions else {}
        
        # State variables
        self.latest_data = None  # Format: (x, y, z, ranges_dict)
        self.history_length = history_length
        self.range_history = {str(aid): deque([0.0]*history_length, maxlen=history_length) 
                              for aid in self.anchor_positions.keys()}
        
        self.fig = plt.figure(figsize=(14, 7)) 
        self.dynamic_lines = {}
        self.history_lines = {} 
        self.ani = None
        
        self._setup_plot()

    def _setup_plot(self):
        """Internal method to build the plot layout."""
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
        
        # --- 1. SPATIAL PLOT (Map) ---
        if self.dimensions == 3:
            self.ax = self.fig.add_subplot(gs[0, 0], projection='3d')
            self.ax.set_zlim(self.z_lim) 
            self.ax.set_zlabel("Z (meters)")
            self.ax.view_init(elev=20, azim=-35)
            self.ax.set_title("Real-Time 3D UWB Localization")
        else:
            self.ax = self.fig.add_subplot(gs[0, 0])
            self.ax.set_aspect('equal', adjustable='box')
            self.ax.set_title("Real-Time 2D UWB Localization")
            # Origin Crosshairs for 2D
            self.ax.axhline(0, color='black', linewidth=1, alpha=0.5)
            self.ax.axvline(0, color='black', linewidth=1, alpha=0.5)

        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")
        self.ax.grid(True, linestyle=':', alpha=0.4)

        # Plot Known Anchors
        for aid, pos in self.anchor_positions.items():
            if self.dimensions == 3:
                self.ax.scatter(pos[0], pos[1], pos[2], color='red', marker='^', s=100, edgecolors='black')
                self.ax.text(pos[0], pos[1], pos[2] + 1.0, f"A{aid}", color='red', weight='bold')
            else:
                self.ax.scatter(pos[0], pos[1], color='red', marker='^', s=100, edgecolors='black', zorder=5)
                self.ax.text(pos[0], pos[1] + 1.0, f"A{aid}", color='red', ha='center', weight='bold')

        # Initialize Tag Graphics
        if self.dimensions == 3:
            self.tag_scatter, = self.ax.plot([], [], [], marker='o', color='blue', ms=12, ls='None', label='Tag', markeredgecolor='white')
            self.info_text = self.ax.text2D(0.02, 0.98, "Waiting for data...", transform=self.ax.transAxes, 
                                            bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top', family='monospace')
        else:
            self.tag_scatter, = self.ax.plot([], [], marker='o', color='blue', ms=12, ls='None', label='Tag', zorder=10, markeredgecolor='white')
            self.info_text = self.ax.text(0.02, 0.98, "Waiting for data...", transform=self.ax.transAxes, 
                                          bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top', family='monospace')

        self.ax.legend(loc="lower right")

        # --- 2. DISTANCE PLOT (Time Series) ---
        self.ax_dist = self.fig.add_subplot(gs[0, 1])
        self.ax_dist.set_title("Anchor Ranges Over Time")
        self.ax_dist.set_xlabel("Time (Frames)")
        self.ax_dist.set_ylabel("Distance (m)")
        self.ax_dist.set_xlim(0, self.history_length)
        
        # Set Y-limit to max possible distance in the room
        max_dist = np.sqrt(self.x_lim[1]**2 + self.y_lim[1]**2 + self.z_lim[1]**2)
        self.ax_dist.set_ylim(0, max_dist) 
        self.ax_dist.grid(True, linestyle=':', alpha=0.6)

        colors = plt.cm.tab10.colors
        for i, aid in enumerate(self.anchor_positions.keys()):
            line, = self.ax_dist.plot([], [], label=f"A{aid}", color=colors[i % len(colors)], linewidth=1.5)
            self.history_lines[str(aid)] = line
            
        self.ax_dist.legend(loc="upper right", ncol=2, fontsize='small')
        plt.tight_layout()

    def update_position(self, x, y, z=0.0, ranges=None):
        """Public method to push new data."""
        self.latest_data = (x, y, z, ranges or {})

    def _update_frame(self, _):
        """Internal callback for animation frames."""
        if not self.latest_data:
            return []

        tag_x, tag_y, tag_z, ranges = self.latest_data

        # 1. Update Distance History
        for aid in self.anchor_positions.keys():
            aid_str = str(aid)
            val = ranges.get(aid_str, self.range_history[aid_str][-1] if self.range_history[aid_str] else 0)
            self.range_history[aid_str].append(val)
            self.history_lines[aid_str].set_data(range(len(self.range_history[aid_str])), list(self.range_history[aid_str]))

        # 2. Update Spatial Position
        valid = (tag_x is not None and tag_y is not None)
        if self.dimensions == 3: valid = valid and (tag_z is not None)

        if valid:
            if self.dimensions == 3:
                self.tag_scatter.set_data_3d([tag_x], [tag_y], [tag_z])
            else:
                self.tag_scatter.set_data([tag_x], [tag_y])
            
            # Update lines connecting Tag to Anchors
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

            # Update Text Overlay
            info = f"TAG POSITION (m)\n----------------\nX: {tag_x:>7.2f}\nY: {tag_y:>7.2f}\nZ: {tag_z:>7.2f}\n\nANCHOR RANGES\n-------------"
            for k, v in ranges.items():
                info += f"\nA{k}: {v:>6.2f}m"
            self.info_text.set_text(info)

        return [self.tag_scatter, self.info_text] + list(self.history_lines.values()) + list(self.dynamic_lines.values())

    def start(self):
        """Starts the animation and displays the plot."""
        use_blit = (self.dimensions == 2)
        self.ani = FuncAnimation(
            self.fig, 
            self._update_frame, 
            interval=50, 
            blit=use_blit, 
            cache_frame_data=False
        )
        plt.show()