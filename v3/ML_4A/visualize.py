import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3
from collections import deque

class LocationVisualizer:
    def __init__(self, dimensions=2, anchor_positions=None, plot_limits=(-1.0, 6.0), history_length=150):
        """
        Initializes the 2D/3D visualization object with a distance history plot.
        
        :param dimensions: 2 for 2D plotting, 3 for 3D plotting.
        :param anchor_positions: Dictionary mapping anchor IDs to (X, Y, Z) coordinates.
        :param plot_limits: Tuple of (min, max) limits for the X and Y axes.
        :param history_length: Number of frames to keep in the distance time-series plot.
        """
        self.dimensions = dimensions
        self.plot_limits = plot_limits
        self.anchor_positions = anchor_positions if anchor_positions else {
            "1": (0.0, 0.0, 2.5),
            "2": (3.0, 0.0, 2.5),
            "3": (3.0, 2.5, 2.5),
        }
        
        # State variable to hold streamed data
        self.latest_data = None  # Format: (x, y, z, ranges_dict)
        
        # Time-series history for the distance plot
        self.history_length = history_length
        self.range_history = {str(aid): deque([0.0]*history_length, maxlen=history_length) 
                              for aid in self.anchor_positions.keys()}
        
        # Increased figure width to accommodate the dual plots
        self.fig = plt.figure(figsize=(14, 7)) 
        self.dynamic_lines = {}
        self.history_lines = {} # Lines for the distance plot
        self.ani = None
        
        self._setup_plot()

    def _setup_plot(self):
        """Internal method to build the plot layout based on chosen dimensions."""
        # Create a 1x2 grid: spatial plot gets more width than the time-series plot
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
        
        # --- 1. SPATIAL PLOT (Map) ---
        if self.dimensions == 3:
            self.ax = self.fig.add_subplot(gs[0, 0], projection='3d')
            self.ax.set_zlim(0, 3.0)
            self.ax.set_zlabel("Z (meters)")
            self.ax.set_title("Real-Time 3D UWB Localization")
        else:
            self.ax = self.fig.add_subplot(gs[0, 0])
            self.ax.set_aspect('equal', adjustable='box')
            self.ax.set_title("Real-Time 2D UWB Localization")

        self.ax.set_xlim(self.plot_limits)
        self.ax.set_ylim(self.plot_limits)
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")
        self.ax.grid(True, linestyle=':', alpha=0.4)

        # Plot Known Anchors
        for aid, pos in self.anchor_positions.items():
            if self.dimensions == 3:
                self.ax.scatter(pos[0], pos[1], pos[2], color='red', marker='^', s=120, zorder=5)
                self.ax.text(pos[0], pos[1], pos[2] + 0.15, f"A{aid}", color='red', weight='bold', ha='center')
            else:
                self.ax.scatter(pos[0], pos[1], color='red', marker='^', s=120, zorder=5)
                self.ax.text(pos[0], pos[1] + 0.15, f"A{aid}", color='red', weight='bold', ha='center')

        # Initialize graphical elements for the Tag
        if self.dimensions == 3:
            self.tag_scatter, = self.ax.plot([], [], [], marker='o', color='blue', markersize=14, linestyle='None', label='Tag', zorder=10)
            self.info_text = self.ax.text2D(
                0.02, 0.02, "Waiting for data...", fontsize=10, family="monospace",
                transform=self.ax.transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                verticalalignment='bottom'
            )
        else:
            self.tag_scatter, = self.ax.plot([], [], marker='o', color='blue', markersize=14, linestyle='None', label='Tag', zorder=10)
            self.info_text = self.ax.text(
                0.02, 0.02, "Waiting for data...", fontsize=10, family="monospace",
                transform=self.ax.transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                verticalalignment='bottom'
            )

        self.ax.legend(loc="upper right")

        # --- 2. DISTANCE BEHAVIOR PLOT (Time Series) ---
        self.ax_dist = self.fig.add_subplot(gs[0, 1])
        self.ax_dist.set_title("Anchor Ranges Over Time")
        self.ax_dist.set_xlabel("Time (Frames)")
        self.ax_dist.set_ylabel("Distance (meters)")
        self.ax_dist.set_xlim(0, self.history_length)
        
        # Adjust Y-limits dynamically or set to a known static max range (e.g., 0 to 10 meters)
        self.ax_dist.set_ylim(0, 10.0) 
        self.ax_dist.grid(True, linestyle=':', alpha=0.6)

        # Create a line for each anchor
        colors = plt.cm.tab10.colors
        for i, aid in enumerate(self.anchor_positions.keys()):
            line, = self.ax_dist.plot([], [], label=f"A{aid}", color=colors[i % len(colors)], linewidth=2)
            self.history_lines[str(aid)] = line
            
        self.ax_dist.legend(loc="upper right")
        
        plt.tight_layout()

    def update_position(self, x, y, z=None, ranges=None):
        """
        Public method to push new coordinate data to the visualizer.
        This can be called safely from another thread.
        """
        self.latest_data = (x, y, z, ranges or {})

    def _update_frame(self, _):
        """Internal callback for Matplotlib's FuncAnimation."""
        artists = [self.tag_scatter, self.info_text] + list(self.dynamic_lines.values()) + list(self.history_lines.values())
        
        if not self.latest_data:
            return artists

        tag_x, tag_y, tag_z, ranges = self.latest_data

        # --- Update Distance Time-Series Data ---
        for aid in self.anchor_positions.keys():
            aid_str = str(aid)
            if ranges and aid_str in ranges:
                self.range_history[aid_str].append(ranges[aid_str])
            else:
                # If no data this frame, duplicate the last known value to keep the graph moving
                last_val = self.range_history[aid_str][-1] if self.range_history[aid_str] else 0
                self.range_history[aid_str].append(last_val)
            
            # Set data for the line
            y_data = list(self.range_history[aid_str])
            x_data = list(range(len(y_data)))
            self.history_lines[aid_str].set_data(x_data, y_data)


        # --- Update Spatial Plot ---
        for line in self.dynamic_lines.values():
            if self.dimensions == 3:
                line.set_data_3d([], [], [])
            else:
                line.set_data([], [])

        dist_text = "Calculated Position:\n"
        str_x = f" X: {tag_x:6.3f} m\n" if tag_x is not None else " X: [DROPPED]\n"
        str_y = f" Y: {tag_y:6.3f} m\n" if tag_y is not None else " Y: [DROPPED]\n"
        str_z = f" Z: {tag_z:6.3f} m\n" if tag_z is not None else " Z: [DROPPED]\n"
        dist_text += str_x + str_y + str_z + "\n"

        if self.dimensions == 3:
            valid_pos = (tag_x is not None and tag_y is not None and tag_z is not None)
        else:
            valid_pos = (tag_x is not None and tag_y is not None)

        if valid_pos:
            if self.dimensions == 3:
                self.tag_scatter.set_data_3d([tag_x], [tag_y], [tag_z])
            else:
                self.tag_scatter.set_data([tag_x], [tag_y])
            
            dist_text += "Reported Ranges:\n"
            for aid, reported_range in ranges.items():
                dist_text += f" Anchor A{aid}: {reported_range:6.3f} m\n"
                
                if aid in self.anchor_positions:
                    ax_pos, ay_pos, az_pos = self.anchor_positions[aid]
                    
                    if aid not in self.dynamic_lines:
                        if self.dimensions == 3:
                            line, = self.ax.plot([], [], [], linestyle='--', color='gray', alpha=0.5, zorder=3)
                        else:
                            line, = self.ax.plot([], [], linestyle='--', color='gray', alpha=0.5, zorder=3)
                        self.dynamic_lines[aid] = line
                    
                    if self.dimensions == 3:
                        self.dynamic_lines[aid].set_data_3d([ax_pos, tag_x], [ay_pos, tag_y], [az_pos, tag_z])
                    else:
                        self.dynamic_lines[aid].set_data([ax_pos, tag_x], [ay_pos, tag_y])
        else:
            if self.dimensions == 3:
                self.tag_scatter.set_data_3d([], [], [])
            else:
                self.tag_scatter.set_data([], [])
            
            dist_text += "!!! POSITION DATA INVALID !!!\n"
            req_axes = "(X,Y,Z)" if self.dimensions == 3 else "(X,Y)"
            dist_text += f"Waiting for complete {req_axes} fix."

        self.info_text.set_text(dist_text)
        return artists

    def start(self):
        """Starts the animation and blocks execution by showing the plot."""
        use_blit = (self.dimensions == 2)
        
        self.ani = FuncAnimation(
            self.fig,
            self._update_frame,
            interval=50,
            cache_frame_data=False,
            blit=use_blit,
        )
        plt.show()