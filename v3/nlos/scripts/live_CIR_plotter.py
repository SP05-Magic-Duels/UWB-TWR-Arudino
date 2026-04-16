import sys
import serial
import struct
import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets, QtCore

# --- Configuration ---
PORT = '/dev/ttyUSB1'
BAUD = 115200
NUM_SAMPLES = 128
HISTORY_DEPTH = 100 
SYNC_HEADER = b'\xEF\xBE\xAD\xDE'

class CIRVisualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("UWB CIR: Heatmap & Instantaneous Profile")
        self.resize(800, 800)
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        # Graphics Layout for multiple plots
        self.win = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.win)
        
        # --- Plot 1: Heatmap (Waterfall) ---
        self.plot_heatmap = self.win.addPlot(title="CIR History (Waterfall)")
        self.plot_heatmap.setLabel('bottom', "Samples (Delay / Distance)")
        self.plot_heatmap.setLabel('left', "Packet History")
        self.img = pg.ImageItem()
        self.plot_heatmap.addItem(self.img)
        
        # Color Map
        colormap = pg.colormap.get('viridis')
        bar = pg.ColorBarItem(values=(0, 1000), colorMap=colormap)
        bar.setImageItem(self.img)

        self.win.nextRow() # Move to the next row for the second plot

        # --- Plot 2: Instantaneous Line Plot ---
        self.plot_line = self.win.addPlot(title="Instantaneous CIR (Latest Packet)")
        self.plot_line.setLabel('bottom', "Samples (Delay / Distance)")
        self.plot_line.setLabel('left', "Amplitude")
        self.plot_line.setYRange(0, 1500) # Prevents Y-axis from jittering. Adjust if needed.
        self.curve = self.plot_line.plot(pen=pg.mkPen('y', width=2))
        
        # Link the X-axes so zooming is synchronized!
        self.plot_line.setXLink(self.plot_heatmap)

        # Data Buffers
        self.data_buffer = np.zeros((HISTORY_DEPTH, NUM_SAMPLES))
        self.latest_samples = np.zeros(NUM_SAMPLES)

        # Serial Setup
        try:
            self.ser = serial.Serial(PORT, BAUD, timeout=0.01)
            print(f"Connected to {PORT}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

        # Timer for fast updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(0)

    def update(self):
        # Read until buffer is mostly empty to catch up to real-time
        while self.ser.in_waiting >= (4 + 8 + NUM_SAMPLES * 2):
            if self.ser.read(1) == b'\xEF':
                if self.ser.read(3) == b'\xBE\xAD\xDE':
                    # Read metadata (Range + FP Index) 
                    metadata_raw = self.ser.read(8)
                    if len(metadata_raw) < 8: continue
                    _, fp_idx = struct.unpack('<ff', metadata_raw)
                    
                    # Read Samples
                    raw_samples = self.ser.read(NUM_SAMPLES * 2)
                    if len(raw_samples) == NUM_SAMPLES * 2:
                        self.latest_samples = struct.unpack(f'<{NUM_SAMPLES}H', raw_samples)
                        
                        # Roll heatmap buffer
                        self.data_buffer = np.roll(self.data_buffer, -1, axis=0)
                        self.data_buffer[-1, :] = self.latest_samples

        # Update visual elements outside the while loop to keep UI smooth
        self.img.setImage(self.data_buffer.T, autoLevels=False)
        self.curve.setData(self.latest_samples)

    def closeEvent(self, event):
        self.ser.close()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    viz = CIRVisualizer()
    viz.show()
    sys.exit(app.exec())