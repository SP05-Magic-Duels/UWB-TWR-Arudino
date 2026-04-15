"""
UWB (N)LOS — Real Data Ingestion, Inference Pipeline & Visualization
Extends: uwb_nlos_cnn_bilstm_tl.py

Sections
--------
1.  RealUWBDataset     — loads CIR from CSV / NPY / HDF5 / serial port / DW1000 SDK
2.  preprocess_cir     — normalise + window raw CIR arrays
3.  InferencePipeline  — wraps a trained model for single-sample or batch prediction
4.  VisualisationSuite — confusion matrix, CIR plots, confidence histogram,
                         real-time classification dashboard (matplotlib)
"""

import time
import struct
import pathlib
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ── re-use model + helpers from the companion script ─────────────────────────
# (copy the CNN_BiLSTM class here or do: from uwb_nlos_cnn_bilstm_tl import ...)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CIR_LEN = 1016       # paper: 1016 time-domain values @ 1 ns resolution
LABELS   = ["LOS", "NLOS"]


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  PRE-PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_cir(
    raw: np.ndarray,
    *,
    target_len: int = CIR_LEN,
    normalize: bool = True,
) -> np.ndarray:
    """
    Prepare a raw CIR array for model input.

    Steps
    -----
    1. Ensure float32.
    2. Pad (zero) or trim to exactly `target_len` samples.
    3. L2-normalise each sample (preserves shape; matches eWINE pre-processing
       described in Section III-A: "normalized magnitude of LOS is greater than
       that of NLOS").

    Args:
        raw:        Array of shape (N, T) or (T,) — one or many CIR waveforms.
        target_len: Expected CIR length (default 1016).
        normalize:  Apply per-sample L2 normalisation.

    Returns:
        Float32 array of shape (N, target_len).
    """
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]          # (1, T)

    N, T = arr.shape

    # Pad / trim
    if T < target_len:
        pad = np.zeros((N, target_len - T), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    elif T > target_len:
        arr = arr[:, :target_len]

    # Normalise
    if normalize:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)   # avoid /0
        arr = arr / norms

    return arr                                       # (N, 1016)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  DATASET — multiple ingestion back-ends
# ═══════════════════════════════════════════════════════════════════════════════

class RealUWBDataset(Dataset):
    """
    Unified Dataset that loads CIR data from four common sources:

        source="csv"     — columns [cir_0..cir_1015, label]
        source="npy"     — two .npy files: X (N×1016) and y (N,)
        source="hdf5"    — HDF5 with datasets "cir" and "label"
        source="binary"  — raw binary stream (little-endian float32 × 1016
                           per sample, optional 1-byte label suffix)

    All samples are pre-processed via `preprocess_cir` before storage.

    Usage
    -----
    >>> ds = RealUWBDataset("office1_cir.csv", source="csv")
    >>> loader = DataLoader(ds, batch_size=64, shuffle=False)
    """

    def __init__(
        self,
        path: str | pathlib.Path,
        source: str = "csv",
        *,
        label_column: str  = "label",
        cir_prefix:   str  = "cir_",
        npy_label_path: str | None = None,
        hdf5_cir_key:  str = "cir",
        hdf5_label_key: str = "label",
        normalize: bool = True,
    ):
        super().__init__()
        path = pathlib.Path(path)

        if source == "csv":
            self.X, self.y = self._from_csv(
                path, label_column=label_column, cir_prefix=cir_prefix
            )
        elif source == "npy":
            self.X, self.y = self._from_npy(path, npy_label_path)
        elif source == "hdf5":
            self.X, self.y = self._from_hdf5(
                path, hdf5_cir_key, hdf5_label_key
            )
        elif source == "binary":
            self.X, self.y = self._from_binary(path)
        else:
            raise ValueError(f"Unknown source '{source}'. "
                             f"Choose csv | npy | hdf5 | binary.")

        self.X = preprocess_cir(self.X, normalize=normalize)

    # ── loaders ──────────────────────────────────────────────────────────────

    @staticmethod
    def _from_csv(path, label_column, cir_prefix):
        import pandas as pd
        df    = pd.read_csv(path)
        cols  = [c for c in df.columns if c.startswith(cir_prefix)]
        if not cols:
            # Fallback: assume all columns except label are CIR
            cols = [c for c in df.columns if c != label_column]
        X = df[cols].to_numpy(dtype=np.float32)
        y = df[label_column].to_numpy(dtype=np.int64) if label_column in df else None
        return X, y

    @staticmethod
    def _from_npy(cir_path, label_path):
        X = np.load(cir_path).astype(np.float32)
        y = np.load(label_path).astype(np.int64) if label_path else None
        return X, y

    @staticmethod
    def _from_hdf5(path, cir_key, label_key):
        import h5py
        with h5py.File(path, "r") as f:
            X = f[cir_key][:].astype(np.float32)
            y = f[label_key][:].astype(np.int64) if label_key in f else None
        return X, y

    @staticmethod
    def _from_binary(path):
        """
        Expects: [float32 × 1016 | uint8 label] repeated.
        Each record = 1016×4 + 1 = 4065 bytes.
        Adjust `record_bytes` if your hardware omits the label byte.
        """
        SAMPLE_FLOATS = CIR_LEN
        LABEL_BYTES   = 1
        RECORD_BYTES  = SAMPLE_FLOATS * 4 + LABEL_BYTES

        data  = pathlib.Path(path).read_bytes()
        n     = len(data) // RECORD_BYTES
        X, y  = np.zeros((n, SAMPLE_FLOATS), dtype=np.float32), np.zeros(n, dtype=np.int64)
        for i in range(n):
            offset = i * RECORD_BYTES
            X[i]   = struct.unpack_from(f"<{SAMPLE_FLOATS}f", data, offset)
            y[i]   = data[offset + SAMPLE_FLOATS * 4]
        return X, y

    # ── Dataset protocol ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)  # (1, 1016)
        if self.y is not None:
            return x, torch.tensor(self.y[idx], dtype=torch.long)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  SERIAL / DW1000 LIVE STREAM
# ═══════════════════════════════════════════════════════════════════════════════

class DW1000SerialReader:
    """
    Read CIR frames from a Decawave DW1000-based module over UART.

    The module must stream frames in the format:
        START_BYTE (0xAA) | float32[1016] LE | END_BYTE (0x55)

    Adjust `start_byte` / `end_byte` and `sample_count` to match your
    firmware's serial protocol.

    Usage
    -----
    >>> reader = DW1000SerialReader("/dev/ttyUSB0", baud=115200)
    >>> for cir in reader.stream(max_frames=100):
    ...     label, conf = pipeline.predict_single(cir)
    """

    START_BYTE    = 0xAA
    END_BYTE      = 0x55
    SAMPLE_COUNT  = CIR_LEN
    FRAME_PAYLOAD = SAMPLE_COUNT * 4   # float32

    def __init__(self, port: str, baud: int = 115200, timeout: float = 1.0):
        try:
            import serial
        except ImportError:
            raise ImportError("Install pyserial:  pip install pyserial")
        self.ser = serial.Serial(port, baud, timeout=timeout)
        print(f"[DW1000] Connected to {port} @ {baud} baud")

    def _read_frame(self) -> np.ndarray | None:
        """Block until one valid CIR frame is received."""
        import serial
        while True:
            byte = self.ser.read(1)
            if not byte:
                return None
            if byte[0] != self.START_BYTE:
                continue
            payload = self.ser.read(self.FRAME_PAYLOAD)
            if len(payload) != self.FRAME_PAYLOAD:
                continue
            end = self.ser.read(1)
            if not end or end[0] != self.END_BYTE:
                continue
            return np.frombuffer(payload, dtype="<f4").astype(np.float32)

    def stream(self, max_frames: int | None = None):
        """Yield pre-processed CIR arrays, one per frame."""
        count = 0
        try:
            while max_frames is None or count < max_frames:
                raw = self._read_frame()
                if raw is None:
                    continue
                yield preprocess_cir(raw)[0]    # (1016,)
                count += 1
        finally:
            self.ser.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  INFERENCE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class InferencePipeline:
    """
    Wraps a trained CNN_BiLSTM model for easy deployment.

    Supports
    --------
    - Single-sample prediction (numpy array → label + confidence)
    - Batch prediction (Dataset or DataLoader → labels + probabilities)
    - Live streaming (iterable of CIR arrays → real-time decisions)
    - Model hot-swap (reload weights without rebuilding the pipeline)

    Usage
    -----
    >>> pipeline = InferencePipeline.from_checkpoint("source_model_office1.pth")
    >>> label, conf = pipeline.predict_single(cir_array_1016)
    >>> results = pipeline.predict_batch(my_dataset)
    """

    def __init__(self, model: nn.Module):
        self.model = model.to(DEVICE)
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, path: str) -> "InferencePipeline":
        """Load a saved state-dict and return a ready pipeline."""
        # Import here to avoid circular dep if used standalone
        from cnn_bilstm import CNN_BiLSTM
        model = CNN_BiLSTM().to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        print(f"[Pipeline] Loaded checkpoint: {path}")
        return cls(model)

    def reload(self, path: str):
        """Hot-swap weights at runtime (useful in long-running services)."""
        self.model.load_state_dict(torch.load(path, map_location=DEVICE))
        self.model.eval()
        print(f"[Pipeline] Reloaded weights from {path}")

    @torch.no_grad()
    def predict_single(
        self,
        cir: np.ndarray,
        *,
        already_preprocessed: bool = False,
    ) -> tuple[str, float]:
        """
        Classify one CIR waveform.

        Args:
            cir:                  1-D array of shape (1016,) or (T,).
            already_preprocessed: Skip normalisation if True.

        Returns:
            (label, confidence)  e.g. ("NLOS", 0.923)
        """
        if not already_preprocessed:
            cir = preprocess_cir(cir)[0]               # (1016,)
        x = torch.tensor(cir, dtype=torch.float32)
        x = x.unsqueeze(0).unsqueeze(0).to(DEVICE)    # (1, 1, 1016)
        logits = self.model(x)
        probs  = torch.softmax(logits, dim=1).squeeze()
        cls    = int(probs.argmax().item())
        return LABELS[cls], float(probs[cls].item())

    @torch.no_grad()
    def predict_batch(
        self,
        source: Dataset | DataLoader,
        *,
        batch_size: int = 64,
        return_probs: bool = False,
    ) -> dict:
        """
        Run inference on an entire dataset.

        Args:
            source:       RealUWBDataset (or any Dataset / DataLoader).
            batch_size:   Ignored if source is already a DataLoader.
            return_probs: Include per-sample probability arrays in output.

        Returns:
            dict with keys: predictions, true_labels (if available), probs
        """
        if isinstance(source, Dataset):
            loader = DataLoader(source, batch_size=batch_size,
                                shuffle=False, pin_memory=True, num_workers=0)
        else:
            loader = source

        all_preds, all_labels, all_probs = [], [], []
        has_labels = True

        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                X, y = batch
                all_labels.extend(y.numpy().tolist())
            else:
                X = batch
                has_labels = False

            X = X.to(DEVICE)
            logits = self.model(X)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = probs.argmax(axis=1)
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

        result = {
            "predictions":  np.array(all_preds),
            "label_names":  [LABELS[p] for p in all_preds],
        }
        if has_labels:
            result["true_labels"] = np.array(all_labels)
        if return_probs:
            result["probs"] = np.array(all_probs)
        return result

    def stream_predict(
        self,
        cir_iterable,
        *,
        print_live: bool = True,
        callback=None,
    ):
        """
        Classify CIR arrays from any iterable in real time.

        Args:
            cir_iterable: Any iterable yielding (1016,) numpy arrays.
                          Pair with DW1000SerialReader.stream() for live use.
            print_live:   Print each result to stdout.
            callback:     Optional callable(label, confidence, cir) — hook
                          for logging, dashboards, or alarm triggers.

        Yields:
            (label, confidence) tuples.
        """
        for cir in cir_iterable:
            label, conf = self.predict_single(cir)
            if print_live:
                tag  = "🔴 NLOS" if label == "NLOS" else "🟢 LOS "
                print(f"  {tag}  conf={conf:.3f}")
            if callback:
                callback(label, conf, cir)
            yield label, conf


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  VISUALISATION SUITE
# ═══════════════════════════════════════════════════════════════════════════════

class VisualisationSuite:
    """
    Five publication-quality plots for model analysis and deployment.

    Methods
    -------
    confusion_matrix(y_true, y_pred)
    cir_comparison(los_cir, nlos_cir)
    confidence_histogram(probs)
    performance_summary(history)
    live_dashboard(pipeline, cir_iterable, max_frames)
    """

    # ── 1. Confusion matrix ──────────────────────────────────────────────────

    @staticmethod
    def confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title:  str = "Confusion Matrix",
        save_path: str | None = None,
    ):
        """Plot a styled confusion matrix (matches paper Tables II–IV)."""
        cm   = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=LABELS)

        fig, ax = plt.subplots(figsize=(5, 4))
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(title, fontsize=13, pad=12)

        # Annotate normalised percentages below raw counts
        total = cm.sum()
        for (r, c), count in np.ndenumerate(cm):
            ax.text(c, r + 0.35, f"({100 * count / total:.1f}%)",
                    ha="center", va="center", fontsize=9, color="gray")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    # ── 2. CIR waveform comparison ───────────────────────────────────────────

    @staticmethod
    def cir_comparison(
        los_cir:  np.ndarray,
        nlos_cir: np.ndarray,
        title:    str = "CIR: LOS vs NLOS",
        save_path: str | None = None,
    ):
        """
        Side-by-side time-domain CIR plots.

        Highlights the paper's key observation (Section II-A):
        "The first and peak paths for the LOS case exhibit a closer and higher
        peak than those of the NLOS case."
        """
        t = np.arange(CIR_LEN) * 1e-9 * 1e9   # nanoseconds

        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        fig.suptitle(title, fontsize=13)

        for ax, cir, label, color in zip(
            axes,
            [los_cir, nlos_cir],
            LABELS,
            ["steelblue", "tomato"],
        ):
            ax.plot(t, cir, lw=0.8, color=color, alpha=0.85)
            peak_idx = int(np.argmax(np.abs(cir)))
            ax.axvline(t[peak_idx], ls="--", lw=1, color="black", alpha=0.5,
                       label=f"Peak @ {t[peak_idx]:.0f} ns")
            ax.set_ylabel("Amplitude", fontsize=10)
            ax.set_title(label, fontsize=11, color=color, loc="left")
            ax.legend(fontsize=9, loc="upper right")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time (ns)", fontsize=10)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    # ── 3. Confidence histogram ───────────────────────────────────────────────

    @staticmethod
    def confidence_histogram(
        probs:     np.ndarray,
        y_true:    np.ndarray | None = None,
        title:     str = "Prediction Confidence Distribution",
        save_path: str | None = None,
    ):
        """
        Histogram of max-class softmax confidence scores.

        If `y_true` is provided, colours correct / incorrect predictions
        separately to reveal model calibration quality.
        """
        max_probs = probs.max(axis=1)

        fig, ax = plt.subplots(figsize=(7, 4))
        bins = np.linspace(0.5, 1.0, 26)

        if y_true is not None:
            preds   = probs.argmax(axis=1)
            correct = max_probs[preds == y_true]
            wrong   = max_probs[preds != y_true]
            ax.hist(correct, bins=bins, alpha=0.7, label="Correct",   color="steelblue")
            ax.hist(wrong,   bins=bins, alpha=0.7, label="Incorrect", color="tomato")
            ax.legend(fontsize=10)
        else:
            ax.hist(max_probs, bins=bins, color="steelblue", alpha=0.8)

        ax.set_xlabel("Confidence (max softmax probability)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    # ── 4. Training curve (from fit() history) ───────────────────────────────

    @staticmethod
    def performance_summary(
        history:   list[dict],
        title:     str = "Training History",
        save_path: str | None = None,
    ):
        """
        Four-panel training dashboard:
        loss, accuracy, precision/recall, and F1 score per epoch.
        """
        epochs = [h["epoch"]    for h in history]
        t_loss = [h["train_loss"] for h in history]
        v_loss = [h["loss"]     for h in history]
        v_acc  = [h["accuracy"] for h in history]
        v_prec = [h["precision"] for h in history]
        v_rec  = [h["recall"]   for h in history]
        v_f1   = [h["f1"]       for h in history]

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        fig.suptitle(title, fontsize=14)

        def _plot(ax, y_list, labels, colors, ylabel, ylim=None):
            for y, lbl, c in zip(y_list, labels, colors):
                ax.plot(epochs, y, label=lbl, color=c, lw=1.5)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_xlabel("Epoch", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            if ylim:
                ax.set_ylim(*ylim)

        _plot(axes[0,0], [t_loss, v_loss], ["Train loss", "Val loss"],
              ["steelblue","tomato"], "Loss")
        _plot(axes[0,1], [v_acc], ["Val accuracy"],
              ["seagreen"], "Accuracy", ylim=(0,1))
        _plot(axes[1,0], [v_prec, v_rec], ["Precision", "Recall"],
              ["darkorange","mediumpurple"], "Score", ylim=(0,1))
        _plot(axes[1,1], [v_f1], ["F1 score"],
              ["steelblue"], "F1", ylim=(0,1))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    # ── 5. Live real-time dashboard ───────────────────────────────────────────

    @staticmethod
    def live_dashboard(
        pipeline:     "InferencePipeline",
        cir_iterable,
        *,
        max_frames:   int   = 200,
        window:       int   = 60,
        update_every: int   = 5,
    ):
        """
        Real-time matplotlib dashboard for live UWB classification.

        Layout
        ------
        ┌──────────────────┬────────────────────┐
        │   Latest CIR     │  Rolling decision  │
        │   waveform       │  timeline          │
        ├──────────────────┼────────────────────┤
        │ Confidence gauge │ LOS/NLOS bar chart │
        └──────────────────┴────────────────────┘

        Pair with DW1000SerialReader.stream() for hardware deployment:
            reader = DW1000SerialReader("/dev/ttyUSB0")
            VisualisationSuite.live_dashboard(pipeline, reader.stream())

        Args:
            pipeline:     InferencePipeline instance.
            cir_iterable: Any iterable yielding (1016,) numpy arrays.
            max_frames:   Stop after this many frames.
            window:       Number of recent decisions shown in timeline.
            update_every: Redraw every N frames (reduces flickering).
        """
        plt.ion()
        fig = plt.figure(figsize=(12, 6), constrained_layout=True)
        fig.suptitle("UWB Real-Time LOS/NLOS Dashboard", fontsize=13)
        gs  = gridspec.GridSpec(2, 2, figure=fig)

        ax_cir   = fig.add_subplot(gs[0, 0])   # latest CIR
        ax_time  = fig.add_subplot(gs[0, 1])   # rolling timeline
        ax_conf  = fig.add_subplot(gs[1, 0])   # confidence gauge
        ax_count = fig.add_subplot(gs[1, 1])   # LOS vs NLOS totals

        t_axis = np.arange(CIR_LEN) * 1e0     # nanoseconds

        history_labels = []
        history_confs  = []
        counts = {"LOS": 0, "NLOS": 0}

        [cir_line]  = ax_cir.plot(t_axis, np.zeros(CIR_LEN), lw=0.8, color="steelblue")
        ax_cir.set_title("Latest CIR", fontsize=10)
        ax_cir.set_xlabel("Time (ns)", fontsize=9)
        ax_cir.set_ylabel("Amplitude", fontsize=9)

        ax_time.set_title("Rolling Decisions", fontsize=10)
        ax_time.set_xlim(0, window)
        ax_time.set_ylim(-0.5, 1.5)
        ax_time.set_yticks([0, 1])
        ax_time.set_yticklabels(LABELS)

        ax_conf.set_title("Confidence", fontsize=10)
        ax_conf.set_xlim(0, 1)
        ax_conf.set_yticks([])
        [conf_bar] = ax_conf.barh([""], [0], color="seagreen")

        bar_chart = ax_count.bar(LABELS, [0, 0], color=["steelblue", "tomato"])
        ax_count.set_title("Cumulative Counts", fontsize=10)

        for frame_idx, cir in enumerate(cir_iterable):
            if frame_idx >= max_frames:
                break

            label, conf = pipeline.predict_single(cir, already_preprocessed=True)
            counts[label] += 1
            history_labels.append(0 if label == "LOS" else 1)
            history_confs.append(conf)

            if frame_idx % update_every == 0:
                # CIR waveform
                cir_line.set_ydata(cir)
                cir_line.set_color("steelblue" if label == "LOS" else "tomato")
                ax_cir.relim(); ax_cir.autoscale_view()

                # Rolling timeline (scatter)
                ax_time.cla()
                ax_time.set_title("Rolling Decisions", fontsize=10)
                ax_time.set_xlim(0, window)
                ax_time.set_ylim(-0.5, 1.5)
                ax_time.set_yticks([0, 1])
                ax_time.set_yticklabels(LABELS)
                recent = history_labels[-window:]
                colors = ["steelblue" if v == 0 else "tomato" for v in recent]
                ax_time.scatter(range(len(recent)), recent, c=colors, s=18, zorder=2)

                # Confidence bar
                conf_bar[0].set_width(conf)
                conf_bar[0].set_color("seagreen" if conf > 0.85 else "goldenrod")
                ax_conf.set_xlim(0, 1)
                ax_conf.set_title(f"Confidence: {conf:.3f} ({label})", fontsize=10)

                # Bar chart
                for bar, lbl in zip(bar_chart, LABELS):
                    bar.set_height(counts[lbl])
                ax_count.set_ylim(0, max(counts.values()) * 1.2 + 1)

                plt.pause(0.001)

        plt.ioff()
        plt.show()
        print(f"\n[Dashboard] Final counts — LOS: {counts['LOS']}, NLOS: {counts['NLOS']}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  END-TO-END DEMO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── (a) Load real data from CSV ──────────────────────────────────────────
    # Uncomment to use a real eWINE CSV (columns: cir_0…cir_1015, label)
    # ds = RealUWBDataset("boiler_room.csv", source="csv")

    # ── Synthetic stand-in (replace with real data above) ───────────────────
    rng = np.random.default_rng(7)
    N   = 1000
    fake_cir    = rng.standard_normal((N, CIR_LEN)).astype(np.float32)
    fake_labels = rng.integers(0, 2, size=N).astype(np.int64)

    from torch.utils.data import TensorDataset
    X_t = torch.tensor(preprocess_cir(fake_cir)).unsqueeze(1)
    y_t = torch.tensor(fake_labels)
    ds  = TensorDataset(X_t, y_t)

    # ── (b) Load trained model & run batch inference ─────────────────────────
    # Replace with real checkpoint:
    # pipeline = InferencePipeline.from_checkpoint("source_model_office1.pth")

    # Dummy model for demo
    from uwb_nlos_cnn_bilstm_tl import CNN_BiLSTM
    dummy_model = CNN_BiLSTM()
    pipeline    = InferencePipeline(dummy_model)

    results = pipeline.predict_batch(ds, return_probs=True)

    # ── (c) Visualise results ─────────────────────────────────────────────────
    viz = VisualisationSuite()

    print("→ Confusion matrix")
    viz.confusion_matrix(
        results["true_labels"],
        results["predictions"],
        title="Boiler Room — CNN-BiLSTM (post-TL)",
    )

    print("→ CIR comparison")
    los_example  = preprocess_cir(fake_cir[fake_labels == 0][0])[0]
    nlos_example = preprocess_cir(fake_cir[fake_labels == 1][0])[0]
    viz.cir_comparison(los_example, nlos_example)

    print("→ Confidence histogram")
    viz.confidence_histogram(
        results["probs"],
        y_true=results["true_labels"],
    )

    # ── (d) Single-sample inference example ──────────────────────────────────
    sample_cir = fake_cir[42]
    label, conf = pipeline.predict_single(sample_cir)
    print(f"\nSample #42 → {label}  (confidence {conf:.3f})")

    # ── (e) Live stream demo (simulated) ─────────────────────────────────────
    def _fake_stream(n=50):
        """Stand-in for DW1000SerialReader.stream()."""
        rng2 = np.random.default_rng(99)
        for _ in range(n):
            yield preprocess_cir(rng2.standard_normal(CIR_LEN).astype(np.float32))[0]
            time.sleep(0.02)   # simulate 50 Hz sample rate

    print("\n→ Live dashboard (50 simulated frames)")
    viz.live_dashboard(pipeline, _fake_stream(50), max_frames=50, window=40)