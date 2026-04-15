"""
UWB (N)LOS Identification via CNN-BiLSTM + Transfer Learning
Based on: Li et al., "UWB (N)LOS Identification Based on Deep Learning and
Transfer Learning", IEEE Communications Letters, Vol. 28, No. 9, Sep 2024.

Architecture Summary:
  - CNN block:    2x [Conv1d → ReLU → MaxPool1d] for spatial feature extraction
  - BiLSTM block: Captures forward + backward temporal dependencies in CIR
  - Classifier:   Dropout → Linear → Softmax (binary: LOS vs NLOS)
  - Transfer Learning: Freeze CNN + BiLSTM, fine-tune only the FC classifier
    on a small target-env dataset (10–30% of full target data)

Dataset: eWINE (open-source)
  - 7 indoor scenes, 42,000 samples total (21k LOS / 21k NLOS)
  - Each CIR sample: 1016 time-domain values @ 1 ns resolution
  - Splits: 70% train / 20% val / 10% test per scene
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class CNN_BiLSTM(nn.Module):
    """
    CNN-BiLSTM for UWB LOS/NLOS classification.

    Input  : (batch, 1, 1016)  — raw 1-D CIR signal
    Output : (batch, 2)        — class logits [LOS, NLOS]

    CNN block
    ---------
    Two Conv1d → ReLU → MaxPool1d stages reduce the 1016-length sequence and
    build a 64-channel feature map, matching Table I of the paper.

    BiLSTM block
    ------------
    The CNN output is treated as a sequence of 64-dim vectors.  A single
    bidirectional LSTM layer (hidden=128 per direction) processes it in both
    temporal directions, yielding richer context than a unidirectional LSTM
    (as noted in Section II-B-2 of the paper).

    Classifier block
    ----------------
    The last BiLSTM hidden state (256-dim = 128 × 2 directions) passes through
    Dropout(0.5) and a fully-connected layer.  This is the layer targeted for
    fine-tuning during Transfer Learning (Section II-B-3).
    """

    def __init__(self, input_size: int = 1016, num_classes: int = 2):
        super().__init__()

        # ── CNN: spatial feature extraction ──────────────────────────────────
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=1,  out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Block 2
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # ── BiLSTM: temporal analysis ─────────────────────────────────────────
        # After CNN the sequence has shape (batch, 64, 254); we permute to
        # (batch, 254, 64) so that LSTM sees 254 time-steps of 64 features.
        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,   # forward + backward → 256-dim output
            batch_first=True,
            dropout=0.0,          # only meaningful for num_layers > 1
        )

        # ── Classifier: transfer-learning target layer ────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 2, num_classes),  # 256 → 2
        )

    # ── Named sub-module groups for convenient freeze / unfreeze ─────────────
    @property
    def feature_extractor_params(self):
        """CNN + BiLSTM parameters (frozen during TL fine-tuning)."""
        return list(self.cnn.parameters()) + list(self.bilstm.parameters())

    @property
    def classifier_params(self):
        """FC classifier parameters (trained during TL fine-tuning)."""
        return list(self.classifier.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("Forward input shape:", x.shape)
        
        # x : (batch, 1, 1016)
        x = self.cnn(x)                     # → (batch, 64, 254)
        x = x.permute(0, 2, 1)              # → (batch, 254, 64)
        x, _ = self.bilstm(x)               # → (batch, 254, 256)
        x = x[:, -1, :]                     # last time-step → (batch, 256)
        x = self.classifier(x)              # → (batch, 2)
        return x                            # raw logits (use CrossEntropyLoss)


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Run one training epoch; return mean loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        X, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
        # X, y = X.to(DEVICE), y.to(DEVICE) --- IGNORE ---
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> dict:
    """Evaluate model; return loss + sklearn metrics dict."""
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    for X, y in loader:
        # print("Batch X shape:", X.shape)
        # print("Batch y shape:", y.shape)
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = model(X)
        total_loss += criterion(logits, y).item() * len(y)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    return {
        "loss":      total_loss / len(loader.dataset),
        "accuracy":  accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall":    recall_score(all_labels, all_preds, zero_division=0),
        "f1":        f1_score(all_labels, all_preds, zero_division=0),
        "confusion": confusion_matrix(all_labels, all_preds),
    }


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    verbose: bool = True,
) -> list[dict]:
    """
    Full training loop with early stopping.

    Returns list of per-epoch metric dicts.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    history, best_val_loss, patience_counter = [], float("inf"), 0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_metrics = evaluate(model, val_loader, criterion)
        scheduler.step(val_metrics["loss"])

        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(row)

        if verbose:
            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_metrics['loss']:.4f}  "
                f"val_acc={val_metrics['accuracy']:.4f}  "
                f"val_f1={val_metrics['f1']:.4f}"
            )

        # Early stopping on validation loss
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}.")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Transfer Learning
# ─────────────────────────────────────────────────────────────────────────────

def prepare_for_transfer(model: nn.Module, reinit_classifier: bool = True) -> nn.Module:
    """
    Implement the paper's fine-tuning strategy (Section II-B-3):

      1. Freeze all CNN and BiLSTM parameters.
      2. Re-initialise (or keep) the classifier's weight gradient so that
         only the last hidden layer is updated on the target dataset.

    Args:
        model:              A pre-trained CNN_BiLSTM instance.
        reinit_classifier:  If True, reset the FC layer's weights before
                            fine-tuning (matches "initializing the weight
                            gradient of the last hidden layer").
    Returns:
        The same model with only classifier parameters trainable.
    """
    # Step 1 – freeze feature extractor
    for param in model.cnn.parameters():
        param.requires_grad = False
    for param in model.bilstm.parameters():
        param.requires_grad = False

    # Step 2 – (optionally) re-initialise classifier weights
    if reinit_classifier:
        for layer in model.classifier.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    # Ensure classifier is trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[TL] Trainable params: {trainable:,} / {total:,}  "
          f"({100 * trainable / total:.1f}%)")
    return model


def fine_tune(
    pretrained_path: str,
    target_train_loader: DataLoader,
    target_val_loader:   DataLoader,
    *,
    epochs: int = 20,
    lr: float = 1e-4,     # lower LR for fine-tuning (paper recommendation)
    verbose: bool = True,
) -> tuple[nn.Module, list[dict]]:
    """
    Load a pre-trained source model and fine-tune it on a small target dataset.

    Mirrors the paper's TL experiment: the classifier layer is reset and
    retrained on 10 / 20 / 30 % of the target-environment data.

    Args:
        pretrained_path:        Path to source model state dict (.pth).
        target_train_loader:    DataLoader for the (small) target training set.
        target_val_loader:      DataLoader for the target validation set.
        epochs:                 Fine-tuning epochs.
        lr:                     Fine-tuning learning rate.

    Returns:
        (fine_tuned_model, training_history)
    """
    tl_model = CNN_BiLSTM().to(DEVICE)
    tl_model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE))
    prepare_for_transfer(tl_model, reinit_classifier=True)
    history = fit(
        tl_model,
        target_train_loader,
        target_val_loader,
        epochs=epochs,
        lr=lr,
        patience=7,
        verbose=verbose,
    )
    return tl_model, history


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers (eWINE-style)
# ─────────────────────────────────────────────────────────────────────────────

def make_loaders(
    cir_array: np.ndarray,
    labels:    np.ndarray,
    *,
    train_frac: float = 0.70,
    val_frac:   float = 0.20,
    batch_size: int   = 64,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split (cir_array, labels) into train/val/test DataLoaders.

    CIR samples are reshaped to (N, 1, 1016) to match Conv1d expectations.

    Args:
        cir_array:   Float array of shape (N, 1016).
        labels:      Long array of shape (N,) with values in {0, 1}.
        train_frac:  Fraction for training (paper uses 0.70).
        val_frac:    Fraction for validation (paper uses 0.20).
        batch_size:  Mini-batch size (paper uses 64).

    Returns:
        (train_loader, val_loader, test_loader)
    """
    assert train_frac + val_frac < 1.0, "Train + val must be < 1.0"

    X = torch.tensor(cir_array, dtype=torch.float32).unsqueeze(1)  # (N,1,1016)
    y = torch.tensor(labels,    dtype=torch.long)

    dataset = TensorDataset(X, y)
    n       = len(dataset)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    n_test  = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    kwargs = dict(batch_size=batch_size, pin_memory=True, num_workers=0)
    return (
        DataLoader(train_ds, shuffle=True,  **kwargs),
        DataLoader(val_ds,   shuffle=False, **kwargs),
        DataLoader(test_ds,  shuffle=False, **kwargs),
    )


def subsample_loader(
    full_loader: DataLoader,
    fraction:    float,
    batch_size:  int = 64,
    seed:        int = 42,
) -> DataLoader:
    """
    Return a DataLoader using only `fraction` of `full_loader`'s dataset.
    Used to simulate the 10 / 20 / 30 % target-data TL experiment.
    """
    dataset = full_loader.dataset
    n_keep  = max(1, int(len(dataset) * fraction))
    n_drop  = len(dataset) - n_keep
    generator = torch.Generator().manual_seed(seed)
    subset, _ = random_split(dataset, [n_keep, n_drop], generator=generator)
    return DataLoader(subset, batch_size=batch_size, shuffle=True,
                      pin_memory=True, num_workers=0)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end demo  (replace synthetic data with real eWINE CIR arrays)
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_scene(n_samples: int = 6000, seed: int = 0) -> tuple:
    """Generate Gaussian noise CIR data as a stand-in for eWINE samples."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, 1016)).astype(np.float32)
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2), dtype=np.int64)
    rng.shuffle(y)
    return X, y


if __name__ == "__main__":

    print("=" * 60)
    print("Phase 1 — Source domain training (office1)")
    print("=" * 60)

    # ── Load / replace with: np.load("office1_cir.npy"), np.load("office1_labels.npy")
    X_src, y_src = _synthetic_scene(n_samples=6000, seed=1)
    src_train, src_val, src_test = make_loaders(X_src, y_src)

    source_model = CNN_BiLSTM().to(DEVICE)
    fit(source_model, src_train, src_val, epochs=50, lr=1e-3, patience=10)

    # Evaluate on held-out test set
    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate(source_model, src_test, criterion)
    print("\n[Source] Test metrics:")
    for k, v in test_metrics.items():
        if k != "confusion":
            print(f"  {k:10s}: {v:.4f}")
    print("  confusion:\n", test_metrics["confusion"])

    # Save source model
    SOURCE_PATH = "source_model_office1.pth"
    torch.save(source_model.state_dict(), SOURCE_PATH)
    print(f"\nSource model saved → {SOURCE_PATH}")

    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 2 — Direct transfer (no retraining, boiler room)")
    print("=" * 60)

    X_tgt, y_tgt = _synthetic_scene(n_samples=6000, seed=42)
    tgt_train, tgt_val, tgt_test = make_loaders(X_tgt, y_tgt)

    direct_model = CNN_BiLSTM().to(DEVICE)
    direct_model.load_state_dict(torch.load(SOURCE_PATH, map_location=DEVICE))
    direct_metrics = evaluate(direct_model, tgt_test, criterion)
    print("[Direct transfer] Test metrics (expect degraded performance):")
    for k, v in direct_metrics.items():
        if k != "confusion":
            print(f"  {k:10s}: {v:.4f}")

    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 3 — Transfer Learning fine-tuning (10% target data)")
    print("=" * 60)

    for frac in (0.10, 0.20, 0.30):
        small_train = subsample_loader(tgt_train, fraction=frac)
        print(f"\n── TL with {int(frac * 100)}% target training data "
              f"({len(small_train.dataset)} samples) ──")

        tl_model, _ = fine_tune(
            SOURCE_PATH,
            small_train,
            tgt_val,
            epochs=20,
            lr=1e-4,
            verbose=False,
        )
        tl_metrics = evaluate(tl_model, tgt_test, criterion)
        print(f"  accuracy : {tl_metrics['accuracy']:.4f}")
        print(f"  f1       : {tl_metrics['f1']:.4f}")
        print(f"  precision: {tl_metrics['precision']:.4f}")
        print(f"  recall   : {tl_metrics['recall']:.4f}")

    print("\nDone.")