from cnn_bilstm import fine_tune, subsample_loader
from viz import InferencePipeline, VisualisationSuite, RealUWBDataset

from torch.utils.data import random_split, DataLoader
import torch
import os

HOME_PATH = os.path.abspath(os.path.dirname(__file__))

ewine_ds_mapping = {
    "office_1.csv": "uwb_dataset_part1.csv",
    "office_2.csv": "uwb_dataset_part2.csv",
    "small_apartment.csv": "uwb_dataset_part3.csv",
    "small_workshop.csv": "uwb_dataset_part4.csv",
    "kitchen_livingroom.csv": "uwb_dataset_part5.csv",
    "bedroom.csv": "uwb_dataset_part6.csv",
    "boiler_room.csv": "uwb_dataset_part7.csv",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    pipeline = InferencePipeline.from_checkpoint("model_boiler_room.pth")
    viz      = VisualisationSuite()

    # Load your room CSV
    csv_selection = "boiler_room.csv"  # Change this to select different rooms
    csv_path = os.path.join("../", "ewine", "dataset", ewine_ds_mapping[csv_selection])
    ds_boiler = RealUWBDataset(csv_path, source="csv", label_column="NLOS", cir_prefix="CIR")

    n = len(ds_boiler)
    n_train, n_val = int(n*0.7), int(n*0.2)
    boiler_train, boiler_val, boiler_test = random_split(
        ds_boiler, [n_train, n_val, n-n_train-n_val]
    )
    boiler_train_loader = DataLoader(boiler_train, batch_size=64, shuffle=True, pin_memory=(DEVICE.type == "cuda"))
    boiler_val_loader   = DataLoader(boiler_val,   batch_size=64,               pin_memory=(DEVICE.type == "cuda"))
    boiler_test_loader  = DataLoader(boiler_test,  batch_size=64,               pin_memory=(DEVICE.type == "cuda"))

    # Fine-tune using only 10% of boiler room data
    small_loader = subsample_loader(boiler_train_loader, fraction=0.10)
    tl_model, tl_history = fine_tune(
        "model_office_1.pth",
        small_loader,
        boiler_val_loader,
        epochs=20,
        lr=1e-4,
    )

    # Batch inference on test set
    results = pipeline.predict_batch(boiler_test_loader, return_probs=True)

    # All four plots
    viz.confusion_matrix(results["true_labels"], results["predictions"], save_path="confusion_matrix.png")
    viz.confidence_histogram(results["probs"], y_true=results["true_labels"], save_path="confidence_histogram.png")
    viz.performance_summary(tl_history, title="Boiler room TL training", save_path="training_history.png")

    # Single-sample
    import pandas as pd
    row = pd.read_csv(csv_path).iloc[0]
    cir = row[[c for c in row.index if c.startswith("cir_")]].to_numpy(dtype="float32")
    label, conf = pipeline.predict_single(cir)
    print(f"→ {label}  ({conf:.3f} confidence)")

    # Live hardware stream (DW1000 connected via USB)
    # from uwb_nlos_inference_viz import DW1000SerialReader
    # reader = DW1000SerialReader("/dev/ttyUSB0", baud=115200)
    # viz.live_dashboard(pipeline, reader.stream(), max_frames=500)

if __name__ == "__main__":
    main()