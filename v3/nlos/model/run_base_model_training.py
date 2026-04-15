from viz import RealUWBDataset
from cnn_bilstm import CNN_BiLSTM, fit, evaluate

import torch, torch.nn as nn
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
    # Load your room CSV
    csv_selection = "office_1.csv"  # Change this to select different rooms
    csv_path = os.path.join("../", "ewine", "dataset", ewine_ds_mapping[csv_selection])

    ds_office = RealUWBDataset(csv_path, source="csv", label_column="NLOS", cir_prefix="CIR")

    # Split into train/val/test (70/20/10 per paper)
    n = len(ds_office)
    n_train, n_val = int(n*0.7), int(n*0.2)
    train_ds, val_ds, test_ds = random_split(ds_office, [n_train, n_val, n-n_train-n_val])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, pin_memory=(DEVICE.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=64,               pin_memory=(DEVICE.type == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=64,               pin_memory=(DEVICE.type == "cuda"))

    # Train
    model = CNN_BiLSTM().to(DEVICE)
    history = fit(model, train_loader, val_loader, epochs=50, lr=1e-3)
    
    # Evaluate and save
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, test_loader, criterion)
    print(f"Test F1: {metrics['f1']:.4f}  Acc: {metrics['accuracy']:.4f}")
    torch.save(model.state_dict(), "model_office_1.pth")

if __name__ == "__main__":
    main()
