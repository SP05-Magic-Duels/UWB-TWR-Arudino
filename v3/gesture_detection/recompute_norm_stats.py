import numpy as np
import os

TEMPLATE_DIR  = "templates/"
NORM_STATS_FILE = "templates/norm_stats.npy"

def recompute_norm_stats():
    all_features = []
    template_files = [
        f for f in os.listdir(TEMPLATE_DIR)
        if f.endswith(".npy") and f != "norm_stats.npy"
    ]

    if not template_files:
        print("No templates found in templates/")
        return

    for file in sorted(template_files):
        data = np.load(os.path.join(TEMPLATE_DIR, file))
        all_features.append(data)
        print(f"  Loaded {file:30s} shape={data.shape}")

    combined = np.vstack(all_features)
    mean = combined.mean(axis=0)
    std  = combined.std(axis=0)
    std[std < 1e-6] = 1.0  # avoid division by zero for constant features

    np.save(NORM_STATS_FILE, {'mean': mean, 'std': std})

    print(f"\nRecomputed from {len(template_files)} template(s), {len(combined)} total frames.")
    print(f"Features: speed, curvature, verticality, turn_magnitude")
    print(f"Mean: {mean}")
    print(f"Std:  {std}")
    print(f"\nSaved to {NORM_STATS_FILE}")

if __name__ == '__main__':
    recompute_norm_stats()
