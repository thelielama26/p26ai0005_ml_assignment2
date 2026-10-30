import os
import joblib
import numpy as np
from sklearn.datasets import fetch_olivetti_faces

def prepare_and_save_data(config, data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    split_path = os.path.join(data_dir, "dataset_split.pkl")
    
    if os.path.exists(split_path):
        return joblib.load(split_path)
    
    faces = fetch_olivetti_faces(shuffle=False)
    images = faces.images  # (400, 64, 64)
    targets = faces.target # (400,)
    
    rng = np.random.default_rng(config["seed"])
    
    train_idx, val_idx, test_idx = [], [], []
    
    # Per identity split (10 images each: 6 train, 2 val, 2 test)
    for identity in range(40):
        id_indices = np.where(targets == identity)[0]
        shuffled = rng.permutation(id_indices)
        train_idx.extend(shuffled[:6])
        val_idx.extend(shuffled[6:8])
        test_idx.extend(shuffled[8:10])
    
    X_train_raw = images[train_idx]
    X_val_raw = images[val_idx]
    X_test_raw = images[test_idx]
    
    # Flatten raw pixels only
    X_train = X_train_raw.reshape(len(train_idx), -1)
    X_val = X_val_raw.reshape(len(val_idx), -1)
    X_test_clean = X_test_raw.reshape(len(test_idx), -1)
    
    y_train = targets[train_idx]
    y_val = targets[val_idx]
    y_test = targets[test_idx]
    
    # Masked Test Set (Rows and columns 24 to 39 set to zero)
    X_test_masked_raw = X_test_raw.copy()
    r_start, r_end = config["dataset"]["mask_box"]
    X_test_masked_raw[:, r_start:r_end, r_start:r_end] = 0.0
    X_test_masked = X_test_masked_raw.reshape(len(test_idx), -1)
    
    dataset = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test_clean": X_test_clean, "X_test_masked": X_test_masked, "y_test": y_test,
        "train_idx": np.array(train_idx), "val_idx": np.array(val_idx), "test_idx": np.array(test_idx)
    }
    
    joblib.dump(dataset, split_path)
    return dataset