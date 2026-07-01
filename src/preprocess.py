import sys
import os
# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

from src.data_loader import download_dataset   # now works

def create_sequences(df, features, seq_len=10):
    sequences = []
    labels = []
    data = df[features].values
    class_labels = df['Class'].values
    for i in range(seq_len, len(data)):
        sequences.append(data[i-seq_len:i])
        labels.append(1 if np.any(class_labels[i-seq_len:i]) else 0)
    return np.array(sequences), np.array(labels)

def preprocess_data():
    raw_file = "data/raw/creditcard.csv"
    if not os.path.exists(raw_file):
        print("Raw data not found. Downloading...")
        download_dataset()

    print("Loading raw data...")
    df = pd.read_csv(raw_file)

    feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    feature_cols_scaled = [f'V{i}' for i in range(1, 29)] + ['Amount_scaled']

    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    X, y = create_sequences(df, feature_cols_scaled, 10)
    print(f"Total sequences created: {X.shape[0]}")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.111, random_state=42, stratify=y_temp
    )

    for name, arr in [
        ("X_train", X_train), ("X_val", X_val), ("X_test", X_test),
        ("y_train", y_train), ("y_val", y_val), ("y_test", y_test)
    ]:
        np.save(f"data/processed/{name}.npy", arr)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    preprocess_data()
