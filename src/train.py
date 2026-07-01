import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.model import train_model, calculate_threshold
from src.preprocess import preprocess_data

def run_training_pipeline():
    try:
        X_train = np.load("data/processed/X_train.npy")
        X_val = np.load("data/processed/X_val.npy")
        print("Loaded preprocessed data.")
    except FileNotFoundError:
        print("Preprocessing data...")
        X_train, X_val, _, _, _, _ = preprocess_data()

    model, history = train_model(X_train, X_val)
    threshold = calculate_threshold(model, X_val)

    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.title('Training History')
    plt.savefig("models/training_plot.png")
    print("Training plot saved to models/training_plot.png")

if __name__ == "__main__":
    run_training_pipeline()
