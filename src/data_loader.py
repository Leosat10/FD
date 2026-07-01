import os
import kagglehub
import pandas as pd
from pathlib import Path

def download_dataset():
    """Download the Credit Card Fraud Detection dataset from Kaggle."""
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_path = os.path.join(path, "creditcard.csv")
    df = pd.read_csv(csv_path)

    output_path = "data/raw/creditcard.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
    return df

if __name__ == "__main__":
    download_dataset()
