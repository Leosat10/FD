import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.models import load_model

def evaluate_model():
    model = load_model("models/autoencoder.keras", compile=False)
    threshold = float(open("models/threshold.txt").read())

    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    reconstructions = model.predict(X_test, verbose=0)
    errors = np.mean(np.square(X_test - reconstructions), axis=(1, 2))
    predictions = (errors > threshold).astype(int)

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(classification_report(y_test, predictions, target_names=['Normal', 'Fraud']))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, errors):.4f}")

    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.savefig("models/confusion_matrix.png")
    print("Confusion matrix saved to models/confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model()
