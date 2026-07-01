import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks

def build_lstm_autoencoder(seq_len=10, n_features=29):
    """
    Build an LSTM Autoencoder for anomaly detection.

    Arguments:
        seq_len: Number of time steps (transactions) in each sequence.
        n_features: Number of input features per time step.

    Returns:
        A compiled Keras Model.
    """
    inputs = layers.Input(shape=(seq_len, n_features))

    # Encoder
    x = layers.LSTM(64, return_sequences=True, dropout=0.2)(inputs)
    x = layers.LSTM(32, return_sequences=False, dropout=0.2)(x)
    latent = layers.Dense(16, activation='relu', name="bottleneck")(x)

    # Decoder
    x = layers.RepeatVector(seq_len)(latent)
    x = layers.LSTM(32, return_sequences=True, dropout=0.2)(x)
    x = layers.LSTM(64, return_sequences=True, dropout=0.2)(x)
    outputs = layers.TimeDistributed(layers.Dense(n_features))(x)

    autoencoder = Model(inputs, outputs)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def train_model(X_train, X_val, epochs=30, batch_size=256):
    """
    Train the autoencoder on the training set.

    Returns:
        model: Trained model.
        history: Training history object.
    """
    model = build_lstm_autoencoder(X_train.shape[1], X_train.shape[2])

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    print("Training LSTM Autoencoder...")
    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    model.save("models/autoencoder.keras")
    print("Model saved to models/autoencoder.keras")
    return model, history

def calculate_threshold(model, X_val, percentile=99.9):
    """
    Compute the reconstruction error threshold at the given percentile.
    """
    reconstructions = model.predict(X_val, verbose=0)
    errors = np.mean(np.square(X_val - reconstructions), axis=(1, 2))
    threshold = np.percentile(errors, percentile)
    with open("models/threshold.txt", "w") as f:
        f.write(str(threshold))
    print(f"Threshold (percentile={percentile}): {threshold:.6f}")
    return threshold

if __name__ == "__main__":
    X_train = np.load("data/processed/X_train.npy")
    X_val = np.load("data/processed/X_val.npy")
    model, hist = train_model(X_train, X_val)
    calculate_threshold(model, X_val)
