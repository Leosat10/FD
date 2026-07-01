# Credit Card Fraud Detection using LSTM Autoencoder

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.16](https://img.shields.io/badge/TensorFlow-2.16-orange.svg)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project implements an **unsupervised anomaly detection system** for credit card fraud using an **LSTM Autoencoder**. Instead of relying on static rules, the model learns the normal spending rhythm of a user from sequences of 10 transactions and flags unusual patterns (e.g., sudden spikes, erratic behavior) as potential fraud.

The trained model is served via a **FastAPI** endpoint, enabling real‑time predictions with an interactive Swagger UI for testing.

## Dataset

The model is trained on the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), which contains 284,807 transactions with only 492 fraud cases (0.172%).

## Features

- **Unsupervised Learning** – No labeled fraud data required; learns normal patterns autonomously.
- **Sequential Modeling** – Captures temporal dependencies using LSTM layers.
- **Real‑time API** – FastAPI endpoint with `/predict`, `/health`, and `/docs`.
- **Production‑Ready Code** – Clean, modular structure with configuration.

## Tech Stack

- **Deep Learning**: TensorFlow 2.16 (Keras)
- **ML Pipeline**: scikit‑learn, NumPy, Pandas
- **API Framework**: FastAPI, Uvicorn
- **Visualization**: Matplotlib, Seaborn
