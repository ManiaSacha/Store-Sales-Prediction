# Store Sales Prediction

This project implements a time series forecasting model to predict store sales for Corporación Favorita, a large Ecuadorian-based grocery retailer. The model uses LSTM neural networks to predict sales across different stores and product families.

## Project Structure

- `prediction.py`: Main script for generating sales predictions
- `prc.py`: Data processing and model definition module
- `requirements.txt`: Python package dependencies
- `best_model.pth`: Trained LSTM model weights

## Features

- Time series forecasting using LSTM neural networks
- GPU acceleration support via PyTorch
- Handles multiple stores and product families
- Incorporates oil price data and holiday information
- Detailed prediction analysis and statistics

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run predictions:
```bash
python prediction.py
```

The script will:
1. Load and preprocess the data
2. Load the trained model
3. Generate predictions for future dates
4. Save predictions in Kaggle submission format
5. Provide detailed analysis of prediction accuracy

## Model Performance

- Training Loss: 0.2034
- RMSLE: 0.1998
- Mean Sales: 437.12
- Standard Deviation: 1129.13

## Requirements

- Python 3.10+
- PyTorch with CUDA support
- NVIDIA GPU (optional, for faster training)
