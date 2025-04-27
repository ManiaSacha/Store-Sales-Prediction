# Store Sales Prediction - Kaggle Competition

This repository contains a PyTorch implementation for the Kaggle competition "Store Sales - Time Series Forecasting".

## Overview
The goal of this competition is to build a model that accurately predicts the unit sales for thousands of items sold at different Favorita stores in Ecuador.

## Project Structure
- `data_processing.py`: Functions for data loading and preprocessing
- `model.py`: PyTorch LSTM model implementation
- `train.py`: Training script
- `predict.py`: Script to generate predictions
- `utils.py`: Utility functions

## Requirements
Required packages are listed in `requirements.txt`. Install with:
```bash
pip install -r requirements.txt
```

## Usage
1. Download the competition data from Kaggle and place it in a `data` folder
2. Run data preprocessing: `python data_processing.py`
3. Train the model: `python train.py`
4. Generate predictions: `python predict.py`

## Model Architecture
This implementation uses LSTM (Long Short-Term Memory) neural networks for time series forecasting. The model takes into account:
- Historical sales data
- Store and product family information
- Oil price data
- Holiday information
