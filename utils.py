import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_log_error

def rmsle(y_true, y_pred):
    """
    Root Mean Squared Logarithmic Error metric.
    """
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    
    # Ensure no negative values
    y_pred = np.maximum(0, y_pred)
    y_true = np.maximum(0, y_true)
    
    # Calculate RMSLE
    log_diff = np.log1p(y_pred) - np.log1p(y_true)
    rmsle_val = np.sqrt(np.mean(np.square(log_diff)))
    return rmsle_val

def rmsle_loss(y_pred, y_true):
    """
    Root Mean Squared Logarithmic Error as a PyTorch loss function.
    """
    # Ensure no negative values
    y_pred = torch.clamp(y_pred, min=0)
    y_true = torch.clamp(y_true, min=0)
    
    # Calculate RMSLE
    log_diff = torch.log1p(y_pred) - torch.log1p(y_true)
    rmsle_val = torch.sqrt(torch.mean(torch.square(log_diff)))
    return rmsle_val

def split_series(series, n_past, n_future):
    """
    Split a time series into past and future sequences.
    
    Args:
        series: Numpy array of shape (n_samples, n_features)
        n_past: Number of past observations
        n_future: Number of future observations
    
    Returns:
        X: Past sequences, shape (n_windows, n_past, n_features)
        y: Future sequences, shape (n_windows, n_future, n_features)
    """
    X, y = [], []
    for window_start in range(len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        
        if future_end > len(series):
            break
            
        # Slicing the past and future parts of the window
        past, future = series[window_start:past_end, :], series[past_end:future_end, :]
        X.append(past)
        y.append(future)
        
    return np.array(X), np.array(y)

def create_dataloader(X, y, batch_size=32, shuffle=True):
    """
    Create PyTorch DataLoader from numpy arrays.
    """
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
