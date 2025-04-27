import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model import get_model
from utils import split_series
from data_processing import load_data

def prepare_test_data(data, processed_data):
    """
    Prepare test data for prediction.
    
    Args:
        data: Dictionary with raw data
        processed_data: Dictionary with processed training data
    
    Returns:
        Test data ready for prediction
    """
    print("Preparing test data...")
    
    # Get the necessary components from processed data
    scaler = processed_data['scaler']
    pivoted_train = processed_data['pivoted_train']
    n_past = processed_data['n_past']
    
    # Get the last n_past days from training data
    last_days = pivoted_train.iloc[-n_past:].values
    
    # Scale the last days using the same scaler
    scaled_last_days = scaler.transform(last_days)
    
    # Reshape for model input (batch_size=1, seq_len=n_past, n_features)
    X_test = scaled_last_days.reshape(1, n_past, scaled_last_days.shape[1])
    
    return X_test

def generate_predictions(model, X_test, device, n_future, processed_data):
    """
    Generate predictions using the trained model.
    
    Args:
        model: Trained PyTorch model
        X_test: Test data input
        device: Device to run prediction on
        n_future: Number of future time steps to predict
        processed_data: Dictionary with processed data
    
    Returns:
        Predictions for the test period
    """
    print("Generating predictions...")
    
    # Move data to device
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Generate predictions
    with torch.no_grad():
        predictions = model(X_test_tensor)
    
    # Move predictions back to CPU and convert to numpy
    predictions = predictions.cpu().numpy()
    
    # Inverse transform to get original scale
    scaler = processed_data['scaler']
    predictions_rescaled = np.zeros_like(predictions)
    
    for i in range(predictions.shape[1]):  # For each time step
        predictions_rescaled[0, i, :] = scaler.inverse_transform(predictions[0, i, :].reshape(1, -1))
    
    return predictions_rescaled

def create_submission(predictions, test_data, pivoted_train, processed_data, save_path='./submission.csv'):
    """
    Create a submission file for Kaggle.
    
    Args:
        predictions: Model predictions
        test_data: Original test data
        pivoted_train: Pivoted training data
        processed_data: Dictionary with processed data
        save_path: Path to save the submission file
    
    Returns:
        Submission dataframe
    """
    print("Creating submission file...")
    
    # Get the column structure from pivoted_train
    columns = pivoted_train.columns
    
    # Add diagnostic prints
    print(f"Predictions shape: {predictions.shape}")
    print(f"Number of unique store/family combinations in pivoted_train: {len(columns)}")
    
    # Encode families with the same encoder used for training
    # This is important! The test data families must be encoded the same way
    ordinal_encoder = processed_data['ordinal_encoder']
    test_data_encoded = test_data.copy()
    
    if 'family' in test_data_encoded.columns and isinstance(test_data_encoded['family'].iloc[0], str):
        print("Encoding test data families...")
        test_data_encoded[['family']] = ordinal_encoder.transform(test_data_encoded[['family']])
    
    # Count successful lookups for diagnostics
    success_count = 0
    zero_pred_count = 0
    
    # Create a dataframe with predictions
    pred_df = pd.DataFrame(index=test_data['id'], columns=['sales'])
    
    # Extract store_nbr and family from each test row and map to prediction
    for _, row in tqdm(test_data_encoded.iterrows(), total=len(test_data_encoded), desc="Processing test rows"):
        store_nbr = row['store_nbr']
        family = row['family']
        date_idx = (row['date'] - test_data_encoded['date'].min()).days
        
        # Find the corresponding column in predictions
        try:
            col_idx = columns.get_loc((store_nbr, family))
            # For the appropriate future date
            time_idx = min(date_idx, predictions.shape[1] - 1)
            sales_pred = predictions[0, time_idx, col_idx]
            # Ensure non-negative
            sales_pred = max(0, sales_pred)
            success_count += 1
            if sales_pred == 0:
                zero_pred_count += 1
        except:
            # If store_nbr and family combination not found, predict 0
            sales_pred = 0
        
        # Add to submission dataframe
        pred_df.loc[row['id'], 'sales'] = sales_pred
    
    print(f"Successful lookups: {success_count} / {len(test_data_encoded)}")
    print(f"Zero predictions: {zero_pred_count} / {len(test_data_encoded)}")
    
    # Save submission file
    pred_df.to_csv(save_path)
    print(f"Submission saved to {save_path}")
    
    return pred_df

if __name__ == "__main__":
    # Check if model and processed data exist
    if not os.path.exists('./models/lstm_model.pth') or not os.path.exists('./preprocessed/processed_data.pt'):
        print("Model or processed data not found. Please run train.py first.")
        exit()
    
    # Load processed data
    processed_data = torch.load('./preprocessed/processed_data.pt', weights_only=False)
    
    # Load raw data
    data = load_data()
    
    # Get model
    model, device = get_model(
        input_dim=processed_data['n_features'],
        hidden_dim=200,
        num_layers=2,
        dropout=0.2
    )
    
    # Load trained model weights
    model.load_state_dict(torch.load('./models/lstm_model.pth'))
    
    # Prepare test data
    X_test = prepare_test_data(data, processed_data)
    
    # Generate predictions
    predictions = generate_predictions(
        model=model,
        X_test=X_test,
        device=device,
        n_future=processed_data['n_future'],
        processed_data=processed_data
    )
    
    # Create submission file
    submission = create_submission(
        predictions=predictions,
        test_data=data['test'],
        pivoted_train=processed_data['pivoted_train'],
        processed_data=processed_data
    )
    
    print("Prediction completed successfully!")
