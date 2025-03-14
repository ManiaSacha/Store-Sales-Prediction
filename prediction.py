import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys

# Import necessary functions and classes from prc.py
from prc import (load_data, prepare_data_for_lstm, StoreSalesLSTM, 
                prepare_future_data, prepare_time_features)

def load_best_model(model, optimizer, filepath='best_model.pth'):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.eval()  # Set model to evaluation mode
        model.to(device)
        print(f"Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
    except FileNotFoundError:
        print(f"Error: Model file {filepath} not found.")
        return False
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False
    return True

def make_predictions(model, data_loader, device, target_scaler=None):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch_features in data_loader:
            # Handle both training and test data formats
            if isinstance(batch_features, tuple):
                batch_features = batch_features[0]
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            predictions.append(outputs.cpu().numpy())
    
    # Concatenate all predictions
    predictions = np.concatenate(predictions, axis=0)
    
    # Inverse transform if scaler provided
    if target_scaler is not None:
        predictions = target_scaler.inverse_transform(predictions)
    
    # Ensure predictions are non-negative
    predictions = np.maximum(predictions, 0)
    return predictions

def prepare_test_data(train_df, stores_df, oil_df, holidays_df, prediction_dates):
    """Prepare test data for future predictions"""
    # Create a DataFrame for future dates
    future_df = pd.DataFrame()
    
    # Get unique combinations of store_nbr and family from training data
    store_family_combos = train_df[['store_nbr', 'family']].drop_duplicates()
    
    # Create rows for each store-family combination for each future date
    all_combinations = []
    for date in prediction_dates:
        for _, row in store_family_combos.iterrows():
            all_combinations.append({
                'date': date,
                'store_nbr': row['store_nbr'],
                'family': row['family']
            })
    
    future_df = pd.DataFrame(all_combinations)
    
    # Sort by date, store_nbr, and family to ensure consistent order
    future_df = future_df.sort_values(['date', 'store_nbr', 'family']).reset_index(drop=True)
    return future_df

def save_predictions(predictions, test_df, output_file='kaggle_submission.csv'):
    """Save predictions in Kaggle submission format"""
    # Ensure test_df is sorted the same way as when creating predictions
    test_df = test_df.sort_values(['date', 'store_nbr', 'family']).reset_index(drop=True)
    
    # Create IDs in the format the competition expects
    test_df['id'] = test_df.apply(
        lambda row: f"store_{int(row['store_nbr'])}_{row['family']}_{row['date'].strftime('%Y-%m-%d')}", 
        axis=1
    )
    
    # Ensure we have the same number of predictions as test samples
    assert len(predictions) == len(test_df), f"Mismatch between predictions ({len(predictions)}) and test samples ({len(test_df)})"
    
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'sales': predictions.flatten()
    })
    
    # Save predictions
    submission_df.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")
    print("\nSample of submission file:")
    print(submission_df.head())
    
    # Print statistics about predictions
    print("\nPrediction Statistics:")
    print(f"Total predictions: {len(predictions)}")
    print(f"Mean predicted sales: {predictions.mean():.2f}")
    print(f"Median predicted sales: {np.median(predictions):.2f}")
    print(f"Min predicted sales: {predictions.min():.2f}")
    print(f"Max predicted sales: {predictions.max():.2f}")
    print(f"Standard deviation: {predictions.std():.2f}")
    print(f"Percentage of zero predictions: {(predictions == 0).mean() * 100:.2f}%")

def analyze_predictions(predictions_df, train_df):
    print("\nDetailed Prediction Analysis:")
    
    # Extract store and family from id column
    predictions_df['store_nbr'] = predictions_df['id'].str.extract(r'store_(\d+)_').astype(int)
    predictions_df['family'] = predictions_df['id'].str.extract(r'_([A-Z ]+)_\d{4}')
    
    # Analyze by store
    store_stats = train_df.groupby('store_nbr')['sales'].agg(['mean', 'std']).reset_index()
    pred_store_stats = predictions_df.groupby('store_nbr')['sales'].mean().reset_index()
    pred_store_stats.columns = ['store_nbr', 'predicted_mean']
    
    store_comparison = pd.merge(store_stats, pred_store_stats, on='store_nbr')
    store_comparison['mean_diff_pct'] = ((store_comparison['predicted_mean'] - store_comparison['mean']) / store_comparison['mean'] * 100)
    
    print("\nStore-level Analysis:")
    print("Average deviation from historical means: {:.2f}%".format(abs(store_comparison['mean_diff_pct']).mean()))
    print("Stores with highest deviation:")
    print(store_comparison.nlargest(5, 'mean_diff_pct')[['store_nbr', 'mean_diff_pct']])
    
    # Analyze by family
    family_stats = train_df.groupby('family')['sales'].agg(['mean', 'std']).reset_index()
    pred_family_stats = predictions_df.groupby('family')['sales'].mean().reset_index()
    pred_family_stats.columns = ['family', 'predicted_mean']
    
    family_comparison = pd.merge(family_stats, pred_family_stats, on='family')
    family_comparison['mean_diff_pct'] = ((family_comparison['predicted_mean'] - family_comparison['mean']) / family_comparison['mean'] * 100)
    
    print("\nProduct Family Analysis:")
    print("Average deviation from historical means: {:.2f}%".format(abs(family_comparison['mean_diff_pct']).mean()))
    print("Families with highest deviation:")
    print(family_comparison.nlargest(5, 'mean_diff_pct')[['family', 'mean_diff_pct']])
    
    # Calculate RMSLE-like metric on historical patterns
    historical_mean = train_df['sales'].mean()
    predicted_mean = predictions_df['sales'].mean()
    rmsle_approx = np.sqrt(np.mean((np.log1p(predicted_mean) - np.log1p(historical_mean))**2))
    print(f"\nApproximate RMSLE against historical patterns: {rmsle_approx:.4f}")

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the data
    train_df, stores_df, oil_df, holidays_df = load_data()
    
    if train_df is not None:
        # Define future dates for prediction
        last_date = train_df['date'].max()
        prediction_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=16,  # Predict 16 days ahead as per competition
            freq='D'
        )
        print(f"\nPredicting sales for dates:\n{prediction_dates}")
        
        # Prepare test data for future predictions first
        print("\nPreparing test data...")
        test_df = prepare_test_data(train_df, stores_df, oil_df, holidays_df, prediction_dates)
        print(f"Created {len(test_df)} test samples")
        
        # Prepare training data to get the scalers and feature columns
        print("\nPreparing training data...")
        train_loader, _, target_scaler, feature_scaler, feature_columns = prepare_data_for_lstm(
            train_df, stores_df, oil_df, holidays_df, is_training=True
        )
        
        # Initialize model parameters
        input_size = next(iter(train_loader))[0].shape[2]  # Number of features
        hidden_size = 128
        num_layers = 2
        output_size = 1
        
        # Create model and optimizer
        model = StoreSalesLSTM(input_size, hidden_size, num_layers, output_size)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Load the best model
        if not load_best_model(model, optimizer):
            print("Failed to load model. Exiting...")
            return
        
        # Prepare features for future predictions
        print("Processing test data features...")
        test_loader = prepare_future_data(
            test_df, stores_df, oil_df, holidays_df, 
            feature_scaler, feature_columns
        )
        
        # Make predictions for future dates
        print("\nMaking predictions for future dates...")
        predictions = make_predictions(model, test_loader, device, target_scaler)
        
        # Compare with training data statistics
        print("\nComparison with training data:")
        print(f"Training data mean sales: {train_df['sales'].mean():.2f}")
        print(f"Training data median sales: {train_df['sales'].median():.2f}")
        print(f"Training data min sales: {train_df['sales'].min():.2f}")
        print(f"Training data max sales: {train_df['sales'].max():.2f}")
        print(f"Training data standard deviation: {train_df['sales'].std():.2f}")
        
        # Save predictions in Kaggle submission format
        save_predictions(predictions, test_df)
        
        # Analyze predictions
        predictions_df = pd.read_csv('kaggle_submission.csv')
        analyze_predictions(predictions_df, train_df)

if __name__ == "__main__":
    main()
