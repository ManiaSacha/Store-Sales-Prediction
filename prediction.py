import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import timedelta
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from pathlib import Path

from prc import (
    load_data, prepare_data_for_lstm, prepare_test_data,
    StoreSalesLSTM, StoreSalesDataset, log_transform, inverse_log_transform
)

def plot_training_history(train_losses, val_losses, save_path='training_history.png'):
    """Plot and save training history"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def load_best_model(model, optimizer=None, filepath='best_model.pth'):
    """
    Load a saved model with careful error handling and verification
    """
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(filepath, map_location=device)
        
        # Verify model architecture compatibility
        expected_keys = set(model.state_dict().keys())
        loaded_keys = set(checkpoint['model_state_dict'].keys())
        
        # Check for mismatches
        if expected_keys != loaded_keys:
            missing_keys = expected_keys - loaded_keys
            extra_keys = loaded_keys - expected_keys
            
            if missing_keys:
                print(f"Warning: Missing keys in loaded model: {missing_keys}")
            if extra_keys:
                print(f"Warning: Extra keys in loaded model: {extra_keys}")
                
            if len(missing_keys) > len(expected_keys) / 2:
                print(f"Error: Model architecture mismatch. Too many missing keys.")
                return False
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Set model to evaluation mode
        model.eval()
        model.to(device)
        
        # Return additional saved data
        additional_data = {k: v for k, v in checkpoint.items() 
                          if k not in ['model_state_dict', 'optimizer_state_dict']}
        
        print(f"Successfully loaded model from {filepath}")
        if 'epoch' in checkpoint and 'loss' in checkpoint:
            print(f"Model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
            
        return True, additional_data
        
    except FileNotFoundError:
        print(f"Error: Model file {filepath} not found.")
        return False, {}
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False, {}

def make_predictions(model, data_loader, device, inv_log_transformer=None):
    """
    Make predictions using the model
    
    Args:
        model: The trained PyTorch model
        data_loader: DataLoader containing the input features
        device: Device to run inference on
        inv_log_transformer: Function to inverse transform the log-scaled predictions
        
    Returns:
        Array of predictions
    """
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
    
    # Inverse transform if available
    if inv_log_transformer is not None:
        predictions = inv_log_transformer(predictions)
    
    # Ensure predictions are non-negative
    predictions = np.maximum(predictions, 0)
    return predictions

def prepare_prediction_data(train_df, stores_df, oil_df, holidays_df, prediction_dates):
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

def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=10, patience=3, device=None):
    """
    Train the model with early stopping and learning rate adjustment
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        model: Model to train
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Maximum number of epochs to train
        patience: Number of epochs to wait for improvement before early stopping
        device: Device to train on
    
    Returns:
        trained model, training history
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    print(f"Training on {device}")
    
    # Initialize learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience//2)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}')
        
        # Check if this is the best model so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            # Save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'best_model.pth')
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Plot and save training history
    plot_training_history(train_losses, val_losses)
    
    return model, {'train_losses': train_losses, 'val_losses': val_losses}

def evaluate_predictions(true_values, predictions, store_ids=None, family_ids=None):
    """
    Evaluate predictions with multiple metrics
    
    Args:
        true_values: Ground truth values
        predictions: Predicted values
        store_ids: Optional store IDs for per-store analysis
        family_ids: Optional family IDs for per-family analysis
    
    Returns:
        Dict of evaluation metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Calculate metrics
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)
    
    # Calculate RMSLE (Root Mean Squared Logarithmic Error)
    # Add small constant to avoid log(0)
    epsilon = 1e-8
    rmsle = np.sqrt(mean_squared_error(
        np.log1p(true_values + epsilon), 
        np.log1p(predictions + epsilon)
    ))
    
    # Mean absolute percentage error with protection against division by zero
    mape = np.mean(np.abs((true_values - predictions) / np.maximum(true_values, epsilon))) * 100
    
    print("\nEvaluation Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"RMSLE: {rmsle:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Additional store or family analysis if IDs provided
    if store_ids is not None and len(store_ids) == len(true_values):
        store_metrics = {}
        for store in np.unique(store_ids):
            mask = store_ids == store
            if np.sum(mask) > 0:
                store_rmse = np.sqrt(mean_squared_error(true_values[mask], predictions[mask]))
                store_metrics[store] = store_rmse
        
        worst_stores = sorted(store_metrics.items(), key=lambda x: x[1], reverse=True)[:5]
        best_stores = sorted(store_metrics.items(), key=lambda x: x[1])[:5]
        
        print("\nStores with highest error:")
        for store, rmse in worst_stores:
            print(f"Store {store}: RMSE = {rmse:.4f}")
        
        print("\nStores with lowest error:")
        for store, rmse in best_stores:
            print(f"Store {store}: RMSE = {rmse:.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'rmsle': rmsle,
        'mape': mape
    }

def main():
    """Main execution function for making predictions"""
    print("Store Sales Prediction System")
    print("-" * 50)
    
    # Check if a saved model exists
    model_path = 'store_sales_model.pth'
    model_exists = Path(model_path).exists()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data
    print("\nLoading data...")
    train_df, stores_df, oil_df, holidays_df = load_data()
    
    if train_df is None:
        print("Failed to load required data. Exiting.")
        return
    
    # Check for existing model
    if model_exists:
        print(f"\nFound existing model at {model_path}")
        
        # Initialize model with placeholder parameters
        # The actual params will be loaded from the saved model
        input_size = 25  # Updated to match the trained model's input size
        hidden_size = 256
        num_layers = 3
        
        model = StoreSalesLSTM(input_size, hidden_size, num_layers)
        optimizer = torch.optim.Adam(model.parameters()) # Placeholder optimizer
        
        # Load model and get saved parameters
        success, saved_data = load_best_model(model, optimizer, model_path)
        
        if not success:
            print("Failed to load model. Training a new one...")
            model_exists = False
        else:
            # Extract needed data from the saved model
            scaler = saved_data.get('scaler')
            feature_columns = saved_data.get('feature_columns')
            store_type_map = saved_data.get('store_type_map')
            family_map = saved_data.get('family_map')
            # Use module-level function even if not found in saved data
            inv_log_transformer = inverse_log_transform
            
            if None in [scaler, feature_columns, store_type_map, family_map]:
                print("Missing required data in saved model. Training a new one...")
                model_exists = False
    
    # If no existing model or failed to load, prepare data and train a new model
    if not model_exists:
        print("\nPreparing data for training...")
        X, y, scaler, feature_columns, store_type_map, family_map, stores_df, _ = prepare_data_for_lstm(
            train_df, stores_df, oil_df, holidays_df
        )
        
        # Use the module-level function for inverse transformation
        inv_log_transformer = inverse_log_transform
        
        # Split data into training and validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        # Create datasets and dataloaders
        train_dataset = StoreSalesDataset(X_train, y_train)
        val_dataset = StoreSalesDataset(X_val, y_val)
        
        batch_size = 64 if torch.cuda.is_available() else 32
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            num_workers=4 if torch.cuda.is_available() else 0
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=4 if torch.cuda.is_available() else 0
        )
        
        # Define model, loss function, and optimizer
        input_size = X.shape[2]  # Number of features
        hidden_size = 256
        num_layers = 3
        
        model = StoreSalesLSTM(input_size, hidden_size, num_layers)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train the model
        print("\nTraining model...")
        model, history = train_model(
            train_loader,
            val_loader,
            model,
            criterion,
            optimizer,
            num_epochs=15,
            patience=5,
            device=device
        )
        
        # Save the model with all necessary data
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler': scaler,
            'feature_columns': feature_columns, 
            'store_type_map': store_type_map,
            'family_map': family_map,
            # Don't save the function directly, we'll use the module-level function
        }, model_path)
        print(f"\nModel saved to {model_path}")
    
    # Prepare test data for prediction
    print("\nPreparing test data...")
    try:
        # Try to load test data from CSV
        test_df = pd.read_csv('test.csv', parse_dates=['date'])
        print("Test data loaded from test.csv")
    except FileNotFoundError:
        # If test.csv doesn't exist, create prediction dates for the next 15 days
        last_date = train_df['date'].max()
        prediction_dates = [last_date + timedelta(days=i+1) for i in range(15)]
        
        # Create test dataframe
        test_df = prepare_prediction_data(train_df, stores_df, oil_df, holidays_df, prediction_dates)
        print(f"Generated prediction data for {len(prediction_dates)} days after {last_date.strftime('%Y-%m-%d')}")
    
    # Prepare test data using the same transformations as training
    X_test, store_family_dates_df = prepare_test_data(
        test_df, stores_df, oil_df, holidays_df,
        scaler, feature_columns, store_type_map, family_map
    )
    
    # Create test dataloader
    test_dataset = torch.FloatTensor(X_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=4 if torch.cuda.is_available() else 0
    )
    
    # Make predictions
    print("\nMaking predictions...")
    print(f"Test data shape: {X_test.shape}")
    print(f"Feature columns: {feature_columns}")
    
    # Debug: Check for all zeros in test data
    if isinstance(X_test, np.ndarray):
        print(f"Test data statistics:")
        print(f"  Mean: {X_test.mean():.6f}")
        print(f"  Min: {X_test.min():.6f}")
        print(f"  Max: {X_test.max():.6f}")
        print(f"  Std: {X_test.std():.6f}")
        print(f"  Zero values: {np.sum(X_test == 0) / X_test.size * 100:.2f}%")
    
    predictions = make_predictions(model, test_loader, device, inv_log_transformer)
    
    # Debug: Check predictions before any processing
    print(f"Raw prediction statistics:")
    print(f"  Mean: {predictions.mean():.6f}")
    print(f"  Min: {predictions.min():.6f}")
    print(f"  Max: {predictions.max():.6f}")
    print(f"  Std: {predictions.std():.6f}")
    print(f"  Zero values: {np.sum(predictions == 0) / predictions.size * 100:.2f}%")
    
    # Align predictions with store-family combinations
    aligned_predictions = np.zeros(len(test_df))
    if len(store_family_dates_df) > 0:
        # Debug information
        print(f"\nPrediction alignment debug:")
        print(f"  store_family_dates_df shape: {store_family_dates_df.shape}")
        print(f"  test_df shape: {test_df.shape}")
        print(f"  First few test dates: {test_df['date'].head().tolist()}")
        print(f"  First few prediction dates: {store_family_dates_df['date'].head().tolist()}")
        
        # If we have detailed store-family dates, use them for alignment
        match_count = 0
        for _, row in store_family_dates_df.iterrows():
            store_nbr = row['store_nbr']
            family = row['family']
            pred_date = row['date']
            pred_index = row['index']
            
            # Find matching rows in test_df - more flexible date matching
            # Try exact match first
            mask = (
                (test_df['store_nbr'] == store_nbr) & 
                (test_df['family'] == family) &
                (test_df['date'] == pred_date)
            )
            
            # If no match, try with the same date
            if not mask.any():
                # Just match store and family, use any date
                mask = (
                    (test_df['store_nbr'] == store_nbr) & 
                    (test_df['family'] == family)
                )
                if mask.any():
                    # Take the first matching record
                    mask_indices = mask.to_numpy().nonzero()[0]
                    if len(mask_indices) > 0:
                        # Set all to false, then just the first one to true
                        mask = pd.Series(False, index=mask.index)
                        mask.iloc[mask_indices[0]] = True
            
            if mask.any():
                match_count += 1
                aligned_predictions[mask] = predictions[pred_index]
                
        print(f"  Found {match_count} matches out of {len(store_family_dates_df)} prediction sequences")
                
    else:
        # If we don't have detailed mapping, assume order is preserved
        # This is a fallback, but it might not align correctly
        print("Warning: No store-family-date mapping available. Using direct 1:1 mapping.")
        for i in range(min(len(predictions), len(aligned_predictions))):
            aligned_predictions[i] = predictions[i]
    
    # Save predictions to CSV
    output_file = 'store_sales_predictions.csv'
    save_predictions(aligned_predictions, test_df, output_file)
    
    # Analyze predictions
    analyze_predictions(pd.read_csv(output_file), train_df)
    
    print("\nPrediction process completed!")

if __name__ == "__main__":
    main()
