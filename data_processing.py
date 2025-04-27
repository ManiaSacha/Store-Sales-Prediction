import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
import torch
from utils import split_series, create_dataloader

def load_data(data_path='./data'):
    """
    Load all the data files for the competition.
    
    Returns:
        Dictionary containing all the dataframes
    """
    print("Loading data files...")
    
    data = {}
    data['oil'] = pd.read_csv(os.path.join(data_path, 'oil.csv'), index_col='date')
    data['holidays_events'] = pd.read_csv(os.path.join(data_path, 'holidays_events.csv'), index_col='date')
    data['stores'] = pd.read_csv(os.path.join(data_path, 'stores.csv'))
    data['transactions'] = pd.read_csv(os.path.join(data_path, 'transactions.csv'))
    
    data['train'] = pd.read_csv(os.path.join(data_path, 'train.csv'), 
                               index_col='id', 
                               parse_dates=['date'], 
                               infer_datetime_format=True)
    
    data['test'] = pd.read_csv(os.path.join(data_path, 'test.csv'),
                              parse_dates=['date'], 
                              infer_datetime_format=True)
    
    print("Data loaded successfully")
    
    # Print basic stats
    print(f"Number of days in train: {data['train']['date'].nunique()}")
    print(f"Number of stores: {data['train']['store_nbr'].nunique()}")
    print(f"Number of product families: {data['train']['family'].nunique()}")
    
    return data

def preprocess_data(data):
    """
    Preprocess the data for model training.
    
    Args:
        data: Dictionary containing dataframes
        
    Returns:
        Processed data ready for model training
    """
    print("Preprocessing data...")
    
    # Drop onpromotion column (can be added later as a feature if needed)
    train_data = data['train'].copy().drop(['onpromotion'], axis=1)
    test_data = data['test'].copy().drop(['onpromotion'], axis=1)
    
    # Encode categorical features
    ordinal_encoder = OrdinalEncoder(dtype=int)
    train_data[['family']] = ordinal_encoder.fit_transform(train_data[['family']])
    test_data[['family']] = ordinal_encoder.transform(test_data[['family']])
    
    # Pivot the data to have a wide format
    # Each column represents a combination of store and product family
    print("Pivoting data...")
    pivoted_train = train_data.pivot(index=['date'], 
                                     columns=['store_nbr', 'family'], 
                                     values='sales')
    
    # Train-validation split
    n_days_train = train_data['date'].nunique()
    train_samples = int(n_days_train * 0.95)  # 95% for training
    
    train_samples_df = pivoted_train[:train_samples]
    valid_samples_df = pivoted_train[train_samples:]
    
    # Scale the data
    print("Scaling data...")
    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(train_samples_df)
    
    scaled_train_samples = minmax_scaler.transform(train_samples_df)
    scaled_validation_samples = minmax_scaler.transform(valid_samples_df)
    
    # Prepare sequences for LSTM
    print("Preparing sequences...")
    n_past = 16  # Number of past time steps to use
    n_future = 16  # Number of future time steps to predict
    n_features = pivoted_train.shape[1]  # Number of features
    
    X_train, y_train = split_series(scaled_train_samples, n_past, n_future)
    X_val, y_val = split_series(scaled_validation_samples, n_past, n_future)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    
    # Create PyTorch dataloaders
    train_loader = create_dataloader(X_train, y_train, batch_size=64, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size=64, shuffle=False)
    
    # Save the preprocessed data for later use
    processed_data = {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'n_features': n_features,
        'n_past': n_past,
        'n_future': n_future,
        'scaler': minmax_scaler,
        'pivoted_train': pivoted_train,
        'test_data': test_data,
        'ordinal_encoder': ordinal_encoder
    }
    
    print("Data preprocessing completed")
    return processed_data

if __name__ == "__main__":
    # If this script is run directly, process the data and save
    data = load_data()
    processed_data = preprocess_data(data)
    
    # Save preprocessed data for later use
    print("Saving preprocessed data...")
    save_dir = './preprocessed'
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the model inputs and scaler
    np.save(os.path.join(save_dir, 'X_train.npy'), processed_data['X_train'])
    np.save(os.path.join(save_dir, 'y_train.npy'), processed_data['y_train'])
    np.save(os.path.join(save_dir, 'X_val.npy'), processed_data['X_val'])
    np.save(os.path.join(save_dir, 'y_val.npy'), processed_data['y_val'])
    
    # Cannot directly save the scaler with numpy, so use torch.save for the entire processed_data
    # Remove dataloaders which can't be easily serialized
    serializable_data = {k: v for k, v in processed_data.items() 
                        if k not in ['train_loader', 'val_loader']}
    torch.save(serializable_data, os.path.join(save_dir, 'processed_data.pt'))
    
    print("Preprocessing complete and data saved!")
