import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

# Check GPU availability
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
else:
    print("No GPU available, using CPU")
    device = torch.device('cpu')

# LSTM Model Definition
class StoreSalesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StoreSalesLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Custom Dataset class
class StoreDataset(Dataset):
    def __init__(self, features, targets=None, seq_length=14):
        self.features = features
        self.targets = targets
        self.seq_length = seq_length
        self.is_test = targets is None

    def __len__(self):
        if self.is_test:
            # For test data, return one prediction per sample
            return len(self.features)
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        if self.is_test:
            # For test data, pad with zeros if we don't have enough history
            if idx < self.seq_length:
                # Create a sequence with zeros padding
                pad_length = self.seq_length - idx - 1
                if pad_length > 0:
                    padding = torch.zeros((pad_length, self.features.shape[1]), dtype=self.features.dtype)
                    sequence = torch.cat([padding, self.features[:idx+1]], dim=0)
                else:
                    sequence = self.features[:idx+1]
            else:
                # Use the last seq_length features
                sequence = self.features[idx-self.seq_length+1:idx+1]
            return sequence
        # For training data, return both features and target
        return (self.features[idx:idx + self.seq_length], 
                self.targets[idx + self.seq_length])

def load_data():
    """
    Load the store sales dataset from CSV files
    Returns:
        train_df, stores_df, oil_df, holidays_df
    """
    try:
        # Load main training data
        train_df = pd.read_csv('train.csv', 
                             parse_dates=['date'],
                             dtype={'store_nbr': 'category',
                                   'family': 'category'})
        print("\nTrain data columns:", train_df.columns.tolist())
        
        # Load store information
        stores_df = pd.read_csv('stores.csv',
                              dtype={'store_nbr': 'category',
                                    'city': 'category',
                                    'state': 'category',
                                    'type': 'category'})
        print("\nStores data columns:", stores_df.columns.tolist())
        print("\nFirst few rows of stores data:")
        print(stores_df.head())
        
        # Load and process oil price data
        oil_df = pd.read_csv('oil.csv', parse_dates=['date'])
        print("\nOil data info:")
        print(oil_df.info())
        print("\nFirst few rows of oil data:")
        print(oil_df.head())
        
        # Handle oil price column
        if 'dcoilwtico' in oil_df.columns:  # This is the actual column name in the dataset
            oil_df['dcoilcv'] = oil_df['dcoilwtico']
        else:
            print("\nAvailable columns in oil_df:", oil_df.columns.tolist())
            raise ValueError("Could not find oil price column. Please check the data format.")
        
        # Load holidays and events data
        holidays_df = pd.read_csv('holidays_events.csv', 
                                parse_dates=['date'],
                                dtype={'type': 'category',
                                      'locale': 'category',
                                      'locale_name': 'category',
                                      'description': 'category'})
        
        print("Data loaded successfully!")
        return train_df, stores_df, oil_df, holidays_df
    
    except FileNotFoundError:
        print("Please ensure you have downloaded the dataset files and placed them in the correct directory.")
        print("You can download the dataset from Kaggle: Store Sales Time Series Forecasting competition")
        return None, None, None, None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None, None, None

def prepare_time_features(df):
    """
    Create time-based features from date column
    """
    df = df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    return df

def process_features(df, stores_df, oil_df, holidays_df, feature_scaler=None, target_scaler=None):
    """Process and scale features for both training and prediction"""
    # Merge datasets
    df = df.merge(stores_df, on='store_nbr', how='left')
    df = df.merge(oil_df[['date', 'dcoilcv']], on='date', how='left')
    
    # Fill missing oil prices
    df['dcoilcv'] = df['dcoilcv'].fillna(method='ffill').fillna(method='bfill')
    
    # Create time features
    df = prepare_time_features(df)
    
    # Convert categorical variables to dummy variables
    df = pd.get_dummies(df, columns=['type', 'family'])
    
    # Select features for the model
    feature_columns = ['dcoilcv', 'year', 'month', 'day', 'day_of_week', 'day_of_year'] + \
                     [col for col in df.columns if col.startswith(('type_', 'family_'))]
    
    # Scale features
    if feature_scaler is None:
        feature_scaler = StandardScaler()
        features = feature_scaler.fit_transform(df[feature_columns])
    else:
        features = feature_scaler.transform(df[feature_columns])
    
    # Scale targets if present
    targets = None
    if 'sales' in df.columns:
        if target_scaler is None:
            target_scaler = StandardScaler()
            targets = target_scaler.fit_transform(df[['sales']])
        else:
            targets = target_scaler.transform(df[['sales']])
    
    return features, targets, feature_scaler, target_scaler, feature_columns

def prepare_data_for_lstm(train_df, stores_df, oil_df, holidays_df, seq_length=14, is_training=True):
    """
    Prepare data for LSTM model
    """
    print("\nShape before merging:")
    print("Train data:", train_df.shape)
    print("Stores data:", stores_df.shape)
    print("Oil data:", oil_df.shape)
    
    # Process features
    features, targets, feature_scaler, target_scaler, feature_columns = process_features(
        train_df, stores_df, oil_df, holidays_df
    )
    
    # Convert to PyTorch tensors
    features = torch.FloatTensor(features)
    
    # For training data
    if is_training:
        targets = torch.FloatTensor(targets)
        dataset = StoreDataset(features, targets, seq_length)
        
        # Split into train and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        batch_size = 128
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                pin_memory=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                              pin_memory=True, num_workers=4)
        
        return train_loader, val_loader, target_scaler, feature_scaler, feature_columns
    
    # For test/prediction data
    else:
        dataset = StoreDataset(features, seq_length=seq_length)
        loader = DataLoader(dataset, batch_size=128, shuffle=False, 
                          pin_memory=True, num_workers=4)
        return loader, None, None, feature_scaler, feature_columns

def prepare_future_data(df, stores_df, oil_df, holidays_df, feature_scaler, feature_columns, seq_length=14):
    """
    Prepare future data for predictions
    """
    # Process features using the same scaler and columns from training
    features, _, _, _, _ = process_features(
        df, stores_df, oil_df, holidays_df,
        feature_scaler=feature_scaler
    )
    
    # Convert to tensor
    features = torch.FloatTensor(features)
    
    # Create dataset and loader
    dataset = StoreDataset(features, targets=None, seq_length=seq_length)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, 
                       pin_memory=True, num_workers=4)
    
    return loader

def get_scaler():
    """Get the feature scaler fitted on training data"""
    train_df, stores_df, oil_df, holidays_df = load_data()
    if train_df is not None:
        train_loader, _, _, feature_scaler, _ = prepare_data_for_lstm(
            train_df, stores_df, oil_df, holidays_df
        )
        return feature_scaler
    return None

def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs, device):
    """
    Train the LSTM model
    """
    model = model.to(device)
    best_val_loss = float('inf')
    print(f"\nTraining on: {device}")
    
    try:
        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                
                # Forward pass
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Add gradient clipping
                optimizer.step()
                
                total_train_loss += loss.item()
            
            # Validation
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(device)
                    batch_targets = batch_targets.to(device)
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    total_val_loss += loss.item()
            
            # Calculate average losses
            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}')
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }, 'best_model.pth')
                print(f'Model saved with validation loss: {best_val_loss:.4f}')
    
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("WARNING: out of memory")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            raise e

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Load the data
    train_df, stores_df, oil_df, holidays_df = load_data()
    
    if train_df is not None:
        print("\nPreparing data for LSTM model...")
        # Prepare data for LSTM
        train_loader, val_loader, target_scaler, feature_scaler, feature_columns = prepare_data_for_lstm(
            train_df, stores_df, oil_df, holidays_df
        )
        
        # Initialize model parameters
        input_size = next(iter(train_loader))[0].shape[2]  # Number of features
        hidden_size = 128  # Increased for GPU
        num_layers = 2
        output_size = 1
        
        # Create model, loss function, and optimizer
        model = StoreSalesLSTM(input_size, hidden_size, num_layers, output_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train the model
        print("\nStarting training...")
        num_epochs = 10
        train_model(
            train_loader, val_loader, model, criterion, optimizer, num_epochs, device
        )
        
        print("\nTraining completed! Best model saved as 'best_model.pth'")

if __name__ == "__main__":
    main()
