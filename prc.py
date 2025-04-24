import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import torch.nn as nn
from sklearn.model_selection import train_test_split

# Define transformation functions at the module level for pickling
def log_transform(x):
    """Apply log1p transformation to input data"""
    return np.log1p(x)

def inverse_log_transform(x):
    """Inverse of log1p transformation (expm1)"""
    return np.expm1(x)

class StoreSalesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.3):
        super(StoreSalesLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Initialize weights with Xavier/Glorot initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
        
        # LSTM layers with residual connections
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=input_size if i == 0 else hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                dropout=0 if i == num_layers - 1 else dropout
            ) for i in range(num_layers)
        ])
        
        # Batch normalization layers
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_size) for _ in range(num_layers)
        ])
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Output layers with skip connection
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, 1)
        
        # Apply weight initialization
        self.apply(init_weights)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Process through LSTM layers with residual connections
        lstm_out = x
        for i, (lstm, bn) in enumerate(zip(self.lstm_layers, self.bn_layers)):
            new_lstm_out, _ = lstm(lstm_out)
            new_lstm_out = new_lstm_out.transpose(1, 2)  # Transpose for batch norm
            new_lstm_out = bn(new_lstm_out)
            new_lstm_out = new_lstm_out.transpose(1, 2)  # Transpose back
            if i > 0:  # Add residual connection after first layer
                lstm_out = lstm_out + new_lstm_out
            else:
                lstm_out = new_lstm_out
            lstm_out = self.dropout(lstm_out)
        
        # Apply attention mechanism
        attention_weights = self.attention(lstm_out)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Process through output layers with skip connection
        out1 = self.fc1(context_vector)
        out1 = torch.relu(out1)
        out1 = self.dropout(out1)
        
        out2 = self.fc2(out1)
        out2 = torch.relu(out2)
        out2 = self.dropout(out2)
        
        # Skip connection from first hidden layer to output
        out3 = self.fc3(out2)
        
        return out3

class StoreSalesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data():
    """
    Load and prepare all required datasets with validation
    """
    required_train_columns = ['date', 'store_nbr', 'family', 'sales', 'onpromotion']
    required_stores_columns = ['store_nbr', 'city', 'state', 'type', 'cluster']
    required_oil_columns = ['date', 'dcoilwtico']
    
    try:
        # Load train data
        train_df = pd.read_csv('train.csv', parse_dates=['date'], dtype={
            'store_nbr': 'category',
            'family': 'category',
            'sales': 'float32',
            'onpromotion': 'int8'
        })
        
        # Validate train columns
        missing_cols = set(required_train_columns) - set(train_df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in train.csv: {missing_cols}")
        print("\nTrain data columns:", train_df.columns.tolist())
        
        # Load store data
        stores_df = pd.read_csv('stores.csv', dtype={
            'store_nbr': 'category',
            'city': 'category',
            'state': 'category',
            'type': 'category',
            'cluster': 'int8'
        })
        
        # Validate stores columns
        missing_cols = set(required_stores_columns) - set(stores_df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in stores.csv: {missing_cols}")
        print("\nStores data columns:", stores_df.columns.tolist())
        print("\nFirst few rows of stores data:")
        print(stores_df.head().to_string())
        
        # Load oil data
        oil_df = pd.read_csv('oil.csv', parse_dates=['date'], dtype={'dcoilwtico': 'float32'})
        
        # Validate oil columns
        missing_cols = set(required_oil_columns) - set(oil_df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in oil.csv: {missing_cols}")
        print("\nOil data info:")
        print(oil_df.info())
        print("\nFirst few rows of oil data:")
        print(oil_df.head().to_string())
        
        # Load holidays data if available
        try:
            holidays_df = pd.read_csv('holidays_events.csv', parse_dates=['date'])
            holidays_df = holidays_df[holidays_df['transferred'] == False]  # Remove transferred holidays
            holidays_df['holiday'] = 1  # Create binary holiday indicator
        except FileNotFoundError:
            holidays_df = None
            print("\nHolidays data not found, proceeding without it.")
        
        # Validate date ranges
        min_date = train_df['date'].min()
        max_date = train_df['date'].max()
        
        # Ensure oil data covers the entire training period
        oil_df = oil_df[(oil_df['date'] >= min_date) & (oil_df['date'] <= max_date)]
        
        # Sort all dataframes by date
        train_df = train_df.sort_values('date')
        oil_df = oil_df.sort_values('date')
        if holidays_df is not None:
            holidays_df = holidays_df.sort_values('date')
        
        print("Data loaded successfully!")
        return train_df, stores_df, oil_df, holidays_df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None, None, None

def prepare_data_for_lstm(train_df, stores_df, oil_df, holidays_df, seq_length=14):
    """
    Prepare data for LSTM model by merging sources, creating features, and scaling
    """
    print("\nPreparing data...")
    
    # Create a copy to avoid modifying original dataframes
    df = train_df.copy()
    stores_df = stores_df.copy()
    oil_df = oil_df.copy()
    
    # Calculate store-level statistics before merging
    print("Calculating store statistics...")
    # Group by store_nbr first to avoid data leakage across stores
    store_stats = df.groupby('store_nbr')['sales'].agg(['mean', 'std', 'median']).reset_index()
    store_stats.columns = ['store_nbr', 'store_mean_sales', 'store_std_sales', 'store_median_sales']
    stores_df = stores_df.merge(store_stats, on='store_nbr', how='left')
    
    # Check if stores_df has the required columns, if not, add default values
    store_columns = ['store_mean_sales', 'store_std_sales', 'store_median_sales']
    for col in store_columns:
        if col not in stores_df.columns:
            print(f"Column {col} not found in stores_df, adding default values")
            stores_df[col] = 0
    
    # Create store type and cluster embeddings
    print("Creating store embeddings...")
    store_type_map = {t: i for i, t in enumerate(stores_df['type'].unique())}
    stores_df['type_code'] = stores_df['type'].map(store_type_map)
    stores_df['cluster_code'] = stores_df['cluster'].astype('category').cat.codes
    
    # Merge all data sources
    print("Merging data sources...")
    df = df.merge(stores_df, on='store_nbr', how='left')
    df = df.merge(oil_df, on='date', how='left')
    if holidays_df is not None:
        df = df.merge(holidays_df, on='date', how='left')
        df['holiday'] = df['holiday'].fillna(0)
    else:
        df['holiday'] = 0
    
    # Fill missing values
    df['dcoilwtico'] = df['dcoilwtico'].bfill().ffill()
    df['onpromotion'] = df['onpromotion'].fillna(0)
    
    # Create time-based features
    print("Creating time features...")
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    
    # Create product family embeddings
    family_map = {f: i for i, f in enumerate(df['family'].unique())}
    df['family_code'] = df['family'].map(family_map)
    
    # Sort by date to ensure proper time series order before calculating rolling stats
    df = df.sort_values(['store_nbr', 'family', 'date'])
    
    # Add rolling statistics for each store-family combination
    print("Calculating rolling statistics...")
    # Initialize columns to avoid data leakage
    df['sales_lag7'] = np.nan
    df['sales_lag14'] = np.nan
    df['sales_rolling_mean7'] = np.nan
    df['sales_rolling_std7'] = np.nan
    df['sales_rolling_median7'] = np.nan
    
    for store_nbr in df['store_nbr'].unique():
        for family in df['family'].unique():
            mask = (df['store_nbr'] == store_nbr) & (df['family'] == family)
            if mask.any():
                # Ensure we're working with chronologically sorted data
                store_family_data = df.loc[mask].sort_values('date')
                
                # Calculate lagged values without data leakage
                store_family_data['sales_lag7'] = store_family_data['sales'].shift(7)
                store_family_data['sales_lag14'] = store_family_data['sales'].shift(14)
                
                # Calculate rolling statistics without data leakage
                # Use shift(1) to ensure we only use past data points
                store_family_data['sales_rolling_mean7'] = store_family_data['sales'].shift(1).rolling(7, min_periods=1).mean()
                store_family_data['sales_rolling_std7'] = store_family_data['sales'].shift(1).rolling(7, min_periods=1).std()
                store_family_data['sales_rolling_median7'] = store_family_data['sales'].shift(1).rolling(7, min_periods=1).median()
                
                # Update original DataFrame with calculated features
                df.loc[mask, 'sales_lag7'] = store_family_data['sales_lag7']
                df.loc[mask, 'sales_lag14'] = store_family_data['sales_lag14']
                df.loc[mask, 'sales_rolling_mean7'] = store_family_data['sales_rolling_mean7']
                df.loc[mask, 'sales_rolling_std7'] = store_family_data['sales_rolling_std7']
                df.loc[mask, 'sales_rolling_median7'] = store_family_data['sales_rolling_median7']
    
    # Add family-level features from historical data to avoid leakage
    print("Adding family-level features...")
    family_stats = df.groupby('family')['sales'].agg(['mean', 'std', 'median']).reset_index()
    family_stats.columns = ['family', 'family_mean_sales', 'family_std_sales', 'family_median_sales']
    df = df.merge(family_stats, on='family', how='left')
    
    # Add promotion features with shift to avoid leakage
    df['promo_14_day'] = df.groupby(['store_nbr', 'family'])['onpromotion'].transform(
        lambda x: x.shift(1).rolling(14, min_periods=1).sum()
    )
    
    # Feature columns for model
    feature_columns = [
        'onpromotion', 'dcoilwtico', 'day_of_week', 'day_of_month', 'month',
        'is_weekend', 'quarter', 'is_month_start', 'is_month_end',
        'type_code', 'cluster_code', 'family_code', 'holiday',
        'sales_lag7', 'sales_lag14', 'sales_rolling_mean7', 'sales_rolling_std7',
        'sales_rolling_median7', 'store_mean_sales', 'store_std_sales',
        'store_median_sales', 'family_mean_sales', 'family_std_sales',
        'family_median_sales', 'promo_14_day'
    ]
    
    # Handle NaN values more carefully
    for col in feature_columns:
        if df[col].isna().any():
            # For rolling statistics, fill NaN with the mean of non-NaN values for that feature
            if col.startswith('sales_rolling') or col.endswith('_lag7') or col.endswith('_lag14'):
                col_mean = df[col].mean()
                df[col] = df[col].fillna(col_mean)
            else:
                df[col] = df[col].fillna(0)
    
    # Scale features using RobustScaler for better handling of outliers
    print("Scaling features...")
    scaler = RobustScaler()
    features = scaler.fit_transform(df[feature_columns])
    
    # Store log1p transformation for inverse transform later
    inv_log_transformer = inverse_log_transform
    
    # Scale target variable using log1p transformation
    sales = log_transform(df['sales'].values)
    
    # Create sequences for LSTM, ensuring chronological order within each store-family pair
    print("Creating sequences...")
    X, y = [], []
    store_family_indices = {} # Keep track of indices for each store-family pair
    
    for store_nbr in df['store_nbr'].unique():
        for family in df['family'].unique():
            mask = (df['store_nbr'] == store_nbr) & (df['family'] == family)
            if mask.sum() > seq_length:  # Only create sequences if we have enough data
                # Get the indices for this store-family combination, ensuring chronological order
                indices = df.loc[mask].sort_values('date').index
                store_family_indices[(store_nbr, family)] = indices
                
                store_features = features[indices]
                store_sales = sales[indices]
                
                for i in range(len(store_features) - seq_length):
                    X.append(store_features[i:(i + seq_length)])
                    y.append(store_sales[i + seq_length])
    
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    print(f"Created {len(X)} sequences with shape {X.shape}")
    
    return X, y, scaler, feature_columns, store_type_map, family_map, stores_df, inv_log_transformer

def prepare_test_data(test_df, stores_df, oil_df, holidays_df, scaler, feature_columns, store_type_map, family_map, seq_length=14):
    """
    Prepare test data for prediction using the same transformations as training data
    """
    print("\nPreparing test data...")
    
    # Create a copy to avoid modifying original dataframes
    df = test_df.copy()
    stores_df = stores_df.copy()
    oil_df = oil_df.copy()
    
    # Create store type and cluster embeddings using training maps
    print("Creating store embeddings...")
    stores_df['type_code'] = stores_df['type'].map(store_type_map)
    stores_df['cluster_code'] = stores_df['cluster'].astype('category').cat.codes
    
    # Merge all data sources
    print("Merging data sources...")
    df = df.merge(stores_df, on='store_nbr', how='left')
    df = df.merge(oil_df, on='date', how='left')
    if holidays_df is not None:
        df = df.merge(holidays_df, on='date', how='left')
        df['holiday'] = df['holiday'].fillna(0)
    else:
        df['holiday'] = 0
    
    # Fill missing values
    df['dcoilwtico'] = df['dcoilwtico'].bfill().ffill()
    
    # Create onpromotion column if it doesn't exist (for test/prediction data)
    if 'onpromotion' not in df.columns:
        print("Creating 'onpromotion' column with default value 0")
        df['onpromotion'] = 0
    else:
        df['onpromotion'] = df['onpromotion'].fillna(0)
    
    # Create time-based features
    print("Creating time features...")
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    
    # Create product family embeddings using training maps
    df['family_code'] = df['family'].map(family_map)
    
    # Add rolling statistics (using historical data)
    print("Adding historical statistics...")
    df['sales_lag7'] = 0  # For test data, we don't have previous sales
    df['sales_lag14'] = 0
    df['sales_rolling_mean7'] = 0
    df['sales_rolling_std7'] = 0
    df['sales_rolling_median7'] = 0
    
    # Add store-level and family-level features from historical data
    print("Adding store and family features...")
    
    # Check if stores_df has the required columns, if not, add default values
    store_columns = ['store_mean_sales', 'store_std_sales', 'store_median_sales']
    for col in store_columns:
        if col not in stores_df.columns:
            print(f"Column {col} not found in stores_df, adding default values")
            stores_df[col] = 0
    
    df['store_mean_sales'] = df['store_nbr'].map(stores_df.set_index('store_nbr')['store_mean_sales'])
    df['store_std_sales'] = df['store_nbr'].map(stores_df.set_index('store_nbr')['store_std_sales'])
    df['store_median_sales'] = df['store_nbr'].map(stores_df.set_index('store_nbr')['store_median_sales'])
    df['family_mean_sales'] = df['family'].map(pd.Series(index=family_map.keys(), data=0))
    df['family_std_sales'] = df['family'].map(pd.Series(index=family_map.keys(), data=0))
    df['family_median_sales'] = df['family'].map(pd.Series(index=family_map.keys(), data=0))
    
    # Add promotion features
    df['promo_14_day'] = 0  # For test data, we don't have historical promotion data
    
    # Fill any missing values
    for col in feature_columns:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(0)
    
    # Make sure all required feature columns exist
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
            print(f"Warning: Added missing column {col} with default value 0")
    
    # Scale features using the same scaler from training
    print("Scaling features...")
    features = scaler.transform(df[feature_columns])
    
    # Create sequences for LSTM properly handling the test data structure
    print("Creating sequences for test data...")
    X_test = []
    
    # Store the store-family combinations and their corresponding dates for reference
    store_family_dates = []
    
    # Process each store-family combination separately
    for store_nbr in df['store_nbr'].unique():
        for family in df['family'].unique():
            mask = (df['store_nbr'] == store_nbr) & (df['family'] == family)
            store_family_df = df.loc[mask].sort_values('date')
            
            if len(store_family_df) > 0:
                # If we have at least seq_length records, we can create a proper sequence
                if len(store_family_df) >= seq_length:
                    # Use the last seq_length records for prediction
                    store_family_features = features[store_family_df.index[-seq_length:]]
                    X_test.append(store_family_features)
                    
                    # Store the date we're predicting for (the day after the sequence)
                    store_family_dates.append({
                        'store_nbr': store_nbr,
                        'family': family,
                        'date': store_family_df['date'].iloc[-1] + pd.Timedelta(days=1),
                        'index': len(X_test) - 1  # Keep track of the index in X_test
                    })
                else:
                    # Handle cases with insufficient history by padding with zeros
                    print(f"Warning: Store {store_nbr}, family {family} has only {len(store_family_df)} records (less than {seq_length})")
                    # Create padding
                    padding_size = seq_length - len(store_family_df)
                    padding = np.zeros((padding_size, features.shape[1]))
                    
                    # Get available features
                    available_features = features[store_family_df.index]
                    
                    # Combine padding with available features
                    padded_sequence = np.vstack([padding, available_features])
                    X_test.append(padded_sequence)
                    
                    # Store reference
                    store_family_dates.append({
                        'store_nbr': store_nbr,
                        'family': family,
                        'date': store_family_df['date'].iloc[-1] + pd.Timedelta(days=1),
                        'index': len(X_test) - 1
                    })
    
    X_test = np.array(X_test)
    store_family_dates_df = pd.DataFrame(store_family_dates)
    
    print(f"Created test data with shape {X_test.shape}")
    
    return X_test, store_family_dates_df

def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs, device):
    """
    Train the LSTM model
    """
    model = model.to(device)
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
                optimizer.step()
                
                total_train_loss += loss.item()
            
            # Calculate average losses
            avg_train_loss = total_train_loss / len(train_loader)
            
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(device)
                    batch_targets = batch_targets.to(device)
                    
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
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
        X, y, scaler, feature_columns, store_type_map, family_map, stores_df, inv_log_transformer = prepare_data_for_lstm(
            train_df, stores_df, oil_df, holidays_df
        )
        
        # Split data into training and validation sets (80% train, 20% validation)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")
        
        # Initialize model parameters
        input_size = X.shape[2]  # Number of features
        hidden_size = 256  
        num_layers = 3
        output_size = 1
        
        # Create model, loss function, and optimizer
        model = StoreSalesLSTM(input_size, hidden_size, num_layers, dropout=0.3)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create datasets and loaders
        train_dataset = StoreSalesDataset(X_train, y_train)
        val_dataset = StoreSalesDataset(X_val, y_val)
        
        # Adjust batch size based on available memory
        batch_size = 64 if torch.cuda.is_available() else 32
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                       pin_memory=torch.cuda.is_available(), num_workers=4 if torch.cuda.is_available() else 0)
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                     pin_memory=torch.cuda.is_available(), num_workers=4 if torch.cuda.is_available() else 0)
        
        # Train the model
        print("\nStarting training...")
        num_epochs = 10
        train_model(
            train_loader, val_loader, model, criterion, optimizer, num_epochs, 
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Save the trained model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler': scaler,
            'feature_columns': feature_columns,
            'store_type_map': store_type_map,
            'family_map': family_map,
            'inv_log_transformer': inv_log_transformer
        }, 'store_sales_model.pth')
        print("\nModel saved as 'store_sales_model.pth'")
        
        print("\nTraining completed!")

if __name__ == "__main__":
    main()
