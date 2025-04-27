import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=200, num_layers=2, output_dim=None, dropout=0.2):
        """
        LSTM model for time series forecasting.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units in LSTM layers
            num_layers: Number of LSTM layers
            output_dim: Number of output features (same as input_dim if None)
            dropout: Dropout probability
        """
        super(LSTMForecaster, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim if output_dim is not None else input_dim
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(hidden_dim)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, self.output_dim)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        # LSTM output: output, (h_n, c_n)
        lstm_out, _ = self.lstm(x)
        
        # Apply batch normalization (need to reshape for batch norm)
        batch_size, seq_len, _ = lstm_out.shape
        lstm_out_reshaped = lstm_out.contiguous().view(batch_size * seq_len, self.hidden_dim)
        bn_out = self.bn(lstm_out_reshaped)
        bn_out = bn_out.view(batch_size, seq_len, self.hidden_dim)
        
        # Apply dropout
        out = self.dropout(bn_out)
        
        # Apply output layer
        out = self.fc(out)
        
        return out

# Helper function to initialize model with GPU support
def get_model(input_dim, hidden_dim=200, num_layers=2, output_dim=None, dropout=0.2):
    """
    Initialize the model and move it to GPU if available.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMForecaster(input_dim, hidden_dim, num_layers, output_dim, dropout)
    model = model.to(device)
    print(f"Model created on device: {device}")
    return model, device
