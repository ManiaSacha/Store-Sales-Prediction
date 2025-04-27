import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import get_model
from utils import rmsle, rmsle_loss
from data_processing import load_data, preprocess_data

def train_model(model, train_loader, val_loader, device, epochs=500, patience=100):
    """
    Train the model with early stopping.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on (cuda/cpu)
        epochs: Maximum number of epochs
        patience: Early stopping patience
        
    Returns:
        Trained model and training history
    """
    print(f"Training on {device}...")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_rmsle': [],
        'val_rmsle': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Start training
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_rmsle_val = 0.0
        train_batches = 0
        
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            # Move data to device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = model(X_batch)
            
            # Calculate loss
            loss = criterion(y_pred, y_batch)
            
            # Calculate RMSLE for monitoring
            batch_rmsle = rmsle_loss(y_pred, y_batch)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_rmsle_val += batch_rmsle.item()
            train_batches += 1
        
        # Calculate average training metrics
        avg_train_loss = train_loss / train_batches
        avg_train_rmsle = train_rmsle_val / train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_rmsle_val = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                # Move data to device
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Forward pass
                y_pred = model(X_batch)
                
                # Calculate loss
                loss = criterion(y_pred, y_batch)
                
                # Calculate RMSLE for monitoring
                batch_rmsle = rmsle_loss(y_pred, y_batch)
                
                # Update metrics
                val_loss += loss.item()
                val_rmsle_val += batch_rmsle.item()
                val_batches += 1
        
        # Calculate average validation metrics
        avg_val_loss = val_loss / val_batches
        avg_val_rmsle = val_rmsle_val / val_batches
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_rmsle'].append(avg_train_rmsle)
        history['val_rmsle'].append(avg_val_rmsle)
        
        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print metrics
        time_taken = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Train RMSLE: {avg_train_rmsle:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val RMSLE: {avg_val_rmsle:.4f}, "
              f"Time: {time_taken:.2f}s")
        
        # Check for early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

def plot_history(history):
    """
    Plot training history.
    """
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot RMSLE
    plt.subplot(1, 2, 2)
    plt.plot(history['train_rmsle'], label='Train RMSLE')
    plt.plot(history['val_rmsle'], label='Validation RMSLE')
    plt.title('Model RMSLE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSLE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./training_history.png')
    plt.show()

def save_model(model, save_dir='./models'):
    """
    Save the model.
    """
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'lstm_model.pth'))
    print(f"Model saved to {os.path.join(save_dir, 'lstm_model.pth')}")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if preprocessed data exists
    if os.path.exists('./preprocessed/processed_data.pt'):
        print("Loading preprocessed data...")
        processed_data = torch.load('./preprocessed/processed_data.pt', weights_only=False) 
        
        # Create dataloaders from saved data
        from utils import create_dataloader
        train_loader = create_dataloader(processed_data['X_train'], processed_data['y_train'], batch_size=64)
        val_loader = create_dataloader(processed_data['X_val'], processed_data['y_val'], batch_size=64, shuffle=False)
        
        n_features = processed_data['n_features']
    else:
        # Load and preprocess data
        data = load_data()
        processed_data = preprocess_data(data)
        
        train_loader = processed_data['train_loader']
        val_loader = processed_data['val_loader']
        n_features = processed_data['n_features']
    
    # Initialize model
    model, device = get_model(
        input_dim=n_features,
        hidden_dim=200,
        num_layers=2,
        dropout=0.2
    )
    
    # Train model
    try:
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=500,
            patience=100
        )
        
        # Plot training history
        plot_history(history)
        
        # Save model
        save_model(model)
        
        print("Training completed successfully!")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        # Try to save the model anyway if it exists
        if 'model' in locals():
            print("Attempting to save the model despite the error...")
            try:
                save_model(model)
                print("Model saved successfully despite error.")
            except Exception as save_error:
                print(f"Could not save model: {save_error}")
