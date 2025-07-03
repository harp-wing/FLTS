# train_container/train.py

import torch
from m1 import LSTM
from shared.train_utils import train
from shared.data_utils import to_torch_dataset

def prepare_data_loaders(X_train_np, y_train_np, X_val_np, y_val_np, num_lags=10, num_features=11, batch_size=64):
    """
    Converts numpy data into PyTorch DataLoader objects.
    """
    train_loader = to_torch_dataset(X_train_np, y_train_np,
                                    num_lags=num_lags,
                                    num_features=num_features,
                                    batch_size=batch_size,
                                    shuffle=False)

    val_loader = to_torch_dataset(X_val_np, y_val_np,
                                  num_lags=num_lags,
                                  num_features=num_features,
                                  batch_size=batch_size,
                                  shuffle=False)

    return train_loader, val_loader

def train_model(X_train_np, y_train_np, X_val_np, y_val_np, config):
    """
    Wrapper for training a single LSTM model.
    """
    device = config.get("device", "cpu")
    num_lags = config.get("num_lags", 10)
    num_features = config.get("num_features", X_train_np.shape[2])
    output_dim = config.get("output_dim", y_train_np.shape[1])
    batch_size = config.get("batch_size", 64)
    epochs = config.get("epochs", 50)
    lr = config.get("lr", 0.001)

    # Create dataloaders
    train_loader, val_loader = prepare_data_loaders(X_train_np, y_train_np, X_val_np, y_val_np,
                                                    num_lags=num_lags, num_features=num_features, batch_size=batch_size)

    # Initialize LSTM
    model = LSTM(input_dim=num_features, num_outputs=output_dim, matrix_rep=True).to(device)

    # Train
    trained_model = train(model, train_loader, val_loader,
                          epochs=epochs,
                          optimizer="adam",
                          lr=lr,
                          criterion="mse",
                          device=device,
                          early_stopping=True)

    return trained_model


