# train_container/train.py
from lstm import LSTM
from shared.logger import log
from logging import INFO
import torch
import numpy as np
from shared.train_utils import train
from torch.utils.data import TensorDataset, DataLoader

def prepare_data_loaders(X_train, y_train, X_test, y_test, batch_size=32, shuffle_train=True):
    """
    Converts numpy data into PyTorch DataLoader objects.
    """
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train_model(X_train_np, y_train_np, X_test_np, y_test_np, config):
    """
    Wrapper for training a single LSTM model.
    """
    device = config.get("device", "cpu")
    num_features = config.get("num_features", X_train_np.shape[2])
    n_exo_features = config.get("num_exogenous_features", 0)
    n_endo_features = num_features - n_exo_features
    hidden_size = config.get("hidden_size", 64)
    num_layers = config.get("num_layers", 1)
    output_size = config.get("output_dim")
    batch_size = config.get("batch_size", 32)
    epochs = config.get("epochs", 50)
    lr = config.get("lr", 0.001)

    # Create dataloaders
    train_loader, test_loader = prepare_data_loaders(X_train_np, y_train_np, X_test_np, y_test_np,
                                                    batch_size=batch_size)

    # Initialize LSTM
    model = LSTM(n_endo_features, n_exo_features, hidden_size, output_size, num_layers).to(device)

    # Train
    trained_model = train(model, train_loader, test_loader,
                          epochs=epochs,
                          optimizer="adam",
                          lr=lr,
                          criterion="mse",
                          device=device,
                          early_stopping=True, patience=50)

    return trained_model
