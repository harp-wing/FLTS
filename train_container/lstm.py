import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, n_endo_features, n_exo_features, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.n_features = n_endo_features + n_exo_features
        self.n_endo_features = n_endo_features
        self.n_exo_features = n_exo_features
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.n_features, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * n_endo_features)
    
    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, n_features).
        
        Returns:
            torch.Tensor: The output tensor from the model.
        """
        # --- Initialize Hidden and Cell States ---
        # The LSTM needs initial hidden and cell states. If not provided, they default to zeros.
        # The shape is (num_layers, batch_size, hidden_size).
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # --- LSTM Forward Pass ---
        # The LSTM returns the output of the entire sequence and the final hidden and cell states.
        # out shape: (batch_size, sequence_length, hidden_size)
        # hn shape: (num_layers, batch_size, hidden_size)
        # cn shape: (num_layers, batch_size, hidden_size)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # --- Fully-Connected Layer ---
        # We are interested in the output of the last time step for many sequence tasks
        # (e.g., classification, next value prediction).
        # out[:, -1, :] selects the output of the last element in the sequence for each batch.
        # Shape of out[:, -1, :]: (batch_size, hidden_size)
        out = self.fc(out[:, -1, :])
        
        # Reshape the output to be (batch_size, output_seq_len, output_features)
        # This matches the shape of our target tensor.
        out = out.view(x.size(0), self.output_size, self.n_endo_features)
        
        return out