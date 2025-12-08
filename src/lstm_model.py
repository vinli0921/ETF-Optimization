import torch
import torch.nn as nn

class LSTMReturnPredictor(nn.Module):
    """
    Simple 1-layer LSTM to predict next-day returns from past returns.
    """
    def __init__(self,input_dim,hidden_dim,num_layers=1):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.lstm=nn.LSTM(input_dim,hidden_dim,num_layers=num_layers,batch_first=True)
        self.fc=nn.Linear(hidden_dim,1)

    def forward(self,x):
        """
        Forward pass:
        x shape: (batch, seq_len, input_dim)
        """
        hidden_0=torch.zeros(self.num_layers,x.size(0),self.hidden_dim)
        cell_0=torch.zeros(self.num_layers,x.size(0),self.hidden_dim)
        # Initialize hidden & cell (zeros)
        # LSTM forward
        out, _=self.lstm(x,(hidden_0, cell_0))

        # Take last time step output
        last_hidden=out[:, -1, :]

        # Linear regression to 1 value
        return self.fc(last_hidden)
