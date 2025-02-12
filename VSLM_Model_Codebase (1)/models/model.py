import torch
import torch.nn as nn

class VSLMModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=1024, output_dim=30522):
        super(VSLMModel, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.activation(x)
        x = self.decoder(x)
        return x
