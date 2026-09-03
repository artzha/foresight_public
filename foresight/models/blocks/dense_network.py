import torch
import torch.nn as nn
from omegaconf import ListConfig, OmegaConf

from einops import rearrange

class DenseNetwork(nn.Module):
    def __init__(self, embedding_dim, out_dim):
        super(DenseNetwork, self).__init__()

        self.embedding_dim = embedding_dim
        self.out_dim = out_dim
        # Convert out_dim to list if it's not already
        if not isinstance(out_dim, list):
            self.out_dim = list(out_dim)
        
        flatten_dim = torch.prod(torch.tensor(self.out_dim))
        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim//4),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//4, self.embedding_dim//16),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//16, flatten_dim)
        )
    
    def forward(self, x):
        assert x.shape[1] == self.embedding_dim, f"Input must have shape [B, {self.embedding_dim}]"
        output = self.network(x)
        if len(self.out_dim) == 2:
            output = rearrange(output, 'b (t a) -> b t a', t=self.out_dim[0], a=self.out_dim[1])
        return output
