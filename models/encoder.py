"""Encoder module for HEA VAE model.

Encodes input alloy compositions into latent space representations.
"""

import mlx.core as mx
import mlx.nn as nn


class Encoder(nn.Module):
    """Encoder network that maps input compositions to latent space.
    
    Args:
        input_dim: Dimension of input composition vector
        hidden_dim: Dimension of hidden layers
        latent_dim: Dimension of latent space
        dropout_rate: Dropout rate for regularization (default: 0.0)
    """
    
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_rate=0.0):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.LeakyReLU(0.2)

    def __call__(self, x):
        """Forward pass through encoder.
        
        Args:
            x: Input composition tensor of shape (batch_size, input_dim)
            
        Returns:
            mean: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
        """
        h = self.activation(self.l1(x))
        h = self.dropout(h)
        h = self.activation(self.l2(h))
        h = self.dropout(h)
        return self.mean(h), self.logvar(h)

