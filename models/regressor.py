"""Property regressor module for HEA VAE model.

Predicts material properties (e.g., stability) from latent representations.
"""

import mlx.nn as nn


class PropertyRegressor(nn.Module):
    """Property regressor network that predicts properties from latent space.
    
    Args:
        latent_dim: Dimension of latent space
        hidden_dim: Dimension of hidden layers
        dropout_rate: Dropout rate for regularization (default: 0.0)
    """
    
    def __init__(self, latent_dim, hidden_dim, dropout_rate=0.0):
        super().__init__()
        self.l1 = nn.Linear(latent_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.l3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.LeakyReLU(0.2)

    def __call__(self, z):
        """Forward pass through regressor.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
            
        Returns:
            Property prediction of shape (batch_size, 1)
        """
        h = self.activation(self.l1(z))
        h = self.dropout(h)
        h = self.activation(self.l2(h))
        h = self.dropout(h)
        return self.l3(h)

