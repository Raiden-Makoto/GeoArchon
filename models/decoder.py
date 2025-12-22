"""Decoder module for HEA VAE model.

Decodes latent space representations back into alloy compositions.
"""

import mlx.core as mx
import mlx.nn as nn


class Decoder(nn.Module):
    """Decoder network that maps latent space to composition space.
    
    Args:
        latent_dim: Dimension of latent space
        hidden_dim: Dimension of hidden layers
        output_dim: Dimension of output composition vector
        dropout_rate: Dropout rate for regularization (default: 0.0)
    """
    
    def __init__(self, latent_dim, hidden_dim, output_dim, dropout_rate=0.0):
        super().__init__()
        self.l1 = nn.Linear(latent_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.LeakyReLU(0.2)

    def __call__(self, z):
        """Forward pass through decoder.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
            
        Returns:
            Composition vector of shape (batch_size, output_dim) with softmax
            applied to ensure valid probability distribution (sums to 1)
        """
        h = self.activation(self.l1(z))
        h = self.dropout(h)
        h = self.activation(self.l2(h))
        h = self.dropout(h)
        # Softmax ensures the output is a valid composition (sums to 1)
        return mx.softmax(self.l3(h), axis=1)

