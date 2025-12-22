"""HEA VAE model for high-entropy alloy property prediction.

This module provides the main HEA_VAE class that combines encoder, decoder,
and property regressor components for variational autoencoder-based material
property prediction.
"""

import mlx.core as mx
import mlx.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .regressor import PropertyRegressor

class HEA_VAE(nn.Module):
    def __init__(self, input_dim=30, latent_dim=4, hidden_dim=512, dropout_rate=0.0):
        super().__init__()
        
        # 1. Encoder (Compress Chemistry -> Latent Map)
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, dropout_rate)
        
        # 2. Decoder (Latent Map -> Reconstruct Chemistry)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, dropout_rate)
        
        # 3. Regressor (Latent Map -> Predict Stability)
        self.regressor = PropertyRegressor(latent_dim, hidden_dim, dropout_rate)

    def reparameterize(self, mean, logvar):
        std = mx.exp(0.5 * logvar)
        eps = mx.random.normal(mean.shape)
        return mean + eps * std

    def __call__(self, x):
        # Forward pass returning all components for loss calculation
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        pred_y = self.regressor(z)
        return recon_x, pred_y, mu, logvar

    def encode(self, x):
        """Maps input alloy to its latent coordinate (mean)."""
        mu, _ = self.encoder(x)
        return mu, None

    def decode(self, z):
        """Generates alloy composition from a latent point."""
        return self.decoder(z)

    def predict(self, z):
        """Predicts stability (energy) for a latent point."""
        return self.regressor(z)