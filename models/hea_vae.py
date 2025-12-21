import mlx.nn as nn
import mlx.core as mx

class HEA_VAE(nn.Module):
    def __init__(
        self,
        input_dim: int=30, # 30 unique elements
        latent_dim: int=4, # Tightened latent space
        hidden_dim: int=512, # Larger hidden dimension for more capacity # or 128
        dropout_rate: float=0.1,
        slope: float=0.2, # works better for VAEs
    ):
        super().__init__()
        self.slope = slope
        self.enc1 = nn.Linear(input_dim, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.dec1 = nn.Linear(latent_dim, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.regressor = [
            nn.Linear(latent_dim, hidden_dim), # 4 -> 512
            nn.LeakyReLU(self.slope),
            self.dropout,
            nn.Linear(hidden_dim, hidden_dim // 2), # 512 -> 256
            nn.LeakyReLU(self.slope),
            self.dropout,
            nn.Linear(hidden_dim // 2, 64), # 256 -> 64
            nn.LeakyReLU(self.slope),
            nn.Linear(64, 1)
        ]

    def encode(self, x):
        h = nn.leaky_relu(self.enc1(x))
        h = nn.leaky_relu(self.enc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        logvar = mx.clip(logvar, -10, 10)
        std = mx.exp(0.5 * logvar)
        eps = mx.random.normal(shape=mu.shape)
        return mu + eps * std

    def decode(self, z):
        h = nn.leaky_relu(self.dec1(z))
        h = nn.leaky_relu(self.dec2(h))
        x = nn.softmax(self.dec_out(h))
        return x

    def predict(self, z):
        h = z
        for layer in self.regressor:
            h = layer(h)
        return h

    def __call__(self, inputs):
        mu, logvar = self.encode(inputs)
        z = self.reparameterize(mu, logvar)
        x = self.decode(z)
        prop = self.predict(z)
        return x, prop, mu, logvar

if __name__ == '__main__':
    model = HEA_VAE()
    print(model)