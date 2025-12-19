import mlx.nn as nn
import mlx.core as mx

class HEA_VAE(nn.Module):
    def __init__(
        self,
        input_dim: int=30, # 30 unique elements
        latent_dim: int=2, #[2, 4, 8]
        hidden_dim: int=128,
    ):
        super().__init__()
        self.enc1 = nn.Linear(input_dim, hidden_dim)
        self.enc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.dec1 = nn.Linear(latent_dim, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, input_dim)

        self.regressor = [
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        ]

    def encode(self, x):
        h = nn.relu(self.enc1(x))
        h = nn.relu(self.enc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparamaterize(mu, logvar):
        std = mx.exp(0.5 * logvar)
        eps = mx.random.uniform(shape=std.size)
        return mu + eps * std

    def decode(self, z):
        h = nn.relu(self.dec1(z))
        h = nn.relu(self.dec2(h))
        x = nn.softmax(self.dec_out(h))
        return x

    def predict(self, z):
        h = z
        for layer in self.regressor:
            h = layer(h)
        return h

    def __call__(self, inputs):
        mu, logvar = self.encode(inputs)
        z = self.reparamaterize(mu, logvar)
        x = self.decoder(z)
        prop = self.predict(z)
        return x, prop, mu, logvar

if __name__ == '__main__':
    model = HEA_VAE()
    print(model)