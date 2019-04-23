import torch
import torch.nn as nn

from .utils import one_hot_encoding


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 1, 8, 8)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class VAE(nn.Module):
    def __init__(self, hidden_units, latent_space_dim, conditional, num_labels, device):
        super(VAE, self).__init__()

        ## Stuff for conditional VAE
        self.conditional = conditional
        if conditional:
            assert num_labels > 0
            hidden_units = hidden_units + num_labels
        else:
            num_labels = 0

        self.device = device
        self.num_labels = num_labels
        self.latent_space_dim = latent_space_dim

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

        self.fc1 = nn.Linear(hidden_units, latent_space_dim)
        self.fc2 = nn.Linear(hidden_units, latent_space_dim)
        self.fc3 = nn.Linear(latent_space_dim+num_labels, 64)

    def _build_encoder(self):
        return nn.Sequential(
            nn.Conv2d(3, out_channels=32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_channels=128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Flatten()
        )

    def _build_decoder(self):
        return nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(1, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=1),
            nn.Sigmoid(),
        )

    def _reparametrize(self, mu, logvar):
        eps = torch.randn(*mu.size(), device=self.device)
        std = logvar.mul(0.5).exp_()
        z = mu + std*eps
        return z

    def _bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self._reparametrize(mu, logvar)
        return z, mu, logvar

    ## Q(z|x, c)
    def _encode(self, x, c):
        h = self.encoder(x)
        if self.conditional:
            c = one_hot_encoding(c, n=self.num_labels)
            h = torch.cat([h, c], dim=1)
        z, mu, logvar = self._bottleneck(h)
        return z, mu, logvar

    ## P(x|z, c)
    def _decode(self, z, c):
        if self.conditional:
            c = one_hot_encoding(c, n=self.num_labels)
            z = torch.cat([z, c], dim=1)
        h = self.fc3(z)
        output = self.decoder(h)
        return output

    def forward(self, x, c=None):
        z, mu, logvar = self._encode(x, c)
        # print("from encoder", z.shape, mu.shape, logvar.shape)
        image = self._decode(z, c)
        return image, mu, logvar

    def inference(self, n=1, c=None):
        z = torch.randn([n, self.latent_space_dim])
        recon_x = self._decode(z, c)
        return recon_x
