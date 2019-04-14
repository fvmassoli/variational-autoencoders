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
    def __init__(self, hidden_unit_size=512, latent_space_dim=100, conditional=False, num_labels=0):
        super(VAE, self).__init__()

        if conditional:
            assert num_labels > 0

        self.conditional = conditional
        self.num_labels = num_labels
        self.latent_space_dim = latent_space_dim

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

        self.fc1 = nn.Linear(hidden_unit_size+num_labels, latent_space_dim)
        self.fc2 = nn.Linear(hidden_unit_size+num_labels, latent_space_dim)
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
        eps = torch.randn(*mu.size()).cuda()
        std = logvar.mul(0.5).exp_()
        z = mu + std*eps
        return z

    def _bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self._reparametrize(mu, logvar)
        return z, mu, logvar

    def _encode(self, x, c):
        if self.conditional:
            c = one_hot_encoding(c, n=self.num_labels)
            x = torch.cat([x, c], dim=1)
        h = self.encoder(x)
        z, mu, logvar = self._bottleneck(h)
        return z, mu, logvar

    def _decode(self, z, c):
        # print("from decoder before fc and torch.cat", z.shape)
        if self.conditional:
            c = one_hot_encoding(c, n=self.num_labels)
            z = torch.cat([z, c], dim=1)
        # print("from decoder before fc and after torch.cat", z.shape)
        h = self.fc3(z)
        # print("from decoder after", h.shape)
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







