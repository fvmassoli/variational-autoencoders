import torch
from models.vae import VAE
from models.perceptual_model import PerceptualModules

import torch.nn.functional as F


class ModelManager(object):
    def __init__(self, conditional, perceptual_loss, num_labels, checkpoint_path, cuda, verbose):
        self.conditional = conditional
        self.num_labels = num_labels
        self.checkpoint_path = checkpoint_path
        self.cuda = cuda
        self.verbose = verbose
        self._vae = self._build_vae()
        self.perceptual_loss = perceptual_loss
        self._perceptual_modules = None
        if perceptual_loss:
            self._perceptual_modules = self._build_perceptual_modules()
        self._print_model_info()

    def _build_vae(self):
        vae = VAE(hidden_units=512, latent_space_dim=100, conditional=self.conditional, num_labels=self.num_labels)
        if self.checkpoint_path is not None:
            vae.load_state_dict(torch.load(self.checkpoint_path))
        if self.cuda:
            vae.cuda()
        return vae

    def _build_perceptual_modules(self):
        perceptual_modules = PerceptualModules(cuda=self.cuda, verbose=self.verbose)
        return perceptual_modules.get_modules()

    def _print_model_info(self):
        if self.verbose > 0:
            print("="*20, "Model info", "="*19)
            print("\t Is model conditional VAE: {}\n"
                  "\t Is using perceptual loss: {}\n".format(self.conditional, self.perceptual_loss))
            print("="*51)

    def get_model(self):
        return self._vae

    def get_perceptual_modules(self):
        return self._perceptual_modules

    def forward(self, x, c):
        return self._vae.forward(x, c)

    def get_loss(self, recon_x, x, mu, logvar):

        alpha = 1

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        if self.perceptual_loss:
            _l = self._perceptual_modules.get_loss(recon_x, x)
            beta = 0.5
        else:
            _l = F.binary_cross_entropy(recon_x, x, size_average=False)
            beta = 1

        return beta*_l + alpha*KLD, _l, KLD
