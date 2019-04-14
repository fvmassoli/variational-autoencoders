import torch
import torch.nn.functional as F


def one_hot_encoding(idx, n):
    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx, 1)
    return onehot


def loss_fn(recon_x, x, mu, logvar):
    # print("from loss function", recon_x.shape, x.shape, mu.shape, logvar.shape)
    BCE = F.binary_cross_entropy(recon_x, x) #, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD
