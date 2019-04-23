import torch


def one_hot_encoding(idx, n, device):
    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n, device=device)
    onehot.scatter_(1, idx, 1)
    return onehot