import torch
import torch.nn.functional as F


def one_hot_encoding(idx, n):
    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).cuda()
    onehot.scatter_(1, idx, 1)
    return onehot