import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


def train(model, optimizer, data_loader, loss_fn, cuda, e, epochs, batch_size):
    for idx, (data, _) in enumerate(tqdm(data_loader), 1):
        if cuda:
            data = data.cuda(non_blocking=True)
        recon_images, mu, logvar = model(data)
        loss, bce, kld = loss_fn(recon_images, data, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    to_print = "Epoch[{}/{}] Loss: {:.3f}\t Reconstruction: {:.3f}\t Dkl: {:.3f}".format(e + 1,
                                                                epochs, loss.item() / batch_size, bce.item() / batch_size,
                                                                kld.item() / batch_size)
    print(to_print)
