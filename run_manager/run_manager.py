from tqdm import tqdm

import torch
from torch.optim import Adam, SGD

from run_manager.utils import visualize_util


class RunManager(object):
    def __init__(self, model_manager, data_manager, logger, optimizer_type, lr, epochs, device, verbose):
        self._model_manager = model_manager
        self._data_manager = data_manager
        self._logger = logger
        self._optimizer_type = optimizer_type
        self._lr = lr
        self._epochs = epochs
        self._device = device
        self._verbose = verbose
        self._bs = data_manager.get_batch_size()
        self._print_run_info()

    def _print_run_info(self):
        if self._verbose > 1:
            print("="*20, "Run info", "="*21)
            print("\t Optimizer:                  {}\n"
                  "\t Learning rate:              {}\n"
                  "\t Number of training epochs:  {}".format(
                    self._optimizer_type, self._lr, self._epochs)
            )
            print("="*51)

    def run(self, train):
        if train:
            self._train()
        else:
            self._infer()

    def _train(self, ):

        if self._optimizer_type == 'adam':
            optimizer = Adam(self._model_manager.get_model_parameters(), lr=self._lr)
        else:
            optimizer = SGD(self._model_manager.get_model_parameters(), lr=self._lr, momentum=0.1, weight_decay=1.e-4)

        self._model_manager.get_model().train()
        train_loader, _ = self._data_manager.get_data_loaders()

        for e in range(self._epochs):

            _l = 0
            _bce = 0
            _dkl = 0

            for idx, (data, labels) in enumerate(tqdm(train_loader), 1):
                data = data.to(device=self._device)
                labels = labels.to(device=self._device)
                recon_images, mu, logvar = self._model_manager.forward(data, labels)
                loss, bce, kld = self._model_manager.get_loss(recon_images, data, mu, logvar)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _l += loss.item()
                _bce += bce.item()
                _dkl += kld.item()

                if (idx + 1) % int(len(train_loader) / 2) == 0:
                    self._logger.save_stats_on_csv(epochs=self._epochs,
                                                   epoch=e+1,
                                                   loss=loss.item(),
                                                   bce=bce.item(),
                                                   kld=kld.item())

            self._logger.save_model(state_dict=self._model_manager.get_model().state_dict())

    def _infer(self):
        self._model_manager.get_model().load_state_dict(torch.load('vae.torch'))
        self._model_manager.get_model().eval()
        samples = [torch.randn(1, 100) for i in range(2)]
        recons = [self._model_manager.get_model().decode(sample.cuda()) for sample in samples]
        recon_images = [r.squeeze(0).permute(1, 2, 0).detach().cpu() for r in recons]

        images, _ = iter(self._data_manager.get_data_loaders()[0]).next()
        recon_images2_, _, _ = self._model_manager.get_model().forward(images.cuda())
        recon_images2 = [r.squeeze(0).permute(1, 2, 0).detach().cpu() for r in recon_images2_]

        images = images.detach().cpu().numpy()
        images = images.transpose(0, 2, 3, 1)

        visualize_util(recon_images, recon_images2, images)
