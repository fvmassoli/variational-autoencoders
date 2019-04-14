import torch
from tqdm import tqdm
from torch.optim import Adam, SGD


class RunManager(object):
    def __init__(self, model, conditional, train_loader, valid_loader, optimizer_type, lr, epochs, cuda):
        self._model = model
        self._conditional = conditional
        self._train_loader = train_loader
        self._valid_loader = valid_loader
        self._optimizer_type = optimizer_type
        self._lr = lr
        self._epochs = epochs
        self._cuda = cuda
        self._batch_size = train_loader.batch_size
        self._print_run_info()

    def _print_run_info(self):
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
            optimizer = Adam(self._model.parameters(), lr=self._lr)
        else:
            optimizer = SGD(self._model.parameters(), lr=self._lr, momentum=0.1, weight_decay=1.e-4)

        self._model.train()
        if self._cuda:
            self._model.cuda()

        for e in enumerate(self._epochs, 1):

            for idx, (data, labels) in enumerate(tqdm(self._train_loader), 1):
                if self._cuda:
                    data = data.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                if self._conditional:
                    recon_images, mu, logvar = self._model(data, labels)
                else:
                    recon_images, mu, logvar = self._model(data)
                loss, bce, kld = self._model.get_loss(recon_images, data, mu, logvar)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            torch.save(self._model.state_dict(), 'vae.torch')

            print("Epoch[{}/{}] Loss: {:.3f}"
                  "\t Reconstruction: {:.3f}"
                  "\t Dkl: {:.3f}".format(e, self._epochs,
                                          loss.item() / self._batch_size,
                                          bce.item() / self._batch_size,
                                          kld.item() / self._batch_size))


    def _infer(self):
        pass
# def check(model):
#     model.load_state_dict(torch.load('vae.torch'))
#     model.cuda()
#     model.eval()
#     samples = [torch.randn(1, 100) for i in range(2)]
#     recons = [model.decode(sample.cuda()) for sample in samples]
#     recon_images = [r.squeeze(0).permute(1, 2, 0).detach().cpu() for r in recons]
#
#     train_dataset = CIFAR10(root='./cifar10',
#                             train=True,
#                             transform=t.Compose([
#                                 t.ToTensor()
#                             ]),
#                             download=False)
#     data_loader = DataLoader(dataset=train_dataset,
#                              batch_size=4,
#                              num_workers=4,
#                              shuffle=True,
#                              pin_memory=torch.cuda.is_available())
#     images, _ = iter(data_loader).next()
#     recon_images2_, _, _ = model(images.cuda())
#     recon_images2 = [r.squeeze(0).permute(1, 2, 0).detach().cpu() for r in recon_images2_]
#
#     images = images.detach().cpu().numpy()
#     images = images.transpose(0, 2, 3, 1)
#
#     plt.subplot(5, 2, 1)
#     plt.imshow(recon_images[0])
#     plt.subplot(5, 2, 2)
#     plt.imshow(recon_images[1])
#
#     plt.subplot(5, 2, 3)
#     plt.imshow(images[0])
#     plt.subplot(5, 2, 4)
#     plt.imshow(recon_images2[0])
#
#     plt.subplot(5, 2, 5)
#     plt.imshow(images[1])
#     plt.subplot(5, 2, 6)
#     plt.imshow(recon_images2[1])
#
#     plt.subplot(5, 2, 7)
#     plt.imshow(images[2])
#     plt.subplot(5, 2, 8)
#     plt.imshow(recon_images2[2])
#
#     plt.subplot(5, 2, 9)
#     plt.imshow(images[3])
#     plt.subplot(5, 2, 10)
#     plt.imshow(recon_images2[3])
#
#     plt.show()
