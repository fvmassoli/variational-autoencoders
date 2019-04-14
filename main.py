import torch

from utils import get_args
from models.model_manager import ModelManager
from data_manager.data_manager import DataManager


def main(args):

    torch.manual_seed(args.randomSeed)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(args.randomSeed)

    data_manager = DataManager(dataset_type=args.datasetType,
                               download=args.download,
                               batch_size=args.batchSize,
                               cuda=cuda)
    train_loader, valid_loader = data_manager.get_data_loaders()

    model_manager = ModelManager(conditional=args.CVAE,
                                 num_labels=len(train_loader.dataset.classes))
    model = model_manager.get_model()

#     run_manager = RunManager(model=model,
#                              train_loader=train_loader,
#                              valid_loader=valid_loader,
#                              optimizer=args.optimizer,
#                              lr=args.learningRate)
#     run_manager.print_run_info()
#     run_manager.run()
#
#
#
#
#     model = VAE()
#     if args.printModel:
#         print(model)
#     if args.training:
#         train_vae(model)
#     else:
#         check(model)
#
#
#
#
#
#
#
#
#
#
#
# def train_vae(model):
#     batch_size = 128
#     train_dataset = CIFAR10(root='./cifar10',
#                             train=True,
#                             transform=t.Compose([
#                                 t.ToTensor()
#                             ]),
#                             download=False)
#     data_loader = DataLoader(dataset=train_dataset,
#                              batch_size=batch_size,
#                              num_workers=4,
#                              shuffle=True,
#                              pin_memory=torch.cuda.is_available())
#
#     cuda = torch.cuda.is_available()
#     if cuda:
#         model.cuda()
#
#     adam = Adam(model.parameters(), lr=1.e-3)
#     sgd = SGD(model.parameters(), lr=1.e-2, momentum=0.1, weight_decay=1.e-4)
#
#     epochs = 80
#     for e in range(epochs):
#         train(model=model, optimizer=adam, data_loader=data_loader,
#               loss_fn=loss_fn, cuda=cuda, e=e, epochs=epochs, batch_size=batch_size)
#     torch.save(model.state_dict(), 'vae.torch')
#
#
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


if __name__ == '__main__':
    args = get_args()
    main(args=args)

    # TODO: content loss
    # TODO: Gaussian Mixture Model as prior
