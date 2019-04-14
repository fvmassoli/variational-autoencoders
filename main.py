import torch

from utils import get_args
from models.model_manager import ModelManager
from run_manager.run_manager import RunManager
from data_manager.data_manager import DataManager


def main(args):

    torch.manual_seed(args.randomSeed)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(args.randomSeed)

    #######################################################################
    ############### Init data manager to handle data ######################
    #######################################################################
    data_manager = DataManager(dataset_type=args.datasetType,
                               download=args.download,
                               batch_size=args.batchSize,
                               cuda=cuda)
    # Get loaders
    train_loader, valid_loader = data_manager.get_data_loaders()

    #######################################################################
    ############### Init data manager to handle data ######################
    #######################################################################
    model_manager = ModelManager(conditional=args.CVAE,
                                 num_labels=len(train_loader.dataset.classes))
    # Get model
    model = model_manager.get_model()

    #######################################################################
    ############### Init data manager to handle data ######################
    #######################################################################
    run_manager = RunManager(model=model,
                             conditional=args.CVAE,
                             train_loader=train_loader,
                             valid_loader=valid_loader,
                             optimizer_type=args.optimizer,
                             lr=args.learningRate,
                             epochs=args.epochs,
                             cuda=cuda)
    # Run
    run_manager.run(args.training)


if __name__ == '__main__':
    args = get_args()
    main(args=args)

    # TODO: content loss
    # TODO: Gaussian Mixture Model as prior
