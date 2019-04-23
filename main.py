from utils import *
from log_manager.logger import Logger
from models.model_manager import ModelManager
from run_manager.run_manager import RunManager
from data_manager.data_manager import DataManager


def main(args):

    device, cuda = init_random_seeds(args.randomSeed, args.cuda, args.verbose)

    #######################################################################
    ########################## Init Logger ################################
    #######################################################################
    logger = Logger(conditional=args.cvae,
                    perceptual_loss=args.perceptualLoss)

    #######################################################################
    ############### Init data manager to handle data ######################
    #######################################################################
    data_manager = DataManager(dataset_type=args.datasetType,
                               batch_size=args.batchSize,
                               cuda=cuda,
                               verbose=args.verbose)

    #######################################################################
    ############### Init data manager to handle data ######################
    #######################################################################
    model_manager = ModelManager(conditional=args.cvae,
                                 perceptual_loss=args.perceptualLoss,
                                 num_labels=data_manager.get_num_labels(),
                                 checkpoint_path=args.checkpointPath,
                                 device=device,
                                 verbose=args.verbose)

    #######################################################################
    ############### Init data manager to handle data ######################
    #######################################################################
    run_manager = RunManager(model_manager=model_manager,
                             data_manager=data_manager,
                             logger=logger,
                             optimizer_type=args.optimizer,
                             lr=args.learningRate,
                             epochs=args.epochs,
                             device=device,
                             verbose=args.verbose)
    # Run
    run_manager.run(args.training)


if __name__ == '__main__':
    args = get_args()
    main(args=args)

    # TODO: Gaussian Mixture Model as prior
