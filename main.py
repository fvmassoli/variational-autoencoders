import json
from utils import *
from log_manager.logger import Logger
from models.model_manager import ModelManager
from run_manager.run_manager import RunManager
from data_manager.data_manager import DataManager


def main(args):

    with open('./config.json') as f:
        d = json.load(f)

    cuda = torch.cuda.is_available() and d["cuda"]
    init_random_seeds(d["random_seed"], cuda)

    #######################################################################
    ########################## Init Logger ################################
    #######################################################################
    logger = Logger(conditional=d["cvae"],
                    perceptual_loss=d["perceptual_loss"])

    #######################################################################
    ############### Init data manager to handle data ######################
    #######################################################################
    data_manager = DataManager(dataset_type=d["dataset_type"],
                               batch_size=args.batchSize,
                               cuda=cuda,
                               verbose=d['verbose'])

    #######################################################################
    ############### Init data manager to handle data ######################
    #######################################################################
    model_manager = ModelManager(conditional=d["cvae"],
                                 perceptual_loss=d["perceptual_loss"],
                                 num_labels=data_manager.get_num_labels(),
                                 checkpoint_path=args.checkpointPath,
                                 cuda=cuda,
                                 verbose=d['verbose'])

    #######################################################################
    ############### Init data manager to handle data ######################
    #######################################################################
    run_manager = RunManager(model_manager=model_manager,
                             data_manager=data_manager,
                             logger=logger,
                             optimizer_type=d["optimizer"],
                             lr=args.learningRate,
                             epochs=args.epochs,
                             cuda=cuda,
                             verbose=d["verbose"])
    # Run
    run_manager.run(args.training)


if __name__ == '__main__':
    args = get_args()
    main(args=args)

    # TODO: Gaussian Mixture Model as prior
