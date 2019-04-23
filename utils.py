import torch

import argparse


def init_random_seeds(random_seed, cuda_, verbose):
    torch.manual_seed(random_seed)
    cuda = cuda_ and torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(random_seed)
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if verbose > 0:
        print("=" * 20, "Device info", "=" * 19)
        print("\t Device: {}\n"
              "\t CUDA:   {}".format(device, cuda))
        print("=" * 51)
    return device, cuda


def get_args():
    parser = argparse.ArgumentParser('VAE')

    parser.add_argument('-cvae', '--cvae', action='store_true', help='Use CVAE (default: False)')
    parser.add_argument('-pl', '--perceptualLoss', action='store_true', help='Use perceptual loss (default: False)')

    parser.add_argument('-t', '--training', action='store_true', help='Start model training (default: False)')

    parser.add_argument('-d', '--datasetType', choices=['mnist', 'cifar10'], default='cifar10',
                        help='Dataset (default: Cifar10)')

    parser.add_argument('-c', '--cuda', action='store_false', help='Move on cuda (default: True)')
    parser.add_argument('-s', '--randomSeed', type=int, default=41, help='Random seed (default: 41)')
    parser.add_argument('-o', '--optimizer', choices=['adam', 'sgd'], default='adam', help='Optimizer (default: Adam)')
    parser.add_argument('-cktp', '--checkpointPath', default=None, help='Checkpoint file path')
    parser.add_argument('-e', '--epochs', type=int, default=1, help='Number of training epochs (default: 1)')
    parser.add_argument('-lr', '--learningRate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('-bs', '--batchSize', type=int, default=256, help='Batch size (default: 256)')
    parser.add_argument('-v', '--verbose', type=int, choices=[0, 1, 2], default=1, help='Verbose level (default: 1)')

    return parser.parse_args()

