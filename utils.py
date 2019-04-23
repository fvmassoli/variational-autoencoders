import torch

import argparse


def init_random_seeds(random_seed, cuda):
    torch.manual_seed(random_seed)
    if cuda:
        torch.cuda.manual_seed(random_seed)


def get_args():
    parser = argparse.ArgumentParser('VAE')

    parser.add_argument('-t', '--training', action='store_true', help='Start model training (default: False)')
    parser.add_argument('-cktp', '--checkpointPath', default=None, help='Checkpoint file path')
    parser.add_argument('-e', '--epochs', default=1, type=int, help='Number of training epochs')
    parser.add_argument('-lr', '--learningRate', default=0.001, help='Learning rate')
    parser.add_argument('-bs', '--batchSize', default=128, help='Batch size')

    return parser.parse_args()

