import torch

import argparse


def init_random_seeds(random_seed, cuda):
    torch.manual_seed(random_seed)
    if cuda:
        torch.cuda.manual_seed(random_seed)


def get_args():
    parser = argparse.ArgumentParser('VAE')

    parser.add_argument('-t', '--training', action='store_true', help='Start model training (default: False)')
    parser.add_argument('-c', '--CVAE', action='store_true', help='Train CVAE (default: False)')
    parser.add_argument('-cktl', '--loadCheckpoint', action='store_true', help='Load model checkpoint (default: False)')
    parser.add_argument('-cktp', '--checkpointPath', help='Checkpoint file path')
    parser.add_argument('-dt', '--datasetType', choices=['mnist', 'cifar10'], default='cifar10',
                        help='Select training dataset')
    parser.add_argument('-d', '--download', action='store_true', help='Download dataset (default: False)')
    parser.add_argument('-e', '--epochs', default=1, type=int, help='Number of training epochs')
    parser.add_argument('-lr', '--learningRate', default=0.001, help='Learning rate')
    parser.add_argument('-o', '--optimizer', choices=['adam', 'sgd'], default='adam', help='Optimizer type')
    parser.add_argument('-bs', '--batchSize', default=128, help='Batch size')
    parser.add_argument('-s', '--randomSeed', default=41, help='Set random seed')

    return parser.parse_args()

