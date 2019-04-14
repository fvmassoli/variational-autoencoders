import os
import csv
import time
import argparse

import torch


current_time = time.strftime("%Y_%m_%d-%H.%M.%S")


def init_random_seeds(random_seed, cuda):
    torch.manual_seed(random_seed)
    if cuda:
        torch.cuda.manual_seed(random_seed)


def create_folders():
    train_main = './training_stats'
    train_sub = os.path.join(train_main, current_time)
    if not os.path.isdir(train_main):
        os.makedirs(train_main)
    if not os.path.isdir(train_sub):
        os.makedirs(train_sub)
    model_ckt_main = './model_ckt'
    model_ckt_sub = os.path.join(model_ckt_main, current_time)
    if not os.path.isdir(model_ckt_main):
        os.makedirs(model_ckt_main)
    if not os.path.isdir(model_ckt_sub):
        os.makedirs(model_ckt_sub)
    inference_main = './inference_results'
    inference_sub = os.path.join(inference_main, current_time)
    if not os.path.isdir(inference_main):
        os.makedirs(inference_main)
    if not os.path.isdir(inference_sub):
        os.makedirs(inference_sub)
    return train_sub, model_ckt_sub, inference_sub

# def create_statistics_csv_file(current_time):
#     """
#     Create empty statistics csv files
#
#     :param stats_folder_path: path to folder for the statistics files
#     :param training_stats_csv_file_name: training stats file name
#     :param validation_stats_csv_file_name: training stats file name
#     :param csv_columns: csv columns name
#     :return: None
#     """
#
#     csv_columns = ['loss', 'acc1', 'lr']
#     directory = os.path.join('stats_output', 'output_'+current_time)
#     if not os.path.isdir(directory):
#         os.makedirs(directory)
#
#     training_stats_csv_file_name = 'training.csv'
#     validation_stats_csv_file_name = 'validation.csv'
#     validation_stats_orig_csv_file_name = 'validation_orig.csv'
#
#     training_stats_csv_file_name = os.path.join(directory, training_stats_csv_file_name)
#     validation_stats_csv_file_name = os.path.join(directory, validation_stats_csv_file_name)
#     validation_stats_orig_csv_file_name = os.path.join(directory, validation_stats_orig_csv_file_name)
#
#     training_stats_file = open(training_stats_csv_file_name, 'w')
#     validation_stats_file = open(validation_stats_csv_file_name, 'w')
#     validation_orig_stats_file = open(validation_stats_orig_csv_file_name, 'w')
#
#     with training_stats_file:
#         writer = csv.writer(training_stats_file)
#         writer.writerow(csv_columns)
#
#     with validation_stats_file:
#         writer = csv.writer(validation_stats_file)
#         writer.writerow(csv_columns)
#
#     with validation_orig_stats_file:
#         writer = csv.writer(validation_orig_stats_file)
#         writer.writerow(csv_columns)
#
#     print("Created files: \n{}\n{}\n{}".format(training_stats_csv_file_name, validation_stats_csv_file_name, validation_stats_orig_csv_file_name))
#
#     return training_stats_csv_file_name, validation_stats_csv_file_name, validation_stats_orig_csv_file_name
#
#
# def save_stats_on_csv(file_name, loss=0, prec_1=0, lr=0):
#     """
#     Writes stats on the csv file
#
#     :param file_name: csv file full path
#     :param loss: average value of the loss
#     :param prec_1: average value of the top 1 accuracy
#     :param prec_5: average value of the top 5 accuracy
#     :return: None
#     """
#     data = [loss, prec_1, lr]
#     filename = open(file_name, 'a')
#     with filename:
#         writer = csv.writer(filename)
#         writer.writerow(data)


def get_args():
    parser = argparse.ArgumentParser('VAE')

    parser.add_argument('-t', '--training', action='store_true', help='Start model training (default: False)')
    parser.add_argument('-c', '--CVAE', action='store_true', help='Train CVAE (default: False)')
    parser.add_argument('-cktl', '--loadCheckpoint', action='store_true', help='Load model checkpoint (default: False)')
    parser.add_argument('-cktp', '--checkpointPath', help='Checkpoint file path')
    parser.add_argument('-dt', '--datasetType', choices=['mnist', 'cifar10'], default='cifar10',
                        help='Select training dataset')
    parser.add_argument('-d', '--download', action='store_true', help='Download dataset (default: False)')
    parser.add_argument('-e', '--epochs', default=1, help='Number of training epochs')
    parser.add_argument('-lr', '--learningRate', default=0.001, help='Learning rate')
    parser.add_argument('-o', '--optimizer', choices=['adam', 'sgd'], default='adam', help='Optimizer type')
    parser.add_argument('-bs', '--batchSize', default=128, help='Batch size')
    parser.add_argument('-s', '--randomSeed', default=41, help='Set random seed')

    return parser.parse_args()

