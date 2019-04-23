import os
import csv
import time

import torch


class Logger(object):
    def __init__(self, conditional, perceptual_loss):
        if conditional and perceptual_loss:
            self._current_time = 'cvae_perceptual_loss_'+time.strftime("%Y_%m_%d-%H.%M.%S")
        elif conditional:
            self._current_time = 'cvae_'+time.strftime("%Y_%m_%d-%H.%M.%S")
        elif perceptual_loss:
            self._current_time = 'perceptual_loss_' + time.strftime("%Y_%m_%d-%H.%M.%S")
        else:
            self._current_time = time.strftime("%Y_%m_%d-%H.%M.%S")
        self._train_main = './training_stats'

        self._train_sub = os.path.join(self._train_main, self._current_time)
        self._model_ckt_main = './model_ckt'
        self._model_ckt_sub = os.path.join(self._model_ckt_main, self._current_time)
        self._inference_main = './inference_results'
        self._inference_sub = os.path.join(self._inference_main, self._current_time)
        self._create_folders()
        self._create_statistics_csv_file()

    def _create_folders(self):
        if not os.path.isdir(self._train_main):
            os.makedirs(self._train_main)
        if not os.path.isdir(self._train_sub):
            os.makedirs(self._train_sub)
        if not os.path.isdir(self._model_ckt_main):
            os.makedirs(self._model_ckt_main)
        if not os.path.isdir(self._model_ckt_sub):
            os.makedirs(self._model_ckt_sub)
        if not os.path.isdir(self._inference_main):
            os.makedirs(self._inference_main)
        if not os.path.isdir(self._inference_sub):
            os.makedirs(self._inference_sub)

    def _create_statistics_csv_file(self):
        """
        Create empty statistics csv files

        """
        csv_columns = ['epoch', 'loss', 'bce', 'kld']
        training_stats_csv_file_name = 'training.csv'
        training_stats_csv_file_name = os.path.join(self._train_sub, training_stats_csv_file_name)
        training_stats_file = open(training_stats_csv_file_name, 'w')
        with training_stats_file:
            writer = csv.writer(training_stats_file)
            writer.writerow(csv_columns)
        self._file_name = training_stats_csv_file_name

    def get_stats_folders(self):
        return self._train_sub, self._model_ckt_sub, self._inference_sub

    def get_csv_file(self):
        return self._file_name

    def save_model(self, state_dict):
        p = os.path.join(self._model_ckt_sub, 'vae.torch')
        torch.save(state_dict, p)
        print("Model checkpoint saved at: {}".format(p))

    def save_stats_on_csv(self, epochs, epoch=0, loss=0, bce=0, kld=0):
        """
        Writes stats on the csv file

        """
        print("Epoch[{}/{}]\t==>\tTotal Loss: {:.3f} --- "
              "Reconstruction/Perceptual loss: {:.3f} --- "
              "Dkl: {:.3f}".format(epoch, epochs, loss, bce, kld))
        data = [epoch, loss, bce, kld]
        filename = open(self._file_name, 'a')
        with filename:
            writer = csv.writer(filename)
            writer.writerow(data)
