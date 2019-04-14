import torch
from models.vae import VAE


class ModelManager(object):
    def __init__(self, conditional, num_labels, load_checkpoint, checkpoint_path):
        self._model = None
        self._load_checkpoint = load_checkpoint
        self._checkpoint_path = checkpoint_path
        # self._build_model(conditional, num_labels)
        self._print_model_info(conditional)

    def _build_model(self, conditional, num_labels):
        self._model = VAE(conditional, num_labels)
        if self._load_checkpoint:
            self._model.load_state_dict(torch.load(self._checkpoint_path))

    def _print_model_info(self, conditional):
        print("="*20, "Model info", "="*19)
        print("\t Is model conditional VAE: {}".format(conditional))
        print("\t Model architecture: ")
        print(self._model)
        print("="*51)

    def get_model(self):
        return self._model
