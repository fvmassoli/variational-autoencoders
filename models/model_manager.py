from models.vae import VAE


class ModelManager(object):
    def __init__(self, conditional, num_labels):
        self._model = None
        # self._build_model(conditional, num_labels)
        self._print_model_info(conditional)

    def _build_model(self, conditional, num_labels):
        self._model = VAE(conditional, num_labels)

    def _print_model_info(self, conditional):
        print("="*20, "Model info", "="*19)
        print("\t Is model conditional VAE: {}".format(conditional))
        print("\t Model architecture: ")
        print(self._model)
        print("="*51)

    def get_model(self):
        return self._model
