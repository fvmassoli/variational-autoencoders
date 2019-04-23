import torch.nn as nn
from torchvision import models


class PerceptualModules(object):
    def __init__(self, device, verbose, layers=[[0, 6], [6, 13], [13, 20], [20, 27], [27, 33], [33, 43]]):
        self.device = device
        self.verbose = verbose
        self.layers = layers
        self.modules = self._init_perceptual_modules()
        self._prin_perceptual_info()

    def _init_perceptual_modules(self):
        vgg16_bn = models.vgg16_bn(pretrained=True).features
        modules = []
        for l in self.layers:
            m = nn.Sequential(*list(vgg16_bn.children())[l[0]:l[1]])
            m.eval()
            m.to(device=self.device)
            modules.append(m)
        return modules

    def _prin_perceptual_info(self):
        if self.verbose > 0:
            print("="*20, "Model info", "="*19)
            for i in range(len(self.modules)):
                print("\t Module: {}\n"
                      "\t Module structure: {}".format(i, self.modules[i]))
            print("="*51)

    def get_loss(self, recon_x, x):
        return 0
