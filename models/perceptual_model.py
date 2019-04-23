import torch.nn as nn
from torchvision import models


class PerceptualModules(object):
    def __init__(self, cuda, verbose, layers=[[0, 6], [6, 13], [13, 20], [20, 27], [27, 33], [33, 43]]):
        self.cuda = cuda
        self.verbose = verbose
        self.layers = layers
        self.modules = self._init_perceptual_modules()
        self.print_modules_info()

    def _init_perceptual_modules(self):
        vgg16_bn = models.vgg16_bn(pretrained=True).features
        modules = []
        for l in self.layers:
            m = nn.Sequential(*list(vgg16_bn.children())[l[0]:l[1]])
            m.eval()
            if self.cuda:
                m.cuda()
            modules.append(m)
        return modules

    def get_modules(self):
        return self.modules

    def print_modules_info(self):
        pass
