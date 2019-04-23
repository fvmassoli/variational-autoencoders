import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class PerceptualModules(object):
    def __init__(self, device, dataset_type, verbose,
                 layers=[[0, 6], [6, 13], [13, 20], [20, 27], [27, 33], [33, 43]]):
        self.device = device
        self.dataset_type = dataset_type
        self.verbose = verbose
        self.layers = layers
        self.perceptual_criterion = nn.MSELoss()
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
        if self.verbose > 1:
            print("="*20, "Model info", "="*19)
            for i in range(len(self.modules)):
                print("\t Module: {}\n"
                      "\t Module structure: {}".format(i, self.modules[i]))
            print("="*51)

    def forward(self, x):
        print('before', x.shape)
        if self.dataset_type == 'mnist':
            x = x.repeat(1, 3, 1, 1)
        print(x.shape)
        o1 = self.modules[0](x)
        o2 = self.modules[1](o1)
        o3 = self.modules[2](o2)
        o4 = self.modules[3](o3)
        o5 = self.modules[4](o4)
        return o1, o2, o3, o4, o5

    def get_loss(self, recon_x, x):
        l_recon = self.forward(recon_x)
        if self.dataset_type == 'mnist':
            pad = (2, 2, 2, 2)
            x = F.pad(input=x, pad=pad, mode='constant', value=0)
        l_orig = self.forward(x)
        loss = sum([self.perceptual_criterion(l_recon[i], l_orig[i]) for i in range(len(l_recon))])
        return loss
