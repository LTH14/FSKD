import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


__all__ = ['vgg']

defaultcfg = {
    11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

class vgg(nn.Module):
    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None, save_feature=False, batch_norm=True):
        super(vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]
        self.cfg =cfg
        self.save_feature = save_feature

        self.feature = self.make_layers(cfg, batch_norm)

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        self.classifier = nn.Linear(cfg[-1], num_classes)
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def add_pwconv(self, batch_norm=True):
        layers = []
        layer_idx = 0
        self.pwconv = []
        for v in self.cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                layer_idx += 1
            else:
                if batch_norm:
                    one_pwconv = nn.Conv2d(v, v, kernel_size=1, stride=1, padding=0, bias=False).cuda()
                    one_pwconv.weight.data.copy_(torch.from_numpy(np.identity(v)).view(v, v, 1, 1))
                    self.pwconv += [one_pwconv]
                    layers += [self.feature[layer_idx], self.feature[layer_idx + 1], one_pwconv, nn.ReLU(inplace=True)]
                    layer_idx += 3
                else:
                    one_pwconv = nn.Conv2d(v, v, kernel_size=1, stride=1, padding=0, bias=False).cuda()
                    one_pwconv.weight.data.copy_(torch.from_numpy(np.identity(v)).view(v, v, 1, 1))
                    self.pwconv += [one_pwconv]
                    layers += [self.feature[layer_idx], one_pwconv, nn.ReLU(inplace=True)]
                    layer_idx += 2
        self.feature = nn.Sequential(*layers)

    def absorb_pwconv(self, batch_norm=True):
        layers = []
        layer_idx = 0
        pwconv_idx = 0
        for v in self.cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                layer_idx += 1
            else:
                if batch_norm:
                    pw = self.pwconv[pwconv_idx].weight.data
                    conv = self.feature[layer_idx]
                    weight = conv.weight.data
                    new_weight = torch.zeros_like(weight)
                    for i in range(3):
                        for j in range(3):
                            new_weight[:, :, i, j] = torch.mm(pw.squeeze(), weight[:, :, i, j].squeeze())
                    conv.weight.data.copy_(new_weight)

                    layers += [conv, self.feature[layer_idx + 1], nn.ReLU(inplace=True)]
                    pwconv_idx += 1
                    layer_idx += 4
                else:
                    pw = self.pwconv[pwconv_idx].weight.data
                    conv = self.feature[layer_idx]
                    weight = conv.weight.data
                    new_weight = torch.zeros_like(weight)
                    for i in range(3):
                        for j in range(3):
                            new_weight[:, :, i, j] = torch.mm(pw.squeeze(), weight[:, :, i, j].squeeze())
                    conv.weight.data.copy_(new_weight)

                    layers += [conv, nn.ReLU(inplace=True)]
                    pwconv_idx += 1
                    layer_idx += 3
        del self.pwconv
        self.feature = nn.Sequential(*layers)

    def forward(self, x):
        self.inter_feature = []
        for layer in self.feature:
            if isinstance(layer, nn.ReLU) and self.save_feature:
                self.inter_feature += [x.clone()]
            x = layer(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

if __name__ == '__main__':
    net = vgg()
    x = Variable(torch.FloatTensor(16, 3, 40, 40))
    y = net(x)
    print(y.data.shape)