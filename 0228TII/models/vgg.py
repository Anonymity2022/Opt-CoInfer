"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn


i=0
z_pruning_num=64

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, partition_id=0, quantization=32, num_class=196,z_architecture_list=None):
        super().__init__()
        self.features = features
        self.quantization=int(pow(2,quantization)-1)
        self.partition_id=partition_id
        if partition_id!=42:
            self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,196)
            )
        else:
            self.classifier = nn.Sequential(
            nn.Linear(int(25088*z_architecture_list[-2]/512), 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,196)
            )

    def forward(self, x):
        output = self.features[:self.partition_id](x)
        _output=output.detach()
        for _ in range(_output.shape[1]):
            _max=torch.max(_output[:,_])
            _min=torch.min(_output[:,_])
            z_d=_max-_min
            _output[:,_]=_min+torch.round((output[:,_]-_min)*self.quantization/(z_d))*((z_d)/self.quantization)
        output.data=_output
        output = self.features[self.partition_id:](output)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    z_tmp_num=0
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            z_tmp_num+=1
            continue
        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]
        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        z_tmp_num+=1
        input_channel = l

    return nn.Sequential(*layers)

def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))

def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))

def vgg16_bn(z_architecture_list,z_partition_id,z_quantization):
    return VGG(make_layers(z_architecture_list, batch_norm=True),z_partition_id,quantization=z_quantization,z_architecture_list=z_architecture_list)

def vgg19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))


