import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt

def pointwiseConvolution(inputChannels, outputChannels):
    return nn.Sequential(
        nn.Conv2d(inputChannels, outputChannels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(outputChannels),
        HardSwishActivation(inplace=True)
    )

def depthwiseConvolution(inputChannels, kernelSize):
    padding = (kernelSize - 1) // 2
    assert 2 * padding == kernelSize - 1, "Given parameters where incorrect. kernel = {}, padding = {}".format(kernelSize, padding)
    return nn.Sequential(
        nn.Conv2d(inputChannels, inputChannels, kernelSize, 1, padding, bias=False, groups=inputChannels),
        nn.BatchNorm2d(inputChannels),
        HardSwishActivation(inplace=True)
    )

def initWeights(module):  
    if isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    else:
        if isinstance(module, nn.Conv2d):
            n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        elif isinstance(module, nn.ConvTranspose2d):
            n = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
        module.weight.data.normal_(0, sqrt(2. / n))
        if module.bias is not None:
            module.bias.data.zero_()

class HardSigmoidActivation(nn.Module):
    def __init__(self, inplace=True):
        super(HardSigmoidActivation, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class HardSwishActivation(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwishActivation, self).__init__()
        self.sigmoid = HardSigmoidActivation(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()

        def relu(relu6):
            if relu6:
                return nn.RELU6(inplace=True)
            else:
                return nn.ReLU(inplace=True)

        def conv_3x3_bn(inputChannels, outputChannels, stride, relu6):
            return nn.Sequential(
                nn.Conv2d(inputChannels, outputChannels, 3, stride, 1, bias=False),
                nn.BatchNorm2d(outputChannels),
                relu(relu6)
            )
        
        def conv_3x3_dw(inputChannels, outputChannels, stride, relu6):
            return nn.Sequential(
                nn.Conv2d(inputChannels, inputChannels, 3, stride, 1, groups=inputChannels, bias=False),
                nn.BatchNorm2d(inputChannels),
                relu(relu6),

                nn.Conv2d(inputChannels, outputChannels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(outputChannels),
                relu(relu6)
            )

        self.model = nn.Sequential(
            conv_3x3_bn(   3,   32, 2),
            conv_3x3_dw(  32,   64, 1),
            conv_3x3_dw(  64,  128, 2),
            conv_3x3_dw( 128,  128, 1),
            conv_3x3_dw( 128,  256, 2),
            conv_3x3_dw( 256,  256, 1),
            conv_3x3_dw( 256,  512, 2),
            conv_3x3_dw( 512,  512, 1),
            conv_3x3_dw( 512,  512, 1),
            conv_3x3_dw( 512,  512, 1),
            conv_3x3_dw( 512,  512, 1),
            conv_3x3_dw( 512,  512, 1),
            conv_3x3_dw( 512, 1024, 2),
            conv_3x3_dw(1024, 1024, 1),
            nn.AvgPool2d(7)
        )
        self.fullyConnected = nn.Linear(1024, 1000)
    
    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fullyConnected(x)
        return x

class DNET(nn.Module):
    def __init__(self, outputSize, usePretrained=False):
        super(DNET, self).__init__()

        self.outputSize = outputSize
        basenet = Base()

        if usePretrained:
            pretrainedPath = "" # TODO: add this later
            pretrained = torch.load(pretrainedPath)
            oldStateDict = pretrained['state_dict']

            from collections import OrderedDict
            newStateDict = OrderedDict()
            for key, value in oldStateDict.items():
                name = key[7:]
                newStateDict[name] = value
            basenet.load_state_dict(newStateDict)
        else:
            basenet.apply(initWeights)

        for idx in range(14):
            setattr(self, 'convolutional{}'.format(idx), basenet.model[idx])

        kernelSize = 5

        self.decodeConvolutional1 = nn.Sequential(
            depthwiseConvolution(1024, kernelSize),
            pointwiseConvolution(1024, 512)
        )
        self.decodeConvolutional2 = nn.Sequential(
            depthwiseConvolution(512, kernelSize),
            pointwiseConvolution(512, 256)
        )
        self.decodeConvolutional3 = nn.Sequential(
            depthwiseConvolution(256, kernelSize),
            pointwiseConvolution(256, 128)
        )
        self.decodeConvolutional4 = nn.Sequential(
            depthwiseConvolution(128, kernelSize),
            pointwiseConvolution(128, 64)
        )
        self.decodeConvolutional5 = nn.Sequential(
            depthwiseConvolution(64, kernelSize),
            pointwiseConvolution(64, 32)
        )
        self.decodeConvolutional6 = pointwiseConvolution(32, 1)

        initWeights(self.decodeConvolutional1)
        initWeights(self.decodeConvolutional2)
        initWeights(self.decodeConvolutional3)
        initWeights(self.decodeConvolutional4)
        initWeights(self.decodeConvolutional5)
        initWeights(self.decodeConvolutional6)

    def forward(self, x):
        for idx in range(14):
            layer = getattr(self, 'convolutional{}'.format(idx))
            x = layer(x)

            if idx == 1:
                x1 = x
            elif idx == 3:
                x2 = x
            elif idx == 5:
                x3 = x

        for idx in range(1, 6):
            layer = getattr(self, 'decodeConvolutional{}'.format(idx))
            x = layer(x)
            x = F.interpolate(x, mode='nearest', scale_factor=2)

            if idx == 4:
                x = x + x1
            elif idx == 3:
                x = x + x2
            elif idx == 2:
                x = x + x3
        
        x = self.decodeConvolutional6(x)
        return x