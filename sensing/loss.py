import torch
import torch.nn as nn
import torch.nn.functional as F

from math import exp

def blurKernel(windowSize, sigma):
    gauss = torch.Tensor([exp(-(x - windowSize // 2) ** 2 / float(2 * sigma ** 2)) for x in range(windowSize)])
    return gauss/gauss.sum()

def createWindow(windowSize, channel=1):
    window1D = blurKernel(windowSize, 1.5).unsqueeze(1)
    window2D = window1D.mm(window1D.t()).float().unsqueeze(0).unsqueeze(0)
    window = window2D.expand(channel, 1, windowSize, windowSize).contiguous()
    return window

# https://en.wikipedia.org/wiki/Structural_similarity
def SSIM(true, pred, valRange, windowSize=11, window=None, sizeAverage=True, full=False):
    L = valRange

    padding = 0
    (_, channel, height, width) = true.size()
    if window is None:
        realSize = min(windowSize, height, width)
        window = createWindow(realSize, channel).to(true.device)

    mu1 = F.conv2d(true, window, padding=padding, groups=channel)
    mu2 = F.conv2d(pred, window, padding=padding, groups=channel)

    mu1sq = mu1.pow(2)
    mu2sq = mu2.pow(2)
    mu12 = mu1 * mu2

    sigma1sq = F.conv2d(true * true, window, padding=padding, groups=channel) - mu1sq
    sigma2sq = F.conv2d(pred * pred, window, padding=padding, groups=channel) - mu2sq
    sigma12 = F.conv2d(true * pred, window, padding=padding, groups=channel) - mu12

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1sq + sigma2sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    map = ((2 * mu12 + C1) * v1) / ((mu1sq + mu2sq + C1) * v2)

    if sizeAverage:
        returnValue = map.mean()
    else:
        returnValue = map.mean(1).mean(1).mean(1)

    if full:
        return returnValue, cs
    return returnValue