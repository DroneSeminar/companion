import torch
import math
import random
import numbers
import types
import collections
import warnings
import numpy as np

import scipy.ndimage.interpolation as interpolation
import scipy.misc as misc

from __future__ import division
from PIL import Image, ImageOps, ImageEnhance
from torch._C import float32

def _isNumpyImage(image):
    return isinstance(image, np.ndarray) and (image.ndim in {2, 3})

def _isPillowImage(image):
    return isinstance(image, Image.Image)

def _isImageTensor(image):
    return torch.is_tensor(image) and image.ndimension() == 3

def adjustBrightness(image, factor):
    if not _isPillowImage(image):
        raise (TypeError("input image should be PIL image. Input was {}".format(type(image))))
    
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(factor)
    return image

def adjustContrast(image, factor):
    if not _isPillowImage(image):
        raise (TypeError("input image should be PIL image. Input was {}".format(type(image))))
    
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(factor)
    return image

def adjustSaturation(image, factor):
    if not _isPillowImage(image):
        raise (TypeError("input image should be PIL image. Input was {}".format(type(image))))
    
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(factor)
    return image

# https://en.wikipedia.org/wiki/Hue
def adjustHue(image, factor):
    if not (-0.5 <= factor <= 0.5):
         raise ValueError('factor is not in range(-0.5, 0.5).'.format(factor))
    
    if not _isPillowImage(image):
        raise (TypeError("input image should be PIL image. Input was {}".format(type(image))))
    
    inputMode = image.mode
    if inputMode in {"L", "1", "I", "F"}:
        return image
    
    h, s, v = image.convert("HSV").split()
    npH = np.array(h, dtype=np.uint8)
    with np.errstate(over="ignore"):
        npH += np.uint8(factor * 255)
    h = Image.fromarray(npH, "L")
    
    image = Image.merge("HSV", (h, s, v)).convert(inputMode)
    return image

# https://en.wikipedia.org/wiki/Gamma_correction
def adjustGamma(image, gamma, gain=1):
    if not _isPillowImage(image):
        raise (TypeError("input image should be PIL image. Input was {}".format(type(image))))

    if gamma < 0:
        raise (ValueError("Gamma should be a non-negative real number"))
    
    inputMode = image.mode
    image = image.convert("RGB")

    npImage = np.array(image, dtype=np.float32)
    npImage = 255 * gain * ((npImage / 255) ** gamma)
    npImage = np.uint8(np.clip(npImage, 0, 255))

    image = Image.fromarray(npImage, "RGB").convert(inputMode)
    return image 

class Compose(object):
    """Composes several transforms together."""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image):
        for transform in self.transforms:
            image = transform(image)
        return image

class ToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
    
    def __call__(self, image):
        if not _isNumpyImage(image):
            raise (TypeError('image should be ndarray. Got {}'.format(type(image))))
        
        if isinstance(image, np.ndarray):
            if image.ndim == 3:
                image = torch.from_numpy(image.transpose((2, 0, 1)).copy())
            elif image.ndim == 2:
                image = torch.from_numpy(image.copy())
            else:
               raise (RuntimeError('image should be ndarray with 2 or 3 dimensions. Got {}'.format(image.ndim))) 
        
        return image.float()

class NormalizeNumpyArray(object):
    """Normalize a ``numpy.ndarray`` with mean and standard deviation."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, image):
        if not _isNumpyImage(image):
            raise (TypeError('image should be ndarray. Got {}'.format(type(image))))

        for i in range(3):
            image[:, :, i] = (image[:, :, i] - self.mean[i] / self.std[i])
        return image

class NormalizeTensor(object):
    """Normalize an tensor image with mean and standard deviation."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        if not _isImageTensor(tensor):
            raise (TypeError('tensor is not a torch image.'))
       
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

class Rotate(object):
    """Rotates the given ``numpy.ndarray``."""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, image):
        # order=0 means nearest-neighbor type interpolation
        return interpolation.rotate(image, self.angle, reshape=False, prefilter=False, order=0)

class Resize(object):
    """Resize the the given ``numpy.ndarray`` to the given size."""

    def __init__(self, size, interpolation='nearest'):
        assert isinstance(size, int) or isinstance(size, float) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image):
        if image.ndim == 3:
            return misc.imresize(image, self.size, self.interpolation)
        elif image.ndim == 2:
            return misc.imresize(image, self.size, self.interpolation, 'F')
        else:
            raise (RuntimeError('image should be ndarray with 2 or 3 dimensions. Got {}'.format(image.ndim)))

class CenterCrop(object):
    """Crops the given ``numpy.ndarray`` at the center."""

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        h = img.shape[0]
        w = img.shape[1]
        th, tw = output_size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return i, j, th, tw

    def __call__(self, image):
        i, j, h, w = self.get_params(image, self.size)
        """
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        """
        if not _isNumpyImage(image):
            raise (TypeError('image should be ndarray. Got {}'.format(type(image))))

        if image.ndim == 3:
            return image[i:i+h, j:j+w, :]
        elif image.ndim == 2:
            return image[i:i + h, j:j + w]
        else:
            raise (RuntimeError('image should be ndarray with 2 or 3 dimensions. Got {}'.format(image.ndim)))

class BottomCrop(object):
    """Crops the given ``numpy.ndarray`` at the bottom."""

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):

        h = img.shape[0]
        w = img.shape[1]
        th, tw = output_size
        i = h - th
        j = int(round((w - tw) / 2.))

        return i, j, th, tw

    def __call__(self, image):
        i, j, h, w = self.get_params(image, self.size)
        """
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        """
        if not _isNumpyImage(image):
            raise (TypeError('image should be ndarray. Got {}'.format(type(image))))

        if image.ndim == 3:
            return image[i:i+h, j:j+w, :]
        elif image.ndim == 2:
            return image[i:i + h, j:j + w]
        else:
            raise (RuntimeError('image should be ndarray with 2 or 3 dimensions. Got {}'.format(image.ndim)))

class Lambda(object):
    """Apply a user-defined lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, image):
        return self.lambd(image)

class HorizontalFlip(object):
    """Horizontally flip the given ``numpy.ndarray``."""

    def __init__(self, doFlip):
        self.flip = doFlip

    def __call__(self, image):
        if not _isNumpyImage(image):
            raise (TypeError('image should be ndarray. Got {}'.format(type(image))))

        if self.doFlip:
            return np.fliplr(image)
        else:
            return image

class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image."""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):

        transforms = []
        if brightness > 0:
            brightnessFactor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(Lambda(lambda image: adjustBrightness(image, brightnessFactor)))

        if contrast > 0:
            contrastFactor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(Lambda(lambda image: adjustContrast(image, contrastFactor)))

        if saturation > 0:
            saturationFactor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(Lambda(lambda image: adjustSaturation(image, saturationFactor)))

        if hue > 0:
            hueFactor = np.random.uniform(-hue, hue)
            transforms.append(Lambda(lambda image: adjustHue(image, hueFactor)))

        np.random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, image):
        if not _isNumpyImage(image):
            raise (TypeError('image should be ndarray. Got {}'.format(type(image))))

        pil = Image.fromarray(image)
        transform = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        return np.array(transform(pil))

class Crop(object):
    """
    Crops the given PIL Image to a rectangular region based on a given
    4-tuple defining the left, upper pixel coordinated, hight and width size.
    """

    def __init__(self, i, j, h, w):
        """
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        """
        self.i = i
        self.j = j
        self.h = h
        self.w = w

    def __call__(self, image):
        i, j, h, w = self.i, self.j, self.h, self.w

        if not _isNumpyImage(image):
            raise (TypeError('image should be ndarray. Got {}'.format(type(image))))
        if image.ndim == 3:
            return image[i:i + h, j:j + w, :]
        elif image.ndim == 2:
            return image[i:i + h, j:j + w]
        else:
            raise (RuntimeError('image should be ndarray with 2 or 3 dimensions. Got {}'.format(image.ndim)))

    def __repr__(self):
        return self.__class__.__name__ + '(i={0},j={1},h={2},w={3})'.format(self.i, self.j, self.h, self.w)