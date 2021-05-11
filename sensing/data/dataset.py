import numpy as np
import data.transforms as transforms

from data.loader import BaseLoader

RAW_WIDTH, RAW_HEIGHT = 640, 480
OUT_WIDTH, OUT_HEIGHT = 224, 224

class Dataset(BaseLoader):
    def __init__(self, root, split, modality):
        self.split = split
        super(BaseLoader, self).__init__(root, split, modality=modality)
        self.outputSize = (OUT_WIDTH, OUT_HEIGHT)
    
    def isImageFile(self, fileName):
        if self.split == "train":
            return (fileName.endswith(".h5") and "00001.h5" not in fileName and "00201.h5" not in fileName)
        elif self.split == "holdout":
            return ('00001.h5' in fileName or '00201.h5' in fileName)
        elif self.split == "validation":
            return (fileName.endswith(".h5"))
        else:
            raise (RuntimeError("Invalid dataset split: " + self.split + "\nSupported dataset splits are: train, validation, holdout"))
        
    def trainTransform(self, image, depth):
        randomScale = np.random.uniform(1.0, 1.5)
        npDepth = depth / randomScale
        randomAngle = np.random.uniform(-5.0, 5.0)
        randomFlip = np.random.uniform(0.0, 1.0)

        # first step of data augmentation
        transform = transforms.Compose([
            transforms.Resize(250.0 / RAW_HEIGHT),
            transforms.Rotate(randomAngle),
            transforms.Resize(randomScale),
            transforms.CenterCrop((OUT_HEIGHT, OUT_WIDTH)),
            transforms.HorizontalFlip(randomFlip)
        ])

        npImage = transform(image)
        npImage = self.colorJitter(npImage)
        npImage = np.asfarray(npImage, dtype='float') / 255
        npDepth = transform(npDepth)

        return npImage, npDepth
    
    def validationTransform(self, image, depth):
        npDepth = depth
        transform = transforms.Compose([
            transforms.Resize(250.0 / RAW_HEIGHT),
            transforms.CenterCrop((228, 304)),
            transforms.Resize(self.outputSize)
        ])
        npImage = transform(image)
        npImage = np.asfarray(npImage, dtype='float') / 255
        npDepth = transform(npDepth)

        return npImage, npDepth