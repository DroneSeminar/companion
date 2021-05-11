import os
import h5py
import numpy as np
import torch.utils.data as data
import data.transforms as transforms

def h5Loader(path):
    h5File = h5py.File(path, "r")
    image = np.array(h5File["rgb"])
    image = np.transpose(image, (1, 2, 0))
    depth = np.array(h5File["depth"])
    return image, depth

class BaseLoader(data.Dataset):
    modalityNames = ["rgb"]
    colorJitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    def isImageFile(self, fileName):
        VALID_IMAGE_EXTENSIONS = [".h5"]
        return any(fileName.endswith(ext) for ext in VALID_IMAGE_EXTENSIONS)

    def findClasses(self, directory):
        classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        classes.sort()
        indices = {classes[i]: i for i in range(len(classes))}
        return classes, indices

    def constructDataset(self, directory, indices):
        images = []
        directory = os.path.expanduser(directory)
        for target in sorted(os.listdir(directory)):
            d = os.path.join(directory, target)
            if not os.path.isdir(d):
                continue

            for root, _, fileNames in sorted(os.walk(d)):
                for fileName in sorted(fileNames):
                    if self.isImageFile(fileName):
                        path = os.path.join(root, fileName)
                        item = (path, indices[target])
                        images.append(item)
        return images

    def __init__(self, root, split, modality="rgb", loader=h5Loader):
        classes, indices = self.findClasses(root)
        images = self.constructDataset(root, indices)
        assert len(images) > 0, "Found 0 images in subfolders of: " + root + "\n"
        print("Found {} images in {} folder.".format(len(images), split))
        self.rootDirectory = root
        self.images = images
        self.classes = classes
        self.indices = indices    
        if split == "train":
            self.transform = self.trainTransform
        elif split == "holdout":
            self.transform = self.validationTransform
        elif split == "validate":
            self.transform = self.validationTransform
        else:
            raise (RuntimeError("Invalid dataset split: " + split + "\n"
                                "Supported dataset splits are: train, validate"))
        self.dataLoader = loader
        assert (modality in self.modality_names), "Invalid modality split: " + modality + "\n" + \
                                "Supported dataset splits are: " + ''.join(self.modality_names)
        self.modality = modality

    def trainTransform(self, image, depth):
        raise (RuntimeError("BaseLoader.trainTransform() is not implemented."))
    
    def validationTransform(self, image, depth):
        raise (RuntimeError("BaseLoader.validationTransform() is not implemented."))

    def __getraw__(self, index):
        path, _ = self.imgs[index]
        image, depth = self.loader(path)
        return image, depth

    def __getitem__(self, index):
        image, depth = self.__getraw__(index)
        if self.transform is not None:
            npImage, npDepth = self.transform(image, depth)
        else:
            raise (RuntimeError("Transform not defined!"))
        
        if self.modality == "rgb":
            npInput = npImage
        
        transformTensor = transforms.ToTensor()
        inputTensor = transformTensor(npInput)

        while inputTensor.dim() < 3:
            inputTensor = inputTensor.unsqueeze(0)
        depthTensor = transformTensor(npDepth)
        depthTensor = depthTensor.unsqueeze(0)

        return inputTensor, depthTensor

    def __len__(self):
        return len(self.images)