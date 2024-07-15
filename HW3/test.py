import numpy as np
import torch
import torchvision

dataset = torchvision.datasets.CIFAR10(root='./cifar10/', train=True, download=True)
labels = np.array(dataset.targets)
# print(labels)
mask = labels == dataset.classes.index('cat')
mask[labels == dataset.classes.index('dog')] = True
images = dataset.data[mask]
labels = np.array(dataset.targets)[mask]
print(labels)
# print(np.all(mask))