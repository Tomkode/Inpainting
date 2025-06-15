import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
from utils import show_tensor_image
import matplotlib.pyplot as plt
BATCH_SIZE = 8
IMG_SIZE = 256

data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ]
data_transform = transforms.Compose(data_transforms)
def load_transformed_dataset():
    image_dir = "E:\\ffhq256x256"
    dataset = datasets.ImageFolder(root=image_dir, transform=data_transform)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                             shuffle=True, drop_last=True)
    return trainloader

def load_cifar_dataset():
    trainset = datasets.CIFAR10(root='./cifar10', train=True,
                                        download=True, transform=data_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                             shuffle=True, drop_last=True)
    return trainloader
def load_specific_image(image_path):
    img = Image.open(image_path)
    return data_transform(img)
