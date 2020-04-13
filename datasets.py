import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

import numpy as np

import os
from PIL import Image
from sklearn.datasets import fetch_olivetti_faces
    

class Datasets(object):
    def __init__(self, dataset_name, batch_size = 100, num_workers = 2, transform = None, shuffle = True):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.shuffle = shuffle
        
    def create(self, path = None):
        print("Dataset :",self.dataset_name)
        if self.transform is None:
                self.transform = transforms.Compose([transforms.ToTensor()])
        
        
        if path is None:
            path = "./"+self.dataset_name+"Dataset/data"
        
        
        if self.dataset_name == "MNIST":
            trainset = torchvision.datasets.MNIST(root = path,
                                       train = True, download = True, transform = self.transform)
            testset = torchvision.datasets.MNIST(root = path,
                                                 train = False, download = True, transform = self.transform)
            classes = list(range(10))
            base_labels = trainset.classes
            
        elif self.dataset_name == "FashionMNIST":
            trainset = torchvision.datasets.FashionMNIST(root = path,
                                       train = True, download = True, transform = self.transform)
            testset = torchvision.datasets.FashionMNIST(root = path,
                                                 train = False, download = True, transform = self.transform)
            classes = list(range(10))
            base_labels = trainset.classes
            
        elif self.dataset_name == "CIFAR10":
            trainset = torchvision.datasets.CIFAR10(root = path,
                                       train = True, download = True, transform = self.transform)
            testset = torchvision.datasets.CIFAR10(root = path,
                                                 train = False, download = True, transform = self.transform)
            classes = list(range(10))
            base_labels = trainset.classes
            
        elif self.dataset_name == "CIFAR100":
            trainset = torchvision.datasets.CIFAR100(root = path,
                                       train = True, download = True, transform = self.transform)
            testset = torchvision.datasets.CIFAR100(root = path,
                                                 train = False, download = True, transform = self.transform)
            classes = list(range(100))
            base_labels = trainset.classes
        
        else:
            raise KeyError("Unknown dataset: {}".format(self.dataset_name))
            
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = self.batch_size,
                        shuffle = self.shuffle, num_workers = self.num_workers)
        
        if testset is not None:
            testloader = torch.utils.data.DataLoader(testset, batch_size = self.batch_size,
                        shuffle = False, num_workers = self.num_workers)
        else:
            testloader = None
            
            
        return [trainloader, testloader, classes, base_labels, trainset, testset]
    
    def worker_init_fn(self, worker_id):                                                          
        np.random.seed(worker_id)