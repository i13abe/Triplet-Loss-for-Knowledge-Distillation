import torch
import numpy as np
from PIL import Image
import umap

class KDTripletDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
        data = dataset.data
        labels = dataset.targets
        if type(labels) is not torch.Tensor:
                labels = torch.tensor(labels)
        
        # make label set 0-9
        labels_set = set(labels.numpy())
        
        # make the indices excepted each classes
        label_to_indices = {label : np.where(labels.numpy() != label)[0] for label in labels_set}
        
        if self.dataset.train:
            self.negative_indices = label_to_indices
        else:
            self.negative_indices = [[np.random.choice(label_to_indices[labels[i].item()])] for i in range(len(data))]
        

            
    def __getitem__(self, index):
        if self.dataset.train:
            img1_2, label1_2 = self.dataset[index]
            if type(label1_2) is not torch.Tensor:
                label1_2 = torch.tensor(label1_2)
            img3, label3 = self.dataset[np.random.choice(self.negative_indices[label1_2.item()])]
        else:
            img1_2, label1_2 = self.dataset[index]
            img3, label3 = self.dataset[self.negative_indices[index][0]]
        
            
        return (img1_2, img3), (label1_2, label3)
    
    def __len__(self):
        return len(self.dataset)