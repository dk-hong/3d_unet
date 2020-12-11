import os
import torch
import numpy as np


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.file_list = os.listdir(path)
        self.file_list = sorted(self.file_list)
        self.transform = transform


    def __len__(self):
        return(len(self.file_list)) 
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        image, target = np.load(os.path.join(self.path, file_name))

        if self.transform:
            image, target = self.transform(image), self.transform(target, label=True)
        else:
            image, target = image, target
        
        image = image.reshape(1, *image.shape)
        label = target.reshape(1, *target.shape)

        return image, label
