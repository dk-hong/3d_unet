import os
import torch


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        file_list = os.listdir(path)
        self.transform = transform

        data = []
        label = []

        for img in file_list:
            tmp_data, tmp_label = torch.load(os.path.join(path, img))
            tmp_data = tmp_data.reshape(1, *tmp_data.shape)
            tmp_label = tmp_label.reshape(1, *tmp_label.shape)

            data.append(tmp_data.type(torch.FloatTensor))
            label.append(tmp_label.type(torch.FloatTensor))

        self.data = torch.stack(data)
        self.label = torch.stack(label)

    def __len__(self):
        return(len(self.data)) 
    
    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.data[idx]), self.transform(self.label[idx])
        return [self.data[idx], self.label[idx]]
