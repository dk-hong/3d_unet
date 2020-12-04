import os
import numpy as np

from torch.utils.data import Dataset

from torchvision.datasets.vision import VisionDataset
from matplotlib import image


# Setting
# 

# TODO:
# 1. normalize: input은 255로 나누고, label은 0 or 1이 되도록
# 2. p2를 train set으로, p7을 test set으로 사용하도록
# 3. Flip은 항상 CPU에서
# 4. CPU에서 Crop & Resize 후 훈련, Crop한 데이터를 불러와서 Resize 후 훈련, Crop & Resize한 데이터를 불러와서 훈련 세 가지를 테스트 할 것


class VoxelFolder(Dataset):
    def __init__(self, path:str, input_size, overlap_size, transform=None):
        super(VoxelFolder, self).__init__()
        # path shoud be 'p*/${angle}'
        self.path = path
        self.input_files, self.label_files = self._find_files(self.path)
        self.transform = transform

        self.D = len(self.input_files)
        self.H, self.W = image.imread(os.path.join(path, self.input_files[0])).shape
        
        if type(input_size) is int:
            self.input_size = (input_size, input_size, input_size)
        elif (type(input_size) is list or type(input_size) is tuple) and len(input_size) == 3:
            self.input_size = input_size
        else:
            raise TypeError('input_size should be int or tuple of three integer values')

        if type(overlap_size) is int:
            self.overlap_size = (overlap_size, overlap_size, overlap_size)
        elif (type(overlap_size) is list or type(overlap_size) is tuple) and len(overlap_size) == 3:
            self.overlap_size = overlap_size
        else:
            raise TypeError('overlap_size should be int or tuple of three integer values')

        assert self.D >= self.input_size[0]
        assert self.H >= self.input_size[1]
        assert self.W >= self.input_size[2]

        assert self.overlap_size[0] <= self.input_size[0]
        assert self.overlap_size[1] <= self.input_size[1]
        assert self.overlap_size[2] <= self.input_size[2]

        # file과 H, W의 index를 저장 후 get_item시에 해당 index의 파일을 직접 열어서
        self.num_voxel_per_depth = self._get_num_voxel_per(self.D, self.input_size[0], self.overlap_size[0])
        self.num_voxel_per_height = self._get_num_voxel_per(self.H, self.input_size[1], self.overlap_size[1])
        self.num_voxel_per_width = self._get_num_voxel_per(self.W, self.input_size[2], self.overlap_size[2])
        
        self.total_voxel_num = self.num_voxel_per_depth * self.num_voxel_per_height * self.num_voxel_per_width
        # print(self.num_voxel_per_depth, self.num_voxel_per_height, self.num_voxel_per_width)
        # print('total_voxel:', self.total_voxel_num)
 
    def _find_files(self, dir:str):
        input_files = [d.name for d in os.scandir(dir) if d.is_file()]
        label_files = [d.name for d in os.scandir(os.path.join(dir, 'label')) if d.is_file()]
        return sorted(input_files), sorted(label_files)
    
    def _get_num_voxel_per(self, base, input_size, overlap_size):
        if (base - input_size) % (input_size - overlap_size) == 0:
            return (base - input_size) // (input_size - overlap_size) + 1
        else:
            return (base - input_size) // (input_size - overlap_size) + 2

    def __len__(self):
        return self.total_voxel_num

    def __getitem__(self, index: int):
        assert self.total_voxel_num > index
        denom = self.num_voxel_per_height * self.num_voxel_per_width
        d = (index // denom) * (self.input_size[0] - self.overlap_size[0])
        h = ((index % denom) // self.num_voxel_per_width) * (self.input_size[1] - self.overlap_size[1])
        w = ((index % denom) % self.num_voxel_per_width) * (self.input_size[2] - self.overlap_size[2])

        if d + self.input_size[0] > self.D:
            d = self.D - self.input_size[0]
        if h + self.input_size[1] > self.H:
            h = self.H - self.input_size[1]
        if w + self.input_size[2] > self.W:
            w = self.W - self.input_size[2]

        returned_image = []
        returned_label = []
        
        for index in range(d, d + self.input_size[0]):
            img = image.imread(os.path.join(self.path, self.input_files[index]))[h: h + self.input_size[1], w: w + self.input_size[2]]
            label = image.imread(os.path.join(self.path, 'label', self.label_files[index]))[h: h + self.input_size[1], w: w + self.input_size[2]]
            if len(label.shape) > 2:
                label = label[:, :, 0]
            returned_image.append(img)
            returned_label.append(label)
        returned_image = np.stack(returned_image).astype(np.float32) / 255
        returned_label = np.stack(returned_label)
        returned_label = (returned_label > 0).astype(np.float32)
        
        if self.transform:
            returned_image = self.transform(returned_image)
            returned_label = self.transform(returned_label, label=True)
        
        returned_image = returned_image.reshape(1, *returned_image.shape)
        returned_label = returned_label.reshape(1, *returned_label.shape)

        return returned_image, returned_label
