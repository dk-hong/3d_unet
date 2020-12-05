import os
import math
import numpy as np

from matplotlib import image

from torch.utils.data import Dataset


class VoxelFolder(Dataset):
    def __init__(self, path:str, input_size, overlap_size, transform=None):
        super(VoxelFolder, self).__init__()
        # path shoud be 'p*/'
        self.angle = ['front']#, 'right', 'top']
        self.input_files, self.label_files, self.D = self._find_files(path)
        self.transform = transform

        self.H, self.W = [], []
        for D in self.D:
            H, W = image.imread(self.input_files[D - 1]).shape
            self.H.append(H)
            self.W.append(W)

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

        for D in self.D:
            assert D >= self.input_size[0]
        for H in self.H:
            assert H >= self.input_size[1]
        for W in self.W:
            assert W >= self.input_size[2]

        assert self.overlap_size[0] < self.input_size[0]
        assert self.overlap_size[1] < self.input_size[1]
        assert self.overlap_size[2] < self.input_size[2]

        # TODO: angle별로 num_voxel_per와 total_voxel_num이 있어야 하며, 전체 voxel_num도 있어야 함
        self.num_voxel_per_depth = []
        self.num_voxel_per_height = []
        self.num_voxel_per_width = []
        self.total_voxel_num_per_angle = []
        self.total_voxel_num = 0

        # file과 H, W의 index를 저장 후 get_item시에 해당 index의 파일을 직접 열어서
        for D, H, W in zip(self.D, self.H, self.W):
            num_voxel_per_depth = self._get_num_voxel_per(D, self.input_size[0], self.overlap_size[0])
            num_voxel_per_height = self._get_num_voxel_per(H, self.input_size[1], self.overlap_size[1])
            num_voxel_per_width = self._get_num_voxel_per(W, self.input_size[2], self.overlap_size[2])
            
            self.num_voxel_per_depth.append(num_voxel_per_depth)
            self.num_voxel_per_height.append(num_voxel_per_height)
            self.num_voxel_per_width.append(num_voxel_per_width)
            
            self.total_voxel_num_per_angle.append(num_voxel_per_depth * num_voxel_per_height * num_voxel_per_width)
            self.total_voxel_num += self.total_voxel_num_per_angle[-1]
 
    def _find_files(self, path):
        input_files = []
        label_files = []
        D = []
        for angle in self.angle:
            dir = os.path.join(path, angle)
            input_files_per_angle = [os.path.join(dir, d.name) for d in os.scandir(dir) if d.is_file()]
            label_files_per_angle = [os.path.join(dir, 'label', d.name) for d in os.scandir(os.path.join(dir, 'label')) if d.is_file()]
            input_files += sorted(input_files_per_angle)
            label_files += sorted(label_files_per_angle)
            D.append(len(input_files_per_angle))
        return input_files, label_files, D
    
    def _get_num_voxel_per(self, base, input_size, overlap_size):
        return math.ceil((base - input_size) / (input_size - overlap_size)) + 1

    def __len__(self):
        return self.total_voxel_num

    def __getitem__(self, index: int):
        assert self.total_voxel_num > index
        index_in_angle = index
        angle = 0
        for angle, i in enumerate(self.total_voxel_num_per_angle):      
            if i <= index_in_angle:
                index_in_angle -= i
            else:
                break

        denom = self.num_voxel_per_height[angle] * self.num_voxel_per_width[angle]
        
        d = (index_in_angle // denom) * (self.input_size[0] - self.overlap_size[0])
        h = ((index_in_angle % denom) // self.num_voxel_per_width[angle]) * (self.input_size[1] - self.overlap_size[1])
        w = ((index_in_angle % denom) % self.num_voxel_per_width[angle]) * (self.input_size[2] - self.overlap_size[2])

        if d + self.input_size[0] > self.D[angle]:
            d = self.D[angle] - self.input_size[0]
        if h + self.input_size[1] > self.H[angle]:
            h = self.H[angle] - self.input_size[1]
        if w + self.input_size[2] > self.W[angle]:
            w = self.W[angle] - self.input_size[2] 

        returned_image = []
        returned_label = []

        num_voxel_before_angle = sum(self.total_voxel_num_per_angle[:angle])
        
        for index in range(num_voxel_before_angle + d, num_voxel_before_angle + d + self.input_size[0]):
            img = image.imread(self.input_files[index])[h: h + self.input_size[1], w: w + self.input_size[2]]
            label = image.imread(self.label_files[index])[h: h + self.input_size[1], w: w + self.input_size[2]]
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
