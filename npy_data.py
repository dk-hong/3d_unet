import os
import math
import torch

import numpy as np
from tqdm import tqdm


def calc_patch_coord(tensor_shape, patch_size=(224, 224, 224), overlap=(112, 112, 112)):
    return_coord = []
    for i in range(math.ceil((tensor_shape[0] - patch_size[0]) / (patch_size[0] - overlap[0]) + 1)):
        for j in range(math.ceil((tensor_shape[1] - patch_size[1])/(patch_size[1] - overlap[1]) + 1)):
            for k in range(math.ceil((tensor_shape[2] - patch_size[2])/(patch_size[2] - overlap[2]) + 1)):
                i_start = i * overlap[0]
                i_end = i * overlap[0] + patch_size[0]
                if i_end > tensor_shape[0]:
                    diff = i_end - tensor_shape[0]
                    i_end = tensor_shape[0]
                    i_start = i_start - diff
                j_start = j * overlap[1]
                j_end = j * overlap[1] + patch_size[1]
                if j_end > tensor_shape[1]:
                    diff = j_end - tensor_shape[1]
                    j_end = tensor_shape[1]
                    j_start = j_start - diff
                k_start = k * overlap[2]
                k_end = k * overlap[2] + patch_size[2]
                if k_end > tensor_shape[2]:
                    diff = k_end - tensor_shape[2]
                    k_end = tensor_shape[2]
                    k_start = k_start - diff
                return_coord.append([i_start, i_end, j_start, j_end, k_start, k_end])
    
    return return_coord


PATCH_SIZE = (224, 224, 224)
OVERLAP_SIZE = (112, 112, 112)
DATA_PATH = './before_resized'


directories = ['p2', 'p7']
positions = ['front']#, 'right', 'top']

for directory in directories:
    for position in positions:
        if directory == 'p2':
            task = 'train'
        elif directory == 'p7':
            task = 'test'
        else:
            print('wrong position')
            exit()

        tensor_data = torch.load(f'./merged_data/{task}/{directory}_{position}.pt')
        tensor_target = torch.load(f'./merged_data/{task}/{directory}_{position}_label.pt')

        tensor_data = tensor_data.numpy()
        tensor_target = tensor_target.numpy()

        
        x = 0

        patch_coord = calc_patch_coord(tensor_data.shape, PATCH_SIZE, OVERLAP_SIZE)

        
        for i, patch_coord in tqdm(enumerate(patch_coord)):
            data_slice = tensor_data[patch_coord[0]: patch_coord[1],\
                                    patch_coord[2]: patch_coord[3],\
                                    patch_coord[4]: patch_coord[5]]

            target_slice = tensor_target[patch_coord[0]: patch_coord[1],\
                                        patch_coord[2]: patch_coord[3],\
                                        patch_coord[4]: patch_coord[5]]

            
            np.save(os.path.join(DATA_PATH + f'/{task}/{i + x:04}'), [data_slice, target_slice])
            
            # torch.save([data_slice, target_slice], os.path.join(DATA_PATH + f'/{task}/{i + x:04}.pt'))
            # torch.save(target_slice, os.path.join(DATA_PATH + f'/{task}/label_{i + x:04}.pt'))

        x += i
