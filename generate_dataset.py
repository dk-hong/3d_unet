import os
import numpy as np

import torch

from matplotlib import image


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

        path = f'./source/{directory}/{position}/'
        file_list = os.listdir(path)
        file_list = sorted(file_list)[:-1]

        input_3d = []
        for i in file_list:
            # scaling and append
            input_3d.append(image.imread(path + i) / 255)

        input_3d = np.array(input_3d)

        print(f'shape: {input_3d.shape}')

        tensor_3d = torch.from_numpy(input_3d).type(torch.FloatTensor)

        print(tensor_3d.shape, tensor_3d.type())
        torch.save(tensor_3d, f'./{task}/{directory}_{position}.pt')
        