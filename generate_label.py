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

        path = f'./source/{directory}/{position}/label/'
        file_list = os.listdir(path)
        file_list = sorted(file_list)

        input_3d = []

        for i in file_list:
            img = image.imread(path + i)
            assert len(img.shape) < 4
            if len(img.shape) == 3:
                img = img[:, :, 0]
            input_3d.append(img)


        input_3d = np.array(input_3d)

        print(f'shape: {input_3d.shape}')

        tensor_3d = torch.from_numpy(input_3d > 0).type(torch.FloatTensor)

        print(tensor_3d.shape, tensor_3d.type())
        torch.save(tensor_3d, f'./{task}/{directory}_{position}_label.pt')
