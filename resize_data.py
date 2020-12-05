import os
import torch
import numpy as np

from tqdm import tqdm
from custom_transforms import Resize, ToTensor, Compose

tasks = ['train', 'test']

for task in tasks:

    load_path = f'./before_resized/{task}'
    save_path = f'./after_resized/{task}'

    file_list = os.listdir(load_path)
    file_list = sorted(file_list)

    t = Compose([ToTensor(), Resize(112)])

    for item in tqdm(file_list):
        image, label = np.load(os.path.join(load_path, item))
        image = t(image)
        label = t(label)
        torch.save([image, label], os.path.join(save_path, item[:-4] + '.pt'))
