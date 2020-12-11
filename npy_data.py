import os
import math
import argparse
import torch

import numpy as np
from tqdm import tqdm
from matplotlib import image

from custom_transforms import Resize


def parse_args():
    parser = argparse.ArgumentParser()
    # 원본 이미지 위치
    parser.add_argument('--source', type=str, default='/data/3d_data/p2')
    # 저장할 위치
    parser.add_argument('--dir', type=str, default='/data/3d_data')
    # task
    parser.add_argument('--task', type=str, default='train')

    # patch size
    parser.add_argument('--patch-size', type=int, default=224)
    # overlap size
    parser.add_argument('--overlap-size', type=int, default=112)
    # resize
    parser.add_argument('--resize_ratio', type=int, default=1)

    return parser.parse_args()


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


def main():
    args = parse_args()

    if args.resize_ratio == 1:
        PATCH_SIZE = (args.patch_size, ) * 3
        OVERLAP_SIZE = (args.overlap_size, ) * 3
    else:
        PATCH_SIZE = (math.ceil(args.patch_size * args.resize_ratio), ) * 3
        OVERLAP_SIZE = (math.ceil(args.overlap_size * args.resize_ratio), ) * 3

    
    os.makedirs(os.path.join(args.dir, f'{args.task}'), exist_ok=True)
    
    angles = ['front', 'right', 'top']

    x = 0
    
    for angle in angles:
        dir = os.path.join(args.source, angle)
        input_files = [os.path.join(dir, d.name) for d in os.scandir(dir) if d.is_file()]
        label_files = [os.path.join(dir, 'label', d.name) for d in os.scandir(os.path.join(dir, 'label')) if d.is_file()]
        input_3d = []
        label_3d = []
        for i in input_files:
            # scaling and append
            input_3d.append(image.imread(i).astype(np.float32) / 255)
        
        for i in label_files:
            label = image.imread(i)
            if len(label.shape) > 2:
                label = label[:, :, 0]

            label_3d.append((label > 0).astype(np.float32))
        
        input_3d = np.array(input_3d)
        label_3d = np.array(label_3d)

        if args.resize_ratio != 1:
            d, h, w = input_3d.shape
            new_d, new_h, new_w = math.ceil(d / 2), math.ceil(h / 2), math.ceil(w / 2)
            input_3d = torch.from_numpy(input_3d)
            label_3d = torch.from_numpy(label_3d)
            resize = Resize((new_d, new_h, new_w))
            input_3d = resize(input_3d)
            label_3d = resize(label_3d, label=False)
            input_3d, label_3d = input_3d.numpy(), label_3d.numpy()


        patch_coord = calc_patch_coord(input_3d.shape, PATCH_SIZE, OVERLAP_SIZE)

        for i, patch_coord in tqdm(enumerate(patch_coord)):
            data_slice = input_3d[patch_coord[0]: patch_coord[1],\
                                patch_coord[2]: patch_coord[3],\
                                patch_coord[4]: patch_coord[5]]

            target_slice = label_3d[patch_coord[0]: patch_coord[1],\
                                    patch_coord[2]: patch_coord[3],\
                                    patch_coord[4]: patch_coord[5]]
            
            np.save(os.path.join(args.dir, f'{args.task}/{i + x:04}'), [data_slice, target_slice])
            
        x += i


if __name__ == '__main__':
    main()