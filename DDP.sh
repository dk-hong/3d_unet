#!/bin/bash

# DDP
python -m torch.distributed.launch --nproc_per_node=1 train_3dunet.py --multiprocessing-distributed