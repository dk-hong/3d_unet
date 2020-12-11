#!/bin/bash

method='A'
workers=4

# DDP
python -m torch.distributed.launch --nproc_per_node=1 train_3dunet.py \
 --multiprocessing-distributed --method "${method}"

# DDP + mixed_precision
# python -m torch.distributed.launch --nproc_per_node=1 train_3dunet.py \
#  --multiprocessing-distributed  --method "${method}"--mixed-precision

# DDP + workers
# python -m torch.distributed.launch --nproc_per_node=1 train_3dunet.py \
#  --multiprocessing-distributed --method "${method}" --n-workers "${workers}"

# DDP + workers + mixed_precision
# python -m torch.distributed.launch --nproc_per_node=1 train_3dunet.py \
#  --multiprocessing-distributed --method "${method}" --n-workers "${workers}" --mixed-precision
