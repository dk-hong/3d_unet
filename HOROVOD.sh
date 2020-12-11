#!/bin/bash

gpu_num=4
method='A'
workers=4

# HOROVOD
horovodrun -np "${gpu_num}" python train_3dunet.py --use-horovod --method "${method}"

# HOROVOD + mixed_precision
# horovodrun -np "${gpu_num}" python train_3dunet.py --use-horovod\
#  --method "${method}" --mixed-precision

# HOROVOD + workers
# horovodrun -np "${gpu_num}" python train_3dunet.py --use-horovod\
#  --method "${method}" --n-workers "${workers}"

# HOROVOD
# horovodrun -np "${gpu_num}" python train_3dunet.py --use-horovod\
#  --method "${method}" --n-workers "${workers}" --mixed-precision