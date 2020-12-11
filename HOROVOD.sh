#!/bin/bash

gpu_num=4

# HOROVOD
horovodrun -np "${gpu_num}" python train_3dunet.py --use-horovod