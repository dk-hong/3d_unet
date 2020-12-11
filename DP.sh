#!/bin/bash

method='A'
workers=4

# DP
python train_3dunet.py

# DP + mixed_precision
# python train_3dunet.py --method "${method}" --mixed-precision

# DP + workers
# python train_3dunet.py --method "${method}" --n-workers "${workers}"

# DP + workers + mixed_precision
# python train_3dunet.py --method "${method}" --n-workers "${workers}" --mixed-precision
