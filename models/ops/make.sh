#!/usr/bin/env bash
# Build the MultiScaleDeformableAttention CUDA op.
# (Cluster users: replace the python line with e.g.
#   srun -p $partition --gres=gpu:1 python setup.py build develop --user )
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-'3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6'}

python setup.py build develop --user

# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

