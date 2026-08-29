#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

EXP_DIR=exps/m2tdiff/r101_m2tdiff
mkdir -p ${EXP_DIR}
PY_ARGS=${@:1}
python -u main.py \
    --backbone resnet101 \
    --epochs 7 \
    --num_feature_levels 1 \
    --num_queries 300 \
    --dilation \
    --batch_size 1 \
    --frames 30 \
    --resume ./exps/singlebaseline/r101/checkpoint0009.pth \
    --lr_drop_epochs 4 6 \
    --num_workers 16 \
    --with_box_refine \
    --dataset_file vid_multi \
    --use_rdqg \
    --diffusion_steps 4 \
    --num_diffusion_trajectories 5 \
    --diff_loss_coef 1.0 \
    --rl_loss_coef 1.0 \
    --use_mgte \
    --graph_layers 2 \
    --knn_k 11 \
    --use_smtd \
    --num_experts 4 \
    --load_balance_coef 0.001 \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T
