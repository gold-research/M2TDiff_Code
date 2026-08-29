#!/usr/bin/env bash
# evaluate an M2TDiff checkpoint on ImageNet VID val.
#
# Usage:
#   ./tools/eval_m2tdiff.sh <exp_dir> [checkpoint_name] [extra main.py args...]
#
#   checkpoint_name defaults to "checkpoint.pth". Component switches are read
#   from environment variables (1 = on, default on) so the same script can
#   evaluate ablation runs, e.g.:
#     USE_RDQG=0 USE_MGTE=0 USE_SMTD=0 ./tools/eval_m2tdiff.sh exps/m2tdiff/r101_A0_baseline
#
#   Results are written to <exp_dir>/eval_<dataset>.json (mAP@0.5, ms/frame, FPS).

set -x
EXP_DIR=${1:?usage: $0 <exp_dir> [checkpoint_name] [extra args]}
CKPT=${2:-checkpoint.pth}
PY_ARGS=${@:3}

T=`date +%m%d%H%M`
mkdir -p ${EXP_DIR}

# Component switches (default: full M2TDiff = A6 config).
RDQG_FLAG=; MGTE_FLAG=; SMTD_FLAG=
[ "${USE_RDQG:-1}" = "1" ] && RDQG_FLAG="--use_rdqg"
[ "${USE_MGTE:-1}" = "1" ] && MGTE_FLAG="--use_mgte"
[ "${USE_SMTD:-1}" = "1" ] && SMTD_FLAG="--use_smtd"
# M2TDiff hyper-parameters (defaults match configs/r101_train_m2tdiff.sh).
: "${FRAMES:=30}"
: "${DIFF_STEPS:=4}"
: "${NUM_TRAJ:=5}"
: "${GRAPH_LAYERS:=2}"
: "${KNN_K:=11}"
: "${NUM_EXPERTS:=4}"
: "${LOAD_BALANCE:=0.001}"

python -u main.py \
    --backbone resnet101 \
    --epochs 7 \
    --eval \
    --num_feature_levels 1 \
    --num_queries 300 \
    --dilation \
    --batch_size 1 \
    --frames ${FRAMES} \
    --diffusion_steps ${DIFF_STEPS} \
    --num_diffusion_trajectories ${NUM_TRAJ} \
    --graph_layers ${GRAPH_LAYERS} \
    --knn_k ${KNN_K} \
    --num_experts ${NUM_EXPERTS} \
    --load_balance_coef ${LOAD_BALANCE} \
    --resume ${EXP_DIR}/${CKPT} \
    --lr_drop_epochs 4 6 \
    --num_workers 16 \
    --with_box_refine \
    --dataset_file vid_multi \
    --output_dir ${EXP_DIR} \
    --infer_seed 0 \
    ${RDQG_FLAG} ${MGTE_FLAG} ${SMTD_FLAG} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.eval.$T
