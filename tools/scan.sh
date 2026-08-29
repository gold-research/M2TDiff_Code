#!/usr/bin/env bash
# single-variable hyper-parameter scan on top of the A6 (full) config.
#
# Usage:
#   SCAN=T   ./tools/scan.sh        # diffusion steps  T in {2,4,6}
#   SCAN=K   ./tools/scan.sh        # trajectories    K in {1,3,5,7}
#   SCAN=knn ./tools/scan.sh        # knn_k           in {1,10,11,21}
#   SCAN=L   ./tools/scan.sh        # graph layers    L in {1,2,3}
#   SCAN=Y   ./tools/scan.sh        # experts         Y in {2,4,8}
#   SCAN=M   ./tools/scan.sh        # window M        in {10,20,30}
#   EPOCHS=15 SCAN=Y ./tools/scan.sh Y4   # single point
#
# Each point is trained (A6 full config + one swept hyper-param) and then
# evaluated via tools/eval_m2tdiff.sh; results land in
# exps/m2tdiff/<tag>/eval_<dataset>.json. A0 (baseline, no components) is the
# regression anchor to compare every sweep point against.

set -x
SCAN=${SCAN:-T}
POINT=${2:-}
EPOCHS=${EPOCHS:-15}
BACKBONE=${BACKBONE:-resnet101}
BASE_CKPT=${BASE_CKPT:-./exps/singlebaseline/r101/checkpoint0009.pth}
BASE_DIR=exps/m2tdiff
T=`date +%m%d%H%M`

case "$SCAN" in
    T)   VALUES="2 4 6";      ;;
    K)   VALUES="1 3 5 7";    ;;
    knn) VALUES="1 10 11 21"; ;;
    L)   VALUES="1 2 3";      ;;
    Y)   VALUES="2 4 8";      ;;
    M)   VALUES="10 20 30";   ;;
    *) echo "unknown SCAN=$SCAN (expect T|K|knn|L|Y|M)"; exit 1 ;;
esac
if [ -n "$POINT" ]; then VALUES="$POINT"; fi

# Sweep the hyper-param into the training command.
extra_flags() {
    case $SCAN in
        T)   echo "--diffusion_steps $1" ;;
        K)   echo "--num_diffusion_trajectories $1" ;;
        knn) echo "--knn_k $1" ;;
        L)   echo "--graph_layers $1" ;;
        Y)   echo "--num_experts $1" ;;
        M)   echo "--frames $1" ;;
    esac
}
# Pass the same value into the evaluation script via env vars.
eval_env() {
    case $SCAN in
        T)   echo "DIFF_STEPS=$1" ;;
        K)   echo "NUM_TRAJ=$1" ;;
        knn) echo "KNN_K=$1" ;;
        L)   echo "GRAPH_LAYERS=$1" ;;
        Y)   echo "NUM_EXPERTS=$1" ;;
        M)   echo "FRAMES=$1" ;;
    esac
}

for v in $VALUES; do
    TAG="${SCAN}${v}"
    EXP_DIR=${BASE_DIR}/${BACKBONE}_${TAG}
    echo "==> scan ${SCAN}=${v} -> ${EXP_DIR}"
    mkdir -p ${EXP_DIR}
    python -u main.py \
        --backbone ${BACKBONE} \
        --epochs ${EPOCHS} \
        --num_feature_levels 1 \
        --num_queries 300 \
        --dilation \
        --batch_size 1 \
        --frames 30 \
        --resume ${BASE_CKPT} \
        --lr_drop_epochs 4 6 \
        --num_workers 16 \
        --with_box_refine \
        --dataset_file vid_multi \
        --use_rdqg --diffusion_steps 4 --num_diffusion_trajectories 5 \
        --use_mgte --graph_layers 2 --knn_k 11 \
        --use_smtd --num_experts 4 --load_balance_coef 0.001 \
        $(extra_flags $v) \
        --output_dir ${EXP_DIR} \
        2>&1 | tee ${EXP_DIR}/log.train.$T
    env $(eval_env $v) bash "$(dirname "$0")/eval_m2tdiff.sh" "${EXP_DIR}"
done

echo "Scan done. Run: python tools/parse_logs.py ${BASE_DIR}"
