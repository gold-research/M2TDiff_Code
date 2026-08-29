#!/usr/bin/env bash
# A0~A6 ablation matrix for M2TDiff (train + evaluate each variant).
#
# Usage:
#   EPOCHS=15 ./tools/ablation.sh            # fast screening (default 15)
#   EPOCHS=50 ./tools/ablation.sh            # final paper numbers
#   ./tools/ablation.sh A6                   # only run one experiment
#   RUN_TRAIN=0 ./tools/ablation.sh          # skip training, evaluate existing ckpts
#   BASE_CKPT=/path/to/pretrained ./tools/ablation.sh
#
# Matrix (A0 is the baseline / regression anchor):
#   A0 baseline | A1 +RDQG | A2 +MGTE | A3 +SMTD | A4 +RDQG+MGTE | A5 +RDQG+SMTD | A6 full
#
# Each variant is evaluated via tools/eval_m2tdiff.sh and writes
# <exp_dir>/eval_<dataset>.json; then run:
#   python tools/parse_logs.py exps/m2tdiff

set -x
EPOCHS=${EPOCHS:-15}
FILTER=${1:-}
RUN_TRAIN=${RUN_TRAIN:-1}
BACKBONE=${BACKBONE:-resnet101}
BASE_CKPT=${BASE_CKPT:-./exps/singlebaseline/r101/checkpoint0009.pth}
BASE_DIR=exps/m2tdiff
T=`date +%m%d%H%M`

case "$FILTER" in
    ""|A0|A1|A2|A3|A4|A5|A6) ;;
    *) echo "unknown filter: $FILTER (expect A0..A6)"; exit 1 ;;
esac

# run_exp <EXP> <TAG> <RDQG_FLAGS> <MGTE_FLAGS> <SMTD_FLAGS>
run_exp() {
    local EXP=$1 TAG=$2 RDQG=$3 MGTE=$4 SMTD=$5
    local EXP_DIR=${BASE_DIR}/${BACKBONE}_${TAG}
    [ -z "$FILTER" ] || [ "$EXP" = "$FILTER" ] || return 0
    echo "==> [$EXP] $TAG  (rdqg=${RDQG:-off} mgte=${MGTE:-off} smtd=${SMTD:-off})"
    mkdir -p ${EXP_DIR}
    if [ "$RUN_TRAIN" = "1" ]; then
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
            ${RDQG} ${MGTE} ${SMTD} \
            --output_dir ${EXP_DIR} \
            2>&1 | tee ${EXP_DIR}/log.train.$T
    fi
    USE_RDQG=$([ -n "$RDQG" ] && echo 1 || echo 0) \
    USE_MGTE=$([ -n "$MGTE" ] && echo 1 || echo 0) \
    USE_SMTD=$([ -n "$SMTD" ] && echo 1 || echo 0) \
    bash "$(dirname "$0")/eval_m2tdiff.sh" "${EXP_DIR}"
}

RDQG="--use_rdqg --diffusion_steps 4 --num_diffusion_trajectories 5"
MGTE="--use_mgte --graph_layers 2 --knn_k 11"
SMTD="--use_smtd --num_experts 4 --load_balance_coef 0.001"

run_exp A0 "A0_baseline"    ""      ""      ""
run_exp A1 "A1_rdqg"        "$RDQG" ""      ""
run_exp A2 "A2_mgte"        ""      "$MGTE" ""
run_exp A3 "A3_smtd"        ""      ""      "$SMTD"
run_exp A4 "A4_rdqg_mgte"   "$RDQG" "$MGTE" ""
run_exp A5 "A5_rdqg_smtd"   "$RDQG" ""      "$SMTD"
run_exp A6 "A6_full"        "$RDQG" "$MGTE" "$SMTD"

echo "Ablation matrix done. Run: python tools/parse_logs.py ${BASE_DIR}"
