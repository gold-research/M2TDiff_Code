# M2TDiff: Multi-Scale MoE-Enhanced Transformer Diffusion Network for Video Object Detection

This repository is the official implementation of **M2TDiff**.

M2TDiff is an end-to-end video object detection framework built on top of
[Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).
It addresses three limitations of DETR-based video object detectors:
**frame-agnostic query initialization, scale-agnostic attention, and
heterogeneity-agnostic decoder transformation**, with three plug-and-play
modules:

1. **RDQG (Reinforcement-Guided Diffusion Query Generator)** — casts
   current-frame query localization as a conditional diffusion denoising task:
   forward diffusion adds Gaussian noise to the GT boxes, and multi-trajectory
   (K trajectories) reverse diffusion guided by a reinforcement-learning-style
   contrastive ranking loss generates content-aware object queries. Inference
   runs a single reverse trajectory (K=1) from random noise boxes
   (`models/rdqg.py`, `models/rdqg_loss.py`).

2. **MGTE (Multi-Scale Graph Interaction Transformer Encoder)** — augments the
   multi-head deformable attention path of each encoder layer with a
   Multi-Scale Dynamic Graph Convolution (MS-DGC) branch over multi-scale
   k-NN graphs (1-NN + 10-NN by default), jointly modeling local structure and
   global context (`models/mgte.py`).

3. **SMTD (Sparsely-Gated Mixture-of-Experts Transformer Decoder)** — a
   Query-aware MoE Block (QMB) with Top-1 sparse routing replaces the FFN in
   each decoder layer, keeping FLOPs roughly on par with the baseline while
   scaling decoder capacity (`models/smtd.py`).

All modules are **optional** and disabled by default — setting
`--use_rdqg / --use_mgte / --use_smtd` turns each module on. With all flags
off, the code reproduces a plain DETR-based multi-frame video baseline exactly.

Detailed model design, training recipes, and ablations are described in
[`M2TDiff_base_model.md`](M2TDiff_base_model.md).

## Main Results

|  Method |  Backbone  | Frame Numbers |  mAP@0.5 | ms/frame |
| :-----: | :--------: | :-----------: | :------: | :------: |
| M2TDiff | ResNet-101 |       30      | **89.2** |   22.1   |
| M2TDiff |  Swin-Base |       30      | **93.0** |   39.6   |

Reported on the ImageNet VID validation set with 4×RTX 5090 (batch size 4);
see [`M2TDiff_base_model.md`](M2TDiff_base_model.md) for details, VisDrone-VID
results, and the ablation study.

`ms/frame` and `FPS` are measured on the ImageNet VID validation set with
`batch_size=1` (reported by `tools/eval_m2tdiff.sh`).

## Updates

* (2026/08) M2TDiff source code released.

## Installation

The codebase is built on top of [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).

### Requirements

* Linux, CUDA>=9.2, GCC>=5.4

* Python>=3.7

  We recommend using Anaconda to create a conda environment:

  ```bash
  conda create -n m2tdiff python=3.7 pip
  conda activate m2tdiff
  ```

* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/))

  ```bash
  conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
  ```

* Other requirements

  ```bash
  pip install -r requirements.txt
  ```

* Build MultiScaleDeformableAttention

  ```bash
  cd ./models/ops
  sh ./make.sh
  ```

## Usage

### Dataset Preparation

1. Download the ILSVRC2015 DET and ILSVRC2015 VID datasets from
   [here](https://image-net.org/challenges/LSVRC/2015/2015-downloads), then
   convert the two datasets to JSON using the
   [MMTracking tools](https://github.com/open-mmlab/mmtracking/blob/master/tools/convert_datasets/ilsvrc/).

   The joint
   [JSON](https://drive.google.com/drive/folders/1cCXY41IFsLT-P06xlPAGptG7sc-zmGKF?usp=sharing)
   of the two datasets is provided. We recommend symlinking the dataset path
   to `datasets/`; the expected layout is:

```text
code_root/
└── data/
    └── vid/
        ├── Data
        │   ├── VID/
        │   └── DET/
        └── annotations/
            ├── imagenet_vid_train.json
            ├── imagenet_vid_train_joint_30.json
            └── imagenet_vid_val.json
```

### Pretraining the Single-Frame Baseline

1. Download the COCO-pretrained weights from
   [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and
   put the checkpoint into:

```text
./exps/our_models/COCO_pretrained_model/
```

2. Train the single-frame baseline, which is used as the resume checkpoint
   of M2TDiff:

```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh $1 r101 $2 configs/r101_train_single.sh
```

### Training M2TDiff

Using the single-frame baseline weights (e.g.
`./exps/singlebaseline/r101/checkpoint0009.pth`, produced by the previous step)
as the resume model:

```bash
# single node, 8 GPUs
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh $1 r101 $2 configs/r101_train_m2tdiff.sh

# or directly (single GPU)
sh configs/r101_train_m2tdiff.sh
```

All RDQG, MGTE, and SMTD hyperparameters are exposed as `main.py` flags;
see `configs/r101_train_m2tdiff.sh` for the recommended values.

### Evaluation

```bash
# Evaluate the M2TDiff checkpoint
# Writes eval_<dataset>.json with mAP@0.5, ms/frame, and FPS
./tools/eval_m2tdiff.sh exps/m2tdiff/r101_m2tdiff checkpoint.pth
```

To evaluate an ablation variant, disable components with `0/1` environment
variables:

```bash
USE_RDQG=0 USE_MGTE=0 USE_SMTD=0 ./tools/eval_m2tdiff.sh exps/m2tdiff/r101_A0_baseline
```

### Ablation Matrix & Hyperparameter Scan

```bash
# A0~A6 ablation matrix (train + evaluate each variant; A0 is the baseline anchor)
EPOCHS=15 ./tools/ablation.sh
EPOCHS=50 ./tools/ablation.sh

# Run a single experiment
./tools/ablation.sh A6

# Single-variable scan over T / K / knn / L / Y / M
SCAN=T   ./tools/scan.sh               # diffusion steps T in {2,4,6}
SCAN=K   ./tools/scan.sh               # trajectories K in {1,3,5,7}
SCAN=knn ./tools/scan.sh               # knn_k in {1,10,11,21}
SCAN=L   ./tools/scan.sh               # graph layers L in {1,2,3}
SCAN=Y   ./tools/scan.sh               # experts Y in {2,4,8}
SCAN=M   ./tools/scan.sh               # window M in {10,20,30}

# Aggregate every eval_*.json into a Markdown table
python tools/parse_logs.py exps/m2tdiff --out docs/experiments.md
```

## New Modules in This Repository

| File                                     | Description                                                                                                   |
| :--------------------------------------- | :------------------------------------------------------------------------------------------------------------ |
| `models/rdqg.py`                         | RDQG: reinforcement-guided diffusion query generator (multi-trajectory training, single-trajectory inference) |
| `models/rdqg_loss.py`                    | SimpleDiffusionLoss (`L_simple`) + ContrastiveRLLoss (`L_C`) + RewardComputer                                 |
| `models/mgte.py`                         | MGTE: multi-scale graph interaction transformer encoder (MS-DGC branch)                                       |
| `models/smtd.py`                         | SMTD: GateNetwork / ExpertFFN / QMB / LoadBalanceLoss                                                         |
| `models/deformable_detr_multi.py`        | Multi-frame detector: RDQG / MGTE / SMTD integration and criterion extensions                                 |
| `models/deformable_transformer_multi.py` | Temporal transformer: QMB routing and route-info collection                                                   |
| `tools/eval_m2tdiff.sh`                  | Evaluation + throughput measurement (ms/frame, FPS)                                                           |
| `tools/ablation.sh`                      | A0~A6 ablation matrix                                                                                         |
| `tools/scan.sh`                          | Single-variable hyperparameter scan                                                                           |
| `tools/parse_logs.py`                    | Aggregate evaluation summaries to `docs/experiments.md`                                                       |
| `configs/r101_train_m2tdiff.sh`          | M2TDiff training recipe                                                                                       |

## Acknowledgement

This project is developed based on the following project. We thank the authors
for releasing their code:

* [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)

## Citing

If you find this work useful in your research, please consider citing:

```bibtex
@article{m2tdiff,
  title   = {M2TDiff: Multi-Scale MoE-Enhanced Transformer Diffusion Network for Video Object Detection},
  author  = {TODO},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2026}
}
```

And the base work:

```bibtex
@inproceedings{zhu2021deformable,
  title     = {Deformable DETR: Deformable Transformers for End-to-End Object Detection},
  author    = {Zhu, Xizhou and Su, Weijie and Lu, Lewei and Li, Bin and Wang, Xiaogang and Dai, Jifeng},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2021}
}
```
