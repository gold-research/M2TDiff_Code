# ------------------------------------------------------------------------
# RDQG losses: L_simple + Contrastive RL (L_C)
# ------------------------------------------------------------------------
# Provides:
#   - RewardComputer      : R(C, Q, G) = max( Softmax(C) x IoU(Q, G) ) per trajectory,
#                           computed with the existing Hungarian matcher (detr matcher)
#   - ContrastiveRLLoss   : L_C = (1/N_pos) * sum_{n in T_pos} log(1 + exp(-(r_hat^n - r_neg_star)))
#   - SimpleDiffusionLoss : L_simple = MSE(eps_theta(z_t, t), eps) over valid GT boxes only
#
# Gradient flow: the reward VALUE is used as a constant for sample selection and
# normalization (mu/sigma/T_pos/T_neg/r_neg_star are detached); the final loss term
# keeps the gradient of r_hat w.r.t. the diffusion outputs (traj_boxes/traj_logits),
# so gradients only flow back through the query-generation network (DenoiseStep).
# ------------------------------------------------------------------------
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from util import box_ops


class RewardComputer(nn.Module):
    """Compute per-trajectory rewards via Hungarian matching + IoU.

    Args:
        matcher: existing HungarianMatcher instance (from build_matcher).

    Inputs:
        traj_boxes : [B, K, N, 4] normalized cxcywh box predictions of each trajectory
        traj_logits: [B, K, N, num_classes] raw (pre-sigmoid) class logits
        targets    : list of B current-frame GT dicts (with 'boxes'/'labels')

    Returns:
        r: [B, K] rewards, r[b, k] = max_i( sigmoid(C_i) * IoU(Q_i, G_matched(i)) )
           over the Hungarian-matched predictions i of trajectory (b, k).
           The value is differentiable w.r.t. traj_boxes/traj_logits.
    """

    def __init__(self, matcher):
        super().__init__()
        self.matcher = matcher

    def forward(self, traj_boxes, traj_logits, targets):
        B, K, N, _ = traj_boxes.shape
        num_classes = traj_logits.shape[-1]
        boxes_f = traj_boxes.reshape(B * K, N, 4)      # [B*K, N, 4]
        logits_f = traj_logits.reshape(B * K, N, num_classes)  # [B*K, N, C]
        tgt_f = [copy.deepcopy(t) for t in targets for _ in range(K)]  # row-major (b, k)

        # matching is discrete -> no gradient needed
        with torch.no_grad():
            indices = self.matcher({'pred_logits': logits_f, 'pred_boxes': boxes_f}, tgt_f)

        rewards = []
        for bi in range(B * K):
            src_idx, tgt_idx = indices[bi]
            if len(src_idx) == 0:
                rewards.append(boxes_f.new_zeros(()))
                continue
            # matched predictions (keep gradient for the RL loss)
            pred_boxes = boxes_f[bi, src_idx]        # [M, 4] cxcywh
            pred_logits = logits_f[bi, src_idx]      # [M, C]
            gt_boxes = tgt_f[bi]['boxes'][tgt_idx]   # [M, 4] cxcywh
            gt_labels = tgt_f[bi]['labels'][tgt_idx]  # [M]

            iou, _ = box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(pred_boxes),
                                     box_ops.box_cxcywh_to_xyxy(gt_boxes))  # [M, M]
            iou_diag = torch.diagonal(iou)            # [M] matched-pair IoU
            cls_score = torch.sigmoid(pred_logits.gather(1, gt_labels.unsqueeze(1)).squeeze(1))  # [M]
            rewards.append((cls_score * iou_diag).max())

        r = torch.stack(rewards).reshape(B, K)
        return r


class ContrastiveRLLoss(nn.Module):
    """Contrastive RL loss (paper Sec. 3.4).

    L_C = (1/N_pos) * sum_{n in T_pos} log(1 + exp(-(r_hat^n - r_neg_star)))

    where r_hat = (r - mu) / sigma over the K trajectories of the batch,
    T_pos = { n : r_hat_n > 0 }, r_neg_star = min_{n in T_neg} r_hat_n.
    """

    def __init__(self, reward_computer):
        super().__init__()
        self.reward = reward_computer

    def forward(self, traj_boxes, traj_logits, targets):
        r = self.reward(traj_boxes, traj_logits, targets)      # [B, K] (differentiable)
        r_det = r.detach()                                     # value used for selection

        mu = r_det.mean()
        sigma = r_det.std().clamp(min=1e-6)
        r_hat_det = (r_det - mu) / sigma

        pos = r_hat_det > 0
        neg = ~pos
        device = r.device
        if not bool(pos.any()) or not bool(neg.any()):
            # no contrastive pair -> zero loss (also safe for K=1 trajectories)
            loss = torch.zeros((), device=device)
        else:
            r_neg_star = r_hat_det[neg].min()                  # reference negative (detached)
            r_hat = (r - mu) / sigma                           # gradient flows via r
            loss = F.softplus(-(r_hat[pos] - r_neg_star)).mean()

        return loss, r_det.mean(), r_det.std()


class SimpleDiffusionLoss(nn.Module):
    """L_simple = MSE(eps_theta(z_t, t), eps) over valid (real GT) boxes only."""

    def forward(self, noise_pred, noise, valid_mask=None):
        if valid_mask is not None:
            v = valid_mask.unsqueeze(-1).expand_as(noise_pred)  # [B, N, 4] bool
            return F.mse_loss(noise_pred[v], noise[v])
        return F.mse_loss(noise_pred, noise)
