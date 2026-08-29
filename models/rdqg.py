# ------------------------------------------------------------------------
# RDQG: Reinforcement-Guided Diffusion Query Generator
# ------------------------------------------------------------------------
# The RDQG replaces the static learnable query_embed with a content-aware
# query generator driven by a small reverse diffusion process over the
# current-frame GT boxes (training) or random noise boxes (inference).
#
# Key components:
#   - NoiseScheduler   : linear beta schedule (beta_start=1e-4, beta_end=0.02)
#   - RoIAlignExtractor: crop region features from current-frame features
#   - DenoiseStep      : self-attn -> RoIAlign -> cross-attn -> FiLM -> heads
#   - NoisePredHead    : epsilon prediction for the L_simple objective
#   - RDQG             : orchestrates the K diffusion trajectories
# ------------------------------------------------------------------------

import math
import random

import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import inverse_sigmoid

try:
    from torchvision.ops import roi_align
    _HAS_ROI_ALIGN = True
except Exception:
    roi_align = None
    _HAS_ROI_ALIGN = False


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class NoiseScheduler(nn.Module):
    """Linear beta schedule following DDPM (Ho et al., 2020).

    beta_start / beta_end are tuned for small T (e.g. 4 steps).
    """

    def __init__(self, num_steps=4, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.num_steps = num_steps
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bar', alpha_bar)

    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: y_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps

        Args:
            x0: [B, N, 4] normalized cxcywh boxes
            t : int in [1, T] (1-indexed)
        Returns:
            (noisy_x, noise)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bar[t - 1]
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise, noise

    def get_time_embedding(self, t, dim=128):
        """Sinusoidal time-step embedding. Returns a vector of shape [dim]."""
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        args = torch.tensor([float(t)], dtype=torch.float32)[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [1, dim]
        return emb[0]


class RoIAlignExtractor(nn.Module):
    """Extract per-query region features via RoIAlign on current-frame features.

    Inputs:
        feat      : [B, C_in, H, W]  (e.g. backbone C5 of the current frame)
        boxes_norm: [B, N, 4] normalized cxcywh boxes in [0, 1]
    Outputs:
        [B, N, d_model] pooled region features
    """

    def __init__(self, d_model=256, in_channels=2048, output_size=7):
        super().__init__()
        assert _HAS_ROI_ALIGN, "torchvision.ops.roi_align is required by RDQG"
        self.output_size = output_size
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=1),
            nn.GroupNorm(32, d_model),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, feat, boxes_norm):
        B, N, _ = boxes_norm.shape
        _, _, H, W = feat.shape
        feat = self.proj(feat)  # [B, d, H, W]

        # cxcywh -> xyxy (normalized) -> absolute pixel coords
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_norm)  # [B, N, 4] in [0,1]
        scale = boxes_xyxy.new_tensor([W, H, W, H])
        boxes_abs = boxes_xyxy.clamp(0.0, 1.0) * scale  # [B, N, 4]

        batch_idx = torch.arange(B, device=feat.device, dtype=torch.float32).view(-1, 1).repeat(1, N).reshape(-1)
        rois = torch.cat([batch_idx.unsqueeze(1), boxes_abs.reshape(-1, 4)], dim=1)  # [B*N, 5]

        out = roi_align(feat, rois, output_size=self.output_size, spatial_scale=1.0, aligned=True)
        out = out.view(B, N, -1, self.output_size, self.output_size)
        out = self.pool(out)  # [B, N, d]
        return out


class FiLMLayer(nn.Module):
    """Feature-wise linear modulation conditioned on the diffusion time step."""

    def __init__(self, time_dim=128, d_model=256, dim_feedforward=1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, 2 * d_model),
        )

    def forward(self, x, t_emb):
        # x: [B, N, C]; t_emb: [D] (broadcast over batch & queries)
        g, b = self.mlp(t_emb).chunk(2, dim=-1)  # each [C]
        return x * g + b


class DenoiseStep(nn.Module):
    """One reverse diffusion step: self-attn -> RoIAlign -> cross-attn -> FiLM -> heads."""

    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1,
                 num_classes=31, time_dim=128, in_channels=2048, output_size=7):
        super().__init__()
        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # region feature interaction
        self.roi_extract = RoIAlignExtractor(d_model, in_channels, output_size)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        # time conditioning
        self.film = FiLMLayer(time_dim, d_model, dim_feedforward)
        # prediction heads (per-step box refinement + confidence)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        self.class_embed = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

        # box head init: predict small residuals
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

    @staticmethod
    def refine_box(y, delta):
        """Iterative box refinement: y_new = sigmoid(inverse_sigmoid(y) + delta)."""
        return (delta + inverse_sigmoid(y.clamp(1e-4, 1.0 - 1e-4))).sigmoid()

    def forward(self, z, y, feat, t_emb):
        """z: [B, N, C]; y: [B, N, 4] cxcywh; feat: [B, C_in, H, W]; t_emb: [D]"""
        # 1) self attention over queries
        z2 = self.self_attn(z.transpose(0, 1), z.transpose(0, 1), z.transpose(0, 1))[0].transpose(0, 1)
        z = self.norm1(z + self.dropout(z2))

        # 2) RoIAlign region features of the current boxes
        roi = self.roi_extract(feat, y)  # [B, N, C]

        # 3) dynamic feature interaction (cross attention, query=z, kv=roi)
        z2 = self.cross_attn(z.transpose(0, 1), roi.transpose(0, 1), roi.transpose(0, 1))[0].transpose(0, 1)
        z = self.norm2(z + self.dropout(z2))

        # 4) FiLM time-step modulation
        z = self.film(z, t_emb)

        # 5) prediction heads
        delta = self.bbox_embed(z)       # [B, N, 4]
        y_new = self.refine_box(y, delta)
        logit = self.class_embed(z)      # [B, N, num_classes]
        return z, y_new, logit


class NoisePredHead(nn.Module):
    """Epsilon predictor for the L_simple objective.

    Predicts the box-space noise epsilon from the noisy query features at a
    randomly sampled step t (training only).
    """

    def __init__(self, d_model=256, time_dim=128, dim_feedforward=1024,
                 in_channels=2048, output_size=7):
        super().__init__()
        self.roi_extract = RoIAlignExtractor(d_model, in_channels, output_size)
        self.film = FiLMLayer(time_dim, d_model, dim_feedforward)
        self.mlp = MLP(d_model, d_model, 4, 2)

    def forward(self, feat, y_t, t_emb):
        z = self.roi_extract(feat, y_t)   # [B, N, C]
        z = self.film(z, t_emb)
        return self.mlp(z)                # [B, N, 4]


class RDQG(nn.Module):
    """Reinforcement-Guided Diffusion Query Generator (paper Sec. 3.1).

    Training: add noise to current-frame GT boxes, run K independent reverse
    trajectories (T -> 1) with DenoiseStep, and additionally sample a random
    step to predict epsilon for L_simple.
    Inference: start from random noise boxes, run a single trajectory (K=1).
    """

    def __init__(self, d_model=256, num_queries=300, num_classes=31,
                 diffusion_steps=4, num_trajectories=5, nhead=8, dim_feedforward=1024,
                 dropout=0.1, time_dim=128, in_channels=2048, roi_output_size=7):
        super().__init__()
        assert _HAS_ROI_ALIGN, "torchvision.ops.roi_align is required by RDQG"
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.T = diffusion_steps
        self.K = num_trajectories

        self.scheduler = NoiseScheduler(num_steps=diffusion_steps)
        self.steps = nn.ModuleList([
            DenoiseStep(d_model, nhead, dim_feedforward, dropout,
                        num_classes, time_dim, in_channels, roi_output_size)
            for _ in range(diffusion_steps)
        ])
        self.noise_head = NoisePredHead(d_model, time_dim, dim_feedforward,
                                        in_channels, roi_output_size)
        # project final query state to [d_model, 2*d_model] (query_embed + tgt split)
        self.query_proj = nn.Linear(d_model, 2 * d_model)

    @torch.no_grad()
    def prepare_gt_boxes(self, targets, device):
        """Pad GT boxes (cxcywh normalized) of the current frame to num_queries.

        Returns:
            boxes: [B, num_queries, 4]
            valid: [B, num_queries] bool mask of real boxes
        """
        B = len(targets)
        boxes = torch.zeros(B, self.num_queries, 4, device=device)
        valid = torch.zeros(B, self.num_queries, dtype=torch.bool, device=device)
        for i, t in enumerate(targets):
            tb = t['boxes']                       # [M, 4] cxcywh normalized
            M = min(tb.shape[0], self.num_queries)
            boxes[i, :M] = tb[:M]
            valid[i, :M] = True
        return boxes, valid

    def _denoise(self, feat, y_start):
        """Run one reverse trajectory (shared by train/inference)."""
        z = self.steps[0].roi_extract(feat, y_start)
        y = y_start
        logit = None
        for t in range(self.T, 0, -1):
            t_emb = self.scheduler.get_time_embedding(t).to(feat.device)
            z, y, logit = self.steps[t - 1](z, y, feat, t_emb)
        return z, y, logit

    @torch.no_grad()
    def infer(self, feat, seed=None):
        """Single-trajectory reverse diffusion for inference.

        Starts from random noise boxes sampled from U(0, 1) (matching the
        training forward-diffusion endpoint) and runs the full T -> 1 denoising
        once (K=1). Pass a fixed seed to make evaluation reproducible.

        Args:
            feat : [B, C_in, H, W] current-frame features
            seed : optional int; when given, the noise boxes are sampled from a
                   seeded generator so that inference is reproducible
        Returns:
            z0  : [B, num_queries, 2*d_model] projected query embedding
            aux : dict with 'rdqg_traj_boxes' [B, 1, N, 4] and
                  'rdqg_traj_logits' [B, 1, N, C] (K=1, kept for log parity)
        """
        B = feat.shape[0]
        aux = {}
        if seed is not None:
            gen = torch.Generator(device=feat.device).manual_seed(seed)
            y_start = torch.rand(B, self.num_queries, 4, device=feat.device, generator=gen)
        else:
            y_start = torch.rand(B, self.num_queries, 4, device=feat.device)
        z_final, y, logit = self._denoise(feat, y_start)
        aux['rdqg_traj_boxes'] = y.unsqueeze(1)   # [B, 1, N, 4]
        aux['rdqg_traj_logits'] = logit.unsqueeze(1)
        z0 = self.query_proj(z_final)             # [B, N, 2C]
        return z0, aux

    def forward(self, feat, boxes=None, valid_mask=None, seed=None):
        """feat: [B, C_in, H, W] current-frame features.

        Args:
            boxes      : [B, N, 4] GT boxes (cxcywh normalized, training only)
            valid_mask : [B, N] bool mask of real boxes (training only)
            seed       : optional int for reproducible inference (K=1)

        Returns:
            z0      : [B, num_queries, 2*d_model] projected query embedding
            aux     : dict of auxiliary tensors for L_simple / RL losses
        """
        B = feat.shape[0]
        aux = {}

        if self.training and boxes is not None:
            # ---- L_simple branch: random step epsilon prediction ----
            t_rand = random.randint(1, self.T)
            t_emb = self.scheduler.get_time_embedding(t_rand).to(feat.device)
            y_t, noise = self.scheduler.q_sample(boxes, t_rand)
            noise_pred = self.noise_head(feat, y_t, t_emb)      # [B, N, 4]
            aux['rdqg_noise_pred'] = noise_pred
            aux['rdqg_noisy_boxes'] = y_t
            aux['rdqg_noise'] = noise
            aux['rdqg_t'] = t_rand
            if valid_mask is not None:
                aux['rdqg_valid_mask'] = valid_mask

            # ---- K independent reverse trajectories ----
            traj_z, traj_boxes, traj_logits = [], [], []
            for _ in range(self.K):
                y_T, _ = self.scheduler.q_sample(boxes, self.T)
                z, y, logit = self._denoise(feat, y_T)
                traj_z.append(z)
                traj_boxes.append(y)
                traj_logits.append(logit)
            z_final = traj_z[0]  # primary trajectory (matches inference K=1)
            aux['rdqg_traj_boxes'] = torch.stack(traj_boxes, 1)    # [B, K, N, 4]
            aux['rdqg_traj_logits'] = torch.stack(traj_logits, 1)  # [B, K, N, C]
        else:
            # ---- inference: random noise boxes, single trajectory (K=1) ----
            z0, infer_aux = self.infer(feat, seed=seed)
            aux.update(infer_aux)
            return z0, aux

        z0 = self.query_proj(z_final)  # [B, N, 2C]
        return z0, aux
