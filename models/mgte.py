# ------------------------------------------------------------------------
# MGTE: Multi-Scale Graph Interaction Transformer Encoder
# ------------------------------------------------------------------------
# The MGTE adds a Multi-Scale Dynamic Graph Convolution (MS-DGC) branch in
# parallel with the existing MHDA (MSDeformAttn) path of each encoder layer.
# The two branches are blended with a learnable soft coefficient:
#       U = lambda * F_MHDA + (1 - lambda) * F_DGC
#
# Key components:
#   - KNNBuilder       : per-scale cosine-similarity graph, chunked topk (k=11)
#   - GraphConvLayer   : symmetric-normalized propagation + LeakyReLU + residual
#   - LayerWiseFusion  : learnable weighted fusion of L graph-conv outputs
#   - DynamicGraphConv : MS-DGC block (per-scale graph, L layers)
#   - MSGraphEncoderLayer : wrapper layer used inside DeformableTransformerEncoderLayer
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch import nn


class KNNBuilder(nn.Module):
    """Build a k-NN adjacency graph from cosine similarity.

    Graph is built per scale, per sample. The similarity matrix is computed
    in column chunks so the dense [S, S] peak memory is avoided.

    Returns:
        idx        : [S, K] neighbor indices (long)
        valid_mask : [S, K] bool, True where the neighbor is not a padded token
    """

    def __init__(self, knn_k=11, chunk=2048):
        super().__init__()
        self.knn_k = knn_k
        self.chunk = chunk

    def build(self, x, padding=None):
        """x: [S, C] single sample, single scale; padding: [S] bool (True=pad)."""
        S, C = x.shape
        k = min(self.knn_k, S)
        if S <= 1:
            idx = torch.zeros(S, k, dtype=torch.long, device=x.device)
            valid = torch.ones(S, k, dtype=torch.bool, device=x.device)
            return idx, valid

        x_norm = F.normalize(x, dim=-1)  # [S, C]
        all_vals, all_idx = [], []
        step = self.chunk
        for start in range(0, S, step):
            end = min(start + step, S)
            sim = x_norm @ x_norm[start:end].T  # [S, step]
            if padding is not None:
                # padded tokens must not be selected as neighbors
                sim = sim.masked_fill(padding[start:end].unsqueeze(0), float('-inf'))
            vals, idx = sim.topk(k, dim=1)
            all_vals.append(vals)
            all_idx.append(idx + start)
        all_vals = torch.cat(all_vals, 1)  # [S, n_chunks * k]
        all_idx = torch.cat(all_idx, 1)
        vals, topk_pos = all_vals.topk(k, dim=1)
        idx = torch.gather(all_idx, 1, topk_pos)
        valid_mask = vals > float('-inf') * 0.5  # ~vals != -inf
        return idx, valid_mask


class GraphConvLayer(nn.Module):
    """One symmetric-normalized graph convolution layer:

        G' = LeakyReLU(D^-1/2 A D^-1/2 G W) + G   (residual)

    A is the binary k-NN adjacency (with self-loop, since cos(x,x)=1 always
    ranks first in topk). D is the degree matrix.
    """

    def __init__(self, d_model, leaky_slope=0.2):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.leaky = nn.LeakyReLU(leaky_slope)

    def forward(self, x, idx, valid_mask):
        """x: [S, C]; idx: [S, K]; valid_mask: [S, K] bool."""
        S, C = x.shape
        K = idx.shape[1]

        # degrees from the binary adjacency (valid neighbors only)
        flat_idx = idx[valid_mask]
        col_deg = torch.zeros(S, device=x.device, dtype=torch.float32).scatter_add(
            0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
        row_deg = valid_mask.sum(1).float().clamp(min=1.0)
        col_deg = col_deg.clamp(min=1.0)

        w = 1.0 / torch.sqrt(row_deg.unsqueeze(1) * col_deg[idx])  # [S, K]
        w = w * valid_mask.float()

        x_nei = x[idx]  # [S, K, C]
        agg = (w.unsqueeze(-1) * x_nei).sum(1)  # [S, C]
        out = self.leaky(self.linear(agg)) + x
        return out


class LayerWiseFusion(nn.Module):
    """Learnable weighted fusion of the L graph-conv outputs (softmax weights)."""

    def __init__(self, num_layers):
        super().__init__()
        self.coef = nn.Parameter(torch.ones(num_layers) / num_layers)

    def forward(self, feats):
        w = self.coef.softmax(0)
        out = 0.0
        for wi, fi in zip(w, feats):
            out = out + wi * fi
        return out


class DynamicGraphConv(nn.Module):
    """MS-DGC block: per-scale k-NN graph, L graph-conv layers, layer-wise fusion."""

    def __init__(self, d_model=256, num_layers=2, knn_k=11, chunk=2048, leaky_slope=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.knn = KNNBuilder(knn_k=knn_k, chunk=chunk)
        self.layers = nn.ModuleList([GraphConvLayer(d_model, leaky_slope) for _ in range(num_layers)])
        self.fusion = LayerWiseFusion(num_layers)

    def _graph_conv_scale(self, x, padding=None):
        """Graph conv for one scale, one sample. x: [S_l, C]."""
        S_l = x.shape[0]
        if S_l <= 1:
            return x
        idx, valid = self.knn.build(x, padding)
        layer_feats = []
        g = x
        for layer in self.layers:
            g = layer(g, idx, valid)
            layer_feats.append(g)
        out = self.fusion(layer_feats)
        if padding is not None:
            out = torch.where(padding.unsqueeze(-1), x, out)
        return out

    def forward(self, src, spatial_shapes, level_start_index, padding_mask=None):
        """src: [B, S, C]; spatial_shapes: [L, 2]; level_start_index: [L].

        Graphs are built independently within each scale (no cross-scale edges).
        Returns [B, S, C] with the same layout as the input.
        """
        B, S, C = src.shape
        outs = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            start = int(level_start_index[lvl].item() if torch.is_tensor(level_start_index) else level_start_index[lvl])
            end = start + int(H) * int(W)
            x = src[:, start:end]  # [B, HW, C]
            pad = padding_mask[:, start:end] if padding_mask is not None else None  # [B, HW]
            feats = []
            for b in range(B):
                xb = x[b]
                padb = pad[b] if pad is not None else None
                feats.append(self._graph_conv_scale(xb, padb))
            outs.append(torch.stack(feats, 0))
        return torch.cat(outs, 1)


class MSGraphEncoderLayer(nn.Module):
    """Wrapper layer: MS-DGC branch, called from DeformableTransformerEncoderLayer."""

    def __init__(self, d_model=256, num_layers=2, knn_k=11, chunk=2048, leaky_slope=0.2):
        super().__init__()
        self.ms_dgc = DynamicGraphConv(d_model, num_layers, knn_k, chunk, leaky_slope)

    def forward(self, src, spatial_shapes, level_start_index, padding_mask=None):
        return self.ms_dgc(src, spatial_shapes, level_start_index, padding_mask)
