# ------------------------------------------------------------------------
# SMTD: Sparsely-Gated Mixture-of-Experts Transformer Decoder
# ------------------------------------------------------------------------
# Implements:
#   - GateNetwork : per-token routing logits
#   - ExpertFFN    : the same FFN structure as the baseline Deformable DETR FFN
#   - QMB          : Query-aware MoE Block replacing the decoder FFN (Top-1 routing)
#   - LoadBalanceLoss : L_aux = Y * sum_i ( f_i * P_i ), switch-style load balance
#
# The MoE replaces only the FFN inside DeformableTransformerDecoderLayer, so the
# self-attention / cross-attention paths are unchanged and pretrained weights keep
# their semantics. Routing is sparse: each token is dispatched to exactly one
# expert, keeping the FLOPs equal to the baseline FFN.
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_activation_fn(activation):
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class GateNetwork(nn.Module):
    """Per-token gating network: z -> logits over Y experts."""

    def __init__(self, d_model, num_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts)
        # initialize bias to zero -> uniform routing prior at start
        nn.init.zeros_(self.gate.bias)

    def forward(self, x):
        # x: [B, N, C]
        return self.gate(x)


class ExpertFFN(nn.Module):
    """FFN with the exact structure of the baseline Deformable DETR FFN
    (linear -> act -> dropout -> linear, without residual; residual is added
    by QMB to keep the outer LayerNorm placement identical to the baseline)."""

    def __init__(self, d_model, d_ffn, dropout, activation="relu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)

    def forward(self, x):
        return self.linear2(self.dropout1(self.activation(self.linear1(x))))


class QMB(nn.Module):
    """Query-aware MoE Block.

    Replaces the decoder FFN:
        gate_logits = G(z);  gate_prob = Softmax(gate_logits)
        route_prob, route_idx = TopK(gate_prob, 1)
        z' = z + Dropout( sum_j onehot(route_idx==j) * expert_j(z) * route_prob )

    Returns (z', route_info) where route_info = (route_idx[B,N], gate_prob[B,N,Y]).
    gate_prob (the full softmax distribution) is returned for LoadBalanceLoss P_i.
    """

    def __init__(self, d_model, d_ffn, num_experts, dropout=0.1, activation="relu"):
        super().__init__()
        self.num_experts = num_experts
        self.gate = GateNetwork(d_model, num_experts)
        self.experts = nn.ModuleList(
            [ExpertFFN(d_model, d_ffn, dropout, activation) for _ in range(num_experts)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        gate_logits = self.gate(x)                 # [B, N, Y]
        gate_prob = F.softmax(gate_logits, dim=-1)  # [B, N, Y] full distr. for L_aux
        route_prob, route_idx = torch.topk(gate_prob, 1, dim=-1)  # [B, N, 1]

        # sparse expert dispatch: each token is processed by exactly one expert
        flat_x = x.reshape(-1, C)
        flat_idx = route_idx.reshape(-1)
        out = torch.zeros_like(flat_x)
        for j in range(self.num_experts):
            sel = flat_idx == j
            if bool(sel.any()):
                out[sel] = self.experts[j](flat_x[sel])
        out = out.reshape(B, N, C)

        # Top-1 softmax(TopK) probability is 1 for the selected expert; keep the
        # multiplication generic (e.g. future Top-2) and consistent with the paper.
        out = out * route_prob
        out = x + self.dropout(out)  # residual, same as baseline FFN

        route_info = (route_idx.squeeze(-1), gate_prob)
        return out, route_info


class LoadBalanceLoss(nn.Module):
    """Switch-style load-balancing auxiliary loss:
        L_aux = Y * sum_{i=1..Y} ( f_i * P_i )
      f_i : fraction of tokens routed to expert i (via argmax Top-1)
      P_i : mean gate probability of expert i over all tokens (via softmax)

    Minimizing L_aux encourages a uniform distribution of tokens over experts.
    """

    def __init__(self, num_experts):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, route_idx, gate_prob):
        # route_idx: [B, N], gate_prob: [B, N, Y]
        one_hot = F.one_hot(route_idx, num_classes=self.num_experts).float()  # [B,N,Y]
        f = one_hot.mean(dim=(0, 1))       # [Y] fraction of tokens per expert
        P = gate_prob.mean(dim=(0, 1))     # [Y] mean routing probability per expert
        aux = self.num_experts * (f * P).sum()
        return aux
