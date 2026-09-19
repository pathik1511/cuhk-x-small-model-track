"""CUHKX_MultiStreamNet: three specialised streams and an adaptive gated fusion head.

Design notes that deviate from a generic spec, each forced by a measurement on this
dataset. See RESULTS.md for the evidence behind every one.

  * GroupNorm everywhere, never BatchNorm. The split is cross-subject, and BN running
    statistics are estimated on the training subjects. GN also makes the batch size and
    the PK sampler independent of normalisation behaviour.
  * The skeleton stream defaults to 4 input channels, not 3. Raw channels are
    root-centred metric 3D; the 4th is floor-referenced height divided by torso length.
    Passing raw 3 reverts the stature fix that was worth +0.0234 OOF.
  * M (bodies) is supported but is 1 in this corpus. Single-actor clips throughout.
  * The 1D stream clamps its dilations to the sequence length. Radar arrives as 12
    numbers at ONE timestep and IMU at 2 source rows; a multi-scale dilated stack over
    T=1 is arithmetic with no signal, so the stream degenerates to an MLP and says so.
  * Visual modalities are a constructor argument, not a fixed list. Depth+IR fused as a
    single 4-channel encoder ("DI") is the measured-best configuration; adding Thermal,
    IMU and Radar measured -0.071 on 3,036 clips.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

H36M_PARENT = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]


def gn(c: int, groups: int = 8) -> nn.GroupNorm:
    return nn.GroupNorm(min(groups, c), c)


def skeleton_adjacency(parents=H36M_PARENT) -> torch.Tensor:
    """Three normalised subsets: self, inward (child to parent), outward."""
    v = len(parents)
    a = torch.zeros(3, v, v)
    a[0] = torch.eye(v)
    for c, p in enumerate(parents):
        if c != p:
            a[1, c, p] = 1.0
            a[2, p, c] = 1.0
    return a / a.sum(-1, keepdim=True).clamp(min=1)


# --------------------------------------------------------------- 1. skeleton ---
class CTRGC(nn.Module):
    """Channel-wise Topology Refinement Graph Convolution.

    A shared topology per subset plus a per-channel refinement inferred from the
    features themselves. The refinement is a rank-1 pairwise difference passed through
    tanh, so the cost is one 1x1 conv over a V x V map, not a dense V^2 matmul per
    channel.
    """

    def __init__(self, cin: int, cout: int, subsets: int = 3, ratio: int = 8):
        super().__init__()
        rel = max(8, cout // ratio)
        self.subsets = subsets
        self.reduce = nn.Conv2d(cin, rel, 1)
        self.expand = nn.Conv2d(rel, cout * subsets, 1)
        self.value = nn.Conv2d(cin, cout * subsets, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.cout = cout

    def forward(self, x, a):                      # x (N,C,T,V), a (S,V,V)
        n, _, t, v = x.shape
        s, c = self.subsets, self.cout
        r = self.reduce(x).mean(2)                                    # (N,rel,V)
        r = torch.tanh(r.unsqueeze(-1) - r.unsqueeze(-2))             # (N,rel,V,V)
        refine = self.expand(r).view(n, s, c, v, v)                   # per-channel topology
        top = a.view(1, s, 1, v, v) + self.alpha * refine
        val = self.value(x).view(n, s, c, t, v)
        return torch.einsum("nsctv,nscvw->nctw", val, top)


class GCNTCN(nn.Module):
    """One CTR-GCN block: graph convolution over V, then a strided conv over T."""

    def __init__(self, cin: int, cout: int, stride: int = 1, subsets: int = 3):
        super().__init__()
        self.gcn, self.n1 = CTRGC(cin, cout, subsets), gn(cout)
        self.tcn = nn.Sequential(
            nn.Conv2d(cout, cout, (5, 1), (stride, 1), (2, 0), groups=cout),
            nn.Conv2d(cout, cout, 1), gn(cout))
        self.res = (nn.Identity() if cin == cout and stride == 1 else
                    nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)), gn(cout)))

    def forward(self, x, a):
        y = F.silu(self.n1(self.gcn(x, a)))
        return F.silu(self.tcn(y) + self.res(x))


class SkeletonStream(nn.Module):
    """Input (B, C, T, V, M). Bodies are folded into the batch and max-pooled back."""

    def __init__(self, cin: int = 4, width: int = 48, dim: int = 256, blocks=(1, 2, 2)):
        super().__init__()
        self.register_buffer("A", skeleton_adjacency())
        self.stem = nn.Sequential(nn.Conv2d(cin, width, 1), gn(width), nn.SiLU())
        layers, c = [], width
        for i, n in enumerate(blocks):
            for j in range(n):
                layers.append(GCNTCN(c, width * 2 ** i, 2 if (j == 0 and i) else 1))
                c = width * 2 ** i
        self.blocks = nn.ModuleList(layers)
        self.head = nn.Linear(c, dim)

    def forward(self, x):                          # (B,C,T,V,M)
        b, c, t, v, m = x.shape
        y = self.stem(x.permute(0, 4, 1, 2, 3).reshape(b * m, c, t, v))
        for blk in self.blocks:
            y = blk(y, self.A)
        y = y.mean((2, 3)).view(b, m, -1).amax(1)
        return self.head(y)


# ----------------------------------------------------------------- 2. visual ---
def temporal_shift(x, t: int, fold_div: int = 8):
    """Zero-parameter temporal mixing. Shifts 2/fold_div of the channels along time."""
    nt, c, h, w = x.shape
    x = x.view(nt // t, t, c, h, w)
    f = max(1, c // fold_div)
    y = torch.zeros_like(x)
    y[:, :-1, :f] = x[:, 1:, :f]
    y[:, 1:, f:2 * f] = x[:, :-1, f:2 * f]
    y[:, :, 2 * f:] = x[:, :, 2 * f:]
    return y.view(nt, c, h, w)


def channel_shuffle(x, groups: int = 2):
    n, c, h, w = x.shape
    return x.view(n, groups, c // groups, h, w).transpose(1, 2).reshape(n, c, h, w)


class ShuffleTSM(nn.Module):
    """ShuffleNetV2 unit with a Temporal Shift Module on the branch input."""

    def __init__(self, cin: int, cout: int, stride: int = 1, shift: bool = True):
        super().__init__()
        self.stride, self.shift = stride, shift
        br = cout // 2
        inp = cin if stride > 1 else cin // 2
        self.main = nn.Sequential(
            nn.Conv2d(inp, br, 1, bias=False), gn(br), nn.SiLU(),
            nn.Conv2d(br, br, 3, stride, 1, groups=br, bias=False), gn(br),
            nn.Conv2d(br, br, 1, bias=False), gn(br), nn.SiLU())
        self.side = nn.Sequential(
            nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False), gn(cin),
            nn.Conv2d(cin, cout - br, 1, bias=False), gn(cout - br), nn.SiLU()
        ) if stride > 1 else nn.Identity()

    def forward(self, x, t: int):
        if self.stride > 1:
            z = temporal_shift(x, t) if self.shift else x
            return torch.cat([self.side(x), self.main(z)], 1)
        a, b = x.chunk(2, 1)
        z = temporal_shift(b, t) if self.shift else b
        return channel_shuffle(torch.cat([a, self.main(z)], 1))


class VisualStream(nn.Module):
    """Input (B, T, C, H, W). One encoder per visual modality group."""

    def __init__(self, cin: int, width: int = 32, dim: int = 256, blocks=(2, 3, 2)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(cin, width, 3, 2, 1, bias=False), gn(width), nn.SiLU(),
            nn.Conv2d(width, width, 3, 2, 1, bias=False), gn(width), nn.SiLU())
        layers, c = [], width
        for i, n in enumerate(blocks):
            for j in range(n):
                out = width * 2 ** (i + 1)
                layers.append(ShuffleTSM(c, out, 2 if j == 0 else 1))
                c = out
        self.blocks = nn.ModuleList(layers)
        self.head = nn.Linear(c, dim)

    def forward(self, x):                          # (B,T,C,H,W)
        b, t = x.shape[:2]
        y = self.stem(x.flatten(0, 1))
        for blk in self.blocks:
            y = blk(y, t)
        y = y.mean((2, 3)).view(b, t, -1).mean(1)  # spatial pool, then temporal pool
        return self.head(y)


# -------------------------------------------------------------- 3. 1D series ---
class SE(nn.Module):
    def __init__(self, c: int, r: int = 8):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, max(4, c // r)), nn.SiLU(),
                                nn.Linear(max(4, c // r), c), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x.mean(-1)).unsqueeze(-1)


class TCNBlock(nn.Module):
    def __init__(self, c: int, dilation: int, k: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c, c, k, padding=dilation * (k - 1) // 2, dilation=dilation,
                      groups=c, bias=False), nn.GroupNorm(min(8, c), c),
            nn.Conv1d(c, c, 1), nn.SiLU(), SE(c))
    def forward(self, x):
        return F.silu(x + self.net(x))


class SeriesStream(nn.Module):
    """Input (B, C_sensor, T_sensor). Dilations are clamped to the sequence length.

    Radar in this corpus is 12 values at a single timestep and IMU is 2 source rows.
    When T is 1 the dilated stack collapses to a pointwise MLP, which `degenerate`
    reports so a caller cannot mistake it for temporal modelling.
    """

    def __init__(self, cin: int, seq_len: int, width: int = 64, dim: int = 256,
                 dilations=(1, 2, 4, 8)):
        super().__init__()
        self.dilations = tuple(d for d in dilations if d < max(seq_len, 2)) or (1,)
        self.degenerate = seq_len <= 2
        self.stem = nn.Sequential(nn.Conv1d(cin, width, 1), nn.GroupNorm(min(8, width), width), nn.SiLU())
        self.blocks = nn.Sequential(*[TCNBlock(width, d) for d in self.dilations])
        self.head = nn.Linear(width, dim)

    def forward(self, x):                          # (B,C,T)
        return self.head(self.blocks(self.stem(x)).mean(-1))


# ----------------------------------------------------------------- 4. fusion ---
class ModalityGate(nn.Module):
    """Residual modality attention over present streams, plus availability embeddings.

    A missing stream contributes its learned 'absent' embedding rather than a zero
    vector, so the aggregate is not pulled toward the origin by absence, and the gate
    is conditioned on the availability pattern itself.
    """

    def __init__(self, n_mod: int, dim: int):
        super().__init__()
        self.absent = nn.Parameter(torch.zeros(n_mod, dim))
        self.present = nn.Parameter(torch.zeros(n_mod, dim))
        self.gate = nn.Sequential(nn.Linear(n_mod * dim + n_mod, dim), nn.SiLU(),
                                  nn.Linear(dim, n_mod), nn.Sigmoid())
        self.proj = nn.Linear(n_mod * dim, dim)

    def forward(self, feats, mask):                # feats (B,M,D), mask (B,M)
        m = mask.unsqueeze(-1)
        z = m * (feats + self.present) + (1 - m) * self.absent
        g = self.gate(torch.cat([z.flatten(1), mask], -1)) * mask   # never gate in an absent stream
        z = z * (1.0 + g).unsqueeze(-1)                             # residual, not replacement
        return self.proj(z.flatten(1))


class CUHKX_MultiStreamNet(nn.Module):
    """Three streams, gated fusion, 40-way head.

    visual: dict name -> input channels, e.g. {"DI": 4} or {"Depth": 3, "IR": 1}
    series: dict name -> (channels, sequence length)
    """

    def __init__(self, n_classes: int = 40, dim: int = 256,
                 visual: dict | None = None, series: dict | None = None,
                 skeleton_ch: int = 4, skeleton: bool = True,
                 visual_width: int = 32, skel_width: int = 48, series_width: int = 64,
                 dropout: float = 0.4):
        super().__init__()
        visual = {"DI": 4} if visual is None else visual
        series = {} if series is None else series
        self.enc = nn.ModuleDict()
        if skeleton:
            self.enc["Skeleton"] = SkeletonStream(skeleton_ch, skel_width, dim)
        for k, c in visual.items():
            self.enc[k] = VisualStream(c, visual_width, dim)
        for k, (c, t) in series.items():
            self.enc[k] = SeriesStream(c, t, series_width, dim)
        self.names = list(self.enc)
        self.fuse = ModalityGate(len(self.names), dim)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout),
                                  nn.Linear(dim, n_classes))

    def forward(self, batch: dict) -> torch.Tensor:
        """batch maps a stream name to its tensor, or omits it / passes None if absent."""
        feats, mask, ref = [], [], None
        for k in self.names:
            x = batch.get(k)
            if x is not None:
                f = self.enc[k](x)
                ref = f
                feats.append(f)
                mask.append(1.0)
            else:
                feats.append(None)
                mask.append(0.0)
        if ref is None:
            raise ValueError("every modality was absent for this sample")
        feats = [f if f is not None else torch.zeros_like(ref) for f in feats]
        m = torch.tensor(mask, device=ref.device, dtype=ref.dtype).expand(ref.size(0), -1)
        return self.head(self.fuse(torch.stack(feats, 1), m))
