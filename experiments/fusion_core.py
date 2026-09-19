"""
fusion_core.py — Dynamic Multimodal Fusion Core for CUHK-X Small Model Track.

Implements masked-attention fusion with an optional MBT-style bottleneck, plus
per-modality auxiliary heads (deep supervision) and learned reliability gating.

DESIGN NOTE, stated honestly:
  MBT's bottleneck is a *compression* device. Its benefit appears when each modality
  contributes MANY tokens (AudioSet: 8x8 spatial x T temporal), where restricting
  cross-modal attention to B bottleneck tokens cuts O(N^2) to O(B*N) and regularizes.
  If every encoder emits a single pooled vector, the "sequence" is 6 tokens long and
  there is nothing to bottleneck — plain masked attention is strictly simpler and
  mathematically equivalent. This module therefore expects encoders to emit K tokens
  each ([B, K, d]), and `n_bottleneck=0` falls back to plain masked attention so the
  two can be ablated against each other with one flag.

MATH — masked attention pooling.
  Given per-modality token sets Z_m in R^{K x d} and availability a_m in {0,1}:

      alpha_m,k = a_m * exp(q^T W_k Z_m,k / sqrt(d))  /  SUM_{m',k'} a_m' exp(...)
      z_fused   = SUM_{m,k} alpha_m,k * W_v Z_m,k

  The mask enters the softmax DENOMINATOR, not the numerator alone. Consequence:
  SUM alpha = 1 regardless of how many modalities fired, so z_fused has identical
  scale whether 6 sensors or 2 fired. This is the property zero-imputation lacks —
  feeding zeros makes "sensor absent" indistinguishable from "sensor read zero",
  which is a legal value for every modality here.

  Deep supervision (aux heads) matters more than the fusion operator at this data
  scale: with 3,036 clips a single fused loss lets the strongest modality monopolize
  gradient flow, and the weak encoders never learn. Per-modality CE forces each to be
  independently discriminative, which is also what makes the fused model degrade
  gracefully when a modality drops out.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e4          # finite, so fp16 stays safe


class MaskedAttention(nn.Module):
    """Multi-head attention where key/value tokens carry an availability mask."""

    def __init__(self, d: int, heads: int = 4, drop: float = 0.1):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q, self.k, self.v = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.o, self.drop = nn.Linear(d, d), nn.Dropout(drop)

    def forward(self, q, kv, mask):
        """q [B,Nq,d]   kv [B,Nk,d]   mask [B,Nk] (1=available) -> [B,Nq,d]"""
        B, Nq, d = q.shape
        Nk = kv.shape[1]
        sh = lambda t, N: t.view(B, N, self.h, self.dk).transpose(1, 2)   # [B,h,N,dk]
        qq, kk, vv = sh(self.q(q), Nq), sh(self.k(kv), Nk), sh(self.v(kv), Nk)
        att = (qq @ kk.transpose(-2, -1)) / self.dk ** 0.5                # [B,h,Nq,Nk]
        att = att.masked_fill(mask[:, None, None, :] == 0, NEG)
        att = self.drop(att.softmax(-1))
        out = (att @ vv).transpose(1, 2).reshape(B, Nq, d)                # [B,Nq,d]
        return self.o(out)


class FusionBlock(nn.Module):
    """Pre-norm block: bottleneck (or CLS) queries attend over available tokens."""

    def __init__(self, d: int, heads: int = 4, mlp: int = 4, drop: float = 0.1):
        super().__init__()
        self.n1, self.attn = nn.LayerNorm(d), MaskedAttention(d, heads, drop)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * mlp), nn.GELU(),
                                nn.Dropout(drop), nn.Linear(d * mlp, d))

    def forward(self, q, kv, mask):
        q = q + self.attn(self.n1(q), kv, mask)
        return q + self.ff(self.n2(q))


class DynamicFusionCore(nn.Module):
    """Fuses a dict of per-modality token tensors, tolerating None / all-absent entries.

    Args
      mod_dims     {"depth": 512, "skeleton": 256, ...} feature width each encoder emits
      d            common fusion width
      n_bottleneck B MBT tokens; 0 -> plain masked attention pooling (ablation switch)
      aux          per-modality auxiliary classifier heads (deep supervision)
      gate         learned per-modality reliability scalar used to reweight aux logits
    """

    def __init__(self, mod_dims: dict[str, int], n_classes: int = 40, d: int = 256,
                 n_bottleneck: int = 4, n_layers: int = 2, heads: int = 4,
                 drop: float = 0.1, aux: bool = True, gate: bool = True):
        super().__init__()
        self.mods = list(mod_dims)
        self.d, self.B, self.use_gate = d, n_bottleneck, gate

        # 1. project every modality into the shared space; add a learned identity code
        self.proj = nn.ModuleDict({m: nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, d))
                                   for m, dm in mod_dims.items()})
        self.mod_emb = nn.Parameter(torch.randn(len(self.mods), d) * 0.02)

        # 2. queries: B bottleneck tokens (MBT) or a single CLS token (plain pooling)
        self.query = nn.Parameter(torch.randn(max(n_bottleneck, 1), d) * 0.02)
        self.blocks = nn.ModuleList([FusionBlock(d, heads, drop=drop) for _ in range(n_layers)])

        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(d, n_classes))
        self.aux = (nn.ModuleDict({m: nn.Linear(d, n_classes) for m in self.mods})
                    if aux else None)
        self.gate = (nn.ModuleDict({m: nn.Linear(d, 1) for m in self.mods})
                     if gate else None)

    # --------------------------------------------------------------------- #
    @staticmethod
    def _tokens(x: torch.Tensor) -> torch.Tensor:
        """Accept [B,d] or [B,K,d]; always return [B,K,d]."""
        return x.unsqueeze(1) if x.dim() == 2 else x

    def forward(self, feats: dict[str, torch.Tensor | None],
                avail: dict[str, torch.Tensor] | None = None) -> dict:
        """
        feats  {mod: [B,d_m] or [B,K_m,d_m] or None}
        avail  optional {mod: [B]} float mask; a modality absent for SOME samples in the
               batch (the common case — you cannot pass None per-sample) is handled here.
        returns {"logits":[B,C], "z":[B,d], "aux":{mod:[B,C]}, "mask":[B,M]}
        """
        ref = next(t for t in feats.values() if t is not None)
        B, dev = ref.shape[0], ref.device

        tok, msk, per_mod = [], [], {}
        for i, m in enumerate(self.mods):
            x = feats.get(m)
            if x is None:                                   # modality absent for whole batch
                continue
            z = self.proj[m](self._tokens(x)) + self.mod_emb[i]        # [B,K_m,d]
            a = (avail[m] if avail and m in avail
                 else torch.ones(B, device=dev, dtype=z.dtype))        # [B]
            tok.append(z)
            msk.append(a[:, None].expand(-1, z.shape[1]))              # [B,K_m]
            per_mod[m] = (z * a[:, None, None]).sum(1) / z.shape[1]    # pooled, [B,d]

        if not tok:                                          # nothing available at all
            z = torch.zeros(B, self.d, device=dev)
            return {"logits": self.head(z), "z": z, "aux": {}, "mask": torch.zeros(B, len(self.mods), device=dev)}

        kv = torch.cat(tok, 1)                               # [B, N=sum K_m, d]
        mask = torch.cat(msk, 1)                             # [B, N]

        # A sample with zero available tokens would make softmax uniform over NEG.
        # Force at least one live slot so attention stays finite; its output is then
        # zeroed by `alive` below.
        alive = (mask.sum(1, keepdim=True) > 0).to(mask.dtype)          # [B,1]
        mask = torch.where(alive.bool(), mask, F.one_hot(torch.zeros(B, dtype=torch.long, device=dev),
                                                        mask.shape[1]).to(mask.dtype))

        q = self.query.unsqueeze(0).expand(B, -1, -1)        # [B, B_tok, d]
        for blk in self.blocks:
            q = blk(q, kv, mask)                             # bottleneck reads modalities
        z = self.norm(q.mean(1)) * alive                     # [B,d]

        out = {"logits": self.head(z), "z": z, "aux": {},
               "mask": torch.stack([(avail[m] if avail and m in avail
                                     else torch.ones(B, device=dev)) if m in per_mod
                                    else torch.zeros(B, device=dev) for m in self.mods], 1)}

        if self.aux is not None:
            for m, zm in per_mod.items():
                lm = self.aux[m](zm)                         # [B,C] per-modality logits
                if self.use_gate:
                    lm = lm * torch.sigmoid(self.gate[m](zm))   # learned reliability in (0,1)
                out["aux"][m] = lm
        return out


def fusion_loss(out: dict, y: torch.Tensor, mods: list[str],
                w_aux: float = 0.3, smooth: float = 0.1) -> torch.Tensor:
    """Fused CE + deep supervision, averaged only over modalities actually present."""
    loss = F.cross_entropy(out["logits"], y, label_smoothing=smooth)
    if out["aux"]:
        parts = []
        for k, m in enumerate(mods):
            if m not in out["aux"]:
                continue
            a = out["mask"][:, k]
            if a.sum() < 1:
                continue
            ce = F.cross_entropy(out["aux"][m], y, label_smoothing=smooth, reduction="none")
            parts.append((ce * a).sum() / a.sum())           # present samples only
        if parts:
            loss = loss + w_aux * torch.stack(parts).mean()
    return loss


if __name__ == "__main__":
    DIMS = {"depth": 512, "ir": 512, "thermal": 512, "skeleton": 256, "imu": 128, "radar": 128}
    for nb in (0, 4):
        core = DynamicFusionCore(DIMS, n_bottleneck=nb)
        n = sum(p.numel() for p in core.parameters())
        print(f"n_bottleneck={nb}: {n/1e6:.3f}M params  {n*4/1e6:.2f} MB")

    core = DynamicFusionCore(DIMS, n_bottleneck=4)
    B = 4
    feats = {"depth": torch.randn(B, 8, 512), "ir": torch.randn(B, 8, 512),
             "thermal": None,                                  # absent for the whole batch
             "skeleton": torch.randn(B, 4, 256), "imu": torch.randn(B, 128),
             "radar": torch.randn(B, 128)}
    avail = {"radar": torch.tensor([1., 0., 1., 0.]),           # per-sample absence
             "imu": torch.tensor([1., 1., 1., 0.])}
    o = core(feats, avail)
    print(f"\nlogits {tuple(o['logits'].shape)}  z {tuple(o['z'].shape)}  "
          f"aux {list(o['aux'])}")
    print(f"mask (rows=samples, cols={list(DIMS)}):\n{o['mask']}")
    y = torch.randint(0, 40, (B,))
    L = fusion_loss(o, y, list(DIMS))
    L.backward()
    g = sum(p.grad.abs().sum() for p in core.parameters() if p.grad is not None)
    print(f"loss {L.item():.4f}   grad flows: {bool(g > 0)}")

    # degenerate case: a sample with nothing available must not produce NaN
    o2 = core({"depth": torch.randn(2, 8, 512)}, {"depth": torch.tensor([1., 0.])})
    print(f"all-absent sample logits finite: {torch.isfinite(o2['logits']).all().item()}  "
          f"z-norm per sample: {o2['z'].norm(dim=1).tolist()}")
