"""VideoMAE ViT-S/16 (Kinetics-400) with a 4-channel DI stem and a 40-class head.

Three traps, each one hit for real before this version existed:

1. THE CHECKPOINT DOES NOT LOAD UNDER transformers 5.x. The published weights store
   attention bias as `q_bias` and `v_bias` with NO key bias (k is deliberately
   unbiased). transformers 5.x wants `query.bias` / `key.bias` / `value.bias`.
   `from_pretrained` prints a load report and then continues with 36 freshly
   initialized bias tensors. The first sanity run returned "belly dancing" for six
   clips of a person pouring a drink because of this. `_load_remapped` fixes it and
   REFUSES on any residual mismatch rather than warning.

2. THE HEAD COUNT. The published config says 16 attention heads; the original
   VideoMAE constructor for ViT-S uses 6. Q/K/V shapes are identical either way, so
   loading succeeds and the computation can still be wrong. `--heads 6` overrides it,
   and `--sanity` is how you decide which is right.

3. THE ADAPTER. DI is 3-channel colorized depth plus 1-channel IR. Rather than
   reinitializing the pretrained patch embedding for 4 channels (which is what
   r2p1d.py does, discarding calibration), a 1x1x1 Conv3d maps 4 -> 3 in front of the
   UNCHANGED 3-channel embedding, initialized to identity on depth-colour with a zero
   column for IR. At step 0 the network sees exactly what it was pretrained on.

  python src/vmae.py --sanity --cache data/cache_y224
  python src/vmae.py --sanity --cache data/cache_y224 --heads 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

MODEL_ID = "MCG-NJU/videomae-small-finetuned-kinetics"
# VideoMAE preprocessing is ImageNet statistics, NOT the Kinetics constants r2p1d uses.
VM_MEAN = (0.485, 0.456, 0.406)
VM_STD = (0.229, 0.224, 0.225)


def _load_remapped(net, model_id: str) -> None:
    from huggingface_hub import hf_hub_download
    try:
        sd = torch.load(hf_hub_download(model_id, "pytorch_model.bin"),
                        map_location="cpu", weights_only=True)
    except Exception:
        from safetensors.torch import load_file
        sd = load_file(hf_hub_download(model_id, "model.safetensors"))

    out, moved = {}, 0
    for k, v in sd.items():
        if k.endswith("attention.q_bias"):
            out[k[:-len("q_bias")] + "query.bias"] = v
            out[k[:-len("q_bias")] + "key.bias"] = torch.zeros_like(v)
            moved += 1
        elif k.endswith("attention.v_bias"):
            out[k[:-len("v_bias")] + "value.bias"] = v
            moved += 1
        else:
            out[k] = v
    miss, unexp = net.load_state_dict(out, strict=False)
    miss = [m for m in miss if "classifier" not in m]
    unexp = [u for u in unexp if "classifier" not in u]
    print(f"[vmae] remapped {moved} q_bias/v_bias -> query/key/value.bias")
    print(f"[vmae] after remap: {len(miss)} missing, {len(unexp)} unexpected")
    if miss or unexp:
        raise SystemExit("REFUSED: checkpoint did not load cleanly.\n"
                         f"  missing:    {miss[:6]}\n  unexpected: {unexp[:6]}")


def _expand_stem(conv: nn.Conv3d, channels: int) -> nn.Conv3d:
    """Widen the pretrained patch embedding from 3 input channels to `channels`.

    Replaces the 4->3 adapter, which FAILED: initialized to a zero column for IR at
    lr 1e-4, its IR weights reached a norm of 0.024 after 40 epochs, i.e. the model
    trained on depth-colour alone while r2p1d used all four channels. The fix is the
    one r2p1d.py::_adapt_stem already proved: copy the pretrained 3-channel weights
    verbatim and seed the extra channel with their mean, so IR starts as a plausible
    image channel rather than as nothing.
    """
    new = nn.Conv3d(channels, conv.out_channels, conv.kernel_size, conv.stride,
                    conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        new.weight[:, :3] = conv.weight
        new.weight[:, 3:] = conv.weight.mean(1, keepdim=True).expand(
            -1, channels - 3, -1, -1, -1)
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


class VideoMAES(nn.Module):
    """Forward takes (B, T, C, H, W), the layout DIClips224 emits."""

    def __init__(self, n_classes: int = 40, channels: int = 4, pretrained: bool = True,
                 model_id: str = MODEL_ID, keep_head: bool = False, heads: int = 0,
                 stem: str = "conv4"):
        super().__init__()
        from transformers import VideoMAEConfig, VideoMAEForVideoClassification
        cfg = VideoMAEConfig.from_pretrained(model_id)
        if heads:
            cfg.num_attention_heads = heads
        net = VideoMAEForVideoClassification(cfg)
        print(f"[vmae] {model_id}  hidden {cfg.hidden_size}  "
              f"heads {cfg.num_attention_heads}  layers {cfg.num_hidden_layers}  "
              f"frames {cfg.num_frames}  tubelet {cfg.tubelet_size}")
        if pretrained:
            _load_remapped(net, model_id)
        self.labels = cfg.id2label if keep_head else None
        if not keep_head:
            net.classifier = nn.Linear(cfg.hidden_size, n_classes)
        self.net = net

        # Expand AFTER loading, so the strict shape check above still applies.
        if stem == "conv4":
            pe = net.videomae.embeddings.patch_embeddings
            pe.projection = _expand_stem(pe.projection, channels)
            pe.num_channels = channels
            cfg.num_channels = channels
            self.adapt = None
            print(f"[vmae] patch embedding widened 3 -> {channels}, "
                  f"4th channel seeded with the RGB mean")
        else:
            self.adapt = nn.Conv3d(channels, 3, 1, bias=False)
            w = torch.zeros(3, channels, 1, 1, 1)
            for i in range(3):
                w[i, i] = 1.0
            with torch.no_grad():
                self.adapt.weight.copy_(w)
        n = sum(p.numel() for p in self.parameters())
        print(f"[vmae] {n:,} params  ({n*2/1e6:.1f} MB fp16, {n*4/1e6:.1f} MB fp32)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.adapt is not None:
            x = self.adapt(x.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        return self.net(pixel_values=x).logits


def sanity(a) -> None:
    """Run the UNTOUCHED K400 head on real clips, adapter at identity."""
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).parent))
    from train_vmae import DIClips224

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = VideoMAES(pretrained=True, keep_head=True, heads=a.heads,
                  stem=a.stem).to(dev).eval()
    df = pd.read_csv(Path(a.cache) / "manifest.csv")
    df = df[(df.split == "train") & df.action_id.ge(0)]
    df = df.groupby("action_id", group_keys=False).head(1).head(a.n)   # distinct actions
    ds = DIClips224(df, a.cache, False, 0, 16, 224)
    for i in range(len(ds)):
        x, y, cid, _ = ds[i]
        with torch.no_grad():
            p = torch.softmax(m(x[None].to(dev)).float(), 1)[0].cpu()
        top = p.topk(5)
        print(f"\n{cid}  true: {df.iloc[i].action_name}")
        print("  " + "  |  ".join(f"{m.labels[int(j)]} {v:.3f}"
                                  for j, v in zip(top.indices, top.values)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sanity", action="store_true")
    p.add_argument("--cache", default="data/cache_y224")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--heads", type=int, default=0,
                   help="override num_attention_heads; 0 keeps the published config")
    p.add_argument("--stem", default="conv4", choices=["conv4", "adapter"])
    a = p.parse_args()
    if a.sanity:
        sanity(a)
    else:
        VideoMAES(heads=a.heads, stem=a.stem)
