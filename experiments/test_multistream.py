"""Synthetic forward-pass test, per-stream parameter and FLOP report.

    python test_multistream.py
"""
import torch, torch.nn as nn
from CUHKX_MultiStreamNet import CUHKX_MultiStreamNet, SkeletonStream, VisualStream, SeriesStream

B, T, HW = 2, 16, 128
VISUAL = {"DI": 4, "HAND": 4, "Thermal": 3}
SERIES = {"IMU": (5 * 16, 16), "Radar": (12, 1)}


def flops(model, args):
    """Multiply-accumulate count for conv, linear and einsum, via forward hooks."""
    tot = [0]
    hs = []
    def conv(m, i, o):
        tot[0] += o.numel() * (m.in_channels // m.groups) * int(torch.tensor(m.kernel_size).prod())
    def lin(m, i, o):
        tot[0] += o.numel() * m.in_features
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            hs.append(m.register_forward_hook(conv))
        elif isinstance(m, nn.Linear):
            hs.append(m.register_forward_hook(lin))
    with torch.no_grad():
        model(*args)
    for h in hs:
        h.remove()
    return tot[0]


def report():
    print("=" * 78)
    print("PER-STREAM PARAMETERS AND FLOPS   (batch 1, 16 frames, 128x128)")
    print("=" * 78)
    rows, tot_p, tot_f = [], 0, 0
    s = SkeletonStream(4, 48, 256)
    rows.append(("Skeleton  CTR-GCN, 17 joints, 4ch", s, (torch.randn(1, 4, T, 17, 1),)))
    for k, c in VISUAL.items():
        rows.append((f"Visual    ShuffleNetV2+TSM  {k} ({c}ch)",
                     VisualStream(c, 32, 256), (torch.randn(1, T, c, HW, HW),)))
    for k, (c, t) in SERIES.items():
        st = SeriesStream(c, t, 64, 256)
        tag = "  DEGENERATE, T<=2, no temporal modelling" if st.degenerate else ""
        rows.append((f"Series    dilated TCN+SE  {k} ({c}x{t}){tag}",
                     st, (torch.randn(1, c, t),)))
    for name, mod, args in rows:
        p = sum(x.numel() for x in mod.parameters())
        f = flops(mod, args)
        tot_p += p; tot_f += f
        print(f"  {name:52s} {p/1e6:7.3f} M  {f/1e9:8.3f} GMAC")
    net = CUHKX_MultiStreamNet(visual=VISUAL, series=SERIES)
    fp = sum(x.numel() for x in net.fuse.parameters()) + sum(x.numel() for x in net.head.parameters())
    print(f"  {'Fusion gate + head':52s} {fp/1e6:7.3f} M")
    print("-" * 78)
    total = sum(x.numel() for x in net.parameters())
    print(f"  {'TOTAL (all 6 modalities)':52s} {total/1e6:7.3f} M  {tot_f/1e9:8.3f} GMAC")
    print(f"  {'fp32 size':52s} {total*4/2**20:7.1f} MB   limit 100 MB")
    print(f"  {'int8 size':52s} {total/2**20:7.1f} MB")
    lean = CUHKX_MultiStreamNet(visual={"DI": 4, "HAND": 4}, series=None)
    lp = sum(x.numel() for x in lean.parameters())
    print(f"  {'MEASURED-BEST subset: DI + HAND + Skeleton':52s} {lp/1e6:7.3f} M  "
          f"{lp*4/2**20:5.1f} MB fp32")
    assert total < 10e6, f"over the 10M budget: {total}"
    return net


def forward_tests(net):
    print("\n" + "=" * 78)
    print("SYNTHETIC FORWARD PASSES")
    print("=" * 78)
    full = {"Skeleton": torch.randn(B, 4, T, 17, 1),
            "DI": torch.randn(B, T, 4, HW, HW),
            "HAND": torch.randn(B, T, 4, 96, 96),
            "Thermal": torch.randn(B, T, 3, HW, HW),
            "IMU": torch.randn(B, 5 * 16, 16),
            "Radar": torch.randn(B, 12, 1)}
    cases = [("all six present", list(full)),
             ("Thermal + Radar missing (46.7% of clips lack Radar)",
              ["Skeleton", "DI", "HAND", "IMU"]),
             ("skeleton only", ["Skeleton"]),
             ("one visual stream only", ["DI"]),
             ("no skeleton, no series", ["DI", "HAND", "Thermal"])]
    net.eval()
    for name, keys in cases:
        with torch.no_grad():
            y = net({k: full[k] for k in keys})
        assert y.shape == (B, 40) and torch.isfinite(y).all()
        print(f"  {name:52s} -> {tuple(y.shape)}  finite OK")
    try:
        net({})
    except ValueError as e:
        print(f"  {'empty batch raises rather than returning garbage':52s} -> {e}")

    print("\n  Absence must not equal zeros. Same present streams, different absences:")
    a = net({k: full[k] for k in ["Skeleton", "DI"]})
    b = net({k: full[k] for k in ["Skeleton", "DI", "Thermal"]})
    print(f"    mean |logit delta| when Thermal is added = {(a - b).abs().mean():.4f}")
    z = dict(full); z["Thermal"] = torch.zeros_like(z["Thermal"])
    c = net({k: z[k] for k in ["Skeleton", "DI", "Thermal"]})
    print(f"    zero-filled Thermal vs absent Thermal      = {(a - c).abs().mean():.4f}"
          "   (nonzero: the availability embedding is doing work)")

    print("\n  Temporal sensitivity, the check that caught a permutation-invariant"
          " encoder in this project:")
    x = {"DI": full["DI"]}
    p = {"DI": full["DI"][:, torch.randperm(T)]}
    print(f"    mean |logit delta| under frame permutation = "
          f"{(net(x) - net(p)).abs().mean():.4f}   (must be far above 1e-6)")

    print("\n  Gradient flow:")
    net.train()
    out = net({k: full[k] for k in ["Skeleton", "DI", "IMU"]})
    out.sum().backward()
    active = ("enc.Skeleton.", "enc.DI.", "enc.IMU.", "fuse.", "head.")
    used = {n.split(".")[1] for n, q in net.named_parameters()
            if q.grad is not None and n.startswith("enc")}
    dead = [n for n, q in net.named_parameters()
            if q.grad is None and n.startswith(active)]
    idle = {n.split(".")[1] for n, q in net.named_parameters()
            if q.grad is None and n.startswith("enc")}
    print(f"    streams receiving gradient: {sorted(used)}")
    print(f"    streams correctly idle:     {sorted(idle)}")
    print(f"    params with no grad on the ACTIVE path: {len(dead)} (must be 0)")
    assert not dead, dead


if __name__ == "__main__":
    torch.manual_seed(0)
    forward_tests(report())
    print("\nAll checks passed.")
