"""
Phân tích khả năng bắt phụ thuộc theo thứ tự (temporal) của LSTM/GRU, so với CNN/MLP của E1.
Tất cả đánh giá trên test set (không augment), dùng best checkpoint của từng run.

Với mỗi cách biểu diễn chuỗi (row / col / patch) và mỗi model:
  - prefix k%:  chỉ cho model thấy k% bước đầu tiên của chuỗi.
                RNN: cắt chuỗi (đọc k bước rồi phân loại).
                CNN/MLP: giữ phần ảnh tương ứng, phần còn lại đặt 0 (= mean sau chuẩn hóa).
  - reversed:   đảo ngược thứ tự các bước (ảnh lật dọc với row, lật ngang với col).
  - shuffled:   hoán vị cố định ngẫu nhiên các bước (cùng một hoán vị cho mọi ảnh).

    python analyze.py                # cần E3/outputs/runs và E1/outputs/runs đã train xong

Ghi ra outputs/temporal.md, outputs/temporal.json, outputs/figs/prefix_acc.png
"""
import glob
import importlib.util
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from data import get_dataloaders
from models import ImageToSequence, RNNClassifier

SEQ_TYPES = ["row", "col", "patch"]
PREFIX_FRACS = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
E1_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "E1")
E1_BASELINES = [("cnn", 1), ("mlp", 0)]   # cấu hình tốt nhất của mỗi model ở E1


def load_e1_models_module():
    spec = importlib.util.spec_from_file_location("e1_models", os.path.join(E1_DIR, "models.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def seq_to_image(seq, seq_type, C=3, H=32, W=32, p=4):
    """Nghịch đảo của ImageToSequence."""
    B = seq.size(0)
    if seq_type == "row":
        return seq.reshape(B, H, C, W).permute(0, 2, 1, 3)
    if seq_type == "col":
        return seq.reshape(B, W, C, H).permute(0, 2, 3, 1)
    x = seq.reshape(B, H // p, W // p, C, p, p).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(B, C, H, W)


class Probe:
    """Bọc một model để đánh giá trên chuỗi đã biến đổi, theo một seq_type cố định."""
    def __init__(self, model, seq_type, is_rnn):
        self.model, self.seq_type, self.is_rnn = model, seq_type, is_rnn
        self.to_seq = ImageToSequence(seq_type)

    @torch.no_grad()
    def logits(self, x, order=None, prefix=None):
        seq = self.to_seq(x)
        if order is not None:
            seq = seq[:, order]
        if prefix is not None and prefix < seq.size(1):
            if self.is_rnn:
                return self.model.forward_seq(seq[:, :prefix])
            seq = seq.clone()
            seq[:, prefix:] = 0
        if self.is_rnn:
            return self.model.forward_seq(seq)
        return self.model(seq_to_image(seq, self.seq_type))


@torch.no_grad()
def run_probes(probe, loader, device, T, perm):
    settings = {f"prefix_{f:g}": dict(prefix=max(1, round(f * T))) for f in PREFIX_FRACS}
    settings["reversed"] = dict(order=torch.arange(T - 1, -1, -1, device=device))
    settings["shuffled"] = dict(order=perm.to(device))
    correct = defaultdict(int)
    n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        for name, kw in settings.items():
            correct[name] += (probe.logits(x, **kw).argmax(1) == y).sum().item()
        n += y.size(0)
    return {k: v / n for k, v in correct.items()}


def load_e3_runs(device):
    runs = []
    for p in sorted(glob.glob("outputs/runs/*/best.pt")):
        rd = os.path.dirname(p)
        with open(os.path.join(rd, "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        mp = cfg["model"]["params"]
        model = RNNClassifier(**mp).to(device).eval()
        model.load_state_dict(torch.load(p, map_location=device)["model"])
        runs.append(dict(name=mp["cell"].upper(), seq_types=[mp["seq_type"]], model=model,
                         is_rnn=True, seed=cfg["train"]["seed"]))
    return runs


def load_e1_runs(device):
    e1 = load_e1_models_module()
    runs = []
    for p in sorted(glob.glob(os.path.join(E1_DIR, "outputs/runs/*/best.pt"))):
        ckpt = torch.load(p, map_location=device)
        cfg = ckpt["config"]
        key = (cfg["model"]["name"], int(cfg["data"]["augment"]))
        if key not in E1_BASELINES:
            continue
        model = e1.build_model(cfg["model"]["name"], **(cfg["model"].get("params") or {}))
        model.to(device).eval().load_state_dict(ckpt["model"])
        runs.append(dict(name=f"E1 {key[0].upper()} (aug{key[1]})", seq_types=SEQ_TYPES,
                         model=model, is_rnn=False, seed=cfg["train"]["seed"]))
    return runs


def mean_std(xs):
    return f"{100 * np.mean(xs):.2f} ± {100 * np.std(xs, ddof=1):.2f}" if len(xs) > 1 \
        else f"{100 * xs[0]:.2f}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_dataloaders(batch_size=500, num_workers=4, split_path=None)

    runs = load_e3_runs(device) + load_e1_runs(device)
    assert runs, "Không tìm thấy checkpoint nào"
    g = torch.Generator().manual_seed(0)
    perms = {s: torch.randperm(ImageToSequence(s).seq_len, generator=g) for s in SEQ_TYPES}

    # results[seq_type][model_name] = list (theo seed) các dict kết quả
    results = defaultdict(lambda: defaultdict(list))
    for r in runs:
        for s in r["seq_types"]:
            T = ImageToSequence(s).seq_len
            res = run_probes(Probe(r["model"], s, r["is_rnn"]), test_loader, device, T, perms[s])
            results[s][r["name"]].append(res)
            print(f"{r['name']:16s} {s:5s} seed {r['seed']:3d} | full {res['prefix_1']:.4f} "
                  f"half {res['prefix_0.5']:.4f} rev {res['reversed']:.4f} "
                  f"shuf {res['shuffled']:.4f}")

    os.makedirs("outputs/figs", exist_ok=True)
    cols = [("prefix_1", "Đủ chuỗi"), ("prefix_0.25", "Prefix 25%"), ("prefix_0.5", "Prefix 50%"),
            ("prefix_0.75", "Prefix 75%"), ("reversed", "Đảo ngược"), ("shuffled", "Xáo trộn")]
    lines = []
    for s in SEQ_TYPES:
        if s not in results:
            continue
        lines += [f"\n### Chuỗi theo `{s}` (T = {ImageToSequence(s).seq_len})\n",
                  "| Model | #Seeds | " + " | ".join(c[1] for c in cols) + " |",
                  "|" + "---|" * (len(cols) + 2)]
        for name, rs in results[s].items():
            lines.append(f"| {name} | {len(rs)} | " +
                         " | ".join(mean_std([r[k] for r in rs]) for k, _ in cols) + " |")
    note = ("\nTest acc (%), mean ± std theo seed. Prefix: RNN đọc k% bước đầu; CNN/MLP thấy "
            "phần ảnh tương ứng, phần còn lại = 0 (mean). Đảo ngược / xáo trộn: chỉ áp ở test, "
            "model train trên thứ tự gốc.\n")
    with open("outputs/temporal.md", "w") as f:
        f.write("\n".join(lines) + "\n" + note)
    with open("outputs/temporal.json", "w") as f:
        json.dump({s: dict(v) for s, v in results.items()}, f, indent=2)
    print("\n".join(lines) + "\n" + note)

    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), squeeze=False)
    for ax, s in zip(axes[0], [s for s in SEQ_TYPES if s in results]):
        for name, rs in results[s].items():
            m = [100 * np.mean([r[f"prefix_{f:g}"] for r in rs]) for f in PREFIX_FRACS]
            ax.plot([100 * f for f in PREFIX_FRACS], m, "o-", label=name)
        ax.set_title(f"sequence = {s}")
        ax.set_xlabel("% bước thời gian được thấy")
        ax.set_ylabel("test acc (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("outputs/figs/prefix_acc.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
