"""
Tổng hợp kết quả A1 (không train, chỉ đọc log và dự đoán đã lưu).

    python evaluate.py

Ghi ra:
  outputs/sweep.md                 val acc tốt nhất theo learning rate (outputs/sweep/)
  outputs/results.{md,csv}         8 cấu hình chính (outputs/runs/), mean ± std nếu nhiều seed
  outputs/errors.md                cặp lớp nhầm nhiều nhất và lớp kém nhất trên test
  outputs/figs/curves.png          val acc theo epoch
  outputs/figs/acc_by_mode.png     test acc theo chế độ huấn luyện
  outputs/figs/acc_vs_cost.png     test acc theo thời gian huấn luyện và số tham số trainable
  outputs/figs/confusion_top.png   ma trận nhầm lẫn trên các lớp hay nhầm nhất (full fine-tune)
  outputs/figs/error_samples_<run>.png  ảnh val bị sai với độ tự tin cao nhất
"""
import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ARCH_NAMES = {"resnet50": "ResNet-50", "deit_s": "DeiT-S"}
MODE_NAMES = {"scratch": "From-scratch", "head": "Freeze backbone", "partial": "Freeze một phần",
              "full": "Full fine-tune"}
MODES = ["scratch", "head", "partial", "full"]
COLORS = {"resnet50": "#4C72B0", "deit_s": "#DD8452"}


def load_runs(root):
    runs = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isfile(os.path.join(d, "log.json")):
            continue
        with open(os.path.join(d, "log.json")) as f:
            log = json.load(f)
        test = None
        if os.path.isfile(os.path.join(d, "test_metrics.json")):
            with open(os.path.join(d, "test_metrics.json")) as f:
                test = json.load(f)
        c = log["config"]
        runs.append(dict(dir=d, log=log, test=test, arch=c["model"]["arch"],
                         mode=c["model"]["mode"], lr=float(c["train"]["lr"]),
                         epochs=c["train"]["epochs"], seed=c["train"]["seed"]))
    return runs


def ms(xs, scale=100, fmt="{:.2f}"):
    xs = [scale * x for x in xs]
    if len(xs) == 1:
        return fmt.format(xs[0])
    return f"{fmt.format(np.mean(xs))} ± {fmt.format(np.std(xs, ddof=1))}"


def sweep_table(runs):
    lines = ["| Model | Chế độ | Epoch | LR | Val acc tốt nhất (%) | Epoch tốt nhất |",
             "|---|---|---|---|---|---|"]
    best = {}
    for r in sorted(runs, key=lambda r: (r["arch"], MODES.index(r["mode"]), r["lr"])):
        key = (r["arch"], r["mode"])
        if key not in best or r["log"]["best_val_acc"] > best[key]["log"]["best_val_acc"]:
            best[key] = r
    for r in sorted(runs, key=lambda r: (r["arch"], MODES.index(r["mode"]), r["lr"])):
        mark = " **(chọn)**" if best[(r["arch"], r["mode"])] is r else ""
        lines.append(f"| {ARCH_NAMES[r['arch']]} | {MODE_NAMES[r['mode']]} | {r['epochs']} | "
                     f"{r['lr']:g} | {100 * r['log']['best_val_acc']:.2f}{mark} | "
                     f"{r['log']['best_epoch']} |")
    return lines, best


def group_main(runs):
    groups = defaultdict(list)
    for r in runs:
        if r["test"] is not None:
            groups[(r["arch"], r["mode"])].append(r)
    return groups


def results_table(groups):
    header = ["Model", "Chế độ", "#Seeds", "LR", "Epoch", "Trainable", "Val acc (%)",
              "Test acc (%)", "Test top-5 (%)", "Test macro-F1 (%)", "Best epoch",
              "Train s/epoch", "Train tổng (phút)", "Peak mem (MB)"]
    rows = []
    for arch in ARCH_NAMES:
        for mode in MODES:
            rs = groups.get((arch, mode))
            if not rs:
                continue
            L = [r["log"] for r in rs]
            rows.append([
                ARCH_NAMES[arch], MODE_NAMES[mode], str(len(rs)), f"{rs[0]['lr']:g}",
                str(rs[0]["epochs"]), f"{L[0]['params_trainable']:,}",
                ms([l["best_val_acc"] for l in L]),
                ms([r["test"]["acc"] for r in rs]), ms([r["test"]["top5"] for r in rs]),
                ms([r["test"]["f1"] for r in rs]),
                "–".join(sorted({str(l["best_epoch"]) for l in L}, key=int)),
                ms([l["time_per_epoch_s"] for l in L], 1, "{:.0f}"),
                ms([l["train_time_s"] / 60 for l in L], 1, "{:.0f}"),
                ms([l["peak_mem_mb"] for l in L], 1, "{:.0f}")])
    return header, rows


def class_names():
    path = os.path.join("data_cache", "food-101", "meta", "classes.txt")
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def error_report(groups, names, k=5):
    lines = []
    for arch in ARCH_NAMES:
        for mode in MODES:
            rs = groups.get((arch, mode))
            if not rs:
                continue
            cm = np.sum([np.array(r["test"]["per_class"]["confusion_matrix"]) for r in rs], 0)
            f1 = np.mean([r["test"]["per_class"]["f1"] for r in rs], 0)
            off = cm.astype(float)
            np.fill_diagonal(off, 0)
            sym = off + off.T
            iu = np.triu_indices(len(names), 1)
            order = np.argsort(-sym[iu])[:k]
            pairs = ", ".join(
                f"{names[iu[0][o]]} ↔ {names[iu[1][o]]} "
                f"{100 * sym[iu][o] / (cm[iu[0][o]].sum() + cm[iu[1][o]].sum()):.1f}%"
                for o in order)
            worst = ", ".join(f"{names[c]} {f1[c]:.2f}" for c in np.argsort(f1)[:k])
            best = ", ".join(f"{names[c]} {f1[c]:.2f}" for c in np.argsort(-f1)[:3])
            lines += [f"**{ARCH_NAMES[arch]}, {MODE_NAMES[mode]}**", "",
                      f"- Cặp nhầm nhiều nhất (gộp hai chiều): {pairs}",
                      f"- F1 thấp nhất: {worst}", f"- F1 cao nhất: {best}", ""]
    return lines


def plot_curves(groups):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    styles = {"scratch": ":", "head": "-.", "partial": "--", "full": "-"}
    for ax, arch in zip(axes, ARCH_NAMES):
        for mode in MODES:
            rs = groups.get((arch, mode))
            if not rs:
                continue
            h = sorted(rs, key=lambda r: r["seed"] != 912)[0]["log"]["history"]
            ax.plot([e["epoch"] for e in h], [100 * e["val_acc"] for e in h],
                    ls=styles[mode], color=COLORS[arch], lw=1.5, label=MODE_NAMES[mode])
        ax.set_title(ARCH_NAMES[arch]); ax.set_xlabel("Epoch"); ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Val accuracy (%)")
    fig.tight_layout()
    fig.savefig("outputs/figs/curves.png", dpi=200)
    plt.close(fig)


def plot_acc_by_mode(groups):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(MODES))
    w = 0.38
    for k, arch in enumerate(ARCH_NAMES):
        vals, errs = [], []
        for mode in MODES:
            accs = [100 * r["test"]["acc"] for r in groups.get((arch, mode), [])]
            vals.append(np.mean(accs) if accs else np.nan)
            errs.append(np.std(accs, ddof=1) if len(accs) > 1 else 0)
        bars = ax.bar(x + (k - 0.5) * w, vals, w, yerr=errs, capsize=2, color=COLORS[arch],
                      label=ARCH_NAMES[arch])
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x, [MODE_NAMES[m] for m in MODES])
    ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(0, 100); ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("outputs/figs/acc_by_mode.png", dpi=200)
    plt.close(fig)


def plot_acc_vs_cost(groups):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    markers = {"scratch": "x", "head": "s", "partial": "^", "full": "o"}
    for (arch, mode), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], MODES.index(kv[0][1]))):
        acc = np.mean([100 * r["test"]["acc"] for r in rs])
        t = np.mean([r["log"]["train_time_s"] / 60 for r in rs])
        p = rs[0]["log"]["params_trainable"] / 1e6
        for ax, xv in zip(axes, (t, p)):
            ax.scatter(xv, acc, color=COLORS[arch], marker=markers[mode], s=70,
                       label=f"{ARCH_NAMES[arch]}, {MODE_NAMES[mode]}")
    axes[0].set_xlabel("Thời gian huấn luyện (phút, trung bình các seed)")
    axes[1].set_xlabel("Tham số trainable (triệu, thang log)"); axes[1].set_xscale("log")
    axes[0].set_ylabel("Test accuracy (%)")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig("outputs/figs/acc_vs_cost.png", dpi=200)
    plt.close(fig)


def plot_confusion(groups, names, n_pairs=8):
    """Ma trận nhầm lẫn (chuẩn hóa theo hàng, % ) trên các lớp thuộc các cặp nhầm nhiều nhất."""
    keys = [("resnet50", "full"), ("deit_s", "full")]
    cms = {k: np.sum([np.array(r["test"]["per_class"]["confusion_matrix"]) for r in groups[k]], 0)
           for k in keys if k in groups}
    if not cms:
        return
    off = sum(cm.astype(float) for cm in cms.values())
    np.fill_diagonal(off, 0)
    sym = off + off.T
    iu = np.triu_indices(len(names), 1)
    cls = []
    for o in np.argsort(-sym[iu])[:n_pairs]:
        for c in (iu[0][o], iu[1][o]):
            if c not in cls:
                cls.append(c)
    fig, axes = plt.subplots(1, len(cms), figsize=(6.5 * len(cms), 5.8))
    axes = np.atleast_1d(axes)
    for ax, (k, cm) in zip(axes, cms.items()):
        norm = 100 * cm / cm.sum(1, keepdims=True)
        sub = norm[np.ix_(cls, cls)]
        ax.imshow(sub, cmap="Blues", vmin=0, vmax=100)
        for i in range(len(cls)):
            for j in range(len(cls)):
                if sub[i, j] >= 0.5:
                    ax.text(j, i, f"{sub[i, j]:.0f}", ha="center", va="center", fontsize=6,
                            color="white" if sub[i, j] > 50 else "black")
        ax.set_xticks(range(len(cls)), [names[c] for c in cls], rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(cls)), [names[c] for c in cls], fontsize=7)
        ax.set_xlabel("Dự đoán"); ax.set_ylabel("Nhãn đúng")
        ax.set_title(f"{ARCH_NAMES[k[0]]}, {MODE_NAMES[k[1]]} (test, % theo hàng, gộp seed)",
                     fontsize=9)
    fig.tight_layout()
    fig.savefig("outputs/figs/confusion_top.png", dpi=200)
    plt.close(fig)


def plot_error_samples(run, names, n=24, ncols=8):
    from data import get_datasets, denormalize
    preds = torch.load(os.path.join(run["dir"], "val_preds.pt"))
    _, val_ds, _ = get_datasets(augment="none", split_path=None)
    p, y, conf = preds["preds"].numpy(), preds["labels"].numpy(), preds["conf"].float().numpy()
    wrong = np.flatnonzero(p != y)
    wrong = wrong[np.argsort(-conf[wrong])][:n]
    nrows = int(np.ceil(len(wrong) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.9, nrows * 2.2))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, wrong):
        x, _ = val_ds[int(i)]
        ax.imshow(denormalize(x).permute(1, 2, 0).numpy())
        ax.set_title(f"T: {names[y[i]]}\nP: {names[p[i]]} ({conf[i]:.2f})", fontsize=6)
    fig.tight_layout()
    name = f"{run['arch']}_{run['mode']}"
    fig.savefig(f"outputs/figs/error_samples_{name}.png", dpi=150)
    plt.close(fig)


def main():
    os.makedirs("outputs/figs", exist_ok=True)
    sweep = load_runs("outputs/sweep")
    if sweep:
        lines, best = sweep_table(sweep)
        with open("outputs/sweep.md", "w") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))
        print("\nLR chọn theo val:")
        for (arch, mode), r in sorted(best.items()):
            print(f"{arch} {mode} {r['lr']:g}")

    main_runs = load_runs("outputs/runs")
    groups = group_main(main_runs)
    if not groups:
        return
    header, rows = results_table(groups)
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    with open("outputs/results.md", "w") as f:
        f.write("\n".join(md) + "\n")
    with open("outputs/results.csv", "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    print("\n" + "\n".join(md))

    names = class_names()
    err = error_report(groups, names)
    with open("outputs/errors.md", "w") as f:
        f.write("\n".join(err) + "\n")
    print("\n" + "\n".join(err))

    plot_curves(groups)
    plot_acc_by_mode(groups)
    plot_acc_vs_cost(groups)
    plot_confusion(groups, names)
    for arch in ARCH_NAMES:
        rs = groups.get((arch, "full"))
        if rs:
            plot_error_samples(sorted(rs, key=lambda r: r["seed"] != 912)[0], names)


if __name__ == "__main__":
    main()
