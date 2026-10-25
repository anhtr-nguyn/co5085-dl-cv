"""
Evaluate model on test set
"""
import csv
import glob
import json
import os
from collections import defaultdict
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml

from data import get_dataloaders
from models import MyTinyViT
from train import evaluate, fmt_metrics
from visualize import plot_confusion_matrix



def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_runs(paths):
    if paths:
        return sorted(paths)
    return sorted(os.path.dirname(p) for p in glob.glob("outputs/runs/*/log.json"))


@torch.no_grad()
def eval_run_on_test(run_dir, test_loader, device):
    with open(os.path.join(run_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    mcfg = cfg["model"]

    model = MyTinyViT(**mcfg["params"]).to(device)
    ckpt = torch.load(os.path.join(run_dir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])

    metrics, preds, labels, _ = evaluate(
        model, test_loader, nn.CrossEntropyLoss(), device,
        return_preds=True, per_class=True)

    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    fig_dir = os.path.join(run_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    with open(os.path.join(run_dir, "log.json")) as f:
        log = json.load(f)
    plot_confusion_matrix(preds.numpy(), labels.numpy(),
                          os.path.join(fig_dir, "test_confusion_matrix.png"),
                          title=f"{log['tag']} | TEST acc {metrics['acc']:.4f}")
    return metrics, log


def make_row(log, test):
    cfg = log["config"]
    p = cfg["model"]["params"]
    arch = f"d{p['embed_dim']}-L{p['num_layers']}-r{p['mlp_ratio']:g}-e{cfg['train']['epochs']}"
    if not p.get("use_cls_token", True):
        arch += "-nocls"
    if not p.get("use_pos_embed", True):
        arch += "-nopos"
    row = dict(
        tokenizer=p["tokenizer_type"], msa=p["msa_type"], arch=arch,
        aug=int(cfg["data"]["augment"]), seed=cfg["train"]["seed"], params=log["params"],
        tokenizer_params=log.get("tokenizer_params"),
        epochs=cfg["train"]["epochs"], best_epoch=log["best_epoch"],
        val_acc=log["best_val_acc"], test_acc=test["acc"], test_f1=test["f1"],
        test_loss=test["loss"], train_time_s=log["train_time_s"],
        time_per_epoch_s=log.get("time_per_epoch_s"),
        peak_mem_mb=log["peak_mem_mb"], tag=log["tag"])
    
    if abs(row["val_acc"] - row["test_acc"]) > 0.03:
        print(f"[warn] {row['tag']}: val {row['val_acc']:.4f} vs test {row['test_acc']:.4f}")
    return row


def mean_std(xs, fmt):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    if len(xs) == 1:
        return fmt.format(xs[0])
    return f"{fmt.format(np.mean(xs))} ± {fmt.format(np.std(xs, ddof=1))}"


def aggregate(rows):
    os.makedirs("outputs", exist_ok=True)

    # Bảng chi tiết từng run
    with open(os.path.join("outputs", "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Bảng tóm tắt: nhóm theo (kiến trúc, tokenizer, msa, aug), gộp các seed
    groups = defaultdict(list)
    for r in rows:
        groups[(r["arch"], r["tokenizer"], r["msa"], r["aug"])].append(r)

    header = ["Arch", "Tokenizer", "MSA", "Aug", "#Seeds", "#Params", "Val acc (%)",
              "Test acc (%)", "Test macro-F1 (%)", "Train s/epoch", "Peak mem (MB)"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for (arch, tok, msa, aug), rs in sorted(groups.items()):
        pct = lambda k: mean_std([100 * r[k] for r in rs], "{:.2f}")
        lines.append("| " + " | ".join([
            arch, tok, msa, str(aug), str(len(rs)), f"{rs[0]['params']:,}",
            pct("val_acc"), pct("test_acc"), pct("test_f1"),
            mean_std([r["time_per_epoch_s"] for r in rs], "{:.2f}"),
            mean_std([r["peak_mem_mb"] for r in rs], "{:.0f}")]) + " |")
    table = "\n".join(lines)
    with open(os.path.join("outputs", "results.md"), "w") as f:
        f.write(table + "\n")
    print("\n" + table)


def plot_overlay(logs, path):
    """Đường cong val của các mô hình trên cùng một hình (train: nét đứt)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for log in logs:
        h = log["history"]
        label = log["tag"].rsplit("_", 1)[0]  # bỏ phần datetime/seed cuối nếu muốn gọn
        val = [x for x in h if "val_loss" in x]
        for ax, key in zip(axes, ["loss", "acc"]):
            line, = ax.plot([x["epoch"] for x in val], [x[f"val_{key}"] for x in val],
                            label=label)
            ax.plot([x["epoch"] for x in h], [x[f"train_{key}"] for x in h], "--",
                    color=line.get_color(), alpha=0.5)
    for ax, title in zip(axes, ["Loss (solid=val, dashed=train)", "Accuracy"]):
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="*", help="mặc định: tất cả trong outputs/runs")
    args = p.parse_args()

    run_dirs = find_runs(args.run_dirs)
    assert run_dirs, "Không tìm thấy run nào"
    device = pick_device()

    # test set không phụ thuộc split/seed -> dựng một lần dùng cho mọi run
    _, _, test_loader = get_dataloaders(batch_size=256, num_workers=4, split_path=None)

    rows, logs = [], []
    for rd in run_dirs:
        test, log = eval_run_on_test(rd, test_loader, device)
        print(f"[{log['tag']}] test | {fmt_metrics(test)}")
        rows.append(make_row(log, test))
        logs.append(log)

    aggregate(rows)
    plot_overlay(logs, os.path.join("outputs", "figs", "compare_curves.png"))


if __name__ == "__main__":
    main()
