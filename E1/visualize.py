"""Vẽ hình cho một run: python visualize.py outputs/runs/<run_name>

Đọc log.json, val_preds.pt, config.yaml trong run dir và ghi ra <run_dir>/figs/:
  - curves.png            loss / accuracy / macro-F1 theo epoch (train vs val)
  - confusion_matrix.png  confusion matrix (chuẩn hoá theo hàng) của best model trên val
  - error_samples.png     các ảnh val bị dự đoán sai với độ tự tin cao nhất
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
import yaml

from data import CLASSES, denormalize, get_dataloaders


def plot_curves(history, path):
    ep = [h["epoch"] for h in history]
    val = [h for h in history if "val_loss" in h]
    val_ep = [h["epoch"] for h in val]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in zip(axes, ["loss", "acc", "f1"],
                              ["Loss", "Accuracy", "Macro F1"]):
        ax.plot(ep, [h[f"train_{key}"] for h in history], label="train")
        ax.plot(val_ep, [h[f"val_{key}"] for h in val], "o-", label="val")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(preds, labels, path, title=""):
    n = len(CLASSES)
    cm = np.bincount(labels * n + preds, minlength=n * n).reshape(n, n)
    cm_norm = cm / cm.sum(1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                    fontsize=7, color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(n), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(n), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_samples(val_ds, preds, labels, conf, path, n_samples=32, ncols=8):
    wrong = np.flatnonzero(preds != labels)
    wrong = wrong[np.argsort(-conf[wrong])][:n_samples]  # sai mà tự tin nhất
    if len(wrong) == 0:
        return
    nrows = int(np.ceil(len(wrong) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.9),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, wrong):
        img, y = val_ds[i]
        assert y == labels[i], "val_preds.pt không khớp thứ tự val set"
        ax.imshow(denormalize(img).permute(1, 2, 0).numpy())
        ax.set_title(f"T: {CLASSES[labels[i]]}\nP: {CLASSES[preds[i]]} ({conf[i]:.2f})",
                     fontsize=7)
    fig.suptitle("Most confident errors (T = true, P = predicted)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def visualize_run(run_dir):
    fig_dir = os.path.join(run_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    with open(os.path.join(run_dir, "log.json")) as f:
        log = json.load(f)
    with open(os.path.join(run_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    vp = torch.load(os.path.join(run_dir, "val_preds.pt"))
    preds, labels, conf = vp["preds"].numpy(), vp["labels"].numpy(), vp["conf"].numpy()

    plot_curves(log["history"], os.path.join(fig_dir, "curves.png"))
    plot_confusion_matrix(preds, labels, os.path.join(fig_dir, "confusion_matrix.png"),
                          title=f"{log['tag']} | best epoch {log['best_epoch']} | "
                                f"val acc {log['best_val_acc']:.4f}")

    # Dựng lại val set theo đúng split của run (không augment, cùng thứ tự)
    _, val_loader, _ = get_dataloaders(
        val_size=cfg["data"]["val_size"], seed=cfg["data"]["split_seed"],
        augment=False, num_workers=0, split_path=None)
    plot_error_samples(val_loader.dataset, preds, labels, conf,
                       os.path.join(fig_dir, "error_samples.png"),
                       n_samples=cfg["output"].get("n_error_samples", 32))
    print(f"figures saved to {fig_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", help="e.g. outputs/runs/cnn_aug0_s912_20261005_101500")
    visualize_run(p.parse_args().run_dir)
