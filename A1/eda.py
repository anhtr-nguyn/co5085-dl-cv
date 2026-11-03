"""EDA cho Food-101: python eda.py

Ghi ra outputs/figs/:
  - samples.png       một ảnh mẫu cho 24 lớp đầu (ảnh gốc, chưa biến đổi)
  - class_dist.png    số ảnh mỗi lớp ở train / val / test
  - image_sizes.png   phân bố chiều rộng, chiều cao, cạnh dài
  - augment.png       một ảnh qua pipeline augmentation của train (8 lần lấy mẫu)
và outputs/eda.json (thống kê kích thước ảnh, số ảnh mỗi split).
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from data import build_transforms, denormalize, get_datasets


def image_size(path):
    with Image.open(path) as im:   # chỉ đọc header
        return im.size


def main():
    os.makedirs("outputs/figs", exist_ok=True)
    tr, va, te = get_datasets()
    base = tr.dataset
    classes = base.classes
    labels = np.asarray(base._labels)
    test_labels = np.asarray(te._labels)

    # 1. Phân bố lớp
    counts = {"train": np.bincount(labels[tr.indices], minlength=len(classes)),
              "val": np.bincount(labels[va.indices], minlength=len(classes)),
              "test": np.bincount(test_labels, minlength=len(classes))}
    fig, ax = plt.subplots(figsize=(14, 3.5))
    bottom = np.zeros(len(classes))
    for name, c in counts.items():
        ax.bar(range(len(classes)), c, bottom=bottom, label=f"{name} ({c.sum():,})")
        bottom += c
    ax.set_xlabel("class index")
    ax.set_ylabel("#images")
    ax.set_xlim(-1, len(classes))
    ax.legend()
    ax.set_title("Food-101: số ảnh mỗi lớp")
    fig.tight_layout()
    fig.savefig("outputs/figs/class_dist.png", dpi=150)
    plt.close(fig)

    # 2. Kích thước ảnh (toàn bộ train + test)
    paths = [str(p) for p in base._image_files] + [str(p) for p in te._image_files]
    with ThreadPoolExecutor(32) as ex:
        sizes = np.array(list(ex.map(image_size, paths, chunksize=256)))
    w, h = sizes[:, 0], sizes[:, 1]
    long_side = sizes.max(1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    for ax, v, t in zip(axes, [w, h, long_side], ["width", "height", "long side"]):
        ax.hist(v, bins=50)
        ax.set_title(f"{t}: median {int(np.median(v))}, min {v.min()}, max {v.max()}")
        ax.set_xlabel("pixels")
    fig.tight_layout()
    fig.savefig("outputs/figs/image_sizes.png", dpi=150)
    plt.close(fig)

    # 3. Ảnh mẫu: một ảnh cho mỗi lớp trong 24 lớp đầu
    fig, axes = plt.subplots(4, 6, figsize=(12, 8.5))
    for c, ax in enumerate(axes.flat):
        i = int(np.flatnonzero(labels == c)[0])
        with Image.open(base._image_files[i]) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(classes[c], fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("outputs/figs/samples.png", dpi=120)
    plt.close(fig)

    # 4. Augmentation của train trên cùng một ảnh
    train_tf, eval_tf = build_transforms()
    with Image.open(base._image_files[0]) as im:
        im = im.convert("RGB")
        torch.manual_seed(0)
        views = [eval_tf(im)] + [train_tf(im) for _ in range(7)]
    fig, axes = plt.subplots(1, 8, figsize=(16, 2.4))
    for k, (ax, v) in enumerate(zip(axes, views)):
        ax.imshow(denormalize(v).permute(1, 2, 0).numpy())
        ax.set_title("eval" if k == 0 else f"train #{k}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("outputs/figs/augment.png", dpi=120)
    plt.close(fig)

    stats = dict(
        n_train=len(tr), n_val=len(va), n_test=len(te), n_classes=len(classes),
        per_class={k: [int(v.min()), int(v.max())] for k, v in counts.items()},
        width=dict(min=int(w.min()), median=float(np.median(w)), max=int(w.max())),
        height=dict(min=int(h.min()), median=float(np.median(h)), max=int(h.max())),
        long_side_eq_512=float((long_side == 512).mean()),
        short_side_lt_224=float((sizes.min(1) < 224).mean()),
    )
    with open("outputs/eda.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
