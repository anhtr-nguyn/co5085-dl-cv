"""
EDA cho KITTI 3D Object Detection (split train/val 3,712/3,769 của Chen et al. 2016).

    python eda.py --root /path/to/KITTI

KITTI root cần có ImageSets/{train,val}.txt và training/{image_2,label_2,calib}/.
Ghi ra outputs/eda.json và outputs/figs/{class_counts,difficulty,depth_hist,samples}.png.
"""
import argparse
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CLASSES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]
EVAL_CLASSES = ["Car", "Pedestrian", "Cyclist"]
LEVELS = ["Easy", "Moderate", "Hard"]
# Ngưỡng chính thức của KITTI: chiều cao box 2D tối thiểu (px), occlusion tối đa, truncation tối đa
LEVEL_RULES = [(40, 0, 0.15), (25, 1, 0.30), (25, 2, 0.50)]


def read_split(root, split):
    with open(os.path.join(root, "ImageSets", f"{split}.txt")) as f:
        return [l.strip() for l in f if l.strip()]


def read_labels(root, idx):
    objs = []
    with open(os.path.join(root, "training", "label_2", f"{idx}.txt")) as f:
        for line in f:
            v = line.split()
            objs.append(dict(cls=v[0], trunc=float(v[1]), occ=int(v[2]),
                             box2d=[float(x) for x in v[4:8]],
                             hwl=[float(x) for x in v[8:11]],
                             loc=[float(x) for x in v[11:14]], ry=float(v[14])))
    return objs


def read_P2(root, idx):
    with open(os.path.join(root, "training", "calib", f"{idx}.txt")) as f:
        for line in f:
            if line.startswith("P2:"):
                return np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)


def difficulty(o):
    """Mức dễ nhất mà object thỏa (0/1/2), -1 nếu ngoài cả ba mức (bị bỏ qua khi đánh giá)."""
    h = o["box2d"][3] - o["box2d"][1]
    for lvl, (min_h, max_occ, max_trunc) in enumerate(LEVEL_RULES):
        if h >= min_h and o["occ"] <= max_occ and o["trunc"] <= max_trunc:
            return lvl
    return -1


def corners_3d(o):
    h, w, l = o["hwl"]
    x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y = [0, 0, 0, 0, -h, -h, -h, -h]
    z = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    c, s = np.cos(o["ry"]), np.sin(o["ry"])
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return R @ np.array([x, y, z]) + np.array(o["loc"])[:, None]


def project(P, pts):
    p = P @ np.vstack([pts, np.ones((1, pts.shape[1]))])
    return p[:2] / p[2:]


def plot_samples(root, ids, path):
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    colors = {"Car": "lime", "Pedestrian": "red", "Cyclist": "cyan"}
    fig, axes = plt.subplots(len(ids), 1, figsize=(12, 3.95 * len(ids)))
    for ax, idx in zip(np.atleast_1d(axes), ids):
        img = Image.open(os.path.join(root, "training", "image_2", f"{idx}.png"))
        P = read_P2(root, idx)
        ax.imshow(img)
        for o in read_labels(root, idx):
            if o["cls"] not in colors or o["loc"][2] < 0.5:
                continue
            uv = project(P, corners_3d(o))
            for a, b in edges:
                ax.plot(uv[0, [a, b]], uv[1, [a, b]], color=colors[o["cls"]], lw=1)
            ax.text(max(uv[0].min(), 2), max(uv[1].min() - 3, 10), f"{o['cls']} {o['loc'][2]:.0f}m",
                    color=colors[o["cls"]], fontsize=7, clip_on=True)
        ax.set_xlim(0, img.width); ax.set_ylim(img.height, 0)
        ax.set_title(f"training/{idx}.png", fontsize=8); ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.97, bottom=0.005, hspace=0.12)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("KITTI_ROOT", "data/KITTI"))
    args = ap.parse_args()
    os.makedirs("outputs/figs", exist_ok=True)

    stats = {}
    depths = {s: {c: [] for c in EVAL_CLASSES} for s in ("train", "val")}
    level_counts = {s: {c: Counter() for c in EVAL_CLASSES} for s in ("train", "val")}
    for split in ("train", "val"):
        ids = read_split(args.root, split)
        cls_count, empty = Counter(), 0
        for idx in ids:
            objs = read_labels(args.root, idx)
            if not any(o["cls"] in EVAL_CLASSES for o in objs):
                empty += 1
            for o in objs:
                cls_count[o["cls"]] += 1
                if o["cls"] in EVAL_CLASSES:
                    depths[split][o["cls"]].append(o["loc"][2])
                    level_counts[split][o["cls"]][difficulty(o)] += 1
        stats[split] = dict(
            images=len(ids), images_without_eval_class=empty,
            objects=dict(sorted(cls_count.items(), key=lambda kv: -kv[1])),
            difficulty={c: {(LEVELS[k] if k >= 0 else "Ignored"): v
                            for k, v in sorted(level_counts[split][c].items())}
                        for c in EVAL_CLASSES},
            depth_m={c: dict(mean=float(np.mean(d)), median=float(np.median(d)),
                             p95=float(np.percentile(d, 95)), max=float(np.max(d)),
                             frac_ge_40m=float(np.mean(np.array(d) >= 40)))
                     for c, d in depths[split].items()})

    all_ids = read_split(args.root, "train") + read_split(args.root, "val")
    sizes = Counter(Image.open(os.path.join(args.root, "training", "image_2", f"{i}.png")).size
                    for i in all_ids)
    stats["image_sizes"] = {f"{w}x{h}": n for (w, h), n in sizes.most_common()}
    stats["test_images"] = len(os.listdir(os.path.join(args.root, "testing", "image_2")))
    with open("outputs/eda.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))

    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = np.arange(len(CLASSES))
    for k, split in enumerate(("train", "val")):
        v = [stats[split]["objects"].get(c, 0) for c in CLASSES]
        bars = ax.bar(x + (k - 0.5) * 0.4, v, 0.4, label=split)
        ax.bar_label(bars, fontsize=7)
    ax.set_xticks(x, CLASSES, rotation=20); ax.set_ylabel("Số object"); ax.set_yscale("log")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig("outputs/figs/class_counts.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, c in zip(axes, EVAL_CLASSES):
        for k, split in enumerate(("train", "val")):
            d = stats[split]["difficulty"][c]
            names = LEVELS + ["Ignored"]
            ax.bar(np.arange(4) + (k - 0.5) * 0.4, [d.get(n, 0) for n in names], 0.4, label=split)
        ax.set_xticks(range(4), names); ax.set_title(c); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Số object (mức dễ nhất thỏa)"); axes[0].legend()
    fig.tight_layout(); fig.savefig("outputs/figs/difficulty.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharex=True)
    for ax, c in zip(axes, EVAL_CLASSES):
        for split in ("train", "val"):
            ax.hist(depths[split][c], bins=np.arange(0, 85, 2.5), alpha=0.6, label=split)
        ax.set_title(c); ax.set_xlabel("Độ sâu z (m)"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Số object"); axes[0].legend()
    fig.tight_layout(); fig.savefig("outputs/figs/depth_hist.png", dpi=150); plt.close(fig)

    plot_samples(args.root, ["000001", "000008", "000021"], "outputs/figs/samples.png")


if __name__ == "__main__":
    main()
