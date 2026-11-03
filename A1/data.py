import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Thống kê ImageNet: backbone pretrained (timm) được train với chuẩn hóa này;
# dùng chung cho cả from-scratch để mọi chế độ có cùng input.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 101


def build_transforms(img_size=224, augment="standard"):
    """
    augment: "none"     -> train dùng cùng biến đổi với eval
             "standard" -> RandomResizedCrop + HorizontalFlip + TrivialAugmentWide
    Trả về train_tf, eval_tf. Val/test luôn: Resize(256) -> CenterCrop(224).
    """
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    if augment == "none":
        return eval_tf, eval_tf
    if augment != "standard":
        raise ValueError(f"Unknown augment '{augment}'")
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


def stratified_split(labels, val_frac=0.1, seed=912):
    """Tách val_frac ảnh mỗi lớp làm validation (phân tầng), trả về train_idx, val_idx đã sắp xếp."""
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        n_val = int(round(val_frac * len(idx)))
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    return np.sort(np.concatenate(train_idx)), np.sort(np.concatenate(val_idx))


def get_datasets(root="./data_cache", img_size=224, augment="standard", val_frac=0.1,
                 seed=912, split_path="outputs/split.npz"):
    train_tf, eval_tf = build_transforms(img_size, augment)
    # Hai dataset cùng ảnh, khác transform: val lấy từ train gốc nhưng không bị augment
    train_full = datasets.Food101(root, split="train", transform=train_tf, download=True)
    train_full_eval = datasets.Food101(root, split="train", transform=eval_tf, download=False)
    test_ds = datasets.Food101(root, split="test", transform=eval_tf, download=False)

    train_idx, val_idx = stratified_split(train_full._labels, val_frac, seed)
    assert len(np.intersect1d(train_idx, val_idx)) == 0

    if split_path:
        os.makedirs(os.path.dirname(split_path) or ".", exist_ok=True)
        np.savez(split_path, train_idx=train_idx, val_idx=val_idx, seed=seed, val_frac=val_frac)

    return Subset(train_full, train_idx), Subset(train_full_eval, val_idx), test_ds


def get_dataloaders(batch_size=128, num_workers=16, seed=912, **kw):
    train_ds, val_ds, test_ds = get_datasets(seed=seed, **kw)
    g = torch.Generator().manual_seed(seed)
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                  persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, generator=g, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Đảo Normalize để vẽ ảnh. Nhận [3,H,W] hoặc [B,3,H,W]."""
    mean = torch.tensor(MEAN, device=x.device)
    std = torch.tensor(STD, device=x.device)
    shape = (1, 3, 1, 1) if x.dim() == 4 else (3, 1, 1)
    return (x * std.view(*shape) + mean.view(*shape)).clamp(0, 1)


if __name__ == "__main__":
    tr, va, te = get_datasets()
    print("sizes:", len(tr), len(va), len(te))
    labels = np.asarray(tr.dataset._labels)
    for name, ds in [("train", tr), ("val", va)]:
        c = np.bincount(labels[ds.indices], minlength=NUM_CLASSES)
        print(f"{name}: per-class min {c.min()} max {c.max()}")

    # Tái lập: chia lại với cùng seed phải ra cùng chỉ số
    a = np.load("outputs/split.npz")["val_idx"]
    _, b = stratified_split(labels, 0.1, 912)
    assert np.array_equal(a, b), "Split không tái lập!"

    x, y = tr[0]
    print("sample:", tuple(x.shape), y)
    print("OK")
