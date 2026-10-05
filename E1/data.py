import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from copy import deepcopy

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")

# Normalization config, ref: https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data
MEAN = (0.491, 0.482, 0.446)  
STD = (0.247, 0.243, 0.261)


def build_transforms(augment: bool = True):
    """
    Return train_tf, val_tf (default non augment for val)
    """
    eval_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ]
    )
    if not augment:
        return eval_tf, eval_tf

    train_tf = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, hue= 0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ]
    )
    return train_tf, eval_tf

def get_dataloaders(root="./data_cache", batch_size=32, val_size=10_000, seed=912, augment = True, num_workers= 4, split_path="outputs/split.npz"):
    train_tf, eval_tf = build_transforms(augment)
    cifar_trainset_with_train_tf = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    cifar_trainset_with_eval_tf = datasets.CIFAR10(root, train=True, download=True, transform=eval_tf)
    cifar_testset = datasets.CIFAR10(root, train=False, download=True, transform=eval_tf)
    total_samples = len(cifar_trainset_with_train_tf)

    perm = np.random.RandomState(seed).permutation(total_samples)
    val_idx, train_idx = perm[:val_size], perm[val_size: ]
    assert len(set(train_idx) & set(val_idx)) == 0

    if split_path:
        os.makedirs(os.path.dirname(split_path) or ".", exist_ok=True)
        np.savez(split_path, train_idx=train_idx, val_idx=val_idx, seed=seed)

    train_ds = Subset(cifar_trainset_with_train_tf, train_idx)
    val_ds = Subset(cifar_trainset_with_eval_tf, val_idx)
    g = torch.Generator().manual_seed(seed)
    common = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=True, persistent_workers=num_workers > 0)
    
    train_loader = DataLoader(train_ds, shuffle=True, generator=g, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(cifar_testset, shuffle=False, **common)
    return train_loader, val_loader, test_loader

def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Đảo Normalize để vẽ ảnh. Nhận [3,H,W] hoặc [B,3,H,W]."""
    mean = torch.tensor(MEAN, device=x.device)
    std = torch.tensor(STD, device=x.device)
    shape = (1, 3, 1, 1) if x.dim() == 4 else (3, 1, 1)
    return (x * std.view(*shape) + mean.view(*shape)).clamp(0, 1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    tr, va, te = get_dataloaders(num_workers=0)
    print("sizes:", len(tr.dataset), len(va.dataset), len(te.dataset))

    x, y = next(iter(tr))
    print("batch:", x.shape, y.dtype)           # [B,3,32,32], torch.int64

    # Phân bố lớp của train và val
    for name, loader in [("train", tr), ("val", va)]:
        ds = loader.dataset
        labels = np.array(ds.dataset.targets)[ds.indices]
        print(name, np.bincount(labels, minlength=10))

    # Tái lập: chạy lại với cùng seed, so sánh chỉ số val
    a = np.load("outputs/split.npz")["val_idx"][:10]
    get_dataloaders(num_workers=0)
    b = np.load("outputs/split.npz")["val_idx"][:10]
    assert (a == b).all(), "Split không tái lập!"

    # Lưới ảnh 8x8
    os.makedirs("outputs/figs", exist_ok=True)
    grid = make_grid(denormalize(x[:64]), nrow=8)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis("off")
    plt.savefig("outputs/figs/samples.png", dpi=150, bbox_inches="tight")
    print("OK")