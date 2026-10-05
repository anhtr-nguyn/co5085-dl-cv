import argparse
import json
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from data import get_dataloaders
from models import build_model, count_params
from utils import set_seed

NUM_CLASSES = 10


def load_config():
    """Đọc YAML config; cho phép ghi đè nhanh từ CLI: --set train.lr=0.01 train.seed=1"""
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="e.g. config/cnn.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override config entries, e.g. train.seed=1 data.augment=true")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for item in args.set:
        key, value = item.split("=", 1)
        *parents, leaf = key.split(".")
        node = cfg
        for k in parents:
            node = node[k]
        if leaf not in node:
            raise KeyError(f"Unknown config key '{key}'")
        node[leaf] = yaml.safe_load(value)
    return cfg


def make_run_dir(cfg):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = (f"{cfg['model']['name']}_aug{int(cfg['data']['augment'])}"
            f"_s{cfg['train']['seed']}_{stamp}")
    run_dir = os.path.join(cfg["output"]["root"], name)
    os.makedirs(run_dir)  # không exist_ok: không bao giờ ghi đè run cũ
    return name, run_dir


def update_confmat(cm, preds, y):
    # cm[i, j] = số mẫu có nhãn thật i, dự đoán j
    cm += torch.bincount(y * NUM_CLASSES + preds,
                         minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES)


def metrics_from_confmat(cm, per_class=False):
    cm = cm.double().cpu()
    tp = cm.diag()
    precision = tp / cm.sum(0).clamp(min=1)
    recall = tp / cm.sum(1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    out = dict(acc=(tp.sum() / cm.sum()).item(), precision=precision.mean().item(),
               recall=recall.mean().item(), f1=f1.mean().item())
    if per_class:
        out["per_class"] = dict(precision=precision.tolist(), recall=recall.tolist(),
                                f1=f1.tolist(), confusion_matrix=cm.long().tolist())
    return out


def fmt_metrics(m):
    return (f"loss {m['loss']:.4f} acc {m['acc']:.4f} P {m['precision']:.4f} "
            f"R {m['recall']:.4f} F1 {m['f1']:.4f}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    # Cộng dồn trên GPU, chỉ .item() một lần cuối epoch (tránh sync mỗi step)
    loss_sum = torch.zeros((), device=device)
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    n = 0
    for x, y in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        loss_sum += loss.detach() * bs
        update_confmat(cm, logits.detach().argmax(1), y)
        n += bs
    return dict(loss=loss_sum.item() / n, **metrics_from_confmat(cm))


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_preds=False, per_class=False):
    """return_preds=True -> (metrics, preds, labels, conf), conf = softmax prob của lớp dự đoán"""
    model.eval()
    loss_sum = torch.zeros((), device=device)
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    n = 0
    all_preds, all_labels, all_conf = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += criterion(logits, y) * y.size(0)
        conf, preds = logits.softmax(1).max(1)
        update_confmat(cm, preds, y)
        n += y.size(0)
        if return_preds:
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            all_conf.append(conf.cpu())
    metrics = dict(loss=loss_sum.item() / n, **metrics_from_confmat(cm, per_class))
    if return_preds:
        return metrics, torch.cat(all_preds), torch.cat(all_labels), torch.cat(all_conf)
    return metrics


def main():
    cfg = load_config()
    mcfg, dcfg, tcfg = cfg["model"], cfg["data"], cfg["train"]
    set_seed(tcfg["seed"])
    device = torch.device("cuda")

    tag, run_dir = make_run_dir(cfg)
    ckpt_path = os.path.join(run_dir, "best.pt")
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)  # config đã áp override, để tái lập run

    # split_path=None: không ghi lại file split (data.py đã lưu rồi),
    # tránh 2 process chạy song song cùng ghi một file
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=dcfg["batch_size"], val_size=dcfg["val_size"], seed=dcfg["split_seed"],
        num_workers=dcfg["num_workers"], augment=dcfg["augment"], split_path=None)

    model = build_model(mcfg["name"], **(mcfg.get("params") or {})).to(device)
    n_params = count_params(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"],
                                 weight_decay=tcfg["weight_decay"])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg["epochs"])
                 if tcfg["sched"] == "cosine" else None)

    print(f"[{tag}] device={device} params={n_params:,} -> {run_dir}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history, best_acc, best_epoch, best_val = [], 0.0, 0, None
    t_start = time.time()
    epochs = tcfg["epochs"]

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device)
        do_eval = epoch % tcfg["eval_every"] == 0 or epoch == epochs
        va = evaluate(model, val_loader, criterion, device, per_class=True) if do_eval else None
        lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()
        dt = time.time() - t0

        record = dict(epoch=epoch, lr=lr, time_s=dt,
                      **{f"train_{k}": v for k, v in tr.items()})
        if va is not None:
            record.update({f"val_{k}": v for k, v in va.items()})
        history.append(record)

        if va is not None and va["acc"] > best_acc:
            best_acc, best_epoch, best_val = va["acc"], epoch, va
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_metrics": va, "config": cfg}, ckpt_path)

        print(f"ep {epoch:3d}/{epochs} | lr {lr:.2e} | {dt:.1f}s\n"
              f"    train | {fmt_metrics(tr)}")
        if va is not None:
            print(f"    val   | {fmt_metrics(va)}{'  *best' if epoch == best_epoch else ''}")

    total_time = time.time() - t_start
    peak_mem = (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None

    # Dự đoán của best model trên val, dùng cho confusion matrix / error samples
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    _, preds, labels, conf = evaluate(model, val_loader, criterion, device, return_preds=True)
    torch.save({"preds": preds, "labels": labels, "conf": conf},
               os.path.join(run_dir, "val_preds.pt"))

    summary = dict(tag=tag, config=cfg, params=n_params, best_epoch=best_epoch,
                   best_val_acc=best_acc, best_val_metrics=best_val,
                   train_time_s=total_time, peak_mem_mb=peak_mem, history=history)
    with open(os.path.join(run_dir, "log.json"), "w") as f:
        json.dump(summary, f, indent=2)

    from visualize import visualize_run
    visualize_run(run_dir)

    print(f"[{tag}] best @ epoch {best_epoch} | {fmt_metrics(best_val)} | "
          f"{total_time:.0f}s | run dir: {run_dir}")


if __name__ == "__main__":
    main()
