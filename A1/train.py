import argparse
import json
import math
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from data import NUM_CLASSES, get_dataloaders
from models import build_model, count_params, set_train_mode


def load_config():
    """Đọc yaml, cho phép override từ CLI: --set model.arch=deit_s model.mode=full train.lr=5e-5"""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/a1.yaml")
    p.add_argument("--set", nargs="*", action="extend", default=[], metavar="KEY=VALUE")
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_run_dir(cfg):
    m = cfg["model"]
    name = f"{m['arch']}_{m['mode']}_s{cfg['train']['seed']}"
    stamp = f"lr{float(cfg['train']['lr']):g}_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(cfg["output"]["root"], f"{name}_{stamp}")
    os.makedirs(run_dir)
    return name, run_dir


def build_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # bias, BN/LN (ndim<2) và cls_token/pos_embed không weight decay
        if p.ndim < 2 or name.endswith(("pos_embed", "cls_token")):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def build_scheduler(optimizer, total_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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
    s = f"loss {m['loss']:.4f} acc {m['acc']:.4f} F1 {m['f1']:.4f}"
    return s + (f" top5 {m['top5']:.4f}" if "top5" in m else "")


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, amp_dtype,
                    grad_clip=0.0):
    set_train_mode(model)                        # phần bị freeze giữ ở eval()
    params = [p for p in model.parameters() if p.requires_grad]
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    n = 0
    for x, y in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
            loss = criterion(logits, y)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step()
        scheduler.step()                         # theo iteration

        bs = y.size(0)
        loss_sum += loss.detach() * bs
        correct += (logits.detach().argmax(1) == y).sum()
        n += bs
    return dict(loss=loss_sum.item() / n, acc=correct.item() / n)


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_dtype, per_class=False, return_preds=False):
    """return_preds=True -> (metrics, dict(preds, labels, conf, top5))"""
    model.eval()
    loss_sum = torch.zeros((), device=device)
    top5 = torch.zeros((), device=device)
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    n = 0
    out = dict(preds=[], labels=[], conf=[], top5=[])
    for x, y in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
        logits = logits.float()
        loss_sum += criterion(logits, y) * y.size(0)
        prob = logits.softmax(1)
        conf, preds = prob.max(1)
        t5 = prob.topk(5, dim=1).indices
        top5 += (t5 == y[:, None]).any(1).sum()
        update_confmat(cm, preds, y)
        n += y.size(0)
        if return_preds:
            for k, v in zip(("preds", "labels", "conf", "top5"), (preds, y, conf, t5)):
                out[k].append(v.cpu())
    metrics = dict(loss=loss_sum.item() / n, top5=top5.item() / n,
                   **metrics_from_confmat(cm, per_class))
    if return_preds:
        return metrics, {k: torch.cat(v) for k, v in out.items()}
    return metrics


def main():
    cfg = load_config()
    mcfg, dcfg, tcfg = cfg["model"], cfg["data"], cfg["train"]
    set_seed(tcfg["seed"])
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    amp_dtype = {"bf16": torch.bfloat16, "none": None}[tcfg["amp"]]

    tag, run_dir = make_run_dir(cfg)
    ckpt_path = os.path.join(run_dir, "best.pt")
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=dcfg["batch_size"], num_workers=dcfg["num_workers"], seed=dcfg["split_seed"],
        augment=dcfg["augment"], split_path=None)

    model = build_model(mcfg["arch"], mcfg["mode"], drop_path=mcfg["drop_path"])
    model = model.to(device).to(memory_format=torch.channels_last)
    n_train, n_total = count_params(model), count_params(model, trainable_only=False)
    criterion = nn.CrossEntropyLoss(label_smoothing=tcfg["label_smoothing"])
    eval_criterion = nn.CrossEntropyLoss()
    # YAML đọc "1e-3" (không có dấu chấm) thành chuỗi
    tcfg["lr"], tcfg["weight_decay"] = float(tcfg["lr"]), float(tcfg["weight_decay"])
    optimizer = build_optimizer(model, tcfg["lr"], tcfg["weight_decay"])

    epochs = tcfg["epochs"]
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    scheduler = build_scheduler(optimizer, total_steps,
                                max(1, round(tcfg["warmup_frac"] * total_steps)))

    print(f"[{tag}] trainable {n_train:,} / {n_total:,} | {epochs} epochs x "
          f"{steps_per_epoch} steps | lr {tcfg['lr']} -> {run_dir}", flush=True)
    torch.cuda.reset_peak_memory_stats()

    history, best_acc, best_epoch, best_val = [], -1.0, 0, None
    t_start = time.time()
    train_time = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device,
                             amp_dtype, tcfg["grad_clip"])
        t1 = time.time()                         # .item() ở cuối epoch đã sync GPU
        train_time += t1 - t0
        va = evaluate(model, val_loader, eval_criterion, device, amp_dtype)
        t2 = time.time()

        record = dict(epoch=epoch, lr=optimizer.param_groups[0]["lr"], train_s=t1 - t0,
                      eval_s=t2 - t1, **{f"train_{k}": v for k, v in tr.items()},
                      **{f"val_{k}": v for k, v in va.items()})
        history.append(record)
        if va["acc"] > best_acc:
            best_acc, best_epoch, best_val = va["acc"], epoch, va
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": cfg}, ckpt_path)
        print(f"ep {epoch:3d}/{epochs} | train {t1 - t0:.0f}s eval {t2 - t1:.0f}s | "
              f"train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | val {fmt_metrics(va)}"
              f"{'  *best' if epoch == best_epoch else ''}", flush=True)

    peak_mem = torch.cuda.max_memory_allocated() / 2**20

    # Test đúng một lần, với checkpoint có val acc cao nhất (không chạy khi dò learning rate)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    val_m, val_out = evaluate(model, val_loader, eval_criterion, device, amp_dtype,
                              per_class=True, return_preds=True)
    torch.save(val_out, os.path.join(run_dir, "val_preds.pt"))
    test_m = None
    if tcfg["eval_test"]:
        test_m, test_out = evaluate(model, test_loader, eval_criterion, device, amp_dtype,
                                    per_class=True, return_preds=True)
        torch.save(test_out, os.path.join(run_dir, "test_preds.pt"))
        with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
            json.dump(test_m, f)

    summary = dict(tag=tag, config=cfg, params_trainable=n_train, params_total=n_total,
                   best_epoch=best_epoch, best_val_acc=best_acc,
                   best_val_metrics={k: v for k, v in val_m.items() if k != "per_class"},
                   train_time_s=train_time, total_time_s=time.time() - t_start,
                   time_per_epoch_s=train_time / epochs, peak_mem_mb=peak_mem,
                   history=history)
    with open(os.path.join(run_dir, "log.json"), "w") as f:
        json.dump(summary, f, indent=2)

    test_s = f"test {fmt_metrics(test_m)}" if test_m else "test skipped"
    print(f"[{tag}] best @ epoch {best_epoch} | val {fmt_metrics(val_m)} | "
          f"{test_s} | train {train_time / 60:.1f} min | "
          f"peak mem {peak_mem:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
