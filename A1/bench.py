"""
Đo tốc độ của ResNet-50 / DeiT-S ở từng chế độ, độc lập với DataLoader
(tuần tự trên một GPU, batch ảnh ngẫu nhiên 3x224x224 đã nằm sẵn trên GPU, bf16 như khi train).

    python bench.py

Ghi ra outputs/bench.md và outputs/bench.json:
  - train step (forward + backward + AdamW step), ms/batch, theo chế độ freeze
  - inference, ảnh/giây (giống nhau cho mọi chế độ của cùng kiến trúc)
  - peak GPU memory của bước train
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn

from models import ARCHS, MODES, build_model, count_params, set_train_mode


def time_it(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def bench_one(arch, mode, batch_size, iters, warmup, device):
    torch.manual_seed(0)
    model = build_model(arch, mode).to(device).to(memory_format=torch.channels_last)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(batch_size, 3, 224, 224, device=device).contiguous(
        memory_format=torch.channels_last)
    y = torch.randint(0, 101, (batch_size,), device=device)

    def train_step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = crit(model(x), y)
        loss.backward()
        opt.step()

    set_train_mode(model)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t_train = time_it(train_step, iters, warmup)
    peak = torch.cuda.max_memory_allocated() / 2**20

    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        t_inf = time_it(lambda: model(x), iters, warmup)
    return dict(arch=arch, mode=mode, trainable=count_params(model), train_ms=1000 * t_train,
                train_img_s=batch_size / t_train, infer_img_s=batch_size / t_inf,
                peak_mem_mb=peak)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")

    results = []
    for arch in ARCHS:
        for mode in ["head", "partial", "full"]:   # scratch có cùng chi phí với full
            r = bench_one(arch, mode, args.batch_size, args.iters, args.warmup, device)
            results.append(r)
            print(f"{arch:8s} {mode:8s} trainable {r['trainable']:>11,} train {r['train_ms']:.1f} "
                  f"ms ({r['train_img_s']:,.0f} img/s) infer {r['infer_img_s']:,.0f} img/s "
                  f"mem {r['peak_mem_mb']:.0f} MB", flush=True)

    header = ["Model", "Chế độ", "Trainable", "Train (ms/batch)", "Train (ảnh/s)",
              "Inference (ảnh/s)", "Peak mem train (MB)"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in results:
        lines.append("| " + " | ".join([
            r["arch"], r["mode"], f"{r['trainable']:,}", f"{r['train_ms']:.1f}",
            f"{r['train_img_s']:,.0f}", f"{r['infer_img_s']:,.0f}", f"{r['peak_mem_mb']:.0f}"])
            + " |")
    note = (f"\nGPU: {torch.cuda.get_device_name()}, batch {args.batch_size}, bf16 autocast, "
            f"{args.iters} iters sau {args.warmup} warmup. From-scratch có cùng chi phí với "
            f"full fine-tune.\n")
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/bench.md", "w") as f:
        f.write("\n".join(lines) + "\n" + note)
    with open("outputs/bench.json", "w") as f:
        json.dump(dict(gpu=torch.cuda.get_device_name(), batch_size=args.batch_size,
                       results=results), f, indent=2)
    print("\n" + "\n".join(lines) + note)


if __name__ == "__main__":
    main()
