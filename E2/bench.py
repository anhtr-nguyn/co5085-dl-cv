"""
Đo tốc độ MyMSA vs nn.MultiheadAttention cho từng tokenizer, độc lập với training
(chạy tuần tự trên một GPU, không bị ảnh hưởng bởi DataLoader hay các job song song).

    python bench.py                     # cấu hình baseline config/tinyvit.yaml
    python bench.py --batch-size 256

Ghi ra outputs/bench.md và outputs/bench.json:
  - train step (forward + backward + optimizer.step), ms/iter
  - inference, ảnh/giây
  - peak GPU memory khi train
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
import yaml

from models import MyTinyViT, count_params


def time_it(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def bench_one(params, batch_size, iters, warmup, device):
    torch.manual_seed(0)
    model = MyTinyViT(**params).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)

    def train_step():
        opt.zero_grad(set_to_none=True)
        crit(model(x), y).backward()
        opt.step()

    model.train()
    torch.cuda.reset_peak_memory_stats()
    t_train = time_it(train_step, iters, warmup)
    peak_mem = torch.cuda.max_memory_allocated() / 2**20

    model.eval()
    with torch.no_grad():
        t_infer = time_it(lambda: model(x), iters, warmup)

    return dict(params=count_params(model), num_tokens=model.tokenizer.num_tokens,
                train_ms=1000 * t_train, infer_img_s=batch_size / t_infer,
                peak_mem_mb=peak_mem)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/tinyvit.yaml")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    args = p.parse_args()

    with open(args.config) as f:
        base = yaml.safe_load(f)["model"]["params"]
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    results = []
    for tok in ["patch", "row", "cnn"]:
        for msa in ["my", "torch"]:
            r = bench_one({**base, "tokenizer_type": tok, "msa_type": msa},
                          args.batch_size, args.iters, args.warmup, device)
            r.update(tokenizer=tok, msa=msa)
            results.append(r)
            print(f"{tok:6s} {msa:6s} tokens={r['num_tokens']:3d} params={r['params']:>8,} "
                  f"train {r['train_ms']:.2f} ms/iter  infer {r['infer_img_s']:,.0f} img/s  "
                  f"mem {r['peak_mem_mb']:.0f} MB")

    header = ["Tokenizer", "MSA", "#Tokens", "#Params", "Train (ms/iter)",
              "Inference (img/s)", "Peak mem (MB)"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in results:
        lines.append("| " + " | ".join([
            r["tokenizer"], r["msa"], str(r["num_tokens"]), f"{r['params']:,}",
            f"{r['train_ms']:.2f}", f"{r['infer_img_s']:,.0f}", f"{r['peak_mem_mb']:.0f}"]) + " |")
    note = (f"\nGPU: {torch.cuda.get_device_name()}, batch {args.batch_size}, "
            f"{args.iters} iters sau {args.warmup} warmup, fp32.\n")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/bench.md", "w") as f:
        f.write("\n".join(lines) + "\n" + note)
    with open("outputs/bench.json", "w") as f:
        json.dump(dict(gpu=torch.cuda.get_device_name(), batch_size=args.batch_size,
                       results=results), f, indent=2)
    print("\n" + "\n".join(lines) + note)


if __name__ == "__main__":
    main()
